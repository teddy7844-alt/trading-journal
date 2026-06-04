"""
Tests for smc_reversal_engine.

Strategy: instead of random data (where you can't predict the answer), we
hand-build tiny candle sequences that contain exactly one known setup, then
assert the engine reacts the way the rules say it should.

The canonical long setup, bar by bar (LTF positional index):
    idx 0: wide "mother" candle
    idx 1: inside bar (sits fully inside the mother)
    idx 2: liquidity purge (dips below the inside bar low)  -> arms the setup
              trigger = inside-bar high, stop = purge low
    idx 3-4: price recovers without breaking the stop
    idx 5: bullish FVG + price reaches the trigger           -> ENTRY
    idx 6: resolves into the take-profit or the stop

Run with:  pytest -q
"""

import numpy as np
import pandas as pd
import pytest

from smc_reversal_engine import (
    SMCAdvancedReversalEngine,
    InstitutionalTimeAndStructureEngine,
    build_price_structure,
    max_consecutive_losses,
)


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def make_ltf(rows, start="2025-01-03 09:30", freq="5min", tz=None):
    """Build an OHLC frame from a list of (Open, High, Low, Close) tuples."""
    idx = pd.date_range(start, periods=len(rows), freq=freq, tz=tz)
    return pd.DataFrame(rows, index=idx, columns=["Open", "High", "Low", "Close"])


def flat_htf(dates, high=200, low=0, tz=None):
    """A daily HTF frame with a constant range -> equilibrium is midpoint."""
    idx = pd.DatetimeIndex(pd.to_datetime(dates))
    if tz is not None:
        idx = idx.tz_localize(tz)
    return pd.DataFrame(
        {"Open": (high + low) / 2, "High": high, "Low": low, "Close": (high + low) / 2},
        index=idx,
    )


# The shared structural candles for idx 0..5 (mother, inside, purge, recover, entry).
# eq broadcast = 100, so every Close below 100 is "discount".
BASE_ROWS = [
    (20, 30, 10, 20),   # 0 mother
    (21, 28, 12, 22),   # 1 inside bar  (H28<H30, L12>L10)
    (20, 24,  8, 11),   # 2 purge       (low 8 < inside low 12) -> trigger=28, stop=8
    (12, 18, 11, 17),   # 3 recover     (low 11 > stop 8)
    (20, 27, 19, 26),   # 4 recover     (FVG not yet: low19 !> high24)
    (30, 40, 33, 39),   # 5 ENTRY       (FVG: low33>high[3]=18, close[4]=26>18; high40>=28)
]
HTF_DATES = ["2025-01-01", "2025-01-02", "2025-01-03"]


# --------------------------------------------------------------------------- #
# Feature-builder tests
# --------------------------------------------------------------------------- #
def test_structure_flags_are_detected():
    df = build_price_structure(make_ltf(BASE_ROWS), flat_htf(HTF_DATES), htf_lookback=1)
    assert bool(df["Is_Inside_Bar"].iloc[2]) is True      # inside bar at idx1, flagged on idx2
    assert bool(df["Is_Liquidity_Purge"].iloc[2]) is True  # the sweep
    assert bool(df["Is_Bullish_FVG"].iloc[5]) is True      # the displacement
    assert bool(df["Is_Discount_Zone"].iloc[5]) is True    # close 39 < eq 100


def test_no_lookahead_uses_previous_closed_htf_bar():
    """HTF equilibrium seen on day N must come from day N-1, never day N."""
    # eq per day = 5, 20, 40 ; after shift each LTF day sees the PRIOR day's eq.
    htf = pd.DataFrame(
        {"Open": [5, 20, 40], "High": [10, 30, 50], "Low": [0, 10, 30],
         "Close": [5, 20, 40]},
        index=pd.to_datetime(["2025-01-01", "2025-01-02", "2025-01-03"]),
    )
    ltf = pd.concat([
        make_ltf(BASE_ROWS, start="2025-01-02 09:30"),
        make_ltf(BASE_ROWS, start="2025-01-03 09:30"),
    ])
    df = build_price_structure(ltf, htf, htf_lookback=1)
    day2 = df.loc["2025-01-02", "HTF_Equilibrium"].unique()
    day3 = df.loc["2025-01-03", "HTF_Equilibrium"].unique()
    assert day2.tolist() == [5]    # day-2 bars see day-1's eq (=5), not their own (=20)
    assert day3.tolist() == [20]   # day-3 bars see day-2's eq (=20), not their own (=40)


def test_merge_asof_fills_intraday_unlike_exact_join():
    """The fix: intraday bars get an HTF value even though timestamps differ."""
    df = build_price_structure(make_ltf(BASE_ROWS), flat_htf(HTF_DATES), htf_lookback=1)
    assert df["HTF_Equilibrium"].notna().all()  # a plain .join() would be all-NaN here


# --------------------------------------------------------------------------- #
# Full-trade tests
# --------------------------------------------------------------------------- #
def test_winning_trade_hits_take_profit():
    rows = BASE_ROWS + [
        (50, 99, 45, 98),   # 6 high 99 >= TP(30+22*3=96) -> WIN
        (60, 70, 55, 65),   # 7 filler
    ]
    eng = SMCAdvancedReversalEngine(rr_ratio=3.0, htf_lookback=1)
    df, m = eng.generate_signals(make_ltf(rows), flat_htf(HTF_DATES))

    assert m["total_trades"] == 1
    assert m["wins"] == 1 and m["losses"] == 0
    assert m["net_return_R"] == pytest.approx(3.0)

    entry = df[df["Signal"] == 1].iloc[0]
    assert entry["Entry_Price"] == pytest.approx(30.0)   # max(open30, trigger28)
    assert entry["Stop_Loss"] == pytest.approx(8.0)
    assert entry["Take_Profit"] == pytest.approx(96.0)


def test_losing_trade_hits_stop():
    rows = BASE_ROWS + [
        (40, 42, 5, 10),    # 6 low 5 <= stop 8 -> LOSS
        (12, 15, 10, 14),   # 7 filler
    ]
    eng = SMCAdvancedReversalEngine(rr_ratio=3.0, htf_lookback=1)
    df, m = eng.generate_signals(make_ltf(rows), flat_htf(HTF_DATES))

    assert m["total_trades"] == 1
    assert m["losses"] == 1 and m["wins"] == 0
    assert m["net_return_R"] == pytest.approx(-1.0)


def test_no_trade_when_price_never_reaches_trigger():
    """Entry must be price-confirmed: a weak idx5 that never tags the trigger."""
    rows = BASE_ROWS[:5] + [
        (20, 26, 19, 25),   # 5 FVG-ish but high 26 < trigger 28 -> no fill
        (24, 27, 22, 26),   # 6 still below trigger
        (24, 27, 22, 26),   # 7
    ]
    eng = SMCAdvancedReversalEngine(rr_ratio=3.0, htf_lookback=1)
    _, m = eng.generate_signals(make_ltf(rows), flat_htf(HTF_DATES))
    assert m["total_trades"] == 0


def test_setup_expires_and_blocks_late_entry():
    """A valid FVG that arrives after the expiry window should not trigger."""
    filler = [(12, 18, 11, 17)] * 30          # long calm stretch, no FVG, stop intact
    rows = BASE_ROWS[:3] + filler + [
        (30, 40, 33, 39),   # FVG + trigger, but ~30 bars after the purge
        (50, 99, 45, 98),
    ]
    eng = SMCAdvancedReversalEngine(rr_ratio=3.0, htf_lookback=1, setup_expiry=20)
    _, m = eng.generate_signals(make_ltf(rows), flat_htf(HTF_DATES))
    assert m["total_trades"] == 0


# --------------------------------------------------------------------------- #
# Time-engine tests
# --------------------------------------------------------------------------- #
def _win_rows():
    return BASE_ROWS + [(50, 99, 45, 98), (60, 70, 55, 65)]


def test_macro_window_allows_trade_inside_window():
    # idx5 entry lands at 09:55 NY (start 09:30 + 5*5min) -> inside 09:50-10:10.
    ltf = make_ltf(_win_rows(), start="2025-01-03 09:30", tz="America/New_York")
    htf = flat_htf(HTF_DATES, tz="America/New_York")
    eng = InstitutionalTimeAndStructureEngine(
        rr_ratio=3.0, htf_lookback=1, macro_windows=[(590, 610)]
    )
    _, m = eng.run_backtest(ltf, htf)
    assert m["total_trades"] == 1 and m["wins"] == 1


def test_macro_window_blocks_trade_outside_window():
    ltf = make_ltf(_win_rows(), start="2025-01-03 09:30", tz="America/New_York")
    htf = flat_htf(HTF_DATES, tz="America/New_York")
    eng = InstitutionalTimeAndStructureEngine(
        rr_ratio=3.0, htf_lookback=1, macro_windows=[(0, 1)]  # 00:00-00:01 only
    )
    _, m = eng.run_backtest(ltf, htf)
    assert m["total_trades"] == 0


# --------------------------------------------------------------------------- #
# Pure-function tests
# --------------------------------------------------------------------------- #
def test_max_consecutive_losses():
    seq = [{"Type": t, "Return": 0} for t in
           ["Loss", "Loss", "Win", "Loss", "Loss", "Loss", "Win"]]
    assert max_consecutive_losses(seq) == 3
    assert max_consecutive_losses([]) == 0
    assert max_consecutive_losses([{"Type": "Win", "Return": 3}]) == 0


def test_trade_invariants_hold_on_random_data():
    """Across noisy data, every logged trade must have stop < entry < target."""
    rng = np.random.default_rng(0)
    idx = pd.date_range("2025-01-01", periods=4000, freq="5min")
    price = 100 + np.cumsum(rng.normal(0, 0.3, len(idx)))
    ltf = pd.DataFrame({
        "Open": price,
        "High": price + np.abs(rng.normal(0.4, 0.2, len(idx))),
        "Low": price - np.abs(rng.normal(0.4, 0.2, len(idx))),
        "Close": price + rng.normal(0, 0.1, len(idx)),
    }, index=idx)
    htf = ltf.resample("1D").agg(
        {"Open": "first", "High": "max", "Low": "min", "Close": "last"}).dropna()

    df, _ = SMCAdvancedReversalEngine(htf_lookback=5).generate_signals(ltf, htf)
    trades = df[df["Signal"] == 1]
    assert (trades["Stop_Loss"] < trades["Entry_Price"]).all()
    assert (trades["Entry_Price"] < trades["Take_Profit"]).all()


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v"]))
