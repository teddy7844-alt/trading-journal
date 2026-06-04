# SMC Reversal Engine — Project Overview

A portable summary of the `smc_reversal_engine.py` work so it can be read on
any device (GitHub mobile, etc.) or dropped into a Claude Project as knowledge.

Branch: `claude/smc-reversal-engine-MY1DN`

---

## What it is, in one sentence

A robot that **backtests one specific trade setup** on historical price data:
*"Buy when cheap price gets a stop-hunt fake-out followed by a strong bounce
(optionally during big-player hours), risk a little below the trap, aim for 3×
the risk, and score how often that would've worked."*

It only takes **long (buy)** trades, and only when a full checklist lines up.

---

## The trade it hunts for

All of these must be true on the same candle before it buys:

1. **Cheap price (discount zone)** — on the big-picture timeframe it finds the
   recent high/low, takes the midpoint, and only buys in the bottom half.
2. **A trap (liquidity sweep)** — a quiet "inside bar" gets undercut, grabbing
   other traders' stop-losses before reversing.
3. **A hard snap-back (Fair Value Gap)** — price jumps up so fast it leaves a
   gap, confirming momentum.
4. **(time engine only) Right clock** — inside an institutional "macro window"
   (e.g. 09:50–10:10 AM New York time).

**Entry** at the inside-bar high (only if price actually trades up to it).
**Stop** just below the trap low. **Target** = risk × `rr_ratio` (default 3 →
a win is +3R, a loss is −1R). Stale setups expire after ~20 candles.

---

## What it outputs

A scorecard: total trades, win rate, net return in "R", and worst losing streak.

---

## The two engines

| Class | What it adds |
|---|---|
| `SMCAdvancedReversalEngine` | Core setup: cheap price + trap + snap-back. |
| `InstitutionalTimeAndStructureEngine` | Same setup **plus** timezone-aware ICT macro windows and optional Fibonacci time-zone projections. |

Both share `build_price_structure()` and return `(df, metrics)`.

---

## Bugs fixed vs. the original prototype

| # | Bug | Fix |
|---|---|---|
| 1 | HTF **lookahead** (used the current, unclosed HTF bar) | `shift(1)` → only the last *closed* HTF bar |
| 2 | Exact-index `join()` → HTF column all-NaN intraday (**zero trades**) | `pd.merge_asof(direction="backward")` |
| 3 | Filled at a price the candle never reached | require `high >= trigger`; fill at `max(open, trigger)` |
| 4 | No exit check on the entry candle | same-bar stop/TP check |
| 5 | Dead auto-cancel branch (`low < itself`) | cancel only on later bars |
| 6 | Setups never expired | `setup_expiry` (default 20 bars) |
| 7 | Macro time filter defined but never used | wired into the entry gate |
| 8 | Metrics printed, not returned | returns a `metrics` dict |
| 9 | `O(n²)` per-iteration `.shift()` | precomputed numpy arrays |

---

## How to test

```bash
pip install -r requirements.txt
pytest -q          # 11 tests -> "11 passed"
```

Tests use **hand-built candles with known answers** (not random data), so a
broken change turns a test red. They cover: pattern detection, the no-lookahead
guarantee, intraday HTF fill, a full +3R win and a −1R loss, price-confirmed
entries, setup expiry, the macro-window filter, and trade invariants
(`stop < entry < target`).

### Running on real data (needs internet)

```python
import yfinance as yf
from smc_reversal_engine import SMCAdvancedReversalEngine
htf = yf.download("BTC-USD", interval="1d", period="2y")
ltf = yf.download("BTC-USD", interval="1h", period="2y")
df, metrics = SMCAdvancedReversalEngine().generate_signals(ltf, htf)
```

---

## Honest caveats

Backtest only. Assumes clean fills, ignores fees/slippage, and past performance
doesn't predict the future. A research/learning tool, not a money printer.

---

## Files

- `smc_reversal_engine.py` — the two engines + shared helpers
- `test_smc_reversal_engine.py` — pytest suite (11 tests)
- `requirements.txt` — pandas, numpy, pytest
