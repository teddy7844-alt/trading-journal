"""
SMC Advanced Reversal Engine
============================

A multi-timeframe Smart Money Concepts backtester that codifies the
"Inside Bar Failure" methodology documented in trading_flowcharts_updated_3.html:

    HTF bias (discount/premium)  ->  liquidity sweep of an inside bar
        ->  lower-timeframe FVG displacement  ->  reversal entry
            ->  stop beyond the swept wick  ->  fixed R:R target

This is a rewrite of the original prototype that fixes:
  * HTF lookahead bias (only *closed* HTF data is used, via shift + merge_asof)
  * the exact-index join that silently produced all-NaN HTF columns
  * unrealistic entry fills (price must actually reach the trigger)
  * missing same-bar exit handling on the entry candle
  * setups never expiring
  * the macro-window time filter never being wired in
  * O(n^2) per-iteration .shift() recomputation
  * metrics being printed instead of returned
"""

import pandas as pd
import numpy as np


def build_price_structure(df_ltf, df_htf, htf_lookback=50):
    """
    Shared SMC feature builder used by every engine in this module.

    Returns a copy of the LTF frame with HTF bias and LTF pattern flags:
    HTF_Equilibrium, Is_Discount_Zone, Is_Inside_Bar, Is_Liquidity_Purge,
    Is_Bullish_FVG. Both inputs need Datetime indices and OHLC columns.
    """
    # 1. HTF DEALING RANGE & BIAS --------------------------------------
    df_htf = df_htf.copy()
    df_htf['Highest_HTF_High'] = df_htf['High'].rolling(htf_lookback).max()
    df_htf['Lowest_HTF_Low'] = df_htf['Low'].rolling(htf_lookback).min()
    equilibrium = (df_htf['Highest_HTF_High'] + df_htf['Lowest_HTF_Low']) / 2

    # Shift by one HTF bar so an LTF bar only ever sees the *previous,
    # fully closed* HTF candle -- this is what removes the lookahead bias.
    df_htf['HTF_Equilibrium'] = equilibrium.shift(1)

    df = df_ltf.copy()

    # merge_asof aligns each LTF timestamp to the most recent HTF row
    # (direction='backward'), which is the correct way to broadcast a
    # coarse timeframe onto a fine one. A plain join() only matches on
    # exact timestamps and would leave HTF_Equilibrium all-NaN intraday.
    htf_eq = df_htf[['HTF_Equilibrium']].dropna()
    df = pd.merge_asof(
        df.sort_index(),
        htf_eq.sort_index(),
        left_index=True,
        right_index=True,
        direction='backward',
    )

    df['Is_Discount_Zone'] = df['Close'] < df['HTF_Equilibrium']

    # 2. INSIDE BAR LIQUIDITY PURGE ------------------------------------
    df['Is_Inside_Bar'] = (
        (df['High'].shift(1) < df['High'].shift(2))
        & (df['Low'].shift(1) > df['Low'].shift(2))
    )
    df['Is_Liquidity_Purge'] = df['Is_Inside_Bar'] & (df['Low'] < df['Low'].shift(1))

    # 3. LTF BULLISH FVG DISPLACEMENT ----------------------------------
    # 3-candle bullish gap: current low above the high from two bars back,
    # with a displacing middle candle that closed above that same high.
    df['Is_Bullish_FVG'] = (
        (df['Low'] > df['High'].shift(2))
        & (df['Close'].shift(1) > df['High'].shift(2))
    )

    return df


def max_consecutive_losses(results):
    """Longest run of consecutive losing trades in a results list/DataFrame."""
    if len(results) == 0:
        return 0
    res_df = results if isinstance(results, pd.DataFrame) else pd.DataFrame(results)
    is_loss = res_df['Type'] == 'Loss'
    if not is_loss.any():
        return 0
    # (~is_loss).cumsum() is constant within a loss run, so grouping losses by
    # that key and taking the largest group size yields the longest streak.
    return int((~is_loss).cumsum()[is_loss].value_counts().max())


class SMCAdvancedReversalEngine:
    def __init__(self, rr_ratio=3.0, htf_lookback=50, setup_expiry=20,
                 macro_window=None):
        """
        Initialize the Advanced Smart Money Concepts Reversal Engine.

        :param rr_ratio:    Risk-to-Reward multiplier for the Take Profit.
        :param htf_lookback: Lookback window for the HTF Dealing Range/Equilibrium.
        :param setup_expiry: Max number of LTF bars a sweep setup stays armed
                             before it is abandoned (0/None = never expire).
        :param macro_window: Optional (start, end) "HH:MM" tuple restricting
                             entries to an algorithmic macro window, e.g.
                             ("09:50", "10:10"). None disables the filter.
        """
        self.rr_ratio = rr_ratio
        self.htf_lookback = htf_lookback
        self.setup_expiry = setup_expiry
        self.macro_window = macro_window

    # ------------------------------------------------------------------ #
    # Feature engineering
    # ------------------------------------------------------------------ #
    def _build_features(self, df_ltf, df_htf):
        """Attach HTF bias and LTF pattern flags to a copy of the LTF frame."""
        df = build_price_structure(df_ltf, df_htf, self.htf_lookback)

        # MACRO TIME WINDOW (optional) ----------------------------------
        if self.macro_window is not None:
            start, end = self.macro_window
            idx = pd.to_datetime(df.index)
            t = idx.strftime('%H:%M')  # zero-padded -> lexicographic == chronological
            df['In_Macro_Window'] = (t >= start) & (t <= end)
        else:
            df['In_Macro_Window'] = True

        return df

    # ------------------------------------------------------------------ #
    # Backtest loop
    # ------------------------------------------------------------------ #
    def generate_signals(self, df_ltf, df_htf):
        """
        Process multi-timeframe data to find sweeps inside the HTF discount
        zone confirmed by an LTF Fair Value Gap displacement.

        Both dataframes must have Datetime indices and 'Open', 'High', 'Low',
        'Close' columns. Returns (df, metrics) where df carries the signal
        columns and metrics is a summary dict.
        """
        df = self._build_features(df_ltf, df_htf)

        df['Signal'] = 0
        df['Entry_Price'] = np.nan
        df['Stop_Loss'] = np.nan
        df['Take_Profit'] = np.nan

        # Vectorize the columns we touch every iteration (avoids O(n^2) .shift).
        high = df['High'].to_numpy()
        low = df['Low'].to_numpy()
        open_ = df['Open'].to_numpy()
        ib_high = df['High'].shift(1).to_numpy()          # inside-bar high
        is_purge = df['Is_Liquidity_Purge'].fillna(False).to_numpy()
        is_discount = df['Is_Discount_Zone'].fillna(False).to_numpy()
        is_fvg = df['Is_Bullish_FVG'].fillna(False).to_numpy()
        in_macro = df['In_Macro_Window'].fillna(False).to_numpy()

        pattern_active = False
        armed_bar = -1
        trigger_level = np.nan
        stop_loss_level = np.nan
        take_profit_level = np.nan
        entry_price = np.nan
        in_position = False

        results = []

        for i in range(2, len(df)):
            # --- A. Manage an open position (exits first) ---------------
            if in_position:
                hit_stop = low[i] <= stop_loss_level
                hit_tp = high[i] >= take_profit_level
                if hit_stop and hit_tp:
                    # Both touched in one bar: assume the stop came first
                    # (conservative -- we can't see intrabar order).
                    results.append({'Type': 'Loss', 'Return': -1.0,
                                    'Timestamp': df.index[i]})
                    in_position = False
                    continue
                if hit_stop:
                    results.append({'Type': 'Loss', 'Return': -1.0,
                                    'Timestamp': df.index[i]})
                    in_position = False
                    continue
                if hit_tp:
                    results.append({'Type': 'Win', 'Return': self.rr_ratio,
                                    'Timestamp': df.index[i]})
                    in_position = False
                    continue

            # --- B. Track setup lifecycle ------------------------------
            if is_purge[i]:
                pattern_active = True
                armed_bar = i
                trigger_level = ib_high[i]   # inside-bar high
                stop_loss_level = low[i]      # purge low (swept wick)

            # Cancel if a later bar breaks below the structural floor.
            if pattern_active and i > armed_bar and low[i] < stop_loss_level:
                pattern_active = False

            # Cancel if the setup has gone stale.
            if (pattern_active and self.setup_expiry
                    and i - armed_bar > self.setup_expiry):
                pattern_active = False

            # --- C. Entry trigger --------------------------------------
            if (pattern_active and not in_position
                    and is_discount[i] and is_fvg[i] and in_macro[i]
                    and high[i] >= trigger_level):       # price must reach it
                in_position = True
                # Stop-entry fill realism: if the bar gapped above the trigger,
                # we fill at the open; otherwise at the trigger level.
                entry_price = max(open_[i], trigger_level)
                risk = entry_price - stop_loss_level
                take_profit_level = entry_price + risk * self.rr_ratio

                df.at[df.index[i], 'Signal'] = 1
                df.at[df.index[i], 'Entry_Price'] = entry_price
                df.at[df.index[i], 'Stop_Loss'] = stop_loss_level
                df.at[df.index[i], 'Take_Profit'] = take_profit_level

                # Same-bar exit check on the entry candle itself.
                if low[i] <= stop_loss_level:
                    results.append({'Type': 'Loss', 'Return': -1.0,
                                    'Timestamp': df.index[i]})
                    in_position = False
                elif high[i] >= take_profit_level:
                    results.append({'Type': 'Win', 'Return': self.rr_ratio,
                                    'Timestamp': df.index[i]})
                    in_position = False

                pattern_active = False

        metrics = self._summarize(results, open_trade=in_position)
        return df, metrics

    # ------------------------------------------------------------------ #
    # Metrics
    # ------------------------------------------------------------------ #
    def _summarize(self, results, open_trade=False):
        """Build and print a performance summary; return it as a dict."""
        metrics = {
            'total_trades': len(results),
            'wins': 0,
            'losses': 0,
            'win_rate': 0.0,
            'net_return_R': 0.0,
            'open_trade_at_end': bool(open_trade),
        }

        if results:
            res_df = pd.DataFrame(results)
            wins = int((res_df['Type'] == 'Win').sum())
            metrics['wins'] = wins
            metrics['losses'] = len(res_df) - wins
            metrics['win_rate'] = wins / len(res_df) * 100
            metrics['net_return_R'] = float(res_df['Return'].sum())

            print("--- Strategy Performance Summary ---")
            print(f"Total Trades: {metrics['total_trades']}")
            print(f"Win Rate: {metrics['win_rate']:.2f}%")
            print(f"Net Return (R-Multiple): {metrics['net_return_R']:.2f}R")
            if open_trade:
                print("Note: one position was still open at the end of the window.")
        else:
            print("No valid structural trades executed in backtest window.")

        return metrics


class InstitutionalTimeAndStructureEngine:
    """
    Unified SMC price-action + temporal-execution engine.

    Layers two time-based filters on top of the shared price structure:
      * intraday ICT macro windows (timezone-aware, EST/EDT), and
      * optional Fibonacci time-zone projections from a seed structural wave.

    Carries the same correctness fixes as SMCAdvancedReversalEngine: no HTF
    lookahead, merge_asof MTF broadcasting, price-confirmed fills, same-bar
    exits, and setup expiry.
    """

    def __init__(self, rr_ratio=3.0, htf_lookback=50, setup_expiry=20,
                 macro_windows=None, source_tz='UTC', market_tz='America/New_York'):
        """
        :param rr_ratio:     Risk-to-Reward multiplier for Take Profits.
        :param htf_lookback: Lookback window for the HTF Dealing Range.
        :param setup_expiry: Bars a sweep setup stays armed (0/None = never).
        :param macro_windows: List of (start_min, end_min) windows in minutes
                              from midnight market-time. Defaults to standard
                              ICT NY AM / Lunch / PM macros.
        :param source_tz:    Timezone to assume for tz-naive input indices.
        :param market_tz:    Timezone the macro windows are expressed in.
        """
        self.rr_ratio = rr_ratio
        self.htf_lookback = htf_lookback
        self.setup_expiry = setup_expiry
        self.source_tz = source_tz
        self.market_tz = market_tz
        # Default to standard ICT Macro windows if none provided.
        self.macro_windows = macro_windows or [
            (590, 610),   # 09:50-10:10 EST (NY AM Macro)
            (710, 730),   # 11:50-12:10 EST (NY Lunch Macro)
            (915, 945),   # 03:15-03:45 EST (NY Close Macro)
        ]

    # ------------------------------------------------------------------ #
    # Time features
    # ------------------------------------------------------------------ #
    def _apply_intraday_time_macros(self, df):
        """Convert the index to market time and tag active macro windows."""
        df = df.copy()
        df.index = pd.to_datetime(df.index)

        if df.index.tz is None:
            df = df.tz_localize(self.source_tz).tz_convert(self.market_tz)
        else:
            df = df.tz_convert(self.market_tz)

        time_minutes = df.index.hour * 60 + df.index.minute
        active = np.zeros(len(df), dtype=bool)
        for start, end in self.macro_windows:
            active |= (time_minutes >= start) & (time_minutes <= end)
        df['Is_Macro_Active'] = active
        return df

    def apply_fibonacci_time_zones(self, df, wave_start_idx, wave_end_idx):
        """
        Project future turning points from the duration of a seed wave.
        Adds a boolean 'Fib_Time_Target' column.
        """
        df = df.copy()
        df['Fib_Time_Target'] = False

        wave_duration = wave_end_idx - wave_start_idx
        for fib in (1, 2, 3, 5, 8, 13, 21, 34, 55):
            target_index = wave_end_idx + int(wave_duration * fib)
            if 0 <= target_index < len(df):
                df.iat[target_index, df.columns.get_loc('Fib_Time_Target')] = True
        return df

    # ------------------------------------------------------------------ #
    # Backtest
    # ------------------------------------------------------------------ #
    def run_backtest(self, df_ltf, df_htf, wave_start_idx=None, wave_end_idx=None):
        """
        Run the integrated backtest: price structure + macro time window +
        optional Fibonacci time alignment. Returns (df, metrics).
        """
        # 1. Pipeline: shared (fixed) price structure, then time layers.
        df = build_price_structure(df_ltf, df_htf, self.htf_lookback)
        df = self._apply_intraday_time_macros(df)

        if wave_start_idx is not None and wave_end_idx is not None:
            df = self.apply_fibonacci_time_zones(df, wave_start_idx, wave_end_idx)
            df['Time_Condition_Met'] = df['Is_Macro_Active'] & df['Fib_Time_Target']
        else:
            df['Time_Condition_Met'] = df['Is_Macro_Active']

        # 2. State-machine execution loop.
        df['Signal'] = 0
        df['Entry_Price'] = np.nan
        df['Stop_Loss'] = np.nan
        df['Take_Profit'] = np.nan

        high = df['High'].to_numpy()
        low = df['Low'].to_numpy()
        open_ = df['Open'].to_numpy()
        ib_high = df['High'].shift(1).to_numpy()
        is_purge = df['Is_Liquidity_Purge'].fillna(False).to_numpy()
        is_discount = df['Is_Discount_Zone'].fillna(False).to_numpy()
        is_fvg = df['Is_Bullish_FVG'].fillna(False).to_numpy()
        time_ok = df['Time_Condition_Met'].fillna(False).to_numpy()

        pattern_active = False
        armed_bar = -1
        trigger_level = stop_loss_level = take_profit_level = entry_price = np.nan
        in_position = False
        results = []

        for i in range(2, len(df)):
            # A. Manage exits first.
            if in_position:
                hit_stop = low[i] <= stop_loss_level
                hit_tp = high[i] >= take_profit_level
                if hit_stop:  # stop assumed first on an ambiguous both-touch bar
                    results.append({'Type': 'Loss', 'Return': -1.0,
                                    'Timestamp': df.index[i]})
                    in_position = False
                    continue
                if hit_tp:
                    results.append({'Type': 'Win', 'Return': self.rr_ratio,
                                    'Timestamp': df.index[i]})
                    in_position = False
                    continue

            # B. Track structural setup state.
            if is_purge[i]:
                pattern_active = True
                armed_bar = i
                trigger_level = ib_high[i]
                stop_loss_level = low[i]

            if pattern_active and i > armed_bar and low[i] < stop_loss_level:
                pattern_active = False
            if (pattern_active and self.setup_expiry
                    and i - armed_bar > self.setup_expiry):
                pattern_active = False

            # C. Integrated price-geometry + time verification.
            if (pattern_active and not in_position
                    and is_discount[i] and is_fvg[i] and time_ok[i]
                    and high[i] >= trigger_level):
                in_position = True
                entry_price = max(open_[i], trigger_level)
                risk = entry_price - stop_loss_level
                take_profit_level = entry_price + risk * self.rr_ratio

                df.at[df.index[i], 'Signal'] = 1
                df.at[df.index[i], 'Entry_Price'] = entry_price
                df.at[df.index[i], 'Stop_Loss'] = stop_loss_level
                df.at[df.index[i], 'Take_Profit'] = take_profit_level

                if low[i] <= stop_loss_level:
                    results.append({'Type': 'Loss', 'Return': -1.0,
                                    'Timestamp': df.index[i]})
                    in_position = False
                elif high[i] >= take_profit_level:
                    results.append({'Type': 'Win', 'Return': self.rr_ratio,
                                    'Timestamp': df.index[i]})
                    in_position = False

                pattern_active = False

        metrics = self._report(results, open_trade=in_position)
        return df, metrics

    # ------------------------------------------------------------------ #
    # Reporting
    # ------------------------------------------------------------------ #
    def _report(self, results, open_trade=False):
        metrics = {
            'total_trades': len(results),
            'wins': 0,
            'losses': 0,
            'win_rate': 0.0,
            'net_return_R': 0.0,
            'max_consecutive_losses': max_consecutive_losses(results),
            'open_trade_at_end': bool(open_trade),
        }

        if results:
            res_df = pd.DataFrame(results)
            wins = int((res_df['Type'] == 'Win').sum())
            metrics['wins'] = wins
            metrics['losses'] = len(res_df) - wins
            metrics['win_rate'] = wins / len(res_df) * 100
            metrics['net_return_R'] = float(res_df['Return'].sum())

            print("\n" + "=" * 40)
            print("  SYSTEM PERFORMANCE METRICS SUMMARY  ")
            print("=" * 40)
            print(f"Total Trades Executed   : {metrics['total_trades']}")
            print(f"Strategy Win Rate       : {metrics['win_rate']:.2f}%")
            print(f"Net Profit (R-Multiple) : {metrics['net_return_R']:.2f}R")
            print(f"Max Consecutive Losses  : {metrics['max_consecutive_losses']}")
            if open_trade:
                print("Open position at window end : 1")
            print("=" * 40 + "\n")
        else:
            print("\n[!] No trades executed within the specified parameters.\n")

        return metrics


# Backwards-compatible alias for the original (misspelled) class name.
SMCAvancedReversalEngine = SMCAdvancedReversalEngine
