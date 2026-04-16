import logging
from datetime import datetime
from functools import reduce

import numpy as np
import talib.abstract as ta
from pandas import DataFrame
from technical import qtpylib

from freqtrade.strategy import IStrategy


logger = logging.getLogger(__name__)


class RLDayTradeStrategy(IStrategy):
    """
    RL Day Trade Strategy for ETH/SOL Futures

    4-Action space (Base4ActionRLEnv):
      0: Neutral, 1: Exit, 2: Long_enter, 3: Short_enter

    Features:
    - S/R Flip detection via fractal-based support/resistance
    - Multi-timeframe analysis (5m, 15m, 1h, 4h)
    - Per-pair leverage (ETH=10x, SOL=5x) and stoploss

    Usage:
      freqtrade backtesting --strategy RLDayTradeStrategy \
        --config user_data/config_daytrade.json \
        --freqaimodel RLDayTrader \
        --timerange 20240101-20260101 --export trades
    """

    minimal_roi = {"0": 0.05, "30": 0.03, "60": 0.01}
    stoploss = -0.03
    use_exit_signal = True
    startup_candle_count: int = 60
    can_short = True
    process_only_new_candles = True

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe = self.freqai.start(dataframe, metadata, self)
        return dataframe

    # ------------------------------------------------------------------
    # Feature Engineering
    # ------------------------------------------------------------------

    def _get_feature_flags(self) -> dict:
        """Get feature flags from config with defaults (all enabled)."""
        return self.freqai_info.get("feature_parameters", 
        {}).get("feature_flags", {})

    def feature_engineering_expand_all(
        self, dataframe: DataFrame, period: int, metadata: dict, **kwargs
    ) -> DataFrame:
        """
        Expanded across: indicator_periods x timeframes x shifted x corr_pairlist.
        Features can be toggled via config feature_flags.expand_all.
        """
        flags = self._get_feature_flags()
        ea = flags.get("expand_all", {})

        if ea.get("rsi", True):
            dataframe["%-rsi-period"] = ta.RSI(dataframe, timeperiod=period)

        if ea.get("adx", True):
            dataframe["%-adx-period"] = ta.ADX(dataframe, timeperiod=period)

        if ea.get("atr_norm", True):
            atr = ta.ATR(dataframe, timeperiod=period)
            dataframe["%-atr_norm-period"] = atr / dataframe["close"]

        if ea.get("rel_vol", True):
            dataframe["%-rel_vol-period"] = (
                dataframe["volume"] / dataframe["volume"].rolling(period).mean()
            )

        return dataframe

    def feature_engineering_expand_basic(
        self, dataframe: DataFrame, metadata: dict, **kwargs
    ) -> DataFrame:
        """
        Expanded across: timeframes x shifted x corr_pairlist (NOT indicator_periods).
        Features can be toggled via config feature_flags.expand_basic.
        """
        flags = self._get_feature_flags()
        eb = flags.get("expand_basic", {})

        # Trend: EMA
        if eb.get("ema_20", True):
            dataframe["%-ema_20"] = ta.EMA(dataframe, timeperiod=20)

        if eb.get("ema_50", True):
            dataframe["%-ema_50"] = ta.EMA(dataframe, timeperiod=50)

        # Bollinger Bands (21, 2.0, Close)
        need_bb = eb.get("bb_percent", True) or eb.get("bb_width", True)
        if need_bb:
            bollinger = qtpylib.bollinger_bands(dataframe["close"], window=21, stds=2.0)
            bb_range = bollinger["upper"] - bollinger["lower"]

            if eb.get("bb_percent", True):
                dataframe["%-bb_percent"] = (
                    (dataframe["close"] - bollinger["lower"]) / bb_range.replace(0, np.nan)
                )
            if eb.get("bb_width", True):
                dataframe["%-bb_width"] = bb_range / bollinger["mid"].replace(0, np.nan)

        # MACD normalized
        need_macd = eb.get("macd_norm", True) or eb.get("macd_hist_norm", True)
        if need_macd:
            macd = ta.MACD(dataframe, fastperiod=12, slowperiod=26, signalperiod=9)

            if eb.get("macd_norm", True):
                dataframe["%-macd_norm"] = macd["macd"] / dataframe["close"]
            if eb.get("macd_hist_norm", True):
                dataframe["%-macd_hist_norm"] = macd["macdhist"] / dataframe["close"]

        # S/R Flip features (6 features)
        if eb.get("sr_features", True):
            dataframe = self._compute_sr_features(dataframe)

        return dataframe

    def feature_engineering_standard(
        self, dataframe: DataFrame, metadata: dict, **kwargs
    ) -> DataFrame:
        """
        Standard features (no expansion).
        Features can be toggled via config feature_flags.standard.
        """
        flags = self._get_feature_flags()
        std = flags.get("standard", {})

        # Required raw OHLCV for RL environment
        if std.get("raw_ohlc", True):
            dataframe["%-raw_close"] = dataframe["close"]
            dataframe["%-raw_open"] = dataframe["open"]
            dataframe["%-raw_high"] = dataframe["high"]
            dataframe["%-raw_low"] = dataframe["low"]

        # Cyclical time encoding
        if std.get("time_cyclical", True):
            hour = dataframe["date"].dt.hour + dataframe["date"].dt.minute / 60.0
            dataframe["%-hour_sin"] = np.sin(2 * np.pi * hour / 24.0)
            dataframe["%-hour_cos"] = np.cos(2 * np.pi * hour / 24.0)

            dow = dataframe["date"].dt.dayofweek.astype(float)
            dataframe["%-dow_sin"] = np.sin(2 * np.pi * dow / 7.0)
            dataframe["%-dow_cos"] = np.cos(2 * np.pi * dow / 7.0)

        return dataframe

    def set_freqai_targets(
        self, dataframe: DataFrame, metadata: dict, **kwargs
    ) -> DataFrame:
        dataframe["&-action"] = 0
        return dataframe

    # ------------------------------------------------------------------
    # Entry / Exit
    # ------------------------------------------------------------------

    def populate_entry_trend(self, df: DataFrame, metadata: dict) -> DataFrame:
        # Long entry: action == 2
        enter_long = [df["do_predict"] == 1, df["&-action"] == 2]
        if enter_long:
            df.loc[
                reduce(lambda x, y: x & y, enter_long),
                ["enter_long", "enter_tag"],
            ] = (1, "rl_long")

        # Short entry: action == 3
        enter_short = [df["do_predict"] == 1, df["&-action"] == 3]
        if enter_short:
            df.loc[
                reduce(lambda x, y: x & y, enter_short),
                ["enter_short", "enter_tag"],
            ] = (1, "rl_short")

        return df

    def populate_exit_trend(self, df: DataFrame, metadata: dict) -> DataFrame:
        # Unified exit: action == 1 (for both long and short)
        exit_cond = [df["do_predict"] == 1, df["&-action"] == 1]
        if exit_cond:
            df.loc[reduce(lambda x, y: x & y, exit_cond), "exit_long"] = 1
            df.loc[reduce(lambda x, y: x & y, exit_cond), "exit_short"] = 1

        return df

    # ------------------------------------------------------------------
    # Callbacks
    # ------------------------------------------------------------------

    def leverage(
        self,
        pair: str,
        current_time: datetime,
        current_rate: float,
        proposed_leverage: float,
        max_leverage: float,
        entry_tag: str | None,
        side: str,
        **kwargs,
    ) -> float:
        """Per-pair leverage: ETH 10x, SOL 5x."""
        if "SOL" in pair:
            return 5.0
        return 10.0

    def custom_stoploss(
        self,
        pair: str,
        trade,
        current_time,
        current_rate,
        current_profit,
        after_fill,
        **kwargs,
    ) -> float | None:
        """Per-pair stoploss: ETH -3%, SOL -2%."""
        if "SOL" in pair:
            return -0.02
        return -0.03

    # ------------------------------------------------------------------
    # S/R Flip Feature Computation
    # ------------------------------------------------------------------

    def _compute_sr_features(
        self, df: DataFrame, fractal_n: int = 5, cluster_atr_mult: float = 0.5
    ) -> DataFrame:
        """
        Compute Support/Resistance Flip features using fractal detection.

        Algorithm (from design doc Section 5.2):
          1. Detect fractal highs/lows (local extremes over 2*N+1 window)
          2. Cluster nearby levels within ATR * cluster_atr_mult
          3. Track nearest support (below) and resistance (above)
          4. Detect breaks (sustained close beyond level)
          5. Detect flips (retest + rejection of broken level)

        Produces 6 features:
          %-dist_to_support, %-dist_to_resistance, %-price_pos_in_sr,
          %-sr_strength, %-sr_flip_bullish, %-sr_flip_bearish
        """
        close = df["close"].values.astype(np.float64)
        high = df["high"].values.astype(np.float64)
        low = df["low"].values.astype(np.float64)
        n = len(df)

        # Fallback for insufficient data
        if n < fractal_n * 2 + 1:
            df["%-dist_to_support"] = 0.0
            df["%-dist_to_resistance"] = 0.0
            df["%-sr_flip_bullish"] = 0.0
            df["%-sr_flip_bearish"] = 0.0
            df["%-price_pos_in_sr"] = 0.5
            df["%-sr_strength"] = 0.0
            return df

        # --- Step 1: Fractal detection (vectorized) ---
        fractal_high_mask = np.ones(n, dtype=bool)
        fractal_low_mask = np.ones(n, dtype=bool)

        for offset in range(1, fractal_n + 1):
            fractal_high_mask[offset:] &= high[offset:] > high[:-offset]
            fractal_high_mask[:n - offset] &= high[:n - offset] > high[offset:]
            fractal_low_mask[offset:] &= low[offset:] < low[:-offset]
            fractal_low_mask[:n - offset] &= low[:n - offset] < low[offset:]

        # Edges cannot be valid fractals
        fractal_high_mask[:fractal_n] = False
        fractal_high_mask[-fractal_n:] = False
        fractal_low_mask[:fractal_n] = False
        fractal_low_mask[-fractal_n:] = False

        # ATR for clustering distance
        atr_series = ta.ATR(df, timeperiod=14)
        atr = atr_series.bfill().values.astype(np.float64)

        # --- Steps 2-5: Level tracking + break/flip detection ---
        nearest_sup = np.full(n, np.nan)
        nearest_res = np.full(n, np.nan)
        sup_strength_arr = np.zeros(n)
        res_strength_arr = np.zeros(n)
        flip_bullish = np.zeros(n)
        flip_bearish = np.zeros(n)

        # Dynamic level containers: list of [level, touch_count]
        res_levels: list[list[float]] = []
        sup_levels: list[list[float]] = []
        max_touches = 1.0

        # Break tracking: (level, tick_index) or None
        broken_sup = None
        broken_res = None

        prev_nearest_sup = np.nan
        prev_nearest_res = np.nan

        for i in range(fractal_n, n):
            # Newly confirmed fractal: position (i - fractal_n) confirmed at tick i
            conf = i - fractal_n
            if conf >= fractal_n:
                cdist = atr[i] * cluster_atr_mult if atr[i] > 0 else 0.0

                if fractal_high_mask[conf]:
                    lvl = high[conf]
                    merged = False
                    if cdist > 0:
                        for entry in res_levels:
                            if abs(lvl - entry[0]) < cdist:
                                cnt = entry[1] + 1
                                entry[0] = (entry[0] * entry[1] + lvl) / cnt
                                entry[1] = cnt
                                if cnt > max_touches:
                                    max_touches = cnt
                                merged = True
                                break
                    if not merged:
                        res_levels.append([lvl, 1.0])

                if fractal_low_mask[conf]:
                    lvl = low[conf]
                    merged = False
                    if cdist > 0:
                        for entry in sup_levels:
                            if abs(lvl - entry[0]) < cdist:
                                cnt = entry[1] + 1
                                entry[0] = (entry[0] * entry[1] + lvl) / cnt
                                entry[1] = cnt
                                if cnt > max_touches:
                                    max_touches = cnt
                                merged = True
                                break
                    if not merged:
                        sup_levels.append([lvl, 1.0])

            c = close[i]

            # Find nearest support (highest level below close)
            best_sup = np.nan
            best_sup_str = 0.0
            for entry in sup_levels:
                if entry[0] < c and (np.isnan(best_sup) or entry[0] > best_sup):
                    best_sup = entry[0]
                    best_sup_str = entry[1]

            # Find nearest resistance (lowest level above close)
            best_res = np.nan
            best_res_str = 0.0
            for entry in res_levels:
                if entry[0] > c and (np.isnan(best_res) or entry[0] < best_res):
                    best_res = entry[0]
                    best_res_str = entry[1]

            nearest_sup[i] = best_sup
            nearest_res[i] = best_res
            sup_strength_arr[i] = best_sup_str
            res_strength_arr[i] = best_res_str

            # --- Break detection ---
            # Support break: previous nearest support is now above close
            if not np.isnan(prev_nearest_sup) and c < prev_nearest_sup:
                broken_sup = (prev_nearest_sup, i)

            # Resistance break: previous nearest resistance is now below close
            if not np.isnan(prev_nearest_res) and c > prev_nearest_res:
                broken_res = (prev_nearest_res, i)

            # --- Flip detection (retest + rejection) ---
            retest_pct = 0.002  # within 0.2% of broken level

            if broken_res is not None and i > broken_res[1]:
                br_level = broken_res[0]
                # Bullish flip: price pulled back to broken resistance (now support)
                # and bounced (current candle higher than previous)
                if br_level > 0 and abs(c - br_level) / br_level < retest_pct:
                    if i > 0 and close[i] > close[i - 1]:
                        flip_bullish[i] = 1.0
                        broken_res = None
                # Expire stale breaks (more than 20 candles ago)
                elif i - broken_res[1] > 20:
                    broken_res = None

            if broken_sup is not None and i > broken_sup[1]:
                br_level = broken_sup[0]
                # Bearish flip: price rallied back to broken support (now resistance)
                # and got rejected (current candle lower than previous)
                if br_level > 0 and abs(c - br_level) / br_level < retest_pct:
                    if i > 0 and close[i] < close[i - 1]:
                        flip_bearish[i] = 1.0
                        broken_sup = None
                elif i - broken_sup[1] > 20:
                    broken_sup = None

            prev_nearest_sup = best_sup
            prev_nearest_res = best_res

        # --- Compute normalized features ---
        close_safe = np.where(close > 0, close, 1.0)

        df["%-dist_to_support"] = np.where(
            ~np.isnan(nearest_sup),
            (close - nearest_sup) / close_safe,
            0.0,
        )
        df["%-dist_to_resistance"] = np.where(
            ~np.isnan(nearest_res),
            (nearest_res - close) / close_safe,
            0.0,
        )

        sr_range = nearest_res - nearest_sup
        valid_range = (~np.isnan(nearest_sup)) & (~np.isnan(nearest_res)) & (sr_range > 0)
        df["%-price_pos_in_sr"] = np.where(
            valid_range,
            (close - nearest_sup) / sr_range,
            0.5,
        )

        combined_strength = np.maximum(sup_strength_arr, res_strength_arr)
        df["%-sr_strength"] = combined_strength / max_touches if max_touches > 0 else 0.0

        df["%-sr_flip_bullish"] = flip_bullish
        df["%-sr_flip_bearish"] = flip_bearish

        return df
