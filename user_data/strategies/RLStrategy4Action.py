import logging
from functools import reduce

import talib.abstract as ta
from pandas import DataFrame
from technical import qtpylib

from freqtrade.strategy import IStrategy


logger = logging.getLogger(__name__)


class RLStrategy4Action(IStrategy):
    """
    Reinforcement Learning Strategy for FreqAI using Base4ActionRLEnv
    
    This strategy uses the ReinforcementLearner4Action model with 4 actions:
    - 0: Neutral
    - 1: Exit (unified exit for both Long and Short)
    - 2: Long_enter
    - 3: Short_enter
    
    Usage:
    freqtrade backtesting --strategy RLStrategy4Action --config user_data/config_dqn_gpu.json --freqaimodel ReinforcementLearner4Action --timerange 20231201-20240130
    """

    minimal_roi = {"0": 0.1, "240": -1}
    
    process_only_new_candles = True
    stoploss = -0.05
    use_exit_signal = True
    startup_candle_count: int = 40
    can_short = True

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        This is the main FreqAI entry point.
        All feature engineering happens through freqai.start().
        
        :param dataframe: strategy dataframe
        :param metadata: metadata of current pair
        """
        dataframe = self.freqai.start(dataframe, metadata, self)
        return dataframe

    def feature_engineering_expand_all(
        self, dataframe: DataFrame, period: int, metadata: dict, **kwargs
    ) -> DataFrame:
        """
        Define features that will be expanded across multiple timeframes and periods.
        All features must be prepended with `%` to be recognized by FreqAI.
        
        :param dataframe: strategy dataframe which will receive the features
        :param period: period of the indicator
        :param metadata: metadata of current pair
        """
        dataframe["%-rsi-period"] = ta.RSI(dataframe, timeperiod=period)
        dataframe["%-mfi-period"] = ta.MFI(dataframe, timeperiod=period)
        dataframe["%-adx-period"] = ta.ADX(dataframe, timeperiod=period)
        dataframe["%-sma-period"] = ta.SMA(dataframe, timeperiod=period)
        dataframe["%-ema-period"] = ta.EMA(dataframe, timeperiod=period)

        dataframe["%-roc-period"] = ta.ROC(dataframe, timeperiod=period)
        
        dataframe["%-relative_volume-period"] = (
            dataframe["volume"] / dataframe["volume"].rolling(period).mean()
        )

        return dataframe

    def feature_engineering_expand_basic(
        self, dataframe: DataFrame, metadata: dict, **kwargs
    ) -> DataFrame:
        """
        Define features that will be expanded but NOT across indicator_periods_candles.
        TradingView BB(21, 2.0, Close) settings.
        
        :param dataframe: strategy dataframe which will receive the features
        :param metadata: metadata of current pair
        """
        # TradingView Bollinger Bands: Length=21, StdDev=2, Source=Close
        bollinger_21 = qtpylib.bollinger_bands(
            dataframe["close"],  # Source: Close (not typical_price)
            window=21,           # Length: 21
            stds=2.0            # StdDev: 2
        )
        
        dataframe["%-bb21_lower"] = bollinger_21["lower"]
        dataframe["%-bb21_mid"] = bollinger_21["mid"]
        dataframe["%-bb21_upper"] = bollinger_21["upper"]
        
        # Bollinger Bands Width (normalized)
        dataframe["%-bb21_width"] = (
            (bollinger_21["upper"] - bollinger_21["lower"]) / bollinger_21["mid"]
        )
        
        # Price position in BB (0 = lower band, 1 = upper band)
        dataframe["%-bb21_percent"] = (
            (dataframe["close"] - bollinger_21["lower"]) / 
            (bollinger_21["upper"] - bollinger_21["lower"])
        )
        
        # Distance from middle band (normalized)
        dataframe["%-bb21_delta"] = (
            (dataframe["close"] - bollinger_21["mid"]) / bollinger_21["mid"]
        )
        
        return dataframe

    def feature_engineering_standard(self, dataframe: DataFrame, metadata: dict, **kwargs) -> DataFrame:
        """
        Define standard features that are NOT expanded.
        REQUIRED for RL models: Add raw OHLCV data.
        
        :param dataframe: strategy dataframe which will receive the features
        :param metadata: metadata of current pair
        """
        # Required for RL models - raw price data
        dataframe["%-raw_close"] = dataframe["close"]
        dataframe["%-raw_open"] = dataframe["open"]
        dataframe["%-raw_high"] = dataframe["high"]
        dataframe["%-raw_low"] = dataframe["low"]
        
        return dataframe

    def set_freqai_targets(self, dataframe: DataFrame, metadata: dict, **kwargs) -> DataFrame:
        """
        For RL models: Set a neutral action placeholder.
        The RL agent will override this with actual actions during training/prediction.
        
        :param dataframe: strategy dataframe which will receive the targets
        :param metadata: metadata of current pair
        """
        # For RL, there are no direct targets. This is neutral until agent sends action.
        dataframe["&-action"] = 0
        return dataframe

    def populate_entry_trend(self, df: DataFrame, metadata: dict) -> DataFrame:
        """
        Define entry conditions based on RL agent actions.
        
        Actions (Base4ActionRLEnv):
        - 0: Neutral
        - 1: Exit (unified exit)
        - 2: Long Enter
        - 3: Short Enter
        
        :param df: strategy dataframe
        :param metadata: metadata of current pair
        """
        # Long entry when action = 2
        enter_long_conditions = [
            df["do_predict"] == 1,
            df["&-action"] == 2
        ]
        
        if enter_long_conditions:
            df.loc[
                reduce(lambda x, y: x & y, enter_long_conditions),
                ["enter_long", "enter_tag"]
            ] = (1, "rl_long")

        # Short entry when action = 3
        enter_short_conditions = [
            df["do_predict"] == 1,
            df["&-action"] == 3
        ]
        
        if enter_short_conditions:
            df.loc[
                reduce(lambda x, y: x & y, enter_short_conditions),
                ["enter_short", "enter_tag"]
            ] = (1, "rl_short")

        return df

    def populate_exit_trend(self, df: DataFrame, metadata: dict) -> DataFrame:
        """
        Define exit conditions based on RL agent actions.
        
        For Base4ActionRLEnv, action = 1 is the unified Exit action for both long and short.
        
        :param df: strategy dataframe
        :param metadata: metadata of current pair
        """
        # Unified exit when action = 1 (for both long and short)
        exit_conditions = [
            df["do_predict"] == 1,
            df["&-action"] == 1
        ]
        
        if exit_conditions:
            # Exit long positions
            df.loc[
                reduce(lambda x, y: x & y, exit_conditions),
                "exit_long"
            ] = 1
            
            # Exit short positions
            df.loc[
                reduce(lambda x, y: x & y, exit_conditions),
                "exit_short"
            ] = 1

        return df
