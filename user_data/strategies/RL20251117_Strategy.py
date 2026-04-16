import logging
from functools import reduce

import talib.abstract as ta
from pandas import DataFrame
from freqtrade.strategy import IStrategy

logger = logging.getLogger(__name__)

class RL20251117_Strategy(IStrategy):
    """
    RL20251117 Strategy
    
    Features:
    1. Trend (EMA 20/50)
    2. Fibonacci Retracement (Rolling 100)
    3. Fair Value Gap (FVG)
    """

    minimal_roi = {"0": 0.1, "240": -1}
    
    process_only_new_candles = True
    stoploss = -0.05
    use_exit_signal = True
    startup_candle_count: int = 100
    can_short = True

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Main FreqAI entry point.
        """
        dataframe = self.freqai.start(dataframe, metadata, self)
        return dataframe

    def feature_engineering_standard(self, dataframe: DataFrame, metadata: dict, **kwargs) -> DataFrame:
        """
        Define standard features for RL model.
        """
        # 1. 趨勢 (Trend) - 使用 EMA 交叉或價格位置
        dataframe['ema_20'] = ta.EMA(dataframe, timeperiod=20)
        dataframe['ema_50'] = ta.EMA(dataframe, timeperiod=50)
        dataframe['trend_up'] = (dataframe['ema_20'] > dataframe['ema_50']).astype(int)

        # 2. 斐波那契 (Fibonacci) - 計算近期高低點的回撤位
        # 假設使用過去 100 根 K 線的高低點
        rolling_high = dataframe['high'].rolling(100).max()
        rolling_low = dataframe['low'].rolling(100).min()
        diff = rolling_high - rolling_low
        # 避免除以零
        diff = diff.replace(0, 0.0000001)
        
        dataframe['fib_0.618'] = rolling_high - (diff * 0.618)
        dataframe['fib_0.5'] = rolling_high - (diff * 0.5)
        
        # 特徵化：當前價格相對於 Fib 0.618 的距離 (正規化)
        dataframe['dist_to_fib_618'] = (dataframe['close'] - dataframe['fib_0.618']) / dataframe['close']

        # 3. 公允價值缺口 (FVG)
        # FVG 定義: (High[n-2] < Low[n]) for Bullish FVG
        dataframe['fvg_bull'] = (
            (dataframe['high'].shift(2) < dataframe['low']) & 
            (dataframe['close'].shift(1) > dataframe['open'].shift(1)) # 中間是紅K(上漲)
        ).astype(int)
        
        # 必須包含原始 OHLCV 供 RL 環境使用
        dataframe["%-raw_close"] = dataframe["close"]
        dataframe["%-raw_open"] = dataframe["open"]
        dataframe["%-raw_high"] = dataframe["high"]
        dataframe["%-raw_low"] = dataframe["low"]

        return dataframe

    def set_freqai_targets(self, dataframe: DataFrame, metadata: dict, **kwargs) -> DataFrame:
        """
        For RL models: Set a neutral action placeholder.
        """
        dataframe["&-action"] = 0
        return dataframe

    def populate_entry_trend(self, df: DataFrame, metadata: dict) -> DataFrame:
        """
        Define entry conditions based on RL agent actions.
        """
        # Long entry when action = 1
        enter_long_conditions = [
            df["do_predict"] == 1,
            df["&-action"] == 1
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
        """
        # Long exit when action = 2
        exit_long_conditions = [
            df["do_predict"] == 1,
            df["&-action"] == 2
        ]
        
        if exit_long_conditions:
            df.loc[
                reduce(lambda x, y: x & y, exit_long_conditions),
                "exit_long"
            ] = 1

        # Short exit when action = 4
        exit_short_conditions = [
            df["do_predict"] == 1,
            df["&-action"] == 4
        ]
        
        if exit_short_conditions:
            df.loc[
                reduce(lambda x, y: x & y, exit_short_conditions),
                "exit_short"
            ] = 1

        return df
