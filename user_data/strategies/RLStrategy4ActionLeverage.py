import logging
from datetime import datetime

from freqtrade.strategy import IStrategy
from RLStrategy4Action import RLStrategy4Action


logger = logging.getLogger(__name__)


class RLStrategy4ActionLeverage(RLStrategy4Action):
    """
    Reinforcement Learning Strategy with 10x Fixed Leverage
    
    Inherits from RLStrategy4Action and adds 10x leverage support.
    Works with RL4ActionLeverage model.
    
    Usage:
    freqtrade backtesting --strategy RLStrategy4ActionLeverage --config user_data/config_rl_10x.json --freqaimodel RL4ActionLeverage --timerange 20230101-20251019 --export trades
    """

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
        """
        Customize leverage for each new trade.
        
        Returns fixed 10x leverage for all trades.
        
        :param pair: Pair that's currently analyzed
        :param current_time: datetime object, containing the current datetime
        :param current_rate: Rate, calculated based on pricing settings
        :param proposed_leverage: A leverage proposed by the bot
        :param max_leverage: Max leverage allowed on this pair
        :param entry_tag: Optional entry_tag (buy_tag) if provided with the buy signal
        :param side: 'long' or 'short' - indicating the direction of the proposed trade
        :return: 10.0 (fixed 10x leverage)
        """
        return 10.0
