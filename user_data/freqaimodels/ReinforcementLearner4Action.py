import logging

import numpy as np

from freqtrade.freqai.prediction_models.ReinforcementLearner import ReinforcementLearner
from freqtrade.freqai.RL.Base4ActionRLEnv import Actions, Base4ActionRLEnv, Positions


logger = logging.getLogger(__name__)


class ReinforcementLearner4Action(ReinforcementLearner):
    """
    Reinforcement Learning Model using Base4ActionRLEnv (4 actions).
    
    This model uses a unified Exit action instead of separate Long_exit and Short_exit.
    
    Actions:
    - 0: Neutral
    - 1: Exit (unified exit for both Long and Short)
    - 2: Long_enter
    - 3: Short_enter
    
    Usage:
    freqtrade backtesting --strategy RLStrategy4Action --config user_data/config_dqn_gpu.json --freqaimodel ReinforcementLearner4Action --timerange 20231201-20240130
    """

    class MyRLEnv(Base4ActionRLEnv):
        """
        Custom environment using Base4ActionRLEnv.
        Unified exit action for both long and short positions.
        """

        def calculate_reward(self, action: int) -> float:
            """
            Calculate reward for the current action.
            
            Reward structure:
            - Invalid action: -2
            - Enter trade: +25
            - Neutral when not in position: -1
            - Exit with profit > profit_aim: PNL * factor * win_reward_factor
            - Exit: PNL * factor
            - Sitting in position doing nothing: -1 * (trade_duration / max_trade_duration)
            
            :param action: int = The action made by the agent
            :return: float = reward value
            """
            # Penalize invalid actions
            if not self._is_valid(action):
                self.tensorboard_log("invalid", category="actions")
                return -2

            pnl = self.get_unrealized_profit()
            rew = np.sign(pnl) * (pnl + 1)
            factor = 100.0

            # Reward agent for entering trades
            if (
                action in (Actions.Long_enter.value, Actions.Short_enter.value)
                and self._position == Positions.Neutral
            ):
                return 25

            # Discourage agent from not entering trades
            if action == Actions.Neutral.value and self._position == Positions.Neutral:
                return -1

            max_trade_duration = self.rl_config.get("max_trade_duration_candles", 300)
            trade_duration = self._current_tick - self._last_trade_tick  # type: ignore

            # Adjust factor based on trade duration
            if trade_duration <= max_trade_duration:
                factor *= 1.5
            elif trade_duration > max_trade_duration:
                factor *= 0.5

            # Discourage sitting in position without action
            if (
                self._position in (Positions.Short, Positions.Long)
                and action == Actions.Neutral.value
            ):
                return -1 * trade_duration / max_trade_duration

            # Unified Exit action for both Long and Short positions
            if action == Actions.Exit.value and self._position == Positions.Long:
                # Log PNL for Long exits
                self.tensorboard_log("long_exit_pnl", value=pnl, category="pnl")
                self.tensorboard_log("exit_pnl", value=pnl, category="pnl")
                
                # Track profitable vs loss exits
                if pnl > 0:
                    self.tensorboard_log("profitable_exit", category="pnl")
                    self.tensorboard_log("profitable_pnl_sum", value=pnl, category="pnl")
                else:
                    self.tensorboard_log("loss_exit", category="pnl")
                    self.tensorboard_log("loss_pnl_sum", value=pnl, category="pnl")
                
                if pnl > self.profit_aim * self.rr:
                    factor *= self.rl_config["model_reward_parameters"].get("win_reward_factor", 2)
                    self.tensorboard_log("big_win_exit", category="pnl")
                
                return float(rew * factor)

            if action == Actions.Exit.value and self._position == Positions.Short:
                # Log PNL for Short exits
                self.tensorboard_log("short_exit_pnl", value=pnl, category="pnl")
                self.tensorboard_log("exit_pnl", value=pnl, category="pnl")
                
                # Track profitable vs loss exits
                if pnl > 0:
                    self.tensorboard_log("profitable_exit", category="pnl")
                    self.tensorboard_log("profitable_pnl_sum", value=pnl, category="pnl")
                else:
                    self.tensorboard_log("loss_exit", category="pnl")
                    self.tensorboard_log("loss_pnl_sum", value=pnl, category="pnl")
                
                if pnl > self.profit_aim * self.rr:
                    factor *= self.rl_config["model_reward_parameters"].get("win_reward_factor", 2)
                    self.tensorboard_log("big_win_exit", category="pnl")
                
                return float(rew * factor)

            return 0.0
