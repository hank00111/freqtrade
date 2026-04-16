import logging

from freqtrade.freqai.prediction_models.ReinforcementLearner import ReinforcementLearner
from freqtrade.freqai.RL.Base4ActionRLEnv import Actions, Base4ActionRLEnv, Positions


logger = logging.getLogger(__name__)


class RLDayTrader(ReinforcementLearner):
    """
    RL Day Trade Model with stepped 1RR reward and SL/ROI simulation.

    Inherits from ReinforcementLearner and overrides MyRLEnv with a
    Base4ActionRLEnv customised for intraday leveraged futures trading.

    v2 design:
    - Stepped reward: 5 fixed levels (+10, +5, 0, -5, -10)
    - Neutral zone: PNL within +/-5% = 0 reward (noise immunity)
    - SL/ROI simulation: force exit at configurable PNL thresholds
    - Liquidation at -50% leveraged PNL (buffer=0.05)
    - No entry reward (0) to prevent overtrading

    rl_config custom parameters:
      leverage              (float) default 10.0
      liquidation_buffer    (float) default 0.05
      simulate_stoploss     (float) optional, e.g. -0.10
      simulate_roi          (float) optional, e.g.  0.10
      max_trade_duration_candles (int) default 96

    Usage:
      freqtrade backtesting --strategy RLDayTradeStrategy \
        --config user_data/config_daytrade_v2.json \
        --freqaimodel RLDayTrader \
        --timerange 20230101-20260401 --export trades
    """

    class MyRLEnv(Base4ActionRLEnv):
        """
        Custom 4-action environment for leveraged intraday trading.
        Stepped 1RR reward with neutral zone, following design doc Section 6.2.
        """

        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self.leverage = self.rl_config.get("leverage", 10.0)
            self.liquidation_buffer = self.rl_config.get("liquidation_buffer", 0.05)
            self._is_liquidated = False

        def reset(self, seed=None):
            self._is_liquidated = False
            return super().reset(seed)

        def step(self, action):
            """
            Override step to simulate SL/ROI/timeout forced exits.

            Priority (highest to lowest):
            1. Liquidation (checked in calculate_reward)
            2. Stoploss: PNL <= simulate_stoploss -> force exit
            3. ROI: PNL >= simulate_roi -> force exit
            4. Timeout: holding >= max_trade_duration_candles -> force exit
            5. Model action
            """
            if self._position != Positions.Neutral and self._last_trade_tick is not None:
                pnl = self.get_unrealized_profit()
                trade_duration = self._current_tick - self._last_trade_tick
                max_dur = self.rl_config.get("max_trade_duration_candles", 96)

                stoploss = self.rl_config.get("simulate_stoploss", None)
                if stoploss is not None and pnl <= stoploss:
                    action = Actions.Exit.value

                roi = self.rl_config.get("simulate_roi", None)
                if roi is not None and pnl >= roi:
                    action = Actions.Exit.value

                if max_dur > 0 and trade_duration >= max_dur:
                    action = Actions.Exit.value

            return super().step(action)

        # --------------------------------------------------------------
        # Leverage-aware PNL
        # --------------------------------------------------------------

        def get_unrealized_profit(self) -> float:
            """Return unrealized PNL amplified by leverage."""
            if self._last_trade_tick is None or self._position == Positions.Neutral:
                return 0.0

            if self._position == Positions.Short:
                current_price = self.add_entry_fee(self.prices.iloc[self._current_tick].open)
                last_trade_price = self.add_exit_fee(self.prices.iloc[self._last_trade_tick].open)
                base_pnl = (last_trade_price - current_price) / last_trade_price
            else:  # Long
                current_price = self.add_exit_fee(self.prices.iloc[self._current_tick].open)
                last_trade_price = self.add_entry_fee(self.prices.iloc[self._last_trade_tick].open)
                base_pnl = (current_price - last_trade_price) / last_trade_price

            return base_pnl * self.leverage

        # --------------------------------------------------------------
        # Liquidation
        # --------------------------------------------------------------

        def _get_base_pnl(self) -> float:
            """Return base (non-leveraged) PNL for liquidation check."""
            if self._last_trade_tick is None or self._position == Positions.Neutral:
                return 0.0

            if self._position == Positions.Short:
                cp = self.add_entry_fee(self.prices.iloc[self._current_tick].open)
                tp = self.add_exit_fee(self.prices.iloc[self._last_trade_tick].open)
                return (tp - cp) / tp
            else:
                cp = self.add_exit_fee(self.prices.iloc[self._current_tick].open)
                tp = self.add_entry_fee(self.prices.iloc[self._last_trade_tick].open)
                return (cp - tp) / tp

        def _check_liquidation(self) -> bool:
            """
            Check if position should be liquidated.

            Liquidation threshold = -(1/leverage - buffer).
            For leverage=10, buffer=0.05: threshold = -0.05 (-5.0% base PNL = -50% leveraged).
            """
            if self._position == Positions.Neutral or self._is_liquidated:
                return False

            base_pnl = self._get_base_pnl()
            threshold = -(1.0 / self.leverage - self.liquidation_buffer)

            if base_pnl <= threshold:
                self._is_liquidated = True
                self._position = Positions.Neutral
                self._total_profit += threshold * self.leverage
                return True

            return False

        # --------------------------------------------------------------
        # Reward function: Stepped 1RR with neutral zone
        # --------------------------------------------------------------

        def calculate_reward(self, action: int) -> float:
            """
            Stepped 1RR reward function per design doc Section 6.2.

            5 fixed reward levels: +10, +5, 0, -5, -10
            Neutral zone: PNL within +/-5% = 0 (noise immunity)
            Hold rewards: +2, +1, 0, -1, -2
            Liquidation: -10

            Priority:
              1. Liquidation (PNL <= -50%)     -> -10
              2. Invalid action                -> -1
              3. Neutral / Entry               ->  0
              4. Hold/Exit by PNL zone (stepped)
            """
            # --- Liquidation check (highest priority) ---
            if self._check_liquidation():
                self.tensorboard_log("liquidation", category="risk")
                return -10.0

            # --- Invalid action ---
            if not self._is_valid(action):
                self.tensorboard_log("invalid", category="actions")
                return -1.0

            # --- Neutral while Neutral ---
            if action == Actions.Neutral.value and self._position == Positions.Neutral:
                return 0.0

            # --- Enter trade ---
            if (
                action in (Actions.Long_enter.value, Actions.Short_enter.value)
                and self._position == Positions.Neutral
            ):
                return 0.0

            pnl = self.get_unrealized_profit()
            trade_duration = self._current_tick - self._last_trade_tick  # type: ignore

            # --- Holding position (Neutral action while in position) ---
            if (
                self._position in (Positions.Short, Positions.Long)
                and action == Actions.Neutral.value
            ):
                if pnl >= 0.10:
                    hold_reward = 2.0
                elif pnl >= 0.05:
                    hold_reward = 1.0
                elif pnl > -0.05:
                    hold_reward = 0.0
                elif pnl > -0.10:
                    hold_reward = -1.0
                else:
                    hold_reward = -2.0

                return hold_reward

            # --- Exit action ---
            if action == Actions.Exit.value and self._position in (
                Positions.Long,
                Positions.Short,
            ):
                if pnl >= 0.10:
                    exit_reward = 10.0
                elif pnl >= 0.05:
                    exit_reward = 5.0
                elif pnl > -0.05:
                    exit_reward = 0.0
                elif pnl > -0.10:
                    exit_reward = -5.0
                else:
                    exit_reward = -10.0

                # Logging
                pos = "long" if self._position == Positions.Long else "short"
                self.tensorboard_log(f"{pos}_exit_pnl", value=pnl, category="pnl")
                self.tensorboard_log("exit_pnl", value=pnl, category="pnl")
                if pnl > 0:
                    self.tensorboard_log("profitable_exit", category="pnl")
                else:
                    self.tensorboard_log("loss_exit", category="pnl")
                self.tensorboard_log("exit_reward", value=exit_reward, category="rewards")
                self.tensorboard_log("trade_duration", value=trade_duration, category="rewards")

                return exit_reward

            return 0.0
