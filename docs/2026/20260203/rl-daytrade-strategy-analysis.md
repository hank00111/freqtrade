# RL Day Trade Strategy - Analysis Document

> Date: 2026-02-03 (Updated: 2026-03-25)
> Status: Training Near Complete (v1-20260307, ETH 167 windows, PPO_167 in progress -- likely final, multiproc, cpu_count=4, n_steps=2048)
> Target: Intraday RL trading strategy for ETH + SOL futures

### Quick Start - Training Command

```powershell
# Activate venv
cd G:\Program\freqtrade
.\.venv\Scripts\Activate.ps1

# 1. Download / update data (incremental, only fetches missing ranges)
freqtrade download-data `
  --config user_data/config_daytrade.json `
  --timerange 20220901-20260303 `
  --timeframe 5m 15m 1h 4h

# 2. Production training (multiproc)
freqtrade backtesting `
  --strategy RLDayTradeStrategy `
  --config user_data/config_daytrade.json `
  --freqaimodel RLDayTrader_multiproc `
  --timerange 20230101-20260303 `
  --export trades

# 3. Single-env debugging (reward function verification)
freqtrade backtesting `
  --strategy RLDayTradeStrategy `
  --config user_data/config_daytrade.json `
  --freqaimodel RLDayTrader `
  --timerange 20250101-20250201 `
  --export trades

# 4. TensorBoard (open in another terminal, keep running during training)
tensorboard --logdir user_data/models/rl-daytrade-v1-20260220
# Open http://localhost:6006 in browser
```

Key config: `cpu_count=4`, `n_steps=2048`, `batch_size=512`, `train_cycles=500` (changed from n_steps=1024/batch=256 at PPO_41)
Data range: 20220901-20260201 (includes startup candle buffer for first window)
Estimated windows: ~150 (75d train / 7d backtest over ~3y1m)

---

## 1. Objectives

| Item | Value |
|------|-------|
| Capital | 100 USDT |
| Leverage | 10x (ETH), consider 5x (SOL) |
| Daily Target | 5~10 USDT net profit (5~10% daily ROI) |
| Style | Intraday, fast in/out, stable small gains |
| Pairs | ETH/USDT:USDT, SOL/USDT:USDT |
| Main Timeframe | 5m |
| Multi-Timeframe | 5m, 15m, 1h, 4h |
| Corr Pairlist | BTC/USDT:USDT |

---

## 2. Feasibility Analysis

### 2.1 Profit Math

```
Capital: 100 USDT
Leverage: 10x -> Effective position 1,000 USDT
Daily target: 5~10 USDT (net) = 5~10% daily ROI

Binance Futures Fee (Taker):
  One-way: 0.04% x 1,000 USDT = 0.4 USDT
  Round-trip: 0.8 USDT/trade

Assuming 3 trades/day:
  Fees: 3 x 0.8 = 2.4 USDT
  Required gross: 7.4~12.4 USDT
  Per-trade capture: 2.5~4.1 USDT
  Price movement needed: 0.25%~0.41% (on 1,000 USDT position)
```

### 2.2 Pair Volatility Comparison

| Metric | ETH | SOL |
|--------|-----|-----|
| 5m avg range | 0.08~0.4% | 0.15~0.8% |
| Daily avg range | 3~6% | 5~12% |
| Liquidity | High | Medium-High |
| BTC correlation | 0.85~0.92 | 0.70~0.85 |
| Trend continuity | Medium | Weak (high false breakouts) |
| 10x liquidation risk | Low-Medium | HIGH |

### 2.3 SOL Risk Warning

SOL can move 2~5% within a single 5m candle in extreme conditions.
With 10x leverage, liquidation threshold is base PNL -5%.
1~2 violent candles could trigger liquidation.

Recommendations for SOL:
- Consider reducing leverage to 5x (liquidation threshold widens to -15%)
- Set tighter stoploss at -0.02 (-2%) via `custom_stoploss()` callback
- Apply stricter risk penalties in reward function

**Implementation note:** Single `stoploss` parameter cannot differentiate
between pairs. Use `custom_stoploss()` callback to return -0.03 for ETH
and -0.02 for SOL. Similarly, `leverage()` callback should return 10.0
for ETH and 5.0 for SOL. However, the RL training environment does NOT
use strategy callbacks (see Section 8.4). Leverage in the env comes from
`rl_config.leverage`. To train SOL with 5x leverage, use a separate
config with a different `identifier` (e.g., "rl-daytrade-sol-v1").

---

## 3. Architecture

### 3.1 File Structure

```
user_data/
  strategies/
    RLDayTradeStrategy.py       # Strategy (features + S/R Flip + entry/exit)
  freqaimodels/
    RLDayTrader.py              # Model (reward + leverage + liquidation)
    RLDayTrader_multiproc.py    # Multi-process version
  config_daytrade.json          # Configuration
```

### 3.2 Inheritance

```
Strategy:
  IStrategy
    -> RLDayTradeStrategy

Model:
  ReinforcementLearner (built-in, Base5ActionRLEnv)
    -> RLDayTrader (override MyRLEnv with Base4ActionRLEnv)
       -> RLDayTrader_multiproc (add SubprocVecEnv)

Environment:
  BaseEnvironment (gym.Env)
    -> Base4ActionRLEnv
       -> MyRLEnv (custom calculate_reward + leverage + liquidation)
```

### 3.3 Action Space: 4-Action

| Action | Value | Description |
|--------|-------|-------------|
| Neutral | 0 | Hold / do nothing |
| Exit | 1 | Unified exit (long or short) |
| Long_enter | 2 | Enter long position |
| Short_enter | 3 | Enter short position |

**4-Action constraint:** Agent cannot flip position directly (e.g., Long -> Short).
Must exit first, then enter the opposite direction (minimum 2 steps to flip).
`Base4ActionRLEnv.is_tradesignal()` enforces:
- `Long_enter` while in Short -> invalid (not a trade signal)
- `Short_enter` while in Long -> invalid (not a trade signal)

**Strategy action mapping (4-Action):**

```python
# populate_entry_trend:
enter_long:  df["&-action"] == 2
enter_short: df["&-action"] == 3

# populate_exit_trend:
exit_long:   df["&-action"] == 1   # unified Exit
exit_short:  df["&-action"] == 1   # same action for both
```

Note: This differs from 5-Action examples in official docs where action==1
is Long_enter. Always verify action values against the environment used.

---

## 4. Feature Engineering

### 4.1 Timeframe Analysis

**Selected: 5m, 15m, 1h, 4h**

| Timeframe | Role | Candles/Day | S/R Value |
|-----------|------|-------------|-----------|
| 5m (main) | Execution signals | 288 | Micro S/R |
| 15m | Short-term structure | 96 | Intraday S/R |
| 1h | Medium-term trend | 24 | Swing S/R |
| 4h | Macro trend context | 6 | Institutional S/R |

**Rejected timeframes:**

1m - Rejected:
- SNR too low, indicators unreliable
- Feature explosion: 1440 candles/day
- 5:1 ratio to main TF, extremely high correlation
- RL agent would overfit to noise

3m - Rejected:
- Too close to 5m (5:3 ratio), marginal information gain
- Near-identical indicator values introduce multicollinearity
- Not universally supported across exchanges

4h - Accepted:
- Provides macro context (dominant trend direction)
- 4h S/R levels are watched by institutional traders
- Low feature cost (only 6 candles/day, 4th TF in expansion)
- Helps agent avoid trading against major trend
- S/R Flip on 4h is highly reliable

### 4.2 feature_engineering_expand_all

Features that expand across: indicator_periods_candles x include_timeframes x include_shifted_candles x include_corr_pairlist

Controlled by `feature_flags.expand_all` in config. Indicators can be
individually toggled on/off without modifying strategy code.

```python
# Indicators (enabled via feature_flags):
RSI       -> Overbought/oversold momentum          [enabled]
ADX       -> Trend strength                         [disabled]
ATR_norm  -> Volatility (ATR / close, scale-invariant) [enabled]
Rel_Vol   -> Relative volume (volume / rolling mean)    [disabled]

# Active indicators: 2 (RSI, ATR_norm)
# Expansion:
# 2 indicators x 2 periods (10, 20) x 4 TF x 3 shifted (0,1,2) x 2 pairs
# = 2 x 2 x 4 x 3 x 2 = 96 features
```

**Disabled rationale (2026-02-07):**
- ADX: Redundant with ATR for trend/volatility context; reduces feature noise
- Rel_Vol: Volume data on futures may be less reliable; reduce dimensionality

### 4.3 feature_engineering_expand_basic

Features that expand across: include_timeframes x include_shifted_candles x include_corr_pairlist (NOT indicator_periods)

```python
# Trend indicators:
EMA_20              -> Short-term trend
EMA_50              -> Medium-term trend

# Bollinger Bands (21, 2.0, Close):
BB_percent          -> Price position in band (0=lower, 1=upper)
BB_width            -> Normalized bandwidth (volatility proxy)

# Momentum:
MACD_norm           -> MACD / close (scale-invariant)
MACD_hist_norm      -> MACD histogram / close

# S/R Flip features (NEW):
dist_to_support     -> (close - nearest_support) / close
dist_to_resistance  -> (nearest_resistance - close) / close
sr_flip_bullish     -> 1 if resistance broken & confirmed as new support
sr_flip_bearish     -> 1 if support broken & confirmed as new resistance
price_pos_in_sr     -> (close - support) / (resistance - support)
sr_strength         -> Normalized touch count of nearest level

# Expansion:
# 12 indicators x 4 TF x 3 shifted x 2 pairs = 288 features
```

### 4.4 feature_engineering_standard

Features that do NOT expand (base timeframe only, no pair expansion):

```python
# Required for RL:
raw_close, raw_open, raw_high, raw_low  -> OHLCV for environment

# Time features:
hour_sin, hour_cos    -> Hour of day (cyclical encoding)
dow_sin, dow_cos      -> Day of week (cyclical encoding)

# Total: 8 features
```

### 4.5 Feature Count Summary

```
expand_all:     96 features  (2 active indicators, ADX/Rel_Vol disabled)
expand_basic:  288 features
standard:        8 features
--------------------------
Total:        ~392 features
```

Note: Exact count depends on `feature_flags` settings in config.
Original design had 4 expand_all indicators (~488 total). Reduced to 2
after initial testing to lower dimensionality.

---

## 5. Support/Resistance Flip Implementation

### 5.1 Core Concept

S/R Flip: When price breaks through a support level, that level becomes
resistance (and vice versa). This is a key concept in price action trading.

```
Bearish Flip (Support -> Resistance):
  1. Identify support level (multiple bounces)
  2. Price breaks below support with momentum
  3. Price retests the broken level from below
  4. Level rejects price -> confirmed as new resistance
  5. Short entry signal

Bullish Flip (Resistance -> Support):
  1. Identify resistance level (multiple rejections)
  2. Price breaks above resistance with momentum
  3. Price retests the broken level from above
  4. Level holds as support -> confirmed as new support
  5. Long entry signal
```

### 5.2 Algorithm Design

```
Step 1: Identify Key Levels via Fractal Detection
  fractal_high = high[i] > max(high[i-N:i]) AND high[i] > max(high[i+1:i+N+1])
  fractal_low  = low[i]  < min(low[i-N:i])  AND low[i]  < min(low[i+1:i+N+1])
  Where N = lookback period (default: 5 for 5m TF)

Step 2: Cluster Nearby Levels
  Merge fractals within ATR * 0.5 distance
  Track touch count per cluster (strength)

Step 3: Identify Nearest S/R
  nearest_support    = max(levels WHERE level < close)
  nearest_resistance = min(levels WHERE level > close)

Step 4: Detect Break (single-candle)
  support_break:    close crosses below previous nearest support
  resistance_break: close crosses above previous nearest resistance

Step 5: Detect Flip (Retest)
  After support_break:
    If price rallies back within 0.2% of broken level -> bearish flip zone
    If price gets rejected (next candle lower) -> confirmed bearish flip
  After resistance_break:
    If price pulls back within 0.2% of broken level -> bullish flip zone
    If price bounces (next candle higher) -> confirmed bullish flip
```

### 5.3 Features for RL Agent

All S/R features are placed in `feature_engineering_expand_basic` so they
expand across timeframes automatically (5m/15m/1h/4h multi-TF S/R).

```python
# Distance features (normalized, scale-invariant):
"%-dist_to_support"      # (close - support) / close
"%-dist_to_resistance"   # (resistance - close) / close

# Position in range:
"%-price_pos_in_sr"      # (close - support) / (resistance - support), 0~1

# Level strength:
"%-sr_strength"          # touches / max_touches, 0~1

# Flip signals:
"%-sr_flip_bullish"      # 1.0 if bullish flip confirmed, else 0.0
"%-sr_flip_bearish"      # 1.0 if bearish flip confirmed, else 0.0
```

### 5.4 Multi-Timeframe S/R Value

Since expand_basic applies across all include_timeframes:

```
5m  S/R -> Micro structure, intraday scalping levels
15m S/R -> Intraday swing levels
1h  S/R -> Daily structure levels
4h  S/R -> Institutional levels, high reliability

Agent receives all levels simultaneously, learns to weigh their importance.
```

---

## 6. Reward Function Design

### 6.1 Design Principles

**Core Philosophy: High-Quality Trading (HQT)**

The reward function is designed around the principle that "not trading IS
the correct choice when there's no edge." In volatile markets, the agent
should wait for high-probability setups rather than trade frequently.

1. Compressed range: [-10, +10] for stable training gradients
2. No entry reward: prevent overtrading exploit (aligned with HQT)
3. Neutral = 0: not trading is neither rewarded nor penalized (aligned with HQT)
4. Continuously differentiable: smooth gradient flow
5. Asymmetric: penalize losses more than reward gains (capital preservation)
6. Time-aware: reward quick profitable exits + forced close at max duration (prop firm rule)

Official guidance: "The best reward functions are ones that are
continuously differentiable, and well scaled."

**Expected behavior under HQT:**
- High Neutral ratio (~60-70%) is acceptable and even desired
- Agent should show few but high-quality trades
- Win rate can be below 50% if avg win >> avg loss
- ep_rew_mean will be naturally negative due to invalid action learning cost

### 6.2 Reward Structure

```
Scenario                              Reward (clamped to [-10, +10])
------------------------------------------------------------------------
Liquidation                           -> -10
Invalid action                        -> -2
Neutral while Neutral                 -> 0 (no penalty for waiting)
Enter trade (Neutral -> Long/Short)   -> 0 (no entry reward)

Holding position + Neutral action:
  Profitable position                 -> +clamp(pnl * 20, 0, 2)
  Losing < profit_aim                 -> clamp(pnl * 30, -3, 0)
  Losing > profit_aim                 -> clamp(pnl * 50, -5, 0)
  (no duration penalty; forced close at max_dur via step() override)

Exit action (voluntary or forced close at max_trade_duration_candles):
  Any profit                          -> +clamp(pnl * 50, 1, 8)
  Quick profit (< 30% max_dur)        -> +clamp(pnl * 50, 1, 8) + 2
  Small loss (> -profit_aim)          -> -1
  Medium loss (> -2x profit_aim)      -> linear(-1, -5)
  Large loss (> -3x profit_aim)       -> -8
  Extreme loss (>= 3x profit_aim)    -> -10

Forced close (prop firm rule):
  At trade_duration >= max_trade_duration_candles, step() overrides
  agent action to Exit. Normal exit reward applies based on PNL at
  that moment. No separate penalty -- the implicit cost is losing
  control of exit timing.
```

**HQT interpretation of reward tiers:**

```
+8~+10  Ideal trade: fast, precise, profitable (HQT goal)
+1~+8   Good trade: profitable exit
 0~+2   Acceptable: waiting or holding profitable position
  -1    Cost of business: small loss exit (disciplined stop-loss)
  -2    Learning cost: invalid action (improves over training)
-3~-5   Warning: loss expanding or medium loss realized
-8~-10  Severe: large loss, liquidation (must avoid)
```

### 6.3 Leverage-Specific Additions

```
- get_unrealized_profit() overridden: base_pnl * leverage
- Liquidation check: base_pnl <= -(1/leverage - buffer)
- Liquidation penalty: -10 (maximum)

Liquidation threshold calculation:
  With leverage=10, liquidation_buffer=0.025:
  threshold = -(1/10 - 0.025) = -(0.1 - 0.025) = -0.075 (-7.5% base PNL)
  In leveraged PNL: -7.5% * 10 = -75%

  Note: existing RL4ActionLeverage.py uses buffer=0.05 (threshold -5%).
  Our buffer=0.025 is tighter (closer to real Binance liquidation),
  giving less margin for error. This is an intentional design choice
  for more accurate simulation.
```

### 6.4 Reward Analysis Under HQT Philosophy (2026-02-08)

**Why ep_rew_mean is naturally negative:**

In a typical episode (~10,000 steps):
- ~65% Neutral (no position): 6,500 steps x 0 = 0
- ~15% Invalid actions: 1,500 steps x -2 = -3,000
- ~20% Active trading: 2,000 steps with mixed rewards

Even if all active trading steps averaged +1.0 (which is very good),
the episode total would be: 0 + (-3,000) + 2,000 = -1,000. The observed
range of -2,000 to -3,900 is consistent with this math. The negative
ep_rew_mean does NOT indicate poor trading quality.

**Structural asymmetries and their HQT purpose:**

| Asymmetry | Values | Purpose |
|-----------|--------|---------|
| Hold: win vs loss multiplier | 20x vs 30~50x | Cut losses faster than riding winners |
| Hold: reward cap | +2 vs -5 | 2.5x loss amplification = capital preservation |
| Hold: no duration penalty | PNL-only | Let trades develop without artificial time pressure |
| Exit: profit vs loss range | +1~+10 vs -1~-10 | Symmetric range but +2 fast bonus tips toward quality |
| Small loss exit | Only -1 | Encourages disciplined stop-loss (acceptable loss) |
| Entry | Always 0 | No incentive to trade = HQT alignment |
| Forced close | At max_dur (48 candles) | Prop firm rule: hard intraday deadline, no exceptions |

**Key metrics to evaluate after backtest (HQT criteria):**

1. Trade count per window (fewer = more selective, aligned with HQT)
2. Win rate of actual trades (not overall step reward)
3. Average profit per winning trade vs average loss per losing trade
4. Profit factor (gross profit / gross loss) -- target > 1.5
5. Max drawdown
6. Trade duration distribution (should cluster < 48 candles)

### 6.5 Implementation Reference (calculate_reward)

Source: `user_data/freqaimodels/RLDayTrader_multiproc.py` MyRLEnv.calculate_reward()

Parameters: `profit_aim=0.05`, `rr=1`, `leverage=10`, `max_trade_duration_candles=48`,
`liquidation_buffer=0.025`

#### 6.5.1 Evaluation Order

The environment processes each step in two phases:

**Phase 1: step() -- action override (before reward)**
```
0. Forced close check: if in position AND trade_duration >= max_dur
   -> override action to Exit (prop firm rule, applied before reward calc)
```

**Phase 2: calculate_reward() -- reward evaluation (strict priority order)**
```
1. Liquidation check       -> -10  (highest priority, checked every step)
2. Invalid action check    -> -2
3. Neutral while Neutral   -> 0
4. Entry (open new trade)  -> 0
5. Holding (Neutral while in position) -> variable
6. Exit action             -> variable (includes forced close exits)
7. Fallback                -> 0
```

#### 6.5.2 Holding Reward Detail

When agent holds a position (action=Neutral while in Long/Short):

```python
if pnl >= 0:
    hold_reward = clip(pnl * 20.0, 0.0, 2.0)
elif abs(pnl) < profit_aim:        # loss < 5% leveraged
    hold_reward = clip(pnl * 30.0, -3.0, 0.0)
else:                               # loss >= 5% leveraged
    hold_reward = clip(pnl * 50.0, -5.0, 0.0)

# No duration penalty. Forced close at max_dur handled by step() override.
return clip(hold_reward, -10.0, 10.0)
```

Numerical examples (leveraged PNL -> hold reward):

| Leveraged PNL | Base Move | Multiplier | Raw | Clipped |
|--------------|-----------|-----------|-----|---------|
| +1% | +0.1% | 20x | +0.20 | +0.20 |
| +5% | +0.5% | 20x | +1.00 | +1.00 |
| +10% | +1.0% | 20x | +2.00 | +2.00 |
| -2% | -0.2% | 30x | -0.60 | -0.60 |
| -5% | -0.5% | 50x | -2.50 | -2.50 |
| -10% | -1.0% | 50x | -5.00 | -5.00 |

Note: Hold reward is purely PNL-based with no time penalty.
The agent can hold for the full max_trade_duration_candles (48)
without any duration-based deduction. At candle 48, step()
forces an Exit action (prop firm rule).

#### 6.5.3 Exit Reward Detail

When agent exits a position (action=Exit while in Long/Short):

```python
if pnl > 0:
    exit_reward = clip(pnl * 50.0, 1.0, 8.0)
    if trade_duration < max_dur * 0.3:    # < 14.4 candles
        exit_reward += 2.0                # quick profit bonus
elif pnl == 0 or abs(pnl) < profit_aim:   # small loss < 5%
    exit_reward = -1.0
elif abs(pnl) < profit_aim * 2:           # medium loss 5~10%
    ratio = (abs(pnl) - profit_aim) / profit_aim
    exit_reward = -1.0 - 4.0 * ratio      # linear interpolation
elif abs(pnl) < profit_aim * 3:           # large loss 10~15%
    exit_reward = -8.0
else:                                      # extreme loss >= 15%
    exit_reward = -10.0

return clip(exit_reward, -10.0, 10.0)
```

Profit exit examples:

| Leveraged PNL | Base Move | Raw (pnl*50) | Clipped | Quick bonus | Total |
|--------------|-----------|-------------|---------|-------------|-------|
| +2% | +0.2% | 1.0 | 1.0 | +2.0 | 3.0 |
| +5% | +0.5% | 2.5 | 2.5 | +2.0 | 4.5 |
| +10% | +1.0% | 5.0 | 5.0 | +2.0 | 7.0 |
| +16%+ | +1.6%+ | 8.0+ | 8.0 | +2.0 | 10.0 |

Loss exit tiers:

| Tier | Leveraged PNL | Base Move | Reward | Description |
|------|--------------|-----------|--------|-------------|
| Small | 0% ~ -5% | 0% ~ -0.5% | -1 | Disciplined stop-loss |
| Medium | -5% ~ -10% | -0.5% ~ -1.0% | -1 to -5 | Linear interpolation |
| Large | -10% ~ -15% | -1.0% ~ -1.5% | -8 | Severe warning |
| Extreme | >= -15% | >= -1.5% | -10 | Maximum penalty |
| Liquidation | -75% (base -7.5%) | -7.5% | -10 | Via _check_liquidation |

#### 6.5.4 Liquidation Mechanism

```python
def _check_liquidation(self):
    base_pnl = self._get_base_pnl()           # without leverage
    threshold = -(1.0 / leverage - buffer)     # -(0.1 - 0.025) = -0.075
    if base_pnl <= threshold:
        self._is_liquidated = True
        self._position = Positions.Neutral
        self._total_profit *= (1 + threshold)  # apply loss to equity
        return True
```

- Triggered when base PNL <= -7.5% (leveraged -75%)
- Position force-closed, `_is_liquidated` flag set
- Flag persists until episode reset (no double-liquidation)
- After liquidation, agent can still open new trades but without
  liquidation protection for remainder of episode

#### 6.5.5 Forced Close Mechanism (Prop Firm Rule)

```python
def step(self, action):
    """Override step to force close at max_trade_duration_candles."""
    if self._position != Positions.Neutral and self._last_trade_tick is not None:
        trade_duration = self._current_tick - self._last_trade_tick
        max_dur = self.rl_config.get("max_trade_duration_candles", 48)
        if max_dur > 0 and trade_duration >= max_dur:
            action = Actions.Exit.value
    return super().step(action)
```

- Implemented in `step()`, runs BEFORE `calculate_reward()`
- At candle 48 (4 hours), agent action is overridden to Exit regardless
  of what the agent chose (Neutral, Long_enter, Short_enter -> all become Exit)
- The overridden Exit action flows through the normal exit reward logic:
  profitable positions receive positive exit reward, losing positions
  receive negative exit reward based on PNL tiers
- No separate forced-close penalty -- the implicit cost is losing
  control of exit timing (agent cannot hold for a better exit)
- Quick profit bonus does NOT apply to forced close (trade_duration
  is at max_dur, which is > 30% max_dur threshold)

Design rationale (prop firm rule):
- Similar to prop trading firms that require all positions closed
  by a deadline (e.g., end of trading day)
- During the allowed holding window (candles 1-47): no time-based
  penalty, hold reward is purely PNL-based
- At the deadline (candle 48): hard close, no exceptions
- Agent learns to manage exit timing within the allowed window
  through the natural consequence of being force-closed at suboptimal PNL

#### 6.5.6 Unrealized Profit Calculation

```python
def get_unrealized_profit(self):
    # Long: (current_price - entry_price) / entry_price * leverage
    # Short: (entry_price - current_price) / entry_price * leverage
    # Both include entry/exit fees (0.04% taker each way)
    return base_pnl * self.leverage
```

All PNL values in the reward function are leveraged (base * 10x).
This means `profit_aim=0.05` corresponds to a 0.5% base price move.

---

## 7. Training Configuration

### 7.1 FreqAI Config

```json
{
  "trading_mode": "futures",
  "margin_mode": "isolated",
  "stake_amount": 100,
  "stake_currency": "USDT",
  "timeframe": "5m",
  "can_short": true,
  "fee": 0.0004,
  "freqai": {
    "enabled": true,
    "purge_old_models": 2,
    "train_period_days": 75,
    "backtest_period_days": 7,
    "live_retrain_hours": 0,
    "continual_learning": false,
    "identifier": "rl-daytrade-v1",
    "conv_width": 10,
    "feature_parameters": {
      "include_timeframes": ["5m", "15m", "1h", "4h"],
      "include_corr_pairlist": ["BTC/USDT:USDT"],
      "indicator_periods_candles": [10, 20],
      "include_shifted_candles": 2,
      "label_period_candles": 20,
      "weight_factor": 0.9,
      "DI_threshold": 0,
      "use_SVM_to_remove_outliers": false,
      "use_DBSCAN_to_remove_outliers": false,
      "principal_component_analysis": false
    },
    "data_split_parameters": {
      "test_size": 0.25,
      "random_state": 1
    }
  }
}
```

**Notes on added fields:**
- `trading_mode`, `margin_mode`, `stake_amount`, `stake_currency`:
  Required for futures trading. Without these, freqtrade runs in spot mode.
- `fee: 0.0004`: Binance Futures taker fee (0.04%). The framework default
  is 0.0015 (0.15%) which is too high. Setting explicitly ensures accurate
  backtesting results.
- `conv_width: 10`: Neural network observation window size. Default is 2.
  With ~392 features, a wider window gives the agent more temporal context.
  Adjust based on training performance.
- `data_split_parameters.test_size: 0.25`: Explicit 75/25 train/test split.
  **Framework default is 0.1 (90/10)**, not 0.25. Must be set explicitly
  to match the training steps calculation in Section 8.2.
- `weight_factor: 0.9`: Creates exponential decay weights favoring recent
  data in the train/test split. Note: these weights affect which data points
  go into the train vs test set, but do NOT weight individual steps during
  RL training (the environment treats all candles equally).

### 7.2 RL Config

```json
{
  "rl_config": {
    "train_cycles": 500,
    "max_trade_duration_candles": 48,
    "model_type": "PPO",
    "policy_type": "MlpPolicy",
    "max_training_drawdown_pct": 0.50,
    "cpu_count": 6,
    "net_arch": [256, 256],
    "randomize_starting_position": true,
    "drop_ohlc_from_features": false,
    "add_state_info": false,
    "leverage": 10.0,
    "liquidation_buffer": 0.025,
    "model_reward_parameters": {
      "rr": 1,
      "profit_aim": 0.05
    }
  }
}
```

### 7.3 Model Training Parameters (PPO)

```json
{
  "model_training_parameters": {
    "learning_rate": 0.0003,
    "gamma": 0.99,
    "batch_size": 1024,
    "n_steps": 4096,
    "n_epochs": 10,
    "gae_lambda": 0.95,
    "clip_range": 0.2,
    "ent_coef": 0.05,
    "vf_coef": 0.5,
    "max_grad_norm": 0.5,
    "normalize_advantage": true,
    "device": "cpu"
  }
}
```

### 7.4 Key Parameter Rationale

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| train_period_days | 75 | Cover multiple market regimes |
| backtest_period_days | 7 | Weekly retraining cycle |
| continual_learning | false | Avoid regime drift accumulation |
| weight_factor | 0.9 | Weight train/test split toward recent data (does not weight RL steps) |
| train_cycles | 500 | Sufficient with 75d data, avoid overfit |
| max_trade_duration_candles | 48 | 4 hours max (intraday constraint) |
| max_training_drawdown_pct | 0.50 | Episode ends at 50% cumulative loss (1 - 0.50 = 0.50 threshold) |
| net_arch | [256, 256] | Smaller network for better generalization |
| profit_aim | 0.05 | 5% leveraged PNL target (= 0.5% base price move at 10x) |
| n_steps | 4096 | Doubled from 2048: 2x rollout buffer (32,768) for more diverse experience per update, same total gradient updates (changed from 2048) |
| batch_size | 1024 | Larger batch for more stable gradient updates (changed from 512) |
| ent_coef | 0.05 | Higher exploration to prevent premature policy collapse (changed from 0.01) |
| clip_range | 0.2 | Standard PPO, moderate update steps |
| learning_rate | 0.0003 | PPO default, good with fewer train_cycles |

### 7.5 Strategy Parameters

```python
minimal_roi = {"0": 0.05, "30": 0.03, "60": 0.01}
stoploss = -0.03
use_exit_signal = True
startup_candle_count = 60
can_short = True
process_only_new_candles = True
```

**startup_candle_count rationale:**
Must be >= max indicator period used in feature_engineering functions.
- EMA_50 in expand_basic requires 50 candles minimum
- MACD default (26+9=35) needs 35 candles
- Set to 60 for safety margin (covers all indicators with buffer)
- FreqAI uses this together with max timeframe (4h) to calculate
  total data download requirement automatically.

**Per-pair stoploss:** The single `stoploss = -0.03` applies to all pairs.
For SOL-specific -0.02 stoploss, implement `custom_stoploss()`:

```python
def custom_stoploss(self, pair: str, trade, current_time, current_rate,
                    current_profit, after_fill, **kwargs) -> float | None:
    if "SOL" in pair:
        return -0.02
    return -0.03
```

---

## 8. RL Framework Constraints (from source code analysis)

### 8.1 Forced Disabled Features

BaseReinforcementLearningModel.unset_outlier_removal() forces:
- use_SVM_to_remove_outliers = False
- use_DBSCAN_to_remove_outliers = False
- DI_threshold = False
- shuffle = False

These are silently disabled even if set to True in config.
Explicitly set them to False/0 to avoid confusion.

### 8.2 Total Training Steps Calculation

```
train_df = 75 days * 288 candles/day * 0.75 (train split, test_size=0.25) = 16,200 candles
total_timesteps = train_cycles * len(train_df) = 500 * 16,200 = 8,100,000
With 6 parallel envs: ~1,350,000 steps per env

PPO rollout structure (n_steps=4096, n_envs=6, batch_size=1024, n_epochs=10):
  Rollout buffer   = 4096 * 6 = 24,576 transitions/rollout
  Minibatches      = 24,576 / 1024 = 24 per epoch
  Updates/rollout  = 24 * 10 = 240 gradient updates
  Total rollouts   = 8,100,000 / 24,576 ≈ 329
  Total updates    = 329 * 240 ≈ 78,960

IMPORTANT: The framework default test_size is 0.1 (not 0.25).
If data_split_parameters.test_size is not explicitly set in config,
the actual calculation would be:
  train_df = 75 * 288 * 0.90 = 19,440 candles
  total_timesteps = 500 * 19,440 = 9,720,000
Ensure data_split_parameters.test_size = 0.25 is set in config
(see Section 7.1) to match this calculation.
```

### 8.3 Required Raw OHLCV

The RL framework requires raw OHLCV in feature_engineering_standard():
```python
dataframe["%-raw_close"] = dataframe["close"]
dataframe["%-raw_open"] = dataframe["open"]
dataframe["%-raw_high"] = dataframe["high"]
dataframe["%-raw_low"] = dataframe["low"]
```

These are used by build_ohlc_price_dataframes() to create the price
environment for the RL agent. Without them, training fails.

### 8.4 RL Environment vs Strategy Callbacks

The RL training environment does NOT use strategy callbacks:
- `leverage()` callback: NOT used in training env
- `custom_stoploss()`: NOT used in training env
- `custom_exit()`: NOT used in training env

This means:
- Leverage must be implemented inside `MyRLEnv` via `self.rl_config.get("leverage")`
- Liquidation checks must be in `MyRLEnv.calculate_reward()` or `step()`
- Per-pair leverage (ETH 10x, SOL 5x) requires separate configs/identifiers

Strategy callbacks ARE used in live/dry_run/backtesting after training,
so they must still be implemented for correct execution.

### 8.5 Multiprocessing Tensorboard Limitation

Source: `ReinforcementLearner_multiproc.py` line 82-83:

```
TENSORBOARD CALLBACK DOES NOT RECOMMENDED TO USE WITH MULTIPLE ENVS,
IT WILL RETURN FALSE INFORMATION, NEVERTHELESS NOT THREAD SAFE WITH SB3!!!
```

**Nuance (verified from source code analysis, 2026-02-07):**

The warning above is about `TensorboardCallback._on_step()` which reads
only `env[0]` via `self.locals["infos"][0]` and `get_attr(...)[0]`.
This affects **custom metrics only**. SB3's own logging is unaffected.

| Metric | Multiproc Status | Reason |
|--------|:---:|--------|
| `rollout/ep_rew_mean` | Reliable | SB3 VecMonitor aggregates all N envs |
| `rollout/ep_len_mean` | Reliable | SB3 VecMonitor aggregates all N envs |
| `train/value_loss` | Reliable | Computed from full PPO batch |
| `train/explained_variance` | Reliable | Computed from full PPO batch |
| `actions/*` | 1/N sample | TensorboardCallback reads env[0] only |
| `pnl/*` | 1/N sample | TensorboardCallback reads env[0] only |
| `risk/*` | 1/N sample | TensorboardCallback reads env[0] only |
| `rewards/*` | 1/N sample | TensorboardCallback reads env[0] only |
| `info/total_profit` | 1/N sample | `self.locals["infos"][0]` |

**Practical guidance:**
- SB3 built-in metrics (reward trend, value loss, episode length) can be
  trusted for training health monitoring even with multiproc.
- Custom metrics (action distribution, PNL, liquidation) are directionally
  useful but represent only 1/N of the training experience (N = cpu_count).
- Use single-env `RLDayTrader.py` only when debugging reward function logic
  (e.g., verifying math in 1-2 windows), not for full evaluation runs.

### 8.6 cpu_count Capping

Source: `BaseReinforcementLearningModel.__init__()` line 47-50:

```python
self.max_threads = min(
    self.freqai_info["rl_config"].get("cpu_count", 1),
    max(int(self.max_system_threads / 2), 1),
)
```

The base class caps `cpu_count` to `system_physical_cores / 2`.
Verify with: `python -c "import os; print(os.cpu_count())"`
(Current system: 24 cores, base cap = 12.)

**Note:** `RLDayTrader_multiproc.__init__()` overrides this cap:

```python
if configured_cpu_count > self.max_threads:
    self.max_threads = configured_cpu_count
```

So `cpu_count: 6` in config is used as-is (6 parallel envs), bypassing
the base class cap. This override was intentional to give explicit control.
Reduced from 8 to 6 to free system resources; training time increases ~33%
but total gradient updates remain ~79K (unchanged).

### 8.7 Implementation Review Findings

Post-implementation cross-reference audit against framework source code.
Reviewed on 2026-02-03.

**CRITICAL fix applied: profit_aim**

Original value `profit_aim: 0.005` was compared against LEVERAGED PNL
(`base_pnl * leverage`). With leverage=10, a 0.5% base move = 0.05
leveraged PNL, which is 10x the threshold. This meant almost every
normal ETH price move fell in the "extreme loss" zone (-10 reward).

Fixed to `profit_aim: 0.05` (5% leveraged = 0.5% base target at 10x).
Loss zone thresholds with the fix:
- Small loss: leveraged PNL < 5% (base < 0.5%) -> -1
- Medium loss: leveraged PNL < 10% (base < 1%) -> linear(-1, -5)
- Large loss: leveraged PNL < 15% (base < 1.5%) -> -8
- Beyond: -10

**Removed: leverage_risk_penalty**

The `leverage_risk_penalty` parameter was read in `__init__` but never
used in `calculate_reward()`. Section 6.3 originally claimed it amplified
holding penalties, but this was never implemented. Removed from code,
config, and documentation to avoid confusion.

**Warnings (no code fix needed):**

1. `max_training_drawdown_pct=0.50` with leverage: `max_drawdown = 1 - 0.50 = 0.50`.
   With additive mode (`stake_amount=100`, not "unlimited"),
   `_total_profit` starts at 1.0. A single 5% base adverse move with 10x
   leverage reduces `_total_profit` to ~0.50, ending the episode immediately.
   This may cause very short training episodes initially. Monitor via
   Tensorboard episode length metrics. Consider raising to 0.70 if episodes
   are too short.

2. Post-liquidation trading: After liquidation, `_is_liquidated=True`
   disables further liquidation checks for the remainder of the episode.
   The agent can open new trades without liquidation protection. The
   drawdown check provides a secondary safety net. This is consistent
   with the existing RL4ActionLeverage template behavior.

3. S/R level lists grow without pruning during `_compute_sr_features()`.
   For 75 days of 5m data (~21,600 candles), expect ~200-500 unique levels
   after clustering. Linear scan per candle is O(n * levels) ~ 10M ops,
   acceptable for one-time feature computation.

---

## 9. Implementation Checklist

Reference: existing strategies in `user_data/strategies/` (RLStrategy4Action.py,
RLStrategy4ActionLeverage.py) and models in `user_data/freqaimodels/`
(RL4ActionLeverage.py, RL4ActionLeverage_multiproc.py) can be used as templates.

- [x] Create config_daytrade.json
  - [x] trading_mode, margin_mode, stake_amount, stake_currency
  - [x] fee: 0.0004 (Binance Futures taker)
  - [x] freqai section with all parameters (Section 7.1)
  - [x] data_split_parameters: test_size=0.25
  - [x] conv_width setting
  - [x] rl_config section (Section 7.2)
  - [x] model_training_parameters (Section 7.3)
- [x] Create RLDayTradeStrategy.py
  - [x] feature_engineering_expand_all (RSI, ATR_norm; ADX/Rel_Vol disabled via feature_flags)
  - [x] feature_engineering_expand_basic (EMA, BB, MACD, S/R Flip features)
  - [x] feature_engineering_standard (raw OHLCV, time features)
  - [x] set_freqai_targets (&-action = 0)
  - [x] populate_entry_trend (4-action: Long=2, Short=3)
  - [x] populate_exit_trend (unified Exit=1 for both long and short)
  - [x] leverage() callback (ETH=10.0, SOL=5.0)
  - [x] custom_stoploss() callback (ETH=-0.03, SOL=-0.02)
  - [x] Verify startup_candle_count >= 60
- [x] Create RLDayTrader.py
  - [x] MyRLEnv (Base4ActionRLEnv)
  - [x] __init__: read leverage, liquidation_buffer from rl_config
  - [x] calculate_reward (compressed [-10, +10])
  - [x] get_unrealized_profit (leverage-aware: base_pnl * leverage)
  - [x] _check_liquidation (threshold: -(1/leverage - buffer))
- [x] Create RLDayTrader_multiproc.py
  - [x] Override max_threads
  - [x] Same MyRLEnv as RLDayTrader
  - [x] Note: Tensorboard unreliable with multiproc (see Section 8.5)
- [x] Download data
  - [x] ETH/USDT:USDT (5m, 15m, 1h, 4h)
  - [ ] SOL/USDT:USDT (5m, 15m, 1h, 4h) (unconfirmed)
  - [x] BTC/USDT:USDT (5m, 15m, 1h, 4h) (corr pair)
- [x] Run backtest v1 (skipped single-env, went directly to multiproc)
  - Using: `RLDayTrader_multiproc`, timerange 20230101-20260101
  - Progress: ETH 72/~146 windows (~49%), ~2.3 hrs/window
  - Identifier: `rl-daytrade-v1`
  - SB3 built-in TensorBoard metrics are reliable (see Section 8.5)
- [x] Iterate on reward function
  - Removed overtime penalty (-1.0 at 70% max_dur)
  - Added prop firm forced close (step() override at max_trade_duration_candles)
  - Adjusted n_steps 2048 -> 4096, cpu_count 8 -> 6
- [ ] Run backtest v1-20260220 (current)
  - Using: `RLDayTrader_multiproc`, timerange 20230101-20260201
  - Identifier: `rl-daytrade-v1-20260220`
  - Progress: ETH 110/~161 windows (~68%), approaching final third
  - Changes from v1: forced close, no overtime penalty, n_steps=4096, cpu_count=6->4
  - Crash fix (2026-02-27): MemoryError at window 63 caused AttributeError crash.
    Applied: gc.collect() between windows, None guard on self.model, sorted() feature check.
    cpu_count reduced 6->4 to prevent Windows page file exhaustion.
  - Device fix (2026-03-01): MODELCLASS.load() defaulted device="auto"->GPU, ignoring config.
    Applied: pass device from model_training_parameters to all 3 .load() call sites.
  - Health @ PPO_82 (6 OK, 2 WARN, 1 FAIL / 9 checks):
    - OK: liquidation 0.2%, win rate 49.3%, value_loss decreasing 0.90x, entropy 64% retained, approx_kl 0.011, clip_fraction 0.111
    - WARN: Neutral 70%, invalid 13.2%
    - FAIL: reward_trend -69.1% (misleading: PPO_78 is 1/N anomaly with only 2220 actions)
  - Health @ PPO_94 (6 OK, 3 WARN, 0 FAIL / 9 checks):
    - OK: reward_trend +7.5%, liquidation 0.2%, win rate 50.0%, entropy 59% retained, approx_kl 0.010, clip_fraction 0.106
    - WARN: Neutral 67%, value_loss 1.24x, invalid 15.0%
    - FAIL: none (first batch with 0 FAIL)
  - total_profit trend: PPO_90=16.75 (ATH!), PPO_91=11.88, PPO_92=15.81, PPO_93=13.35, PPO_94=12.88
  - PPO_90 (reliable, 14935 actions): win rate=55.2%, Neutral=63.9%, total_profit=16.75 (new ATH!)
  - PPO_91 (reliable, 15875 actions): win rate=52.6%, Neutral=63.2%, total_profit=11.88
  - PPO_92 (reliable, 13959 actions): win rate=57.2%, Neutral=66.9%, total_profit=15.81
  - PPO_93 (reliable, 11780 actions): win rate=52.3%, Neutral=65.0%, total_profit=13.35
  - PPO_94 (reliable, 9592 actions): win rate=50.0%, Neutral=66.7%, total_profit=12.88
  - No 1/N anomaly in this batch (all windows 9500-15900 actions)
  - Liquidations: PPO_90-94 each had 1 per window (consistent)
  - Entropy 59% at PPO_94 (slightly lower than prior 64%, still healthy)
  - Value loss: 34.6->43.1 (1.24x) at PPO_94, mildly elevated
  - Neutral ratio: 63-67% range, recovered from PPO_82's 70% peak
  - ep_rew_mean range: -1584~-2895, PPO_94=-1584 is one of the best observed
  - Health @ PPO_110 (7 OK, 1 WARN, 1 FAIL / 9 checks):
    - OK: liquidation 0.2%, win rate 54.4%, policy balanced (Neutral=58%), value_loss decreasing 0.67x, entropy 67% retained, approx_kl 0.013, clip_fraction 0.152
    - WARN: invalid 18.3%
    - FAIL: reward_trend -32.4% (ep_rew_mean: -2225 -> -2947; see analysis below)
  - PPO_106 (reliable, 7749 actions): win rate=56.5%, Neutral=56.0%, total_profit=7.82
  - PPO_107 (reliable, 4773 actions): win rate=57.1%, Neutral=45.6%, total_profit=6.40
  - PPO_108 (reliable, 12713 actions): win rate=58.6%, Neutral=49.7%, total_profit=18.29 (new ATH!)
  - PPO_109 (reliable, 6991 actions): win rate=59.8%, Neutral=50.4%, total_profit=9.23
  - PPO_110 (reliable, 8096 actions): win rate=54.4%, Neutral=47.1%, total_profit=8.29
  - No 1/N anomaly in this batch (all windows 4700-12700 actions)
  - Liquidations: PPO_106-108,110 each had 1; PPO_109 had 0 (first zero-liquidation window!)
  - total_profit trend: PPO_108=18.29 surpasses PPO_90=16.75 as new all-time high
  - Neutral ratio: 45-58% range, significantly dropped from PPO_90-94's 63-67%
  - Win rate: 54-60% range, improved from PPO_90-94's 50-57%
  - reward_trend FAIL analysis: ep_rew_mean decline is expected because lower Neutral % means
    more trades attempted, more invalid action penalties incurred. The actual trade quality
    (win rate, total_profit) is improving. This is a healthy sign of more aggressive trading.
  - OOM fix (2026-03-04): Expanded try/except to cover entire backtest window block
    (populate_indicators, DataFrame slicing, training, prediction). Previously only
    self.train() was protected, causing MemoryError at other points to crash the process.
  - Memory optimization (2026-03-04): Recurring OOM at window 117+ (only 26.1 MiB failed).
    Root cause: 116 cached prediction file loads + Windows Python allocator not returning memory.
    Config changes to reduce RAM footprint:
    - `reduce_df_footprint`: false -> true (float64->float32, halves DataFrame RAM)
    - `cpu_count`: 4 -> 2 -> 4 (reverted, speed priority; still safe with other optimizations)
    - `n_steps`: 4096 -> 2048 -> 1024 (rollout buffer 244 -> 122 -> 61 MiB)
    - `batch_size`: 1024 -> 256 (compensate n_steps reduction, restore 16 minibatches/epoch)
    - `pair_whitelist`: removed SOL, ETH only (reduce initial DataFrame ~33%)
    - PPO update impact: 160 gradient updates/rollout restored (buffer/batch * epochs = 4096/256 * 10)
    - Total gradient updates per window: 80K (500 cycles * 160), same as original n_steps=4096 config
    - Previous values for reference: cpu_count=4, n_steps=4096, batch_size=1024
    - timerange extended to 20260303 (from 20260201)
  - New identifier v1-20260307 (2026-03-07): loading 110 cached windows from v1-20260220
    still caused OOM at window 111. Changed to fresh identifier to avoid cache loading overhead.
  - v1-20260307 progress (PPO_1-6, 2026-03-08):
    - total_profit: all positive, PPO_4=4.28 best so far
    - Win rate: trending up (41% -> 44% -> 45% -> 44% -> 47.9%)
    - Neutral%: 61-76%, PPO_6=76% most conservative (small sample 2148 actions)
    - Liquidations: 0 in last 3 windows (PPO_4-6)
    - Entropy retained: stable ~44% at PPO_6, not yet at 30% warning threshold
    - value_loss: PPO_6 spiked 2.19x (25.6->56.2), FAIL -- monitor if transient
  - v1-20260307 progress (PPO_11-14, 2026-03-09):
    - total_profit: all positive (0.51, 2.20, 2.38, 0.78), win rate <50% but RR ratio good
    - Win rate: 37-44% (PPO_11=16.2% is 1/N anomaly, 627 actions; PPO_12-13 = 41-44%)
    - Entropy: recovered to 65% retained (was 44% at PPO_6) -- healthy, ent_coef stays 0.05
    - value_loss: FAIL -> WARN (1.81x), PPO_6 spike was transient as expected
    - policy_collapse: resolved, Neutral back to 59% (was 76% at PPO_6)
    - Liquidations: 0 in PPO_13-14, 1 each in PPO_11-12
    - invalid_actions: 17.3%, slightly higher than before
    - No OOM in 15 windows -- n_steps=1024 memory strategy confirmed working
    - 15/166 windows (~9%), continue to ~30 for full evaluation
  - v1-20260307 progress (PPO_18-22, 2026-03-10):
    - Health: 7 OK, 2 WARN, 0 FAIL -- best result yet (was 5 OK / 3 WARN at PPO_14)
    - total_profit: all positive, PPO_20=3.37 strongest window
    - Win rate: WARN -> OK, 44.8% at PPO_22 (trending up from 37% at PPO_14)
    - value_loss: WARN -> OK (0.73x), fully resolved
    - Entropy: stable 61% retained, healthy
    - Neutral%: 57-71%, PPO_22=71% (mild WARN, expected per HQT)
    - invalid_actions: improved 17.3% -> 13.1% (still WARN)
    - Liquidations: 1 in PPO_21, 0 in PPO_18-20,22
    - No OOM in 22 windows -- memory strategy fully confirmed
    - 22/166 windows (~13%), approaching 30-window evaluation milestone
    - Note: direct comparison with v1-20260220 PPO_106-110 unfair (different market periods)
- [x] Evaluate results (v1-20260307 reached 50 windows, see below)
- [x] Monitor entropy: stable 61%, ent_coef 0.05 sufficient
- [x] Monitor value_loss: fully resolved (FAIL -> WARN -> OK)
- [ ] (Optional) Create separate config for SOL with leverage=5.0
  - n_steps impact analysis (2026-03-12): compared first 32 windows of v1-20260220 vs v1-20260307.
    Rollout experience diversity (24,576 vs 4,096 transitions) is the key driver of policy quality,
    not gradient update count (both 160/rollout). v1-20260220 avg total_profit 4.19 vs v1-20260307 1.52.
  - Config change at PPO_41 (2026-03-13): n_steps 1024->2048, batch_size 256->512.
    Rollout buffer doubles to 8,192 transitions. Gradient updates still 160/rollout.
  - v1-20260307 progress (PPO_33-40, 2026-03-13, last windows under n_steps=1024):
    - total_profit: all positive but declining, avg 0.86 (down from 1.17 at PPO_21-32)
    - Win rate: unstable, 4/8 windows below 35%
    - Entropy: stable ~59%, no issue
    - Liquidations: 0 in all 8 windows
    - Health: 4 OK / 5 WARN / 0 FAIL (reward_trend, win_rate, value_loss newly WARN)
  - v1-20260307 progress (PPO_41-44, n_steps=2048 transition period):
    - Entropy crashed to 40-43% (from ~58%), transition adaptation phase
    - total_profit avg 0.75, Neutral% rose to 75-83%
    - Liquidations: PPO_42 and PPO_44 each had 1 (transition instability)
    - Health: 5 OK / 4 WARN / 0 FAIL
  - v1-20260307 progress (PPO_46-50, 2026-03-13, n_steps=2048 stabilized):
    - Entropy recovered to 52-58% -- transition dip confirmed temporary
    - total_profit avg 1.33 (PPO_46-48 all 1.4+), clear improvement over n_steps=1024 late period
    - Win rate: PPO_47=40.8%, approaching OK threshold
    - Liquidations: 0 in all 5 windows
    - Health: 6 OK / 2 WARN / 1 FAIL
    - FAIL is policy_collapse Neutral=87% at PPO_49 -- small sample (667 actions), not real collapse
    - ent_coef 0.05 still sufficient, no adjustment needed
    - 50/~166 windows (~30%), next milestone PPO_55-60
    - If total_profit stable 1.3+ and entropy stable 50%+, consider n_steps=4096
  - v1-20260307 progress (PPO_53-56, 2026-03-14, n_steps=2048 growth phase):
    - n_steps=2048 effect fully materialized -- three clear phases:
      - PPO_41-44 (adaptation): avg profit 0.75, entropy 40-43%
      - PPO_45-52 (stabilization): avg profit 1.55, entropy 52-58%
      - PPO_53-56 (growth): avg profit **3.57**, entropy 53-63%, win_rate avg 44.7%
    - PPO_56: total_profit=**5.29** (new ATH, surpasses PPO_4=4.28), win_rate=**48.2%**
    - PPO_54: total_profit=4.37, win_rate=44.9%, 946 trades (highest trade count)
    - Liquidations: 0 since PPO_46, 11 consecutive clean windows
    - reward_trend FAIL (-43%): false alarm per Section 6.4 analysis -- ep_rew_mean declines
      because lower Neutral% = more trading = more invalid penalties, actual quality is best ever
    - Health: 5 OK / 3 WARN / 1 FAIL (FAIL is reward_trend false alarm)
    - n_steps 1024->2048 adjustment confirmed successful
    - 56/~166 windows (~34%), next milestone PPO_65
    - If total_profit stable 3.0+ and zero liquidations maintained, consider n_steps=4096 at PPO_80+
  - v1-20260307 progress (PPO_56-60, 2026-03-14, n_steps=2048 maturation phase):
    - Health: **7 OK / 2 WARN / 0 FAIL** -- best since PPO_22, previous FAIL resolved
    - PPO_59: win_rate=**49.5%** (v1-20260307 all-time best), total_profit=2.27
    - PPO_58: total_profit=3.37, 940 trades, win_rate=44.7%
    - Neutral% dropped to **61%** (PPO_59) -- model trading more aggressively (was 87% at PPO_49)
    - Entropy 60-70% -- highest ever, PPO_57=70.1% peak
    - Liquidations: PPO_59 had 1 (first since PPO_46, 0.4% rate, within OK threshold)
    - Invalid actions rose to 16.7% -- natural side effect of more aggressive trading
    - Four phases now identified:
      - adaptation (PPO_41-44): entropy 40-43%, profit avg 0.75
      - stabilization (PPO_45-52): entropy 52-58%, profit avg 1.55
      - growth (PPO_53-56): entropy 53-63%, profit avg 3.57
      - maturation (PPO_56-59): entropy 60-70%, profit avg 2.49, win_rate avg 46.5%
    - 60/~166 windows (~36%), next milestone PPO_65
    - Maintain current config, no adjustment needed
  - v1-20260307 progress (PPO_61-65, 2026-03-15, n_steps=2048 peak phase):
    - Health: **8 OK / 1 WARN / 0 FAIL** -- all-time best health
    - policy_collapse WARN resolved: Neutral=**58%** (first time OK, was 87% at PPO_49)
    - Only remaining WARN: invalid_actions 18.1% (structural, due to aggressive trading)
    - PPO_62: total_profit=**5.80** (new ATH), win_rate=46.4%, 1283 trades
    - PPO_63: total_profit=4.80, PPO_64: total_profit=3.86 -- three consecutive 3.86+ windows
    - Win rate stable at 46.2-46.6% across large-sample windows (871-1283 trades)
    - Entropy 66% at PPO_64 -- excellent exploration maintained
    - Liquidations: 1 per window (PPO_61-64), rate 0.1% due to high trade volume
    - Five phases now identified (added peak phase):
      - adaptation (PPO_41-44): profit avg 0.75
      - stabilization (PPO_45-52): profit avg 1.55
      - growth (PPO_53-56): profit avg 3.57
      - maturation (PPO_56-59): profit avg 2.49, win_rate 46.5%
      - peak (PPO_61-64): profit avg **4.08**, win_rate 46.2%, Neutral 58%
    - 65/~166 windows (~39%), next milestone PPO_80
    - Maintain current config, consider n_steps=4096 at PPO_80+ if profit stable 3.0+
  - v1-20260307 progress (PPO_65-76, 2026-03-16, n_steps=2048 breakthrough phase):
    - Health: **8 OK / 1 WARN / 0 FAIL** -- maintained best
    - PPO_73: total_profit=**12.32** (new ATH, 2x previous ATH), win_rate=**53.6%**, 931 trades
    - PPO_74: total_profit=10.03, win_rate=50.1%, 951 trades -- consecutive double-digit profit
    - PPO_75: total_profit=5.51, win_rate=52.2%, Neutral=**44%** (most aggressive ever)
    - Win rate first time stable >50%: 5/6 windows PPO_70-75 above 50%
    - Approaching v1-20260220 PPO_106-110 levels (avg profit 10 vs 9.3, win rate 57% vs 52%)
    - Entropy 69-75% -- all-time high, model still has room to improve
    - Invalid actions trending up: 22-29%, PPO_75=28.7% nearing 25% FAIL threshold
    - Liquidations: ~1 per window, rate 0.1-0.3%, acceptable at high trade volume
    - Comparison with v1-20260220 PPO_106-110:
      - v1-20260220: avg profit 10.01, win_rate 56.5%, entropy 64.9%
      - v1-20260307 PPO_73-75: avg profit 9.29, win_rate 52.0%, entropy 71.5%
      - v1-20260307 higher entropy suggests further improvement potential
    - 76/~166 windows (~46%), evaluate n_steps=4096 at PPO_80
    - Monitor invalid_actions: if crosses 25% consistently, may need reward adjustment
  - v1-20260307 progress (PPO_76-85, 2026-03-17):
    - Health: **7 OK / 2 WARN / 0 FAIL**
    - PPO_76-79 extended high peak: PPO_79=10.92, PPO_78=10.04, PPO_77=8.90, PPO_76=8.78
    - PPO_73-79 combined: avg profit **9.49**, avg win_rate **51.8%** -- surpassed v1-20260220
    - PPO_80-84 stabilized lower: avg profit 4.27, avg win_rate 45.7%
      - Normal market-driven variation, not a regression (still 3x n_steps=1024 levels)
    - policy_collapse WARN returned: Neutral=60% (was 44% at PPO_75, oscillates with market)
    - Invalid actions self-corrected: 28.7% (PPO_75) -> 16.8% (PPO_84), FAIL risk resolved
    - approx_kl and clip_fraction trending up (0.021 and 0.177), approaching WARN thresholds
    - 85/~166 windows (~51%), past halfway
    - Defer n_steps=4096 evaluation to PPO_90-95, wait for recovery from current dip
  - v1-20260307 progress (PPO_90-94, 2026-03-18, profit recovery confirmed):
    - Health: **7 OK / 2 WARN / 0 FAIL**
    - WARNs: value_loss (1.49x, new), invalid_actions (18.9%, persistent)
    - PPO_91: total_profit=**11.21** (2nd highest single window after PPO_73=12.32 ATH)
    - PPO_93: total_profit=9.18, PPO_94: total_profit=9.73 -- consistent high output
    - PPO_90-94 avg profit **6.57**, avg win_rate **~51.4%** -- clear recovery from PPO_80-84 dip (4.27)
    - Small sample caveat: PPO_90 (852 actions) and PPO_92 (1034 actions) show low profit (1.33/1.38) due to 1/N sampling
    - approx_kl self-corrected: 0.021 -> **0.0113** (was approaching WARN, now healthy)
    - clip_fraction self-corrected: 0.177 -> **0.145** (was approaching WARN, now comfortable)
    - policy_collapse resolved: Neutral=**55%** (was 60% WARN at PPO_84), back to balanced
    - Entropy improved: **70%** retained (was 63-65% at PPO_80-84)
    - Value loss new WARN: 25.84 -> 38.48 (1.49x), likely market regime shift, monitor
    - Liquidations: 1 per window consistently
    - 94/~166 windows (~57%)
    - n_steps=4096 evaluation: profit recovered (6.57 > 5.0 threshold) but value_loss WARN active
    - Defer n_steps=4096 to PPO_100-105 after value_loss stabilizes
  - v1-20260307 progress (PPO_101-105, 2026-03-18, zero liquidation milestone):
    - Health: **6 OK / 3 WARN / 0 FAIL**
    - WARNs: policy_collapse (Neutral=61%, returned), value_loss (1.37x), invalid_actions (16.7%)
    - **Zero liquidations in all 5 windows** -- first time in v1-20260307 history
    - PPO_103: total_profit=4.05 (batch best), PPO_102: 3.72, PPO_101: 2.51
    - PPO_101-105 avg profit **3.01**, avg win_rate **46.2%** -- conservative phase
    - All windows positive (2.32-4.05), no loss windows
    - PPO_102 transient spike: approx_kl=0.036, clip_fraction=0.276 (both above WARN)
      - Self-corrected by PPO_103 (kl=0.013, clip=0.135), PPO_105 excellent (kl=0.012, clip=0.112)
    - Entropy declining: 70% (PPO_94) -> 61% (PPO_105), -9pp over 11 windows, monitor
    - Neutral% rose back to 61% (WARN), model more conservative than PPO_90-94 (55%)
    - Value loss still WARN but slightly improved (1.49x -> 1.37x)
    - Invalid actions improved: 18.9% -> 16.7%
    - Profit vs risk trade-off: lower profit but zero liquidations suggests better risk control
    - 105/~166 windows (~63%)
    - n_steps=4096 evaluation at PPO_100-105: NOT recommended (avg profit 3.01 < 5.0, entropy declining)
    - Defer n_steps=4096 to PPO_115-120, wait for profit rebound + entropy stabilization
    - Monitor entropy: if continues <55%, consider ent_coef 0.05 -> 0.06
  - v1-20260307 progress (PPO_111-114, 2026-03-20, profit rebound to peak levels):
    - Health: **6 OK / 3 WARN / 0 FAIL** (PPO_115 in progress, health based on PPO_114)
    - WARNs: policy_collapse (Neutral=61%), value_loss (1.04x, nearly resolved), invalid_actions (17.3%)
    - PPO_114: total_profit=**10.82** (3rd highest ever), win_rate=51.0%, 14849 actions (reliable)
    - PPO_112: total_profit=**10.39**, PPO_113: 9.32, PPO_111: 7.18
    - PPO_111-114 avg profit **9.43**, avg win_rate **51.7%** -- matches PPO_73-79 peak (9.49)
    - PPO_111 win_rate=**57.6%** -- v1-20260307 all-time highest (medium sample 5896 actions)
    - Entropy recovered: 61% (PPO_105) -> **67%** (PPO_114), declining trend fully reversed
    - Value loss nearly resolved: 1.37x -> **1.04x**, no longer a practical concern
    - Liquidations: PPO_112 and PPO_113 each had 1 (back to ~0.5/window, normal for active trading)
    - approx_kl: last=0.019, approaching 0.02 WARN threshold, mean=0.015 still healthy
    - Training cycle pattern confirmed: peak -> dip -> recovery -> conservative -> **rebound**
    - 115/~166 windows (~69%)
    - n_steps=4096 evaluation at PPO_115-120: NOT recommended (system performing excellently at n_steps=2048)
    - Maintain current config through completion, consider n_steps=4096 for v2 initial config
  - v1-20260307 progress (PPO_117-121, 2026-03-21, new ATH):
    - Health: **6 OK / 3 WARN / 0 FAIL**
    - WARNs: liquidation_rate (9.1%, PPO_121 tiny sample artifact), value_loss (1.17x), invalid_actions (22.1%, tiny sample)
    - PPO_121 extreme 1/N artifact: only 127 actions (11 trades) from env[0], health metrics unreliable
    - PPO_118: total_profit=**16.30** -- **new all-time high** (surpasses PPO_73=12.32 by 32%)
      - Large reliable sample (13898 actions, 1410 trades), win_rate=52.3%, Neutral=56.9%
    - PPO_119: total_profit=10.97, win_rate=55.7%
    - PPO_120: total_profit=9.87, win_rate=**56.2%** (large/medium sample best)
    - PPO_117-120 avg profit **9.72**, avg win_rate **52.8%** -- surpasses all previous peaks
    - Win rate trending up: PPO_118=52.3%, PPO_119=55.7%, PPO_120=56.2%
    - Entropy very strong: PPO_117=71.6%, PPO_120=**72.3%** (near all-time high)
    - policy_collapse resolved: Neutral=56% (OK), down from 61% WARN
    - approx_kl improved: 0.019 -> 0.013, comfortable margin
    - clip_fraction improved: 0.168 -> 0.139
    - Value loss absolute level rising (41-49 range vs 25-31 at PPO_114), but ratio only 1.17x
    - Liquidations: 1 per window consistently (PPO_117-121), normal for active trading
    - 121/~166 windows (~73%)
    - Maintain n_steps=2048 through completion, no config changes needed
  - v1-20260307 progress (PPO_121-123, 2026-03-21, high-quality conservative phase):
    - Health: **11 OK / 2 WARN / 1 FAIL** (14 checks after analyze-rl upgrade)
    - FAIL: reward_trend (-24.5%, HQT false alarm -- more trading = more invalid penalties)
    - WARNs: policy_collapse (Neutral=65%), invalid_actions (19.0%)
    - Unique pattern: **profit declining but win rate at all-time high**
      - PPO_121-123 avg profit 3.73 (down from ATH peak 9.72)
      - PPO_123 win_rate=**61.4%** -- v1-20260307 all-time highest
      - Model trading less frequently but with much higher quality
    - PPO_122: tiny sample (274 actions, 25 trades), total_profit=1.08
    - PPO_123: total_profit=3.88, win_rate=61.4%, Neutral=65%, 2392 actions
    - Value loss fully resolved: 0.87x (decreasing, was 1.49x at PPO_94)
    - Entropy strong: 67-74% range, PPO_122=74.2% near all-time high
    - explained_variance: 0.69-0.88 range (value function predictive, new metric)
    - Long/Short bias: 54%L/46%S (balanced, new metric)
    - Liquidations: 1 per window consistently
    - 123/~166 windows (~74%)
    - Training cycle updated: peak -> dip -> recovery -> conservative -> rebound -> ATH -> **HQ conservative**
    - Maintain current config, expect recovery around PPO_125-130
  - v1-20260307 progress (PPO_128-131, 2026-03-22, profit recovery from HQ conservative):
    - Health: **13 OK / 1 WARN / 0 FAIL** -- best since 14-check upgrade
    - WARN: invalid_actions (20.7%, only remaining WARN)
    - Previous FAIL (reward_trend) resolved: +40.7% (was -24.5%)
    - Previous WARN (policy_collapse) resolved: Neutral=59% (was 65%)
    - PPO_130: total_profit=**12.90** -- 2nd highest ever (after PPO_118=16.30)
      - Large reliable sample (12129 actions, 1200 trades), win_rate=51.0%
    - PPO_131: total_profit=8.88, win_rate=52.9%, 6617 actions (reliable)
    - PPO_128-131 avg profit **6.51** (+75% from HQ conservative 3.73)
    - Excluding small-sample PPO_128 (546 actions): avg profit **8.39**
    - Win rate normalized to 52.9% (from anomalous 61.4% HQ phase)
    - Value loss excellent: 0.57x (decreasing)
    - Entropy stable: 64-69% range
    - explained_variance: 0.811 (improved from 0.692, value function stronger)
    - Long/Short bias: 49%L/51%S (perfectly balanced)
    - Liquidations: 1 per window consistently
    - PPO_132 in progress (162/989 iterations), value_loss spike 93.35 (early-training, monitor)
    - 132/~166 windows (~80%), entering final fifth
    - Training cycle continues: HQ conservative -> **recovery** (confirmed)
    - Maintain current config through completion
  - v1-20260307 progress (PPO_132-135, 2026-03-22, new ATH surge):
    - Health: **12 OK / 1 WARN / 1 FAIL** (PPO_136 in progress, health based on PPO_135)
    - FAIL: invalid_actions (**25.5%**, first time crossing 25% FAIL threshold)
    - WARN: value_loss (1.37x, returned from OK)
    - PPO_135: total_profit=**17.56** -- **new all-time high** (surpasses PPO_118=16.30 by 7.7%)
      - Large reliable sample (11758 actions, 1393 trades), win_rate=54.4%
    - PPO_134: total_profit=**15.75** (3rd highest), 14227 actions, Neutral=46.2% (most aggressive)
    - PPO_132-135 avg profit **13.22** -- **highest batch average ever** (prev best 9.72)
    - Invalid actions FAIL is structural: low Neutral% (46-51%) = more trade attempts = more invalid
      - High profits confirm valid trades are extremely profitable, not a model degradation
    - PPO_134 transient spikes: approx_kl=0.022, clip_fraction=0.228 (both above WARN, self-corrected)
    - Value loss returned to WARN (1.37x) due to higher return variance in high-profit period
    - PPO_132 value_loss spike (93.35 from prior check) self-resolved to 32.87 (0.97x)
    - Entropy: 68-71% range, healthy
    - explained_variance: 0.71-0.76 (predictive)
    - Long/Short: 56%L/44%S (balanced)
    - 136/~166 windows (~82%)
    - Maintain current config, invalid_actions FAIL expected to self-correct as Neutral% oscillates
  - v1-20260307 progress (PPO_142-145, 2026-03-23, post-ATH plateau):
    - Health: **11 OK / 3 WARN / 0 FAIL** (PPO_146 in progress, health based on PPO_145)
    - WARNs: reward_trend (-19.7%, near FAIL), value_loss (1.25x), invalid_actions (22.9%)
    - **Invalid actions self-corrected from FAIL (25.5%) back to WARN (22.9%)** -- as predicted
    - **PPO_142-144: three consecutive zero-liquidation windows** (2nd time in v1-20260307)
    - PPO_145: total_profit=**10.25**, win_rate=52.6%, 11919 actions (reliable)
    - PPO_142: total_profit=9.71 (zero liquidations), PPO_144: 8.89 (zero liquidations)
    - PPO_142-145 avg profit **7.67** -- stable post-ATH plateau, all windows positive
    - PPO_143: total_profit=1.84, Neutral=67.4% (conservative window, small sample 4429 actions)
    - PPO_142 transient: approx_kl=0.020, clip_fraction=0.198 (both at WARN edge, self-corrected)
    - reward_trend WARN (-19.7%): includes PPO_146 in-progress, likely improves when complete
    - Entropy trend very positive: +13.0pp (62%->75%), exploration capacity increasing
    - explained_variance: 0.75 (predictive, stable)
    - Long/Short: 53%L/47%S (balanced)
    - Liquidations: only 1 in 4 completed windows (PPO_145), best risk control since PPO_101-105
    - 146/~166 windows (~88%), ~20 windows remaining
    - Maintain current config through completion
  - v1-20260307 progress (PPO_146-147, 2026-03-24, late-stage conservative):
    - Health: **13 OK / 1 WARN / 0 FAIL** (PPO_148 in progress, health based on PPO_147)
    - Only WARN: invalid_actions (23.3%) -- stable, structural
    - **Two previous WARNs resolved**: reward_trend OK (+53.5%), value_loss OK (0.83x)
    - PPO_146: total_profit=1.51, win_rate=43.0%, 1115 actions (small sample)
    - PPO_147: total_profit=2.88, **win_rate=61.7%** (79W/49L), 1342 actions (small sample)
    - PPO_146-147 avg profit **2.20** -- lower than plateau (7.67), fewer sampled trades
    - Small sample sizes (1115, 1342 vs typical 10K+): env[0] last-episode timing artifact, not data-end effect
    - PPO_146 value_loss spike 2.26x (34->77, FAIL level) -- self-corrected in PPO_147 to 0.83x
    - Entropy trend positive: +9.9pp across batch (69%->79%)
    - explained_variance: 0.774 (predictive, stable)
    - Long/Short: 56%L/44%S (balanced)
    - approx_kl: 0.0174, clip_fraction: 0.146 (both comfortable)
    - Liquidations: 1 each in PPO_146, PPO_147 (small sample effect)
    - 148/~166 windows (~89%), ~18 windows remaining
    - Maintain current config through completion
  - v1-20260307 progress (PPO_152-156, 2026-03-24, stable convergence):
    - Health: **11 OK / 3 WARN / 0 FAIL** (all 5 windows complete, health based on PPO_156)
    - WARNs: policy_collapse (Neutral=65%), value_loss (1.04x), invalid_actions (18.8%)
    - **New WARN: policy_collapse** -- Neutral=65% just over 60% threshold, more selective trading
    - **New WARN: value_loss** -- 1.04x mild; PPO_155 spike 2.67x (FAIL level) self-corrected in PPO_156
    - **Invalid actions improved**: 23.3% -> **18.8%** (-4.5pp), consistent with higher Neutral%
    - PPO_152-156 profit ascending: 1.05 -> 2.13 -> 2.63 -> 3.28 -> **4.17** (positive trend)
    - PPO_152-156 avg profit **2.65** -- all windows positive, stable convergence phase
    - Win rate: 45-50% range, avg ~47.4% (balanced, sustainable)
    - **Every window has exactly 1 liquidation** -- consistent low-level risk, liquidation rate 0.3%
    - PPO_155: value_loss spike 2.67x + entropy retained only 53.7% (lowest in batch) -- transient
    - Explained variance declining: 0.781 -> 0.612 (still OK >0.5, monitor)
    - Long/Short: 49%L/51%S (most balanced phase in entire training)
    - Entropy trend stable: -1.4pp across batch (64%->62%)
    - approx_kl: 0.013, clip_fraction: 0.132 (both very comfortable)
    - Sample sizes recovered: 2558-7024 (vs PPO_146-147's ~1300)
    - 156/~166 windows (~94%), ~10 windows remaining
    - Maintain current config through completion
  - v1-20260307 progress (PPO_163-166, 2026-03-25, final surge):
    - Health: **11 OK / 2 WARN / 1 FAIL** (PPO_167 in progress ~12%, health based on PPO_166)
    - FAIL: invalid_actions (25.3%) -- back to FAIL from WARN, linked to aggressive trading
    - WARNs: approx_kl (mean=0.023, last=0.036), clip_fraction (mean=0.216, last=0.248)
    - **Two previous WARNs resolved**: policy_collapse OK (Neutral=50%), value_loss OK (0.77x)
    - **PPO_164: total_profit=13.19** -- 2nd highest ever (after PPO_135=17.56 ATH)
      - Zero liquidations, 50/50 Long/Short, 11554 actions, explained_var=0.837
    - **PPO_163-164: consecutive zero-liquidation windows**
    - PPO_163: total_profit=2.63, win_rate=60.7%, 511 actions (timing artifact)
    - PPO_165: total_profit=2.75, win_rate=46.3%, Neutral=47.3% (most aggressive in batch)
    - PPO_166: total_profit=5.12, win_rate=54.9%, Neutral=49.6%
    - PPO_163-166 avg profit **5.92** -- strong rebound from convergence phase (2.65)
    - approx_kl/clip_fraction elevated across PPO_163-166 (not just last step) -- regime adaptation
      - PPO_166 approx_kl=0.036 and clip_fraction=0.248 are all-time highs
      - Still well below FAIL thresholds (0.05 / 0.4), training ending so no intervention needed
    - Invalid actions FAIL (25.3%): same pattern as PPO_132-135 -- low Neutral% + aggressive trading
    - Entropy trend stable: +1.1pp (63%->64%)
    - Explained variance: 0.599 (OK, continued decline from 0.612)
    - Long/Short: 44%L/56%S (slightly Short-biased, within OK)
    - **167 total windows, PPO_167 likely final window** -- original estimate was ~166
    - ETH training essentially complete
    - Next steps: **migrated to `docs/2026/20260401/rl-daytrade-strategy-analysis-0401.md`** (v2 improvement plan, 12 items)
  - total_profit metric analysis (2026-03-22):
    - total_profit = per-episode cumulative realized leveraged PNL from env[0]
    - Formula (additive, stake_amount=100): `_total_profit += base_pnl * leverage` on each Exit
    - Starts at 1.0 each episode, so PPO_135=17.56 means +16.56 leveraged PNL accumulated
    - Liquidation uses multiplicative formula (`*= 0.925`) regardless of compound_trades flag
    - This multiplicative liquidation in additive mode is a potential inconsistency for v2 review
    - Unrealized profit does NOT affect total_profit (only used for drawdown kill-switch)

---

## 10. Timeframe Analysis Detail

### 10.1 Why NOT 1-Minute

| Factor | Analysis |
|--------|----------|
| Signal-to-noise ratio | Extremely low at 1m; most price moves are noise |
| Indicator reliability | RSI/ADX/MACD on 1m produce frequent false signals |
| Feature explosion | 1440 candles/day, massive data for training |
| Correlation with 5m | 5:1 ratio, ~95%+ correlation, near-zero marginal info |
| Training cost | Data processing time increases significantly |
| Overfit risk | Agent learns noise patterns that don't generalize |
| RL environment | More steps per episode but each step is less meaningful |

Conclusion: 1m adds computational cost without meaningful information gain.
The 5m candle already incorporates the micro-structure of its constituent
1m candles through OHLC values.

### 10.2 Why NOT 3-Minute

| Factor | Analysis |
|--------|----------|
| Proximity to 5m | 5:3 ratio, indicators nearly identical |
| Multicollinearity | 3m RSI and 5m RSI correlation > 0.95 |
| Exchange support | Not universally supported (Binance yes, others vary) |
| Information gain | Marginal at best, effectively redundant with 5m |
| Feature cost | Same expansion cost as any other TF |
| Practical value | No unique structural information vs 5m |

Conclusion: 3m is too close to 5m. The information overlap is nearly
complete. If sub-5m granularity were needed, 1m would be the only
meaningful option (though still not recommended).

### 10.3 Why YES 4-Hour

| Factor | Analysis |
|--------|----------|
| Macro trend context | Identifies dominant market direction |
| S/R significance | 4h S/R levels are watched by institutions |
| Low correlation with 5m | 1:48 ratio, genuinely different information |
| Feature cost | Only 6 candles/day, minimal expansion cost |
| Trend filter | Helps agent avoid counter-trend trades |
| S/R Flip reliability | 4h flip zones have much higher hit rate |
| Training data | 75 days = 450 candles, sufficient for indicators |

Conclusion: 4h provides the strongest cost-to-value ratio of any
additional timeframe. It gives the agent a "macro lens" that complements
the 5m "micro lens" without redundancy.

### 10.4 Optimal Timeframe Hierarchy

```
4h  -> "What is the dominant trend?" (macro filter)
1h  -> "What is the current swing direction?" (medium context)
15m -> "What is the intraday structure?" (short-term context)
5m  -> "When exactly to enter/exit?" (execution)

Information flow: 4h trend -> 1h direction -> 15m structure -> 5m timing
Each step provides approximately 3-4x more granularity.
This logarithmic spacing maximizes information diversity.
```

### 10.5 Final Timeframe Decision

```
include_timeframes: ["5m", "15m", "1h", "4h"]

Rejected: 1m (noise), 3m (redundant with 5m)
Accepted: 4h (macro context, high-value S/R, minimal cost)
```

---

## 11. References

### 11.1 Local Documentation (docs/)

The following official Freqtrade docs are in this repository.
Use these as the primary reference during implementation.

| Doc File | Content | Relevance |
|----------|---------|-----------|
| [freqai-reinforcement-learning.md](../../freqai-reinforcement-learning.md) | RL agent/environment architecture, custom reward function, action spaces (3/4/5), Tensorboard, state info, `calculate_reward()` examples | **Core** - Primary reference for RL model/env design |
| [freqai-feature-engineering.md](../../freqai-feature-engineering.md) | `feature_engineering_expand_all/basic/standard()`, `set_freqai_targets()`, feature naming (`%` prefix), metadata usage, data pipeline, outlier detection (DI/SVM/DBSCAN), PCA, weight_factor | **Core** - Primary reference for feature engineering |
| [freqai-parameter-table.md](../../freqai-parameter-table.md) | Complete parameter reference: general config, feature_parameters, data_split_parameters, rl_config (train_cycles, max_trade_duration_candles, model_type, net_arch, etc.), PyTorch params | **Core** - Config parameter lookup |
| [freqai-configuration.md](../../freqai-configuration.md) | Minimum config setup, building a FreqAI strategy, `populate_indicators()` with `self.freqai.start()`, prediction models, dynamic target threshold | **Core** - Strategy structure reference |
| [freqai-running.md](../../freqai-running.md) | Backtesting with FreqAI, live/dry run, model retraining, continual learning, data download | **Core** - Training/backtesting workflow |
| [freqai.md](../../freqai.md) | FreqAI overview, supported models, general architecture | Context |
| [freqai-developers.md](../../freqai-developers.md) | Advanced: custom models, custom environments, extending the framework | Reference for custom model class |
| [leverage.md](../../leverage.md) | Futures trading, margin modes (isolated/cross), shorting, pair naming (`base/quote:settle`), liquidation, leverage risks | **Core** - Leverage/futures config |
| [stoploss.md](../../stoploss.md) | Stoploss configuration, stoploss on exchange, trailing stoploss, `stoploss_price_type` for futures | **Core** - Risk management |
| [strategy-customization.md](../../strategy-customization.md) | IStrategy interface, entry/exit signals, `populate_entry_trend()`, `populate_exit_trend()`, order types | **Core** - Strategy base class |
| [strategy-callbacks.md](../../strategy-callbacks.md) | `leverage()`, `custom_stoploss()`, `custom_exit()`, `confirm_trade_entry()` | **Core** - Leverage callback implementation |
| [strategy-advanced.md](../../strategy-advanced.md) | Advanced strategy features, informative pairs | Reference |
| [configuration.md](../../configuration.md) | General Freqtrade config, trading_mode, margin_mode, stake_amount, dry_run | Reference |
| [data-download.md](../../data-download.md) | `freqtrade download-data` command, exchange data, timeframes | Reference for data preparation |
| [backtesting.md](../../backtesting.md) | Backtesting commands, timerange, results analysis | Reference for evaluation |

### 11.2 Key Quotes from Official Docs

#### Reward Function Design (freqai-reinforcement-learning.md)

> "The best reward functions are ones that are continuously differentiable,
> and well scaled. In other words, adding a single large negative penalty
> to a rare event is not a good idea, and the neural net will not be able
> to learn that function. Instead, it is better to add a small negative
> penalty to a common event."

Implication: Our [-10, +10] compressed reward with linear/exponential
scaling follows this guidance.

#### RL Environment Limitation (freqai-reinforcement-learning.md)

> "The RL training environment is much more simplified. It does not
> incorporate any of the complicated strategy logic, such as callbacks
> like custom_exit, custom_stoploss, leverage controls, etc."

Implication: Leverage simulation and liquidation checks MUST be
implemented inside MyRLEnv, not in the strategy callbacks.

#### Agent Exploitation Risk (freqai-reinforcement-learning.md)

> "Classifiers and Regressors have strengths that RL does not have such
> as robust predictions. Improperly trained RL agents may find 'cheats'
> and 'tricks' to maximize reward without actually winning any trades."

Implication: No entry reward (set to 0) to prevent overtrading exploit.

#### Continual Learning Warning (freqai-parameter-table.md)

> "Beware that this is currently a naive approach to incremental learning,
> and it has a high probability of overfitting/getting stuck in local
> minima while the market moves away from your model."

Implication: Confirms our decision to set continual_learning = false.

#### RL Forced Outlier Removal Disable (source code)

BaseReinforcementLearningModel.unset_outlier_removal() at
`freqtrade/freqai/RL/BaseReinforcementLearningModel.py:81-97` forces:
- use_SVM_to_remove_outliers -> False
- use_DBSCAN_to_remove_outliers -> False
- DI_threshold -> False
- shuffle -> False

Not documented in official docs, discovered from source code analysis.
Set these explicitly to False/0 in config to avoid confusion.

#### Feature Expansion Formula (freqai-feature-engineering.md)

expand_all total = indicator_periods_candles * include_timeframes
                   * include_shifted_candles * include_corr_pairs

expand_basic total = include_timeframes * include_shifted_candles
                     * include_corr_pairs

standard total = as defined (no expansion)

#### Leverage Warning (leverage.md)

> "Do not trade with a leverage > 1 using a strategy that hasn't shown
> positive results in a live run using the spot market."

Implication: Must validate strategy in dry_run before going live
with 10x leverage.

### 11.3 Source Code References

#### Framework Files

| File | Key Content |
|------|-------------|
| `freqtrade/freqai/freqai_interface.py` | IFreqaiModel base class, training pipeline, sliding window |
| `freqtrade/freqai/data_kitchen.py` | Data split, feature filtering, weight_factor (default test_size=0.1) |
| `freqtrade/freqai/data_drawer.py` | Model persistence, historic predictions |
| `freqtrade/freqai/RL/BaseEnvironment.py` | Base gym.Env, positions, actions, unrealized profit, fee handling |
| `freqtrade/freqai/RL/Base4ActionRLEnv.py` | 4-action step/is_tradesignal/is_valid logic, position flip constraint |
| `freqtrade/freqai/RL/BaseReinforcementLearningModel.py` | RL train flow, unset_outlier_removal, predict, pack_env_dict |
| `freqtrade/freqai/prediction_models/ReinforcementLearner.py` | fit() with continual_learning, default MyRLEnv (Base5Action) |
| `freqtrade/freqai/prediction_models/ReinforcementLearner_multiproc.py` | SubprocVecEnv, Tensorboard thread-safety warning |
| `freqtrade/templates/FreqaiExampleStrategy.py` | Official strategy template |
| `config_examples/config_freqai.example.json` | Official config example (includes trading_mode, margin_mode) |

#### Existing Custom Models (user_data/freqaimodels/) - Use as Templates

| File | Key Content |
|------|-------------|
| `ReinforcementLearner4Action.py` | Base 4-action model, multi-tier reward, no leverage |
| `RL4ActionLeverage.py` | 10x leverage, liquidation check (buffer=0.05), -1000 penalty |
| `ReinforcementLearner4Action_multiproc.py` | Multiproc 4-action, configurable reward params |
| `RL4ActionLeverage_multiproc.py` | Most advanced: multiproc + leverage + dense rewards + 3-tier loss zones |

#### Existing Custom Strategies (user_data/strategies/) - Use as Templates

| File | Key Content |
|------|-------------|
| `RLStrategy4Action.py` | 4-action strategy with expand_all/basic/standard, entry/exit mapping |
| `RLStrategy4ActionLeverage.py` | Extends RLStrategy4Action with leverage() callback returning 10.0 |

### 11.4 External References

| Resource | URL |
|----------|-----|
| FreqAI RL Docs (web) | https://www.freqtrade.io/en/stable/freqai-reinforcement-learning/ |
| FreqAI Config Docs (web) | https://www.freqtrade.io/en/stable/freqai-configuration/ |
| FreqAI Parameter Table (web) | https://www.freqtrade.io/en/stable/freqai-parameter-table/ |
| Stable Baselines3 PPO | https://stable-baselines3.readthedocs.io/en/master/modules/ppo.html |
| SB3 Custom Policy | https://stable-baselines3.readthedocs.io/en/master/guide/custom_policy.html |
| S/R in ML (MDPI 2022) | https://www.mdpi.com/2227-7390/10/20/3888 |
| ML S/R Detection (K-Means) | https://github.com/judopro/Stock_Support_Resistance_ML |
| Technical Indicators in ML (arXiv 2024) | https://arxiv.org/html/2412.15448v1 |
| S/R Flip Trading Strategy | https://tradingcoach.co.in/price-action-flips-support-resistance-trading-strategy/ |
| SR Flip Definition | https://forexbee.co/sr-flip-in-forex/ |
