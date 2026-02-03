# RL Day Trade Strategy - Analysis Document

> Date: 2026-02-03
> Status: Analysis Phase
> Target: Intraday RL trading strategy for ETH + SOL futures

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
- Set tighter stoploss at -0.02 (-2%)
- Apply stricter risk penalties in reward function

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

```python
# Indicators (4):
RSI       -> Overbought/oversold momentum
ADX       -> Trend strength
ATR_norm  -> Volatility (ATR / close, scale-invariant)
Rel_Vol   -> Relative volume (volume / rolling mean)

# Expansion:
# 4 indicators x 2 periods (10, 20) x 4 TF x 3 shifted (0,1,2) x 2 pairs
# = 4 x 2 x 4 x 3 x 2 = 192 features
```

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
expand_all:    192 features
expand_basic:  288 features
standard:        8 features
--------------------------
Total:        ~488 features
```

Note: Exact count depends on final indicator list and config parameters.

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

Step 4: Detect Break
  support_break:    close < support AND close[N_prev] > support (sustained)
  resistance_break: close > resistance AND close[N_prev] < resistance (sustained)

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

1. Compressed range: [-10, +10] for stable training gradients
2. No entry reward: prevent overtrading exploit
3. Continuously differentiable: smooth gradient flow
4. Asymmetric: penalize losses more than reward gains
5. Time-aware: reward quick profitable exits

Official guidance: "The best reward functions are ones that are
continuously differentiable, and well scaled."

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
  Duration > 70% max                  -> additional -1

Exit action:
  Any profit                          -> +clamp(pnl * 50, 1, 8)
  Quick profit (< 30% max_dur)        -> +clamp(pnl * 50, 1, 8) + 2
  Small loss (> -profit_aim)          -> -1
  Medium loss (> -2x profit_aim)      -> linear(-1, -5)
  Large loss (> -3x profit_aim)       -> -8
```

### 6.3 Leverage-Specific Additions

```
- get_unrealized_profit() overridden: base_pnl * leverage
- Liquidation check: base_pnl <= -(1/leverage - buffer)
- Liquidation penalty: -10 (maximum)
- Holding penalty amplified by (1 + leverage_risk_penalty)
```

---

## 7. Training Configuration

### 7.1 FreqAI Config

```json
{
  "freqai": {
    "enabled": true,
    "purge_old_models": 2,
    "train_period_days": 75,
    "backtest_period_days": 7,
    "live_retrain_hours": 0,
    "continual_learning": false,
    "identifier": "rl-daytrade-v1",
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
    }
  }
}
```

### 7.2 RL Config

```json
{
  "rl_config": {
    "train_cycles": 500,
    "max_trade_duration_candles": 48,
    "model_type": "PPO",
    "policy_type": "MlpPolicy",
    "max_training_drawdown_pct": 0.50,
    "cpu_count": 16,
    "net_arch": [256, 256],
    "randomize_starting_position": true,
    "drop_ohlc_from_features": false,
    "add_state_info": false,
    "leverage": 10.0,
    "liquidation_buffer": 0.025,
    "leverage_risk_penalty": 0.10,
    "model_reward_parameters": {
      "rr": 1,
      "profit_aim": 0.005
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
    "batch_size": 512,
    "n_steps": 2048,
    "n_epochs": 10,
    "gae_lambda": 0.95,
    "clip_range": 0.2,
    "ent_coef": 0.01,
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
| weight_factor | 0.9 | Favor recent data in training |
| train_cycles | 500 | Sufficient with 75d data, avoid overfit |
| max_trade_duration_candles | 48 | 4 hours max (intraday constraint) |
| max_training_drawdown_pct | 0.50 | Strict for intraday risk tolerance |
| net_arch | [256, 256] | Smaller network for better generalization |
| profit_aim | 0.005 | 0.5% price move target |
| ent_coef | 0.01 | Low exploration, faster convergence |
| clip_range | 0.2 | Standard PPO, moderate update steps |
| learning_rate | 0.0003 | PPO default, good with fewer train_cycles |

### 7.5 Strategy Parameters

```python
minimal_roi = {"0": 0.05, "30": 0.03, "60": 0.01}
stoploss = -0.03
use_exit_signal = True
startup_candle_count = 40
can_short = True
process_only_new_candles = True
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
train_df = 75 days * 288 candles/day * 0.75 (train split) = 16,200 candles
total_timesteps = train_cycles * len(train_df) = 500 * 16,200 = 8,100,000
With 16 parallel envs: ~506,250 steps per env
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

---

## 9. Implementation Checklist

- [ ] Create RLDayTradeStrategy.py
  - [ ] feature_engineering_expand_all (RSI, ADX, ATR_norm, Rel_Vol)
  - [ ] feature_engineering_expand_basic (EMA, BB, MACD, S/R Flip features)
  - [ ] feature_engineering_standard (raw OHLCV, time features)
  - [ ] set_freqai_targets (&-action = 0)
  - [ ] populate_entry_trend (4-action mapping)
  - [ ] populate_exit_trend (unified exit)
  - [ ] leverage() method
- [ ] Create RLDayTrader.py
  - [ ] MyRLEnv (Base4ActionRLEnv)
  - [ ] calculate_reward (compressed [-10, +10])
  - [ ] get_unrealized_profit (leverage-aware)
  - [ ] _check_liquidation
- [ ] Create RLDayTrader_multiproc.py
  - [ ] Override max_threads
  - [ ] Same MyRLEnv as RLDayTrader
- [ ] Create config_daytrade.json
  - [ ] Full config with all parameters
- [ ] Download data
  - [ ] ETH/USDT:USDT (5m, 15m, 1h, 4h)
  - [ ] SOL/USDT:USDT (5m, 15m, 1h, 4h)
  - [ ] BTC/USDT:USDT (5m, 15m, 1h, 4h) (corr pair)
- [ ] Run backtest
- [ ] Evaluate results
- [ ] Iterate on reward function

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

| File | Key Content |
|------|-------------|
| `freqtrade/freqai/freqai_interface.py` | IFreqaiModel base class, training pipeline, sliding window |
| `freqtrade/freqai/data_kitchen.py` | Data split, feature filtering, weight_factor |
| `freqtrade/freqai/data_drawer.py` | Model persistence, historic predictions |
| `freqtrade/freqai/RL/BaseEnvironment.py` | Base gym.Env, positions, actions, unrealized profit |
| `freqtrade/freqai/RL/Base4ActionRLEnv.py` | 4-action step/is_tradesignal/is_valid logic |
| `freqtrade/freqai/RL/BaseReinforcementLearningModel.py` | RL train flow, unset_outlier_removal, predict, multiproc |
| `freqtrade/freqai/prediction_models/ReinforcementLearner.py` | fit() with continual_learning, default MyRLEnv |
| `freqtrade/templates/FreqaiExampleStrategy.py` | Official strategy template |
| `config_examples/config_freqai.example.json` | Official config example |

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
