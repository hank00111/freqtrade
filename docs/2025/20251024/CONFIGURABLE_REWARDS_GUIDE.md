# Configurable Reward Parameters Guide

**Date**: 2025-10-24  
**Model**: RL4ActionLeverage_multiproc  
**Config**: config_rl_10x_v2.json

## Overview

All reward function parameters are now configurable through `config_rl_10x_v2.json`. This allows you to tune the training behavior without modifying code, making it easier to optimize the balance between `explained_variance` (learning quality) and profitability.

---

## Configuration Location

Edit `user_data/config_rl_10x_v2.json` → `freqai.rl_config.model_reward_parameters`:

```json
"model_reward_parameters": {
    "rr": 1,
    "profit_aim": 0.025,
    "win_reward_factor": 1.8,
    "tier_multipliers": {
        "small_win": 8, 
        "medium_win": 12,
        "big_win": 16,
        "huge_win": 32,
        "mega_win": 64
    },
    "base_factor": 50.0,
    "immediate_feedback_multiplier": 2.0,
    "time_reward_max": 3.0,
    "time_reward_slope": 2.0,
    "risk_adjustment_cap": 3.0,
    "base_profit_bonus": 5,
    "enter_trade_reward": 25,
    "neutral_penalty": -1,
    "liquidation_penalty": -1000.0,
    "invalid_action_penalty": -2,
    "leverage_loss_multiplier": 2.0
}
```

---

## Parameter Reference

### 1. Base Reward Scaling

#### `base_factor` (default: 50.0)
- **Purpose**: Main reward scale multiplier
- **Impact**: Higher = stronger rewards/penalties
- **Tuning**:
  - Too high (>100): Causes value_loss explosion, low explained_variance
  - Too low (<20): Weak learning signal, slow convergence
  - Sweet spot: 30-70 depending on profit_aim
- **Example**: If explained_variance < 0.5 after 100 cycles, reduce to 30-40

#### `base_profit_bonus` (default: 5)
- **Purpose**: Fixed reward added to profitable exits
- **Impact**: Encourages taking profits even if small
- **Tuning**:
  - Increase (10-20): If model exits too early
  - Decrease (2-3): If model holds losses hoping for recovery
- **Example**: If win_rate > 70% but avg_profit < profit_aim, increase to 15

---

### 2. Dense Rewards (Holding Position)

#### `immediate_feedback_multiplier` (default: 2.0)
- **Purpose**: Reward/penalty multiplier during holding periods
- **Impact**: Provides continuous feedback on unrealized PNL
- **Tuning**:
  - Increase (3.0-5.0): If model exits winners too early
  - Decrease (1.0-1.5): If model holds losers too long
- **Example**: If avg_holding_duration < 10 candles, increase to 4.0

---

### 3. Time-based Rewards

#### `time_reward_max` (default: 3.0)
- **Purpose**: Maximum time efficiency bonus multiplier
- **Impact**: Rewards quick profits
- **Formula**: `time_multiplier = time_reward_max - (time_reward_slope * normalized_duration)`
- **Tuning**:
  - Increase (4.0-5.0): To strongly favor fast trades
  - Decrease (2.0-2.5): If model exits too early
- **Example**: If avg_trade_duration > max_trade_duration, increase to 4.0

#### `time_reward_slope` (default: 2.0)
- **Purpose**: How fast time bonus decays with duration
- **Impact**: Controls penalty for slow trades
- **Tuning**:
  - Increase (3.0-4.0): Penalize slow trades more heavily
  - Decrease (1.0-1.5): More tolerant of longer holdings
- **Example**: If model holds overnight losing trades, increase to 3.5

---

### 4. Risk-adjusted Rewards

#### `risk_adjustment_cap` (default: 3.0)
- **Purpose**: Maximum Sharpe-like risk adjustment bonus
- **Impact**: Rewards profit relative to time exposed
- **Tuning**:
  - Increase (4.0-5.0): Strongly favor time-efficient profits
  - Decrease (2.0): If model becomes too aggressive
- **Example**: If Sharpe ratio < 1.0, increase to 4.0

---

### 5. Entry/Exit Behavior

#### `enter_trade_reward` (default: 25)
- **Purpose**: Reward for entering Long/Short from Neutral
- **Impact**: Encourages taking positions
- **Tuning**:
  - Increase (30-50): If model stays Neutral >50% of time
  - Decrease (10-20): If model over-trades (>10 trades/day)
- **Example**: If total_trades < 50 in backtest, increase to 40

#### `neutral_penalty` (default: -1)
- **Purpose**: Penalty for staying Neutral when no position
- **Impact**: Discourages passivity
- **Tuning**:
  - Increase magnitude (-2 to -5): If model rarely enters trades
  - Decrease magnitude (-0.5): If model enters too frequently
- **Example**: If entry_rate < 5%, increase to -3

---

### 6. Loss Penalties

#### `leverage_loss_multiplier` (default: 2.0)
- **Purpose**: Extra penalty multiplier for losing trades
- **Impact**: Asymmetric risk (losses hurt more than gains help)
- **Formula**: `loss_penalty = leverage_loss_multiplier + leverage_risk_penalty`
- **Tuning**:
  - Increase (3.0-5.0): If model takes too many risky trades
  - Decrease (1.0-1.5): If model becomes too conservative
- **Example**: If loss_rate > 40%, increase to 3.5

#### `liquidation_penalty` (default: -1000.0)
- **Purpose**: Severe penalty for liquidation events
- **Impact**: Teaches model to avoid catastrophic losses
- **Tuning**:
  - **CRITICAL**: Should always be very negative
  - Increase magnitude (-2000): If ANY liquidations occur
  - Never reduce below -500
- **Example**: If liquidation_count > 0, increase to -2000

---

### 7. Action Validity

#### `invalid_action_penalty` (default: -2)
- **Purpose**: Penalty for invalid actions (e.g., enter Long when already Long)
- **Impact**: Should be minimal as proper training has few invalid actions
- **Tuning**:
  - Rarely needs adjustment
  - Increase magnitude (-5) only if invalid_action_rate > 5%
- **Example**: Keep at -2 unless major issues

---

## Optimization Strategies

### Strategy 1: Improve Explained Variance (Value Loss Too High)

**Problem**: `explained_variance < 0.5`, `value_loss > 1e6`

**Solution**: Reduce reward magnitude
```json
"base_factor": 30.0,                    // ↓ from 50.0
"tier_multipliers": {
    "mega_win": 40,                      // ↓ from 64
    "huge_win": 20,                      // ↓ from 32
    "big_win": 10,                       // ↓ from 16
    "medium_win": 8,                     // ↓ from 12
    "small_win": 5                       // ↓ from 8
},
"base_profit_bonus": 3,                  // ↓ from 5
"liquidation_penalty": -500.0            // ↑ from -1000.0 (less extreme)
```

**Expected**: `explained_variance → 0.7-0.9`, `value_loss → 1e3-1e4`

---

### Strategy 2: Improve Profitability (High Learning Quality but Low Profit)

**Problem**: `explained_variance > 0.8` but `total_profit < 0` or `win_rate < 45%`

**Solution**: Amplify profit incentives
```json
"base_factor": 70.0,                    // ↑ from 50.0
"immediate_feedback_multiplier": 4.0,   // ↑ from 2.0
"time_reward_max": 4.0,                 // ↑ from 3.0
"risk_adjustment_cap": 4.0,             // ↑ from 3.0
"base_profit_bonus": 10,                // ↑ from 5
"enter_trade_reward": 35                // ↑ from 25
```

**Expected**: More aggressive profit-seeking behavior

---

### Strategy 3: Balance Learning and Profit (Dual Optimization)

**Problem**: Need both high `explained_variance` AND profitability

**Solution**: Moderate scaling with dense rewards
```json
"base_factor": 40.0,                    // Moderate
"immediate_feedback_multiplier": 3.0,   // Strong holding incentive
"time_reward_max": 3.5,                 // Reward fast profits
"time_reward_slope": 2.5,               // Penalize slow trades
"risk_adjustment_cap": 3.5,             // Sharpe-like bonus
"base_profit_bonus": 7,                 // Moderate fixed bonus
"leverage_loss_multiplier": 2.5         // Asymmetric risk
```

**Expected**: `explained_variance → 0.75-0.85`, `Sharpe ratio > 1.5`

---

### Strategy 4: Reduce Overtrading (Too Many Entries)

**Problem**: `total_trades > 500` in backtest, low profit per trade

**Solution**: Discourage frivolous entries
```json
"enter_trade_reward": 15,               // ↓ from 25
"neutral_penalty": -0.5,                // ↓ from -1
"base_profit_bonus": 10,                // ↑ from 5 (reward quality over quantity)
"leverage_loss_multiplier": 3.0         // ↑ from 2.0 (punish bad trades)
```

**Expected**: Fewer but higher quality trades

---

### Strategy 5: Increase Trading Frequency (Too Passive)

**Problem**: `total_trades < 50`, model stays Neutral >60% of time

**Solution**: Encourage activity
```json
"enter_trade_reward": 40,               // ↑ from 25
"neutral_penalty": -3,                  // ↑ from -1
"immediate_feedback_multiplier": 3.0,   // ↑ from 2.0
"base_profit_bonus": 5                  // Keep moderate to avoid overtrading
```

**Expected**: Higher entry rate (10-20% of time in position)

---

## Monitoring and Adjustment

### TensorBoard Metrics to Watch

1. **Learning Quality**:
   - `explained_variance`: Should reach 0.7-0.9+ (primary target)
   - `value_loss`: Should decrease to 1e3-1e4 (secondary)

2. **Profitability**:
   - `total_profit_sum_10x`: Cumulative profit across all exits
   - `profitable_exit_10x_count`: Number of winning trades
   - `loss_exit_10x_count`: Number of losing trades

3. **Behavior**:
   - `time_multiplier`: Average time efficiency bonus
   - `risk_adjustment`: Average risk-adjusted multiplier
   - `liquidation_count`: MUST be 0 (critical)

4. **Trade Distribution**:
   - `mega_win_10x`, `huge_win_10x`, `big_win_10x`: Frequency of big wins
   - `leverage_loss_penalty`: How often losses trigger extra penalty

### Adjustment Timeline

- **0-50 cycles (2-4 hours)**: 
  - Check `explained_variance` - should be >0.3
  - If <0.3: Reduce `base_factor` by 30%
  
- **50-150 cycles (4-10 hours)**:
  - Check `explained_variance` - should be >0.6
  - Check `total_profit` trend - should be positive
  - Adjust tier_multipliers if profitability is an issue

- **150-300 cycles (10-20 hours)**:
  - Fine-tune time_reward and risk_adjustment parameters
  - Monitor trade frequency and duration
  - Aim for `explained_variance > 0.8`, `Sharpe > 1.5`

---

## Common Issues and Fixes

### Issue 1: Value Loss Explodes Again

**Symptom**: `value_loss > 1e6` after 100 cycles

**Diagnosis**: Reward scale still too high

**Fix**:
```json
"base_factor": 25.0,        // Aggressive reduction
"tier_multipliers": {
    "mega_win": 30,         // Cut all tiers by 40-50%
    "huge_win": 16,
    "big_win": 8,
    "medium_win": 6,
    "small_win": 4
}
```

---

### Issue 2: Model Never Enters Trades

**Symptom**: `total_trades < 20` in entire backtest

**Diagnosis**: Entry reward too weak or Neutral penalty too weak

**Fix**:
```json
"enter_trade_reward": 50,
"neutral_penalty": -5
```

---

### Issue 3: Model Exits Winners Too Early

**Symptom**: `avg_profit < profit_aim`, high `win_rate` but low total profit

**Diagnosis**: Not enough incentive to hold winning positions

**Fix**:
```json
"immediate_feedback_multiplier": 5.0,
"base_profit_bonus": 15,
"time_reward_max": 4.0
```

---

### Issue 4: Model Holds Losers Too Long

**Symptom**: Large drawdowns, `avg_loss > 2 * profit_aim`

**Diagnosis**: Insufficient loss penalty or time penalty

**Fix**:
```json
"leverage_loss_multiplier": 4.0,
"time_reward_slope": 3.5,
"immediate_feedback_multiplier": 1.5  // Reduce positive feedback on losses
```

---

## Best Practices

1. **Change One Parameter at a Time**: When tuning, adjust 1-2 related parameters and retrain to see impact

2. **Always Clear Old Models**: Run `Remove-Item -Path "user_data\models\rl-10x-v2-256*" -Recurse -Force` before retraining with new config

3. **Monitor First 50 Cycles**: If `explained_variance < 0.3` after 50 cycles, stop and reduce reward scale

4. **Document Your Experiments**: Keep a log of parameter changes and their effects

5. **Use Conservative Tiers First**: Start with lower tier_multipliers and increase only if profitability is proven

6. **Liquidation = Critical Failure**: ANY liquidation means you need to reduce risk or improve loss penalty

---

## Summary

These configurable parameters give you fine-grained control over the reward function without touching code. The key is balancing:

- **Learning Quality** (explained_variance): Controlled by base_factor and tier_multipliers
- **Profitability**: Controlled by dense rewards (immediate_feedback), time rewards, and risk adjustment
- **Trading Behavior**: Controlled by entry rewards, neutral penalty, and loss multipliers

Start with the default values, monitor TensorBoard closely in the first 50 cycles, and make targeted adjustments based on observed behavior.

**Key Principle**: Better to undertrain with good explained_variance (0.8+) than overtrain with exploding rewards (explained_variance <0.3)
