---
name: analyze-rl
description: Analyze RL training health from TensorBoard logs
argument-hint: "[model-id] [last-n-windows]"
disable-model-invocation: true
allowed-tools: Bash(python *), Read, Glob
---

# RL Training Health Analysis

Analyze the training health of an RL model by parsing its TensorBoard logs.

## Step 1: Collect data

Run the analysis script (output is injected below):

!`python scripts/analyze_rl_training.py $ARGUMENTS`

## Step 2: Interpret results

Analyze the JSON output from Step 1 and produce a clear report.

### Report structure

1. **Status summary** - One-line overall status (e.g., "3 OK, 2 WARN, 1 FAIL")

2. **Health checks** - For each check, explain:
   - Current status (OK / WARN / FAIL) and the metric values
   - What this means for training quality
   - If WARN or FAIL: likely root cause and suggested fix

3. **Cross-window trends** - Compare metrics across training windows:
   - Is reward improving, stagnant, or degrading?
   - Is the agent learning to avoid liquidations?
   - Is action diversity improving or collapsing?

4. **Recommendations** - Actionable next steps, ordered by priority:
   - For each suggestion, reference specific config parameters with their JSON paths
   - Example: "Increase `rl_config.max_trade_duration_candles` from 48 to 72"

### Health check reference

| Check | OK | WARN | FAIL |
|-------|-----|------|------|
| Reward trend | Positive slope | Flat or slight decline | Declining > 20% |
| Liquidation rate | < 5% of episodes | 5-15% | > 15% |
| Win rate | > 40% | 25-40% | < 25% |
| Policy collapse | No action > 60% | One action 60-80% | One action > 80% |
| Value loss | Decreasing or stable | Increasing < 2x | Increasing > 2x or NaN |
| Invalid actions | < 10% | 10-25% | > 25% |

### Context for analysis

The reward function is defined in `user_data/freqaimodels/RLDayTrader.py`:
- Compressed reward range [-10, +10]
- Liquidation = -10 (worst)
- Invalid action = -2
- Entry/Neutral while neutral = 0 (no overtrading exploit)
- Profitable exit = clamp(pnl * 50, 1, 8) + quick bonus
- Loss exit = -1 to -10 depending on severity

Key config parameters that affect training:
- `rl_config.leverage` - Leverage multiplier (affects PNL and liquidation)
- `rl_config.liquidation_buffer` - Buffer before liquidation threshold
- `rl_config.max_trade_duration_candles` - Max hold time before penalty
- `model_training_parameters.learning_rate` - PPO learning rate
- `model_training_parameters.net_arch` - Network architecture
- `freqai.train_period_days` - Training window size
