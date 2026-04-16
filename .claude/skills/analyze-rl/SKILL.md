---
name: analyze-rl
description: Analyze RL training health from TensorBoard logs
argument-hint: "[model-id] [last-n-windows]"
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
   - Is there a Long/Short directional bias? (may be valid in trending markets)

4. **Recommendations** - Actionable next steps, ordered by priority:
   - For each suggestion, reference specific config parameters with their JSON paths
   - Example: "Increase `rl_config.max_trade_duration_candles` from 48 to 72"

### Health check reference

| Check | OK | WARN | FAIL |
|-------|-----|------|------|
| Reward trend | Positive slope | Flat or slight decline | Declining > 20% (caveat: may reflect market regime change, not learning degradation) |
| Liquidation rate | < 5% of episodes | 5-15% | > 15% |
| Win rate | > 40% | 25-40% | < 25% |
| Policy collapse | No action > 60% | One action 60-80% | One action > 80% |
| Value loss | Decreasing or stable (within window) | Increasing < 2x (within window) | Increasing > 2x or NaN |
| Entropy loss | > 50% of initial retained | 20-50% retained | < 20% retained or near 0 |
| Invalid actions | < 10% | 10-25% | > 25% |
| Approx KL | < 0.02 | 0.02-0.05 | > 0.05 |
| Clip fraction | < 0.2 | 0.2-0.4 | > 0.4 |
| Sample size | >= 1000 actions from env[0] | 200-999 actions | < 200 (health checks unreliable) |
| Profit trend | All windows positive | Some windows negative | Average profit negative |
| Entropy trend | Cross-window decline < 8pp | Decline 8-15pp | Decline > 15pp |
| Explained variance | > 0.5 (predictive) | 0-0.5 (noisy) | < 0 (worse than random) |
| Long/Short bias | < 65% dominant side | 65-80% | > 80% |

Note: Invalid action rate counts `actions/invalid` against total actions logged by
`Base4ActionRLEnv.step()`. Action counts include invalid attempts (action name is
logged before validity check). Sample size, profit trend, and entropy trend are
cross-window checks; others evaluate the latest complete window only.

### Context for analysis

The reward function is defined in `user_data/freqaimodels/RLDayTrader.py`:
- Compressed reward range [-10, +10]
- Liquidation = -10 (worst)
- Invalid action = -2
- Entry/Neutral while neutral = 0 (no overtrading exploit)
- Holding + profitable: clamp(pnl * 20, 0, 2)
- Holding + small loss (< profit_aim): clamp(pnl * 30, -3, 0)
- Holding + large loss (>= profit_aim): clamp(pnl * 50, -5, 0)
- Note: profit_aim compares against leveraged PNL (base_pnl * leverage). E.g., leverage=10, profit_aim=0.05 means 0.5% base price move
- Profitable exit = clamp(pnl * 50, 1, 8) + quick bonus
- Loss exit = -1 to -10 depending on severity
- Forced close: step() overrides action to Exit at max_trade_duration_candles (prop firm rule)

Multiproc sampling caveat:
- SB3 built-in metrics (ep_rew_mean, value_loss, entropy_loss, approx_kl, clip_fraction) are reliable (aggregated by VecMonitor)
- Custom metrics (actions/*, pnl/*, risk/*) only sample env[0] out of N envs -- treat as directional indicators, not exact values

Key config parameters that affect training:
- `rl_config.leverage` - Leverage multiplier (affects PNL and liquidation)
- `rl_config.liquidation_buffer` - Buffer before liquidation threshold
- `rl_config.max_trade_duration_candles` - Max hold time before penalty
- `model_training_parameters.learning_rate` - PPO learning rate
- `model_training_parameters.net_arch` - Network architecture
- `freqai.train_period_days` - Training window size
