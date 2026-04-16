# RL Day Trade Strategy - v3 Improvement Plan

> Date: 2026-04-03
> Based on: v2-20260401 training analysis (PPO_44/189, 23% complete)
> Problem: Policy collapse (Neutral=83% FAIL), entropy decline (-17.4pp FAIL)
> Goal: Resolve structural reward bias, improve PPO training efficiency

---

## 1. Problem Summary

### v2 Current Status (PPO_44)

| Metric | Value | Status |
|--------|-------|--------|
| Neutral% | 83% | FAIL (>80%) |
| Entropy retained | 42% | FAIL (-17.4pp) |
| Long/Short bias | 50/50 | OK (resolved) |
| Win rate | 42.1% | OK |
| Profit avg | 1.52 | OK (all 44 windows positive) |
| Liquidation rate | 0.0% | OK |
| Explained variance | 0.826 | OK |

### Root Cause Analysis (by priority)

1. **Reward design flaw** -- Neutral while Neutral = 0 is a "safe harbor"; the model
   learns that not trading is the rational optimum because risk-free reward (0) beats
   the expected value of entering trades (exposure to -1/-2/-5/-10 penalties)

2. **Neutral zone too wide** -- +/-5% leveraged PNL (+/-0.5% base price) covers most
   short-term ETH price movements. Most hold steps produce 0 reward = no learning signal.

3. **PPO hyperparameters too conservative** -- No `target_kl` protection, `n_epochs=10`
   causes rollout overfitting, fixed `ent_coef` cannot adapt to training phases.

4. **No entropy management** -- Static `ent_coef=0.05` with no scheduling or adaptive
   adjustment. As policy converges, entropy bonus becomes insufficient.

---

## 2. Theoretical Background

### 2.1 PPO Loss Function

```
L_total = L_clip - c1 * L_VF + c2 * H(pi)

L_clip  = min(r_t * A_t, clip(r_t, 1-eps, 1+eps) * A_t)
L_VF    = (V(s) - V_target)^2
H(pi)   = -SUM_a [ pi(a|s) * log(pi(a|s)) ]   (entropy bonus)
```

Parameters:
- `c1 = vf_coef` (v2: 0.5)
- `c2 = ent_coef` (v2: 0.05)
- `eps = clip_range` (v2: 0.2)

For Discrete(4) action space:
- H_max = log(4) = 1.386 nats (uniform distribution)
- H_min = 0 (deterministic policy)
- Healthy range: H > 0.4 * H_max = 0.55 nats

### 2.2 Policy Collapse Mechanism

Self-reinforcing feedback loop:
1. One action (Neutral) gets slightly more probability mass
2. Gets sampled more frequently during rollouts
3. If it avoids negative reward (which Neutral always does), probability increases
4. Other actions are starved of samples, their advantages become poorly estimated
5. Policy converges to near-deterministic Neutral

### 2.3 Collapse Detection Formula

```
collapse_ratio = current_entropy / log(n_actions)
```

| collapse_ratio | Status |
|----------------|--------|
| > 0.5 | Healthy |
| 0.3 - 0.5 | Warning |
| < 0.3 | Critical -- policy collapse imminent |

v2 at PPO_44: collapse_ratio = 0.62 / 1.386 = **0.45 (Warning)**

### 2.4 Key Research References

| Paper | Year | Key Finding |
|-------|------|-------------|
| No Representation, No Trust (NeurIPS) | 2024 | Feature rank deterioration causes PPO collapse; PFO auxiliary loss fixes it |
| Plasticity Loss in On-Policy Deep RL (NeurIPS) | 2024 | L2 regularization prevents network from losing learning ability |
| Entropy Scheduling in RL (OpenReview) | 2025 | Linear decay degrades performance; stable-then-decay is best |
| Tracking Drift: Variation-Aware Entropy (arXiv) | 2026 | Adaptive entropy for non-stationary environments (markets) |
| REPO: Entropy-preserving RL (ICLR) | 2026 | Regularization stabilizes entropy throughout training |
| Risk-Aware RL Reward for Financial Trading | 2025 | Composite reward with downside risk penalty outperforms single-metric |
| Sharpe Ratio Based Reward in DRL (Springer) | 2023 | Sharpe ratio as reward signal for risk-adjusted learning |
| Imitate Optimal Policy: Action Collapse | 2025 | Action collapse in policy gradient methods analysis |

---

## 3. v3 Improvements

### 3.1 Reward Function Changes (Priority: Critical)

#### 3.1.1 Neutral Penalty (opportunity cost)

**Current (v2):** Neutral while Neutral = 0 (safe harbor)
**Proposed (v3):** Neutral while Neutral = **-0.1** (opportunity cost)

```python
# v2 (RLDayTrader.py line 172-173)
if action == Actions.Neutral.value and self._position == Positions.Neutral:
    return 0.0

# v3
if action == Actions.Neutral.value and self._position == Positions.Neutral:
    return -0.1  # opportunity cost -- incentivize trade exploration
```

Rationale: Multiple trading RL papers and FreqAI official docs recommend penalizing
inaction. The penalty should be small enough to not force bad trades, but large enough
to overcome the "safe harbor" bias. -0.1 is 1/10 of the invalid action penalty (-1.0),
making it a gentle nudge.

Alternative to evaluate: Escalating penalty based on consecutive neutral steps:
```python
if action == Actions.Neutral.value and self._position == Positions.Neutral:
    self._neutral_streak += 1
    if self._neutral_streak > 50:  # ~4 hours at 5min candles
        return -0.2
    return -0.1
```

Status: [ ] Implement  [ ] Test  [ ] Validate

#### 3.1.2 Narrow Neutral Zone (+/-5% -> +/-2%)

**Current (v2):** PNL between -5% and +5% leveraged = 0 reward (no signal)
**Proposed (v3):** PNL between -2% and +2% leveraged = 0 reward

```python
# v2 hold rewards (RLDayTrader.py lines 190-199)
if pnl >= 0.10:     hold_reward = 2.0
elif pnl >= 0.05:   hold_reward = 1.0
elif pnl > -0.05:   hold_reward = 0.0   # <- neutral zone too wide
elif pnl > -0.10:   hold_reward = -1.0
else:                hold_reward = -2.0

# v3 hold rewards (narrower neutral zone + more levels)
if pnl >= 0.10:     hold_reward = 2.0
elif pnl >= 0.05:   hold_reward = 1.5
elif pnl >= 0.02:   hold_reward = 0.5   # new: small profit signal
elif pnl > -0.02:   hold_reward = 0.0   # narrowed neutral zone
elif pnl > -0.05:   hold_reward = -0.5  # new: small loss signal
elif pnl > -0.10:   hold_reward = -1.5
else:                hold_reward = -2.0

# v3 exit rewards (same narrowing)
if pnl >= 0.10:     exit_reward = 10.0
elif pnl >= 0.05:   exit_reward = 5.0
elif pnl >= 0.02:   exit_reward = 2.0   # new: small profit exit
elif pnl > -0.02:   exit_reward = 0.0   # narrowed
elif pnl > -0.05:   exit_reward = -2.0  # new: small loss exit
elif pnl > -0.10:   exit_reward = -5.0
else:                exit_reward = -10.0
```

Rationale: With 10x leverage, +/-2% leveraged = +/-0.2% base price. ETH 5min candles
typically move 0.1-0.3%, so most trades will produce non-zero learning signals within
2-6 candles. This dramatically increases gradient information density.

Note: Increases reward levels from 5 to 7, maintaining the stepped design philosophy.

Status: [ ] Implement  [ ] Test  [ ] Validate

#### 3.1.3 Entry Bonus (optional, evaluate separately)

**Current (v2):** Enter trade = 0
**Proposed (v3):** Enter trade = **+0.5**

```python
# v2 (RLDayTrader.py lines 176-180)
if action in (Long_enter, Short_enter) and position == Neutral:
    return 0.0

# v3
if action in (Long_enter, Short_enter) and position == Neutral:
    return 0.5  # small incentive to attempt trades
```

Risk: Could cause over-trading if too high. Start with 0.5 and evaluate.
May not be needed if 3.1.1 (Neutral penalty) already solves the collapse.

Status: [ ] Evaluate need after 3.1.1 + 3.1.2

---

### 3.2 PPO Hyperparameter Changes (Priority: High)

#### 3.2.1 Config changes (config_daytrade_v3.json)

```json
{
  "model_training_parameters": {
    "learning_rate": 0.0003,
    "gamma": 0.99,
    "batch_size": 512,
    "n_steps": 2048,
    "n_epochs": 5,
    "gae_lambda": 0.95,
    "clip_range": 0.2,
    "ent_coef": 0.08,
    "vf_coef": 0.5,
    "max_grad_norm": 0.5,
    "normalize_advantage": true,
    "target_kl": 0.015,
    "device": "cpu"
  }
}
```

Changes from v2:

| Parameter | v2 | v3 | Rationale |
|-----------|-----|-----|-----------|
| `batch_size` | 1024 | **512** | Smaller batch = more gradient noise = more exploration |
| `n_epochs` | 10 | **5** | Reduce rollout overfitting that triggers entropy collapse |
| `ent_coef` | 0.05 | **0.08** | Increase exploration pressure against Neutral bias |
| `target_kl` | None | **0.015** | Early-stop gradient updates when KL too high; prevents runaway policy changes |

Unchanged: learning_rate, gamma, n_steps, gae_lambda, clip_range, vf_coef, max_grad_norm

Status: [ ] Create config  [ ] Test  [ ] Validate

#### 3.2.2 L2 Regularization (anti-plasticity-loss)

```json
{
  "policy_kwargs": {
    "optimizer_kwargs": {
      "weight_decay": 1e-5
    }
  }
}
```

Rationale: NeurIPS 2024 paper shows PPO in non-stationary environments (like financial
markets) suffers plasticity loss -- the network gradually loses ability to learn new
patterns. L2 regularization (weight_decay) prevents weight norms from growing unbounded.

Integration point: Check if freqtrade's `ReinforcementLearner` passes `policy_kwargs`
to the PPO constructor. If not, may need code change.

Status: [ ] Verify integration point  [ ] Implement  [ ] Test

---

### 3.3 Entropy Management (Priority: Medium)

#### 3.3.1 Adaptive Entropy Callback

```python
from stable_baselines3.common.callbacks import BaseCallback
import numpy as np

class AdaptiveEntropyCallback(BaseCallback):
    """
    Monitor actual entropy and adjust ent_coef to prevent policy collapse.
    Uses multiplicative update inspired by SAC's automatic entropy tuning.

    For Discrete(4): max_entropy = ln(4) = 1.386
    Target: 50% of max = 0.693 (prevents collapse while allowing specialization)
    """
    def __init__(
        self,
        target_entropy: float = 0.693,
        adjustment_lr: float = 0.01,
        min_ent_coef: float = 0.01,
        max_ent_coef: float = 0.5,
        verbose: int = 0,
    ):
        super().__init__(verbose)
        self.target_entropy = target_entropy
        self.lr = adjustment_lr
        self.min_ent_coef = min_ent_coef
        self.max_ent_coef = max_ent_coef

    def _on_step(self) -> bool:
        return True

    def _on_rollout_start(self) -> None:
        """Fires after previous train() -- can read logged entropy."""
        entropy_loss = self.logger.name_to_value.get("train/entropy_loss", None)
        if entropy_loss is None:
            return

        actual_entropy = -entropy_loss  # SB3 logs -mean(entropy)

        # If entropy < target: increase ent_coef (more exploration)
        # If entropy > target: decrease ent_coef (allow exploitation)
        error = self.target_entropy - actual_entropy
        multiplier = np.exp(self.lr * error)
        new_ent_coef = float(np.clip(
            self.model.ent_coef * multiplier,
            self.min_ent_coef,
            self.max_ent_coef,
        ))

        self.model.ent_coef = new_ent_coef
        self.logger.record("train/ent_coef", new_ent_coef)
        self.logger.record("train/actual_entropy", actual_entropy)
```

Integration point: Add callback to `model.learn()` call in `ReinforcementLearner.fit()`.

Callback lifecycle (confirmed from SB3 source):
```
collect_rollouts()
  -> callback._on_rollout_start()   # read previous entropy
  -> callback._on_step() per step
  -> callback._on_rollout_end()
train()                              # uses updated ent_coef
```

Status: [ ] Implement callback  [ ] Integrate with freqtrade  [ ] Test  [ ] Validate

#### 3.3.2 Simple Schedule (alternative to adaptive)

If adaptive is too complex, use a stable-then-decay schedule:

```python
class StableThenDecayEntCoef(BaseCallback):
    """Keep ent_coef constant for first half, then decay."""
    def __init__(self, initial=0.08, final=0.02, decay_start=0.5):
        super().__init__()
        self.initial = initial
        self.final = final
        self.decay_start = decay_start

    def _on_rollout_end(self) -> None:
        progress = 1.0 - self.model._current_progress_remaining  # 0.0 -> 1.0
        if progress < self.decay_start:
            self.model.ent_coef = self.initial
        else:
            decay_progress = (progress - self.decay_start) / (1.0 - self.decay_start)
            self.model.ent_coef = self.initial + (self.final - self.initial) * decay_progress
```

Research finding: "Stable-then-decay" outperforms linear decay. Linear decay
"consistently degrades performance" by prematurely suppressing early exploration.

Status: [ ] Implement  [ ] Test

---

### 3.4 Action Masking (Priority: Low, evaluate if 3.1-3.3 insufficient)

Freqtrade already supports MaskablePPO via sb3_contrib. The env needs to implement
`action_masks()` returning a boolean array for valid actions.

Potential use: Limit consecutive Neutral actions in specific market conditions.

```python
def action_masks(self) -> np.ndarray:
    """Return boolean mask: [Long_enter, Short_enter, Exit, Neutral]"""
    mask = np.ones(4, dtype=bool)

    # If neutral for too long, force trade consideration
    if (self._position == Positions.Neutral
            and self._neutral_streak > 100):
        mask[3] = False  # disable Neutral

    return mask
```

Caveat: Changes the action distribution denominator. Entropy metrics become harder to
compare across states with different masks. Use only as last resort.

Config change: `"model_type": "MaskablePPO"` (instead of "PPO")

Status: [ ] Evaluate need after 3.1-3.3

---

### 3.5 Advanced: SAC-style Learnable Entropy (Priority: Low, research)

Subclass PPO to add a learnable `log_ent_coef` tensor optimized via gradient descent,
similar to SAC's automatic entropy tuning. This is the most sophisticated approach
but requires the most code changes.

Key formula:
```
ent_coef_loss = -log(alpha) * (actual_entropy - target_entropy)
```
Where alpha = exp(log_ent_coef) is the learnable entropy coefficient.

This would require:
- Custom PPO subclass overriding `train()`
- Additional optimizer for log_ent_coef
- Integration with freqtrade's model loading pipeline

Status: [ ] Research feasibility  [ ] Evaluate if simpler approaches fail

---

## 4. Implementation Plan

### Phase 1: Quick Wins (config only, new training run)

- [ ] Create `config_daytrade_v3.json` with updated hyperparameters
- [ ] `target_kl`: None -> 0.015
- [ ] `n_epochs`: 10 -> 5
- [ ] `ent_coef`: 0.05 -> 0.08
- [ ] `batch_size`: 1024 -> 512

### Phase 2: Reward Redesign (code change)

- [ ] Create `RLDayTrader_v3.py` (copy from v2)
- [ ] Implement Neutral penalty: 0 -> -0.1
- [ ] Narrow neutral zone: +/-5% -> +/-2%
- [ ] Add intermediate reward levels (5 -> 7 levels)
- [ ] Unit test reward function

### Phase 3: Entropy Management (code change)

- [ ] Implement AdaptiveEntropyCallback
- [ ] Integrate callback into ReinforcementLearner.fit()
- [ ] Add L2 regularization (weight_decay=1e-5)
- [ ] Verify TensorBoard logs include ent_coef and actual_entropy

### Phase 4: Training & Validation

- [ ] Run v3 training: full 189 windows
- [ ] Monitor checkpoints every 10 windows
- [ ] Compare v2 vs v3 metrics at equivalent windows
- [ ] Run backtest at PPO_100 (early validation)
- [ ] Run final backtest after training completion

### Phase 5: Optional Enhancements (if needed)

- [ ] Entry bonus (+0.5) -- only if Neutral% still > 70% after Phase 1-2
- [ ] Action masking via MaskablePPO -- only if entropy still collapsing
- [ ] SAC-style learnable entropy -- only if all above fail

---

## 5. Success Criteria

| Metric | v2 (PPO_44) | v3 Target | Notes |
|--------|-------------|-----------|-------|
| Neutral% | 83% (FAIL) | **50-65%** | Active but selective trading |
| Entropy retained | 42% | **> 50%** | Healthy exploration |
| collapse_ratio | 0.45 | **> 0.5** | H / log(4) |
| Win rate | 42.1% | **> 40%** | Maintain quality |
| Profit (all windows) | All positive | **All positive** | No regression |
| Liquidation rate | 0.0% | **< 2%** | Maintain safety |
| Backtest | TBD | **> 0%** | v1 was -90%; v2 TBD |

---

## 6. v2 vs v3 Config Diff Summary

| Parameter | v2 | v3 | Category |
|-----------|-----|-----|----------|
| Neutral while Neutral reward | 0 | **-0.1** | Reward |
| Hold neutral zone | +/-5% | **+/-2%** | Reward |
| Reward levels | 5 | **7** | Reward |
| `ent_coef` | 0.05 | **0.08** | PPO |
| `n_epochs` | 10 | **5** | PPO |
| `batch_size` | 1024 | **512** | PPO |
| `target_kl` | None | **0.015** | PPO |
| `weight_decay` | None | **1e-5** | PPO |
| Entropy callback | None | **Adaptive** | Code |
| Model file | RLDayTrader.py | **RLDayTrader_v3.py** | Code |
| Config file | config_daytrade_v2.json | **config_daytrade_v3.json** | Config |
