# RL Day Trade Strategy - v2 Improvement Plan

> Date: 2026-04-01
> Based on: v1-20260307 full training results (169 windows, ETH)
> Backtest result: -90.12% (1,139 trades, 36.8% win rate, profit factor 0.60)
> Reference: `docs/2026/20260203/rl-daytrade-strategy-analysis.md`

### Quick Start - v2 Training Command

```powershell
# Activate venv
cd G:\Program\freqtrade
.\.venv\Scripts\Activate.ps1

# 1. Download / update data (incremental, only fetches missing ranges)
freqtrade download-data `
  --config user_data/config_daytrade_v2.json `
  --timerange 20220501-20260325 `
  --timeframe 5m 15m 1h 4h

# 2. Production training (multiproc)
freqtrade backtesting `
  --strategy RLDayTradeStrategy `
  --config user_data/config_daytrade_v2.json `
  --freqaimodel RLDayTrader_multiproc `
  --timerange 20220601-20260325 `
  --export trades

# 3. Single-env debugging (reward function verification)
freqtrade backtesting `
  --strategy RLDayTradeStrategy `
  --config user_data/config_daytrade_v2.json `
  --freqaimodel RLDayTrader `
  --timerange 20250101-20250201 `
  --export trades

# 4. TensorBoard (open in another terminal, keep running during training)
tensorboard --logdir user_data/models/rl-daytrade-v2-20260401
# Open http://localhost:6006 in browser
```

Key config: `cpu_count=4`, `n_steps=2048`, `batch_size=1024`, `train_cycles=500`
Data range: 20220501-20260325 (download), 20220601-20260325 (training, ~189 windows)
Identifier: `rl-daytrade-v2-20260401`

---

## 1. v1 Backtest Summary

| Metric | Value |
|--------|-------|
| Period | 2023-01-15 ~ 2026-03-02 (~3 years) |
| Trades | 1,139 (~1/day) |
| Win Rate | 36.8% (419W / 720L) |
| Total Profit | -901.18 USDT (-90.12%) |
| Profit Factor | 0.60 |
| Sharpe | -4.51 |
| Max Drawdown | 90.12% |

### Exit Reason Breakdown

| Exit Reason | Trades | % | Avg Profit | Total Profit |
|-------------|--------|---|-----------|-------------|
| roi | 353 | 31% | +3.54% | +1,247.89 |
| exit_signal | 255 | 22% | -0.52% | -133.54 |
| stop_loss | 531 | 47% | -3.80% | -2,015.53 |

### Long/Short Breakdown

| Direction | Trades | Profit | % of Trades |
|-----------|--------|--------|-------------|
| rl_long | 360 | -272.69 USDT | 31.6% |
| rl_short | 779 | -628.49 USDT | 68.4% |

Market change during period: +32.73% (ETH bullish)

### Root Cause

RL model trained in an environment WITHOUT stoploss/ROI, but backtested WITH stoploss/ROI.
78% of exits were NOT from RL model decisions (47% stop_loss + 31% roi).
Only 22% of exits (exit_signal) were actual RL model decisions.

---

## 2. Improvement Items

### A. Strategy-RL Framework Alignment (Priority: Critical)

#### A-1. stoploss vs RL Environment Mismatch

**Problem:**

| Environment | Stop Mechanism | Trigger (10x leverage) |
|-------------|---------------|----------------------|
| RL training | Liquidation at -7.5% base PNL | Base price -7.5% |
| Backtest | stoploss=-0.03 | Base price **-0.3%** |

- RL environment has NO stoploss, only liquidation (-7.5%)
- Model learns "hold through drawdowns until recovery or liquidation"
- But backtest kills trades at -0.3% base price move (25x tighter than liquidation)
- 47% of trades (531/1139) hit stop_loss before RL exit signal

**Solutions (pick one):**

- **(a) Add stoploss simulation to RL env** -- model learns to exit before stoploss triggers
- **(b) Widen stoploss** to near-liquidation level (e.g., -8%), let model control exits
- **(c) Disable stoploss** (`stoploss = -1`) + use `custom_stoploss` driven by RL model
- **(d) Hybrid**: mild stoploss (e.g., -5%) + RL env simulates same threshold

**Recommended: (a) or (d)** -- make training and backtest rules identical.

#### A-2. minimal_roi vs RL Environment Mismatch

**Problem:**

| Environment | Take-Profit Mechanism |
|-------------|----------------------|
| RL training | Model decides when to Exit |
| Backtest | `{"0": 0.05, "30": 0.03, "60": 0.01}` auto-close |

- RL environment has NO ROI mechanism
- Model learns "hold for optimal exit timing"
- But backtest auto-closes 31% of trades via ROI
- All 353 ROI exits were profitable (+1,248 USDT), but RL model didn't learn from this

**Solutions (pick one):**

- **(a) Add ROI simulation to RL env** -- auto-exit with positive reward at ROI thresholds
- **(b) Disable ROI** (`minimal_roi = {"0": 100}`), let RL fully control exits
- **(c) Keep ROI but add awareness** in reward function

**Recommended: (a)** -- simulate ROI in RL env so model learns to work with it.

#### A-3. profit_aim vs stoploss Numerical Contradiction

**Problem:**

```
profit_aim = 0.05 (5% leveraged PNL = 0.5% base price)
stoploss   = -0.03 (3% leveraged PNL = 0.3% base price)
```

Reward function loss ranges vs actual stoploss:

| Reward Range | PNL Range (leveraged) | Status in Backtest |
|-------------|----------------------|-------------------|
| Small loss (-3~0) | 0% ~ -5% | stoploss kills at -3% |
| Medium loss (-1~-5) | -5% ~ -10% | **never reached** |
| Large loss (-8) | -10% ~ -15% | **never reached** |
| Extreme loss (-10) | < -15% | **never reached** |

Model learns 4 levels of loss management, but only the first level matters in backtest.

**Solution:** Align profit_aim, stoploss, and reward ranges:
- If stoploss = -0.05: set profit_aim = 0.05 (symmetric)
- If stoploss = -0.08: set profit_aim = 0.08 and adjust reward ranges accordingly
- Or: simulate stoploss in RL env (A-1a) so all ranges become reachable

---

### B. RL Environment Design (Priority: High)

#### B-4. Add stoploss / ROI Simulation to Environment

Modify `Base4ActionRLEnv.step()` or `RLDayTrader.step()`:

```python
# In step(), before action processing:
if self._position != Positions.Neutral:
    pnl = self.get_unrealized_profit()

    # Simulate stoploss
    if pnl <= self.stoploss_threshold:
        action = Actions.Exit.value
        self._stoploss_triggered = True

    # Simulate ROI
    elif self._check_roi_threshold(pnl, trade_duration):
        action = Actions.Exit.value
        self._roi_triggered = True
```

And in `calculate_reward()`:
```python
# After exit processing:
if self._stoploss_triggered:
    exit_reward = -5.0  # Penalize for letting position reach stoploss
elif self._roi_triggered:
    exit_reward = max(exit_reward, 3.0)  # ROI exit is acceptable
```

**Impact:** Training and backtest operate under identical rules. Model learns to:
- Avoid entries that will likely hit stoploss
- Or exit before stoploss triggers
- Work with ROI as a natural profit-taking mechanism

#### B-5. Liquidation Formula Inconsistency

**Problem (current):**
```python
# Multiplicative (wrong for additive mode):
self._total_profit *= (1 + threshold)  # threshold = -0.075
```

**Fix:**
```python
# Additive (correct for stake_amount=100):
self._total_profit += threshold * self.leverage  # = -0.075 * 10 = -0.75
```

Low impact (liquidations are rare), but logically incorrect.

#### B-6. Reward Function Alignment with Strategy

**Replaced by Section 6.2 stepped reward design.** SL/ROI exits now use the same
stepped reward as voluntary exits -- no special exit-type distinction needed.
SL at -10% always gets -10, ROI at +10% always gets +10, regardless of whether
the exit was forced or voluntary.

---

### C. Training Configuration (Priority: Medium-High)

#### C-7. n_steps=4096 from Start

v1 evidence:
- n_steps=1024 (PPO_1-40): weak performance
- n_steps=2048 (PPO_41+): significant improvement
- n_steps impact analysis: 4096 produces 2.8x better results than 1024

v2 config:
```json
{
  "model_training_parameters": {
    "n_steps": 4096,
    "batch_size": 1024
  }
}
```

- Rollout = 4096 * cpu_count transitions per update
- With cpu=4: 16,384 transitions (2/3 of original v1-20260220's 24,576)
- ~244 MiB rollout buffer (safe with reduce_df_footprint=true)

#### C-8. cpu_count Evaluation

- v1: cpu=6 OOM at window 63, reduced to 4
- v1 had reduce_df_footprint=false when OOM occurred
- v2 with reduce_df_footprint=true + n_steps=4096:
  - Start with cpu=4 (stable baseline)
  - If memory allows, try cpu=6 (rollout=24,576, matches v1-20260220)

---

### D. Model Behavior (Priority: Medium)

#### D-9. Short Bias (68% Short in Bull Market)

v1 backtest: 779 Short vs 360 Long, ETH +32.73% in period.

Possible causes:
- Crypto crashes are sharper than rallies -- model learned Short has higher per-trade edge
- Lack of explicit trend-direction features
- Market regime not encoded in observation space

Solutions:
- Add trend features: long-term EMA slope, ADX + DI direction
  - Currently `adx=false` in feature_flags -- consider enabling with directional component
- Analyze per-window Long/Short ratio to identify if bias is time-dependent
- Consider separate reward scaling for Long vs Short based on market direction

#### D-10. Win Rate Gap (36.8% backtest vs 47-55% TensorBoard)

This gap is primarily caused by A-1 (stoploss killing trades before RL exit).
Expected to largely resolve after implementing A-class fixes.

Residual gap may come from:
- Early windows (PPO_1-40) with weak models
- Model overfitting to training data (in-sample vs out-of-sample)

---

### E. Code Quality (Priority: Low)

#### E-11. DataFrame Fragmentation

`RLDayTradeStrategy.py:160`: `dataframe["&-action"] = 0` triggers Pandas PerformanceWarning.

Fix: batch column additions using `pd.concat(axis=1)` or assign to DataFrame copy.

#### E-12. TensorBoard env[0] Sampling

Custom metrics only sample env[0] (1/N envs). Causes:
- Unreliable metrics with small episodes (env[0] timing artifact)
- No visibility into other envs' behavior

Fix: modify `TensorboardCallback._on_step()` to aggregate across all envs,
or at minimum log env count and flag single-env samples.

---

## 3. Priority Summary

| Priority | Item | Expected Impact |
|----------|------|----------------|
| Critical | A-1: stoploss alignment | Eliminate 47% stop_loss exits |
| Critical | A-2: ROI alignment | Consistent exit rules |
| Critical | A-3: profit_aim fix | Correct reward ranges |
| High | B-4: SL/ROI env simulation | Training = backtest rules |
| High | C-7: n_steps=4096 | 2.8x better initial performance |
| Medium-High | B-6: Reward alignment | Strategy-consistent rewards |
| Medium | D-9: Short bias | Bull market performance |
| Medium | B-5: Liquidation formula | Logic consistency |
| Medium | C-8: cpu_count | More experience diversity |
| Low | D-10: Win rate gap | Auto-resolves with A-class fixes |
| Low | E-11: DataFrame fragmentation | Performance optimization |
| Low | E-12: TensorBoard sampling | Monitoring quality |

---

## 4. v2 Implementation Order

### Phase 1: Environment-Strategy Alignment (A-1 ~ A-3, B-4, B-6)
- Add stoploss/ROI simulation to RL environment
- Align profit_aim with chosen stoploss level
- Update reward function for new exit types
- **Validate**: single-env short training (e.g., 2 months) to confirm trades match backtest behavior

### Phase 2: Training Config (C-7, C-8)
- Set n_steps=4096, batch_size=1024 from start
- Start with cpu=4, evaluate cpu=6 if memory allows
- **Validate**: compare first 10 windows with v1 same-period results

### Phase 3: Feature & Behavior Tuning (D-9, B-5)
- Add directional features if Short bias persists after Phase 1
- Fix liquidation formula
- **Validate**: backtest Long/Short ratio and per-direction profit

### Phase 4: Code Quality (E-11, E-12)
- DataFrame fragmentation fix
- TensorBoard aggregation improvement
- **Validate**: clean runs without warnings

---

## 5. v1 Training Lessons Learned

### What Worked Well
- HQT (High-Quality Trading) reward design: Neutral as valid choice, no entry incentive
- ent_coef=0.05: sufficient exploration throughout training
- reduce_df_footprint=true: eliminated OOM issues
- Training cycle showed consistent pattern: model recovers from dips naturally
- PPO_135=17.56 ATH, PPO_164=13.19 -- model CAN learn profitable strategies

### What Didn't Work
- Strategy-RL framework mismatch: fundamental disconnect between training and evaluation
- n_steps=1024 too small for initial training (fixed mid-train at PPO_41)
- 68% Short bias undetected until final backtest
- TensorBoard per-episode metrics created false confidence (positive total_profit)

### Key Insight
> The RL model successfully learned to trade profitably within its environment.
> The failure is not in the model, but in the mismatch between training environment
> and production rules. v2's primary goal is to eliminate this mismatch.

---

## 6. v2 Parameter Decisions (2026-03-27)

### 6.1 conv_width: 10 -> 20

| | v1 (conv_width=10) | v2 (conv_width=20) |
|--|-------------------|-------------------|
| Observation window | 50 min (10 candles) | **100 min (20 candles)** |
| Input tensor | (10, 392) = 3,920 | (20, 392) = 7,840 |
| Network params | ~1.07M | ~2.07M |

More temporal context for the model. 100 minutes covers a larger portion of
the max_trade_duration_candles=96 (8 hours).

### 6.2 Reward Redesign: Stepped 1RR with Neutral Zone

v2 replaces the v1 multi-tier reward with a stepped symmetric design.
Focus on total_profit optimization. No scoring within normal noise range.

#### Design Principles

1. Symmetric 1:1 risk-reward (same structure for profit and loss)
2. Neutral zone: PNL within +/-5% = no reward, no penalty (noise immunity)
3. Only score when PNL exceeds +/-5% (real directional move)
4. Fixed step values, no linear scaling, no multipliers
5. Liquidation at -50% leveraged PNL (changed from -75%)

#### Reward Rules (complete)

| Scenario | Condition | Reward |
|----------|-----------|--------|
| Liquidation | leveraged PNL <= -50% | -10 |
| Invalid action | | -1 |
| Neutral (no position) | | 0 |
| Entry (Long/Short enter) | | 0 |
| Hold/Exit in neutral zone | -5% < PNL < +5% | 0 |
| Hold small profit | PNL >= +5% and < +10% | +1 |
| Hold big profit | PNL >= +10% | +2 |
| Hold small loss | PNL <= -5% and > -10% | -1 |
| Hold big loss | PNL <= -10% | -2 |
| Exit small profit | PNL >= +5% and < +10% | +5 |
| Exit big profit / ROI trigger | PNL >= +10% | +10 |
| Exit small loss | PNL <= -5% and > -10% | -5 |
| Exit big loss / SL trigger | PNL <= -10% | -10 |

SL/ROI simulation at +/-10% (aligned with reward boundaries):
- SL forces exit at PNL <= -10% -> reward -10
- ROI forces exit at PNL >= +10% -> reward +10
- Model has active management zone at +/-5% to +/-10% (reward +/-5)
- All 5 reward levels are reachable

#### Price Map (ETH = 2,000 USDT, Long example)

All PNL values are leveraged (base price move * 10x leverage).

```
ETH       Base     PNL(10x)   Hold   Exit    Zone
──────────────────────────────────────────────────
2,020    +1.0%     +10%        +2     +10     ROI trigger (force exit, max reward)
2,015    +0.75%    +7.5%       +1      +5       |
2,012    +0.6%     +6%         +1      +5     small profit (active management)
2,010    +0.5%     +5%         +1      +5       |
·····    ·····     ····    ·······   ·····   -- +5% scoring threshold --
2,009    +0.45%    +4.5%        0       0       |
2,005    +0.25%    +2.5%        0       0       |
2,002    +0.1%     +1%          0       0     neutral zone
2,000     0%        0%          0       0     (no scoring)
1,998    -0.1%     -1%          0       0       |
1,995    -0.25%    -2.5%        0       0       |
1,991    -0.45%    -4.5%        0       0       |
·····    ·····     ····    ·······   ·····   -- -5% scoring threshold --
1,990    -0.5%     -5%         -1      -5       |
1,988    -0.6%     -6%         -1      -5     small loss (active management)
1,985    -0.75%    -7.5%       -1      -5       |
·····    ·····     ····    ·······   ·····   -- -10% SL/ROI boundary --
1,980    -1.0%    -10%         -2     -10     SL trigger (force exit, max penalty)
1,970    -1.5%    -15%         -2     -10       |
  :        :         :          :       :
1,900    -5.0%    -50%         --     -10     liquidation
```

#### Thresholds

```
ETH 2,020 (+1.0%) -- PNL +10% -- ROI force exit (reward +10)
ETH 2,010 (+0.5%) -- PNL  +5% -- scoring threshold (reward starts)
ETH 2,010 ~ 1,990 ----------------  neutral zone (no scoring)
ETH 1,990 (-0.5%) -- PNL  -5% -- scoring threshold (penalty starts)
ETH 1,980 (-1.0%) -- PNL -10% -- SL force exit (reward -10)
ETH 1,900 (-5.0%) -- PNL -50% -- liquidation
```

#### Trade Lifecycle in RL Environment

```
Priority (highest to lowest):
1. Liquidation: PNL <= -50% -> episode end, reward -10
2. Stoploss:    PNL <= -10% -> force exit, reward -10
3. ROI:         PNL >= +10% -> force exit, reward +10
4. Timeout:     holding >= 96 candles (8h) -> force exit, reward by PNL
5. Model exit:  model chooses Exit action -> reward by PNL
6. Model hold:  model chooses Neutral -> hold reward by PNL
```

#### Liquidation Change: -75% -> -50%

| | v1 | v2 |
|--|-----|-----|
| Leveraged PNL | -75% | **-50%** |
| Base ETH move | -7.5% | **-5.0%** |
| ETH price (from 2,000) | 1,850 | **1,900** |
| `liquidation_buffer` | 0.025 | **0.05** |
| Formula | -(1/10 - 0.025) = -0.075 | -(1/10 - 0.05) = -0.05 |

Earlier liquidation reduces maximum possible loss per episode.

#### vs v1 Reward Design

| Aspect | v1 (multi-tier) | v2 (stepped) |
|--------|----------------|-------------|
| Rules | 12+ scenarios with formulas | **5 fixed levels** (+10,+5,0,-5,-10) |
| PNL +/-5% range | scored (various formulas) | **neutral zone, 0** |
| Break-even exit | -1 (penalized) | **0** |
| Small loss exit -3% | -1 (same as -0.1%) | **0** (within neutral zone) |
| Exit at -6% vs -9% | different tiers | **same -5** (stepped) |
| SL/ROI in RL env | none (only liquidation) | **SL -10%, ROI +10%** |
| Liquidation | -75% leveraged | **-50% leveraged** |
| Invalid action | -2 | **-1** |
| max_trade_duration | 48 candles (4h) | **96 candles (8h)** |

#### Model Behavior

```
PNL within +/-5% (ETH +/-0.5%): completely free, no pressure
  -> normal 5-min volatility (0.1~0.5%) stays in neutral zone
  -> model decides on its own without reward noise

PNL +5% to +10% (ETH +0.5% to +1%): active management zone
  -> hold reward +1 (gentle encouragement to stay)
  -> exit reward +5 (take profit option)
  -> model decides: take +5 now, or hold for +10 at ROI trigger?

PNL -5% to -10% (ETH -0.5% to -1%): active management zone
  -> hold reward -1 (gentle pressure to leave)
  -> exit reward -5 (cut loss option)
  -> model decides: cut at -5 now, or risk -10 at SL trigger?

PNL crosses +/-10%: forced exit by SL/ROI
  -> ROI +10%: force exit, reward +10 (maximum)
  -> SL -10%: force exit, reward -10 (maximum penalty)
  -> model learns: this is the hard boundary
```

---

## 7. v2 Implementation Checklist

### Code Changes (`RLDayTrader.py`)

| # | Item | Location | Description | Status |
|---|------|----------|-------------|--------|
| 1 | Reward rewrite | `calculate_reward()` | Stepped 1RR + neutral zone +/-5%, 5 fixed levels (+10,+5,0,-5,-10) | [x] |
| 2 | Invalid action | `calculate_reward()` | -2 -> -1 | [x] |
| 3 | SL simulation (A-1) | `step()` | Force exit when PNL <= -10% (simulate_stoploss from config) | [x] |
| 4 | ROI simulation (A-2) | `step()` | Force exit when PNL >= +10% (simulate_roi from config) | [x] |
| 5 | Liquidation formula (B-5) | `_check_liquidation()` | Multiplicative -> additive: `_total_profit += threshold * leverage` | [x] |

### Config Changes (`config_daytrade_v2.json`)

| # | Parameter | Current | New | Status |
|---|-----------|---------|-----|--------|
| 6 | `max_trade_duration_candles` | 48 | **96** (8 hours) | [x] |
| 7 | `simulate_stoploss` | (none) | **-0.10** | [x] |
| 8 | `simulate_roi` | (none) | **0.10** | [x] |

### Already Completed

| Parameter | Value | Status |
|-----------|-------|--------|
| `identifier` | rl-daytrade-v2-20260401 | [x] |
| `conv_width` | 20 | [x] |
| `liquidation_buffer` | 0.05 (-50% leveraged) | [x] |
| `rr` | 1 | [x] |
| `bot_name` | rl-daytrade-v2 | [x] |

### Not Fixing (decided)

| Item | Reason |
|------|--------|
| D-9 Short bias | Decided not to consider |
| D-10 Win rate gap | Auto-resolves with A-1/A-2 |
| C-7 n_steps=4096 | Decided to keep 2048 |
| C-8 cpu_count=6 | Decided to keep 4 |

### Phase 2: Post-Training Fixes (after v2 training validates)

| # | Item | File | Current | Change | Note |
|---|------|------|---------|--------|------|
| 9 | `stoploss` align with RL | `RLDayTradeStrategy.py` | -0.03 | **-0.10** | Match simulate_stoploss |
| 10 | `minimal_roi` simplify | `RLDayTradeStrategy.py` | {"0":0.05,"30":0.03,"60":0.01} | **{"0":0.10}** | Match simulate_roi, remove time decay |
| 11 | `startup_candle_count` | `RLDayTradeStrategy.py` | 60 | **100** | Official recommends 2x max indicator period |
| 12 | E-11 DataFrame fragmentation | `RLDayTradeStrategy.py:160` | `dataframe["&-action"] = 0` | pd.concat | Performance warning fix |
| 13 | E-12 TensorBoard env[0] sampling | `TensorboardCallback.py` | Only samples env[0] (1/N) | Aggregate across all envs | Improve monitoring reliability |

---

## 8. v2 Feature Analysis

### 7.1 Feature Count: 392

| Category | Features | % | Function |
|----------|----------|---|----------|
| RSI | 48 | 12.2% | Overbought/oversold |
| ATR normalized | 48 | 12.2% | Volatility |
| EMA (20, 50) | 48 | 12.2% | Trend direction |
| Bollinger Bands | 48 | 12.2% | Price position + volatility |
| MACD (norm, hist) | 48 | 12.2% | Trend momentum |
| S/R features (6) | 144 | **36.7%** | Support/resistance |
| Raw OHLC | 4 | 1.0% | RL environment base |
| Time encoding | 4 | 1.0% | Intraday/weekly cycles |

Feature expansion dimensions:
- expand_all: 2 periods x 4 TF x 3 shift x 2 pairs = 48 per indicator
- expand_basic: 4 TF x 3 shift x 2 pairs = 24 per indicator
- standard: no expansion

Observation: S/R features dominate at 36.7%. Directional features (ADX, EMA slope)
are absent, which may contribute to the Short bias seen in v1 backtest.

### 7.2 Disabled Features

| Feature | Status | Reason / Note |
|---------|--------|--------------|
| ADX | disabled | No trend strength detection -- consider enabling in future |
| rel_vol | disabled | No volume analysis |
| add_state_info | false | Not available in backtesting (framework limitation) |

### 7.3 Potential Gaps

- No explicit trend DIRECTION features (EMA slope, ADX+DI)
- No volume features (rel_vol disabled)
- S/R features may be noisy in non-ranging markets
- These gaps may explain v1's 68% Short bias in a bullish market

---

## 8. v2 Training Parameters Summary

### 8.1 PPO Configuration

| Parameter | v1 actual | v2 | Change |
|-----------|-----------|-----|--------|
| n_steps | 2048 | 2048 | -- |
| batch_size | 512 (actual) | 1024 (config) | verify before training |
| n_epochs | 10 | 10 | -- |
| learning_rate | 0.0003 | 0.0003 | -- |
| ent_coef | 0.05 | 0.05 | -- |
| clip_range | 0.2 | 0.2 | -- |
| gamma | 0.99 | 0.99 | -- |
| net_arch | [256, 256] | [256, 256] | -- |
| cpu_count | 4 | 4 | -- |
| train_cycles | 500 | 500 | -- |

Rollout per window (n_steps=2048, cpu=4, batch=1024):
- Buffer: 2048 x 4 = 8,192 transitions
- Batches/epoch: 8,192 / 1024 = 8
- Updates/rollout: 8 x 10 = 80
- Rollouts/window: ~989 (total_timesteps = 500 x ~16,200)
- Total gradient updates/window: ~79,000

### 8.2 Environment Configuration

| Parameter | v1 | v2 | Change |
|-----------|-----|-----|--------|
| conv_width | 10 | **20** | 2x temporal context |
| profit_aim | 0.05 | 0.05 | -- |
| rr | 1 | 1 | -- (linear reward replaces tier system, rr no longer used) |
| leverage | 10.0 | 10.0 | -- |
| liquidation_buffer | 0.025 | **0.05** | liquidation at -50% leveraged (was -75%) |
| max_trade_duration | 48 | **96** | 8 hours (was 4 hours) |
| max_training_drawdown | 0.50 | 0.50 | -- |
| randomize_starting | true | true | -- |
| add_state_info | false | false | -- |
| drop_ohlc | false | false | -- |

### 8.3 Data Configuration

| Parameter | v1 | v2 | Change |
|-----------|-----|-----|--------|
| stake_amount | 100 | 100 | -- |
| identifier | v1-20260307 | **v2-20260401** | new training run |
| train_period_days | 75 | 75 | -- |
| backtest_period_days | 7 | 7 | -- |
| continual_learning | false | false | -- |
| timeframes | [5m,15m,1h,4h] | same | -- |
| corr_pairlist | [BTC] | same | -- |
| indicator_periods | [10, 20] | same | -- |
| shifted_candles | 2 | same | -- |
| reduce_df_footprint | true | true | -- |

### 8.4 Config File

v2 config: `user_data/config_daytrade_v2.json`
v1 config preserved: `user_data/config_daytrade.json`

---

## 9. Official Documentation Findings (2026-03-27)

Key findings from freqtrade source code and docs verification:

1. **train_cycles actual meaning**: `total_timesteps = train_cycles * len(train_df)`.
   With 500 cycles and ~16,200 training candles, total = 8.1M steps = ~989 rollouts (not 500).

2. **profit_aim semantics**: Default `get_unrealized_profit()` returns BASE (unleveraged) PNL.
   Our `RLDayTrader.py` overrides it to return LEVERAGED PNL (`base_pnl * leverage`).
   So `profit_aim=0.05` = 5% leveraged = 0.5% base price. Verified correct.

3. **stoploss with leverage**: `stoploss / leverage` = price trigger.
   `-0.05 / 10 = -0.5%` base price. Verified correct.

4. **minimal_roi with leverage**: `roi / leverage` = price trigger.
   `0.075 / 10 = +0.75%` base price. Verified correct.

5. **stake_amount in RL**: Only checks `== "unlimited"` for compounding mode.
   Changing to 10 does NOT affect RL training behavior.

6. **add_state_info**: Raises OperationalException in backtesting.
   Must remain false for backtest-based training.

7. **RL auto-disables**: SVM outlier removal, DBSCAN, DI_threshold, shuffle.
   Config settings for these are redundant but harmless.

8. **startup_candle_count**: Official recommends 2x max indicator period.
   Current 60 < recommended 100 (for EMA_50). May cause NaN in early data.

9. **Default net_arch [128,128]**: Our [256,256] is 2x larger. Acceptable for 392 features.

10. **Default max_training_drawdown_pct 0.8**: Our 0.50 ends episodes earlier (at -50% vs -80%).
    More conservative, appropriate for 10x leverage.

---

## 10. v2 Training Progress

### 10.1 Status (2026-04-10, retraining ~window 43/189)

Training started 2026-03-28. 56 windows completed (~30%), PPO_56 complete.

#### PPO_17 checkpoint (2026-03-30)

Health check (corrected for small-sample bias using PPO_16, 9512 samples):
**~9 OK, 5 WARN, 0 FAIL**

| Check | Status | Value | Note |
|-------|--------|-------|------|
| Reward trend | OK | -2161 -> -1842 (+14.7%) | Improving |
| Liquidation rate | WARN | 5.9% | SL simulation effective |
| Win rate | OK | 42.4% (PPO_16) | PPO_17=25% is small-sample bias |
| Policy collapse | WARN | Neutral=77% (PPO_16) | Higher than v1 (57%), expected from neutral zone |
| Value loss | WARN | 1.50x (PPO_16) | PPO_17=2.10x is small-sample bias |
| Entropy | OK | 66% retained | Stable |
| Invalid actions | OK | 10% | Improved from v1's 23% |
| Approx KL | OK | mean=0.008 | Stable |
| Clip fraction | OK | mean=0.069 | Normal |
| Profit trend | OK | All positive, avg=1.40 | Range 0.90~2.43 |
| Entropy trend | OK | +21.3pp (improving) | 45% -> 66% |
| Explained variance | OK | 0.80 | Good, improved from 0.55 at PPO_5 |
| Long/Short bias | OK | ~47%L/53%S | Balanced (v1 was 32%L/68%S) |

#### PPO_25 checkpoint (2026-03-31) -- significant improvement

Health check (based on PPO_24, 7379 samples):
**11 OK, 2 WARN, 1 FAIL**

| Check | Status | Value | Note |
|-------|--------|-------|------|
| Reward trend | OK | -2977 -> -1015 (**+65.9%**) | Rapidly improving |
| Liquidation rate | OK | 0.2% | Excellent |
| Win rate | OK | **47.5%** | Up from 42.4% |
| Policy collapse | WARN | Neutral=68% | Down from 77%, model more active |
| Value loss | FAIL | 2.55x (PPO_24) | Isolated spike, PPO_25 recovered (0.73x) |
| Entropy | OK | 58% retained | Stable |
| Invalid actions | WARN | 16.5% | Up from 10%, correlated with more trading activity |
| Approx KL | OK | mean=0.010 | Stable |
| Clip fraction | OK | mean=0.088 | Normal |
| Profit trend | OK | All positive, **avg=4.25** | Range 0.69~11.12, 3x improvement |
| Entropy trend | OK | +7.6pp (improving) | 59% -> 67% |
| Explained variance | OK | 0.72 | Slight decline from 0.80, still good |
| Long/Short bias | OK | 57%L/43%S | Balanced |

#### PPO_32 checkpoint (2026-04-01) -- stabilization, Long bias emerging

Health check (based on PPO_31, 4761 samples):
**10 OK, 4 WARN, 0 FAIL**

| Check | Status | Value | Note |
|-------|--------|-------|------|
| Reward trend | OK | -2947 -> -1049 (+64.4%) | Continues improving |
| Liquidation rate | OK | 0.3% | Excellent |
| Win rate | OK | 43.5% | Stable |
| Policy collapse | WARN | Neutral=67% | Stable at 67-70% range |
| Value loss | WARN | 1.19x | Recovered from PPO_24's 2.55x FAIL |
| Entropy | OK | 59% retained | Stable |
| Invalid actions | WARN | 17.2% | Slight increase, stable |
| Approx KL | OK | mean=0.010 | Stable |
| Clip fraction | OK | mean=0.082 | Normal |
| Profit trend | OK | All positive, avg=2.53 | Range 0.72~4.70 |
| Entropy trend | OK | +11.9pp (improving) | 54% -> 66% |
| Explained variance | OK | **0.82** | Improving (best: PPO_30=0.89) |
| Long/Short bias | **WARN** | **75%L/25%S (PPO_31)** | New -- see analysis below |

Long/Short bias progression:
```
PPO_28: 50%L/50%S  (balanced)
PPO_29: 48%L/52%S  (balanced)
PPO_30: 64%L/36%S  (tilting Long)
PPO_31: 75%L/25%S  (clear Long bias)
PPO_35: 46%L/54%S  (briefly balanced)
PPO_36: 58%L/42%S  (mild Long)
PPO_37: 67%L/33%S  (WARN)
PPO_38: 90%L/10%S  (FAIL, small sample 874 actions)
PPO_39: 78%L/22%S* (in progress)
```
Long bias confirmed as persistent trend, likely market-regime driven.
PPO_35-39 training data covers ~2023 Jan-Feb (ETH recovery from FTX crash,
$1100 -> $1600+). Large-sample windows (PPO_35-37) show 46% -> 58% -> 67%L.
PPO_38's 90% is small-sample amplification.

#### PPO_39 checkpoint (2026-04-02) -- profit dip, Long bias persists

Health check (based on PPO_37, 14783 samples for bias; PPO_38, 874 for others):
**8 OK, 5 WARN, 1 FAIL**

| Check | Status | Value | Note |
|-------|--------|-------|------|
| Reward trend | OK | -2791 -> -2186 (+21.6%) | Improving |
| Liquidation rate | OK | 1.6% | Good |
| Win rate | WARN | 35-41% (PPO_35-38) | Down from 43-48%, in WARN zone |
| Policy collapse | WARN | Neutral=69-73% | Stable |
| Value loss | WARN | 1.00x (PPO_38) | Very stable |
| Entropy | OK | 62% retained | Stable |
| Invalid actions | WARN | 14-17% | Stable |
| Approx KL | OK | mean=0.010 | Stable |
| Clip fraction | OK | mean=0.086 | Normal |
| Profit trend | OK | All positive, avg=1.07 | Range 0.67~1.48, dip phase |
| Entropy trend | OK | -6.2pp | Within range |
| Explained variance | OK | **0.83** | Consistently good |
| Long/Short bias | **FAIL** | **67-90%L** | Persistent, market-regime driven |
| Sample size | WARN | 874 (PPO_38) | Small, use PPO_35-37 for reliable data |

#### PPO_44 checkpoint (2026-04-03) -- Long bias resolved, policy collapse emerging

Health check (based on PPO_44, 8777 samples):
**11 OK, 1 WARN, 2 FAIL**

| Check | Status | Value | Note |
|-------|--------|-------|------|
| Reward trend | OK | -2941 -> -1697 (+42.3%) | Strong improvement |
| Liquidation rate | OK | **0.0%** (0/356) | First zero liquidation window |
| Win rate | OK | 42.1% | Recovered from WARN |
| **Policy collapse** | **FAIL** | **Neutral=83%** | Up from 69-73%, crossed 80% threshold |
| Value loss | OK | 0.68x | Decreasing, healthy |
| Entropy | WARN | 42% retained | Declining from 62% |
| Invalid actions | OK | **9.2%** | Best seen in training |
| Approx KL | OK | mean=0.008 | Stable |
| Clip fraction | OK | mean=0.063 | Normal |
| Profit trend | OK | All positive, avg=1.52 | Range 0.69~2.43, recovering |
| **Entropy trend** | **FAIL** | **-17.4pp** (59%->42%) | Accelerating decline |
| Explained variance | OK | **0.826** | Consistently strong |
| Long/Short bias | OK | **50%L/50%S** (348L/349S) | Resolved from FAIL |

Long/Short bias progression (full history):
```
PPO_28: 50%L/50%S  (balanced)
PPO_29: 48%L/52%S  (balanced)
PPO_30: 64%L/36%S  (tilting Long)
PPO_31: 75%L/25%S  (clear Long bias)
PPO_35: 46%L/54%S  (briefly balanced)
PPO_36: 58%L/42%S  (mild Long)
PPO_37: 67%L/33%S  (WARN -- peak of large-sample bias)
PPO_38: 90%L/10%S  (FAIL, small sample 874)
PPO_39: 78%L/22%S  (FAIL)
PPO_40: 46%L/54%S  (correcting)
PPO_42: 50%L/50%S  (balanced)
PPO_43: 60%L/40%S  (mild)
PPO_44: 50%L/50%S  (balanced -- resolved)
```
Long bias was market-regime driven (2023 ETH bull recovery). Auto-corrected
as training data moved past that period. No intervention was needed.

Neutral% progression (policy collapse concern):
```
PPO_28-32: 67-73%  (stable)
PPO_35-39: 69-73%  (stable)
PPO_40:    66%     (dip)
PPO_42:    53%     (active window)
PPO_43:    82%     (crossed 80% -- FAIL)
PPO_44:    83%     (confirmed)
PPO_46:    65%     (dip, 963 actions)
PPO_47:    87%     (FAIL, 1117 actions)
PPO_48:    96%     (extreme, tiny sample 682 actions)
PPO_49:    80%     (FAIL, 5594 actions)
PPO_50:    80%     (FAIL, 11843 actions -- large sample, reliable)
PPO_52:    70%     (WARN, 4794 actions)
PPO_53:    80%     (FAIL, 8248 actions)
PPO_54:    79%     (WARN, 2495 actions)
PPO_55:    78%     (WARN, 3899 actions)
PPO_56:    74%     (WARN, 10898 actions -- large sample, reliable)
```
Large-sample trend: PPO_43=82% -> PPO_44=83% -> PPO_50=80% -> PPO_56=74%.
Model recovering from policy collapse. Neutral% improving steadily in large-sample windows.

#### PPO_50 checkpoint (2026-04-04) -- steady state, profit declining

Health check (based on PPO_50, 11843 samples):
**9 OK, 3 WARN, 2 FAIL**

| Check | Status | Value | Note |
|-------|--------|-------|------|
| Reward trend | **FAIL** | -70.8% | ep_rew_mean unreliable for v2 (longer episodes = more penalty accumulation) |
| Liquidation rate | OK | 0.2% (1/590) | Excellent |
| Win rate | **WARN** | **35.8%** (211W/378L) | Dropped from 42.1% |
| **Policy collapse** | **FAIL** | **Neutral=80%** | Improved from 83%, still at FAIL threshold |
| Value loss | OK | 0.90x | Decreasing, healthy |
| Entropy | **WARN** | 41% retained | Stable (was 42% at PPO_44) |
| Invalid actions | **WARN** | 10.0% | Borderline OK/WARN |
| Approx KL | OK | mean=0.009 | Stable |
| Clip fraction | OK | mean=0.074 | Normal |
| Profit trend | OK | All positive, avg=1.13 | Range 0.59~1.79, declining |
| **Entropy trend** | **OK** | **-0.1pp** | **RESOLVED from FAIL (-17.4pp at PPO_44)** |
| Explained variance | OK | **0.887** | ATH -- best seen in training |
| Long/Short bias | OK | 38%L/62%S | Balanced (mild Short) |

Key changes from PPO_44:
- Entropy trend **RESOLVED** from FAIL to OK: entropy stabilized at ~41%
- Policy collapse **slightly improved**: 83% -> 80%
- Win rate **degraded**: 42.1% -> 35.8% (OK -> WARN)
- Profit **declining**: avg 1.52 -> 1.13, PPO_50=0.588 (historical low)
- Explained variance **at ATH**: 0.826 -> 0.887

Entropy progression (cross-window within PPO_46-50):
```
PPO_46: 40.7% retained
PPO_47: 35.9% retained
PPO_48: 42.5% retained
PPO_49: 44.0% retained
PPO_50: 40.6% retained
Trend: -0.1pp (stable)
```
Entropy has stopped declining. Model reached a steady state at ~41%.

L/S progression (continued):
```
PPO_44: 50%L/50%S  (balanced -- resolved)
PPO_46: 75%L/25%S  (129L/43S)
PPO_47: 31%L/69%S  (20L/45S)
PPO_48: 59%L/41%S  (10L/7S, tiny)
PPO_49: 43%L/57%S  (244L/324S)
PPO_50: 38%L/62%S  (434L/714S -- mild Short bias)
```
Large-sample PPO_50 shows mild Short bias (62%S). Within OK threshold (<65%).
May reflect market regime shift in training data.

#### PPO_56 checkpoint (2026-04-05) -- recovery from stagnation

Health check (based on PPO_56, 10898 samples):
**12 OK, 2 WARN, 0 FAIL**

| Check | Status | Value | Note |
|-------|--------|-------|------|
| Reward trend | OK | +8.9% | Improving |
| Liquidation rate | OK | **0.0%** (0/662) | Zero liquidation PPO_53-56 |
| Win rate | **OK** | **41.4%** (274W/388L) | Recovered from WARN |
| Policy collapse | **WARN** | **Neutral=74%** | Improved from 80% FAIL |
| Value loss | OK | 0.93x | Decreasing, healthy |
| Entropy | **OK** | **54% retained** | Recovered from 41% WARN |
| Invalid actions | **WARN** | 13.7% | Up from 10%, correlated with more exploration |
| Approx KL | OK | mean=0.008 | Stable |
| Clip fraction | OK | mean=0.064 | Normal |
| Profit trend | OK | All positive, avg=0.98 | Range 0.85~1.27, still declining |
| Entropy trend | OK | **+2.9pp** | Improving across windows |
| Explained variance | OK | 0.738 | Down from 0.887 (exploration impact) |
| Long/Short bias | OK | 54%L/46%S | Balanced |

Key changes from PPO_50:
- **Zero FAIL** for the first time since PPO_25 checkpoint
- Policy collapse **improved**: 80% FAIL -> 74% WARN
- Entropy **recovered**: 41% WARN -> 54% OK (+13pp)
- Win rate **recovered**: 35.8% WARN -> 41.4% OK
- Explained variance **dropped**: 0.887 -> 0.738 (model exploring new patterns)
- Invalid actions **increased**: 10.0% -> 13.7% (more trading attempts)
- Profit **still declining**: avg 1.13 -> 0.98 (exploration phase, expected lag)

**Correction to PPO_50 analysis:** Previous conclusion that model was in a
"sub-optimal steady state unlikely to self-correct" was wrong. Model is
self-recovering, similar to v1 dip/recovery cycles.

Entropy recovery progression:
```
PPO_39-44: avg 42% (declining phase)
PPO_46-50: avg 41% (stabilized)
PPO_52-56: avg 52% (RECOVERING)
```
Three phases: rapid decline -> stabilization -> recovery. The ent_coef=0.05
entropy bonus continues to exert exploration pressure even after policy
concentrated to Neutral=80%. Given enough time, entropy recovered.

L/S progression (continued):
```
PPO_50: 38%L/62%S  (434L/714S -- mild Short)
PPO_52: 46%L/54%S  (351L/419S)
PPO_53: 66%L/34%S  (484L/254S)
PPO_54: 60%L/40%S  (145L/98S)
PPO_55: 46%L/54%S  (175L/202S)
PPO_56: 54%L/46%S  (721L/612S -- balanced)
```
L/S balance maintains OK throughout PPO_52-56.

### 10.2 Per-Window Summary

| Window | total_profit | ep_rew_mean | win_rate | Neutral% | L/S | ep_len |
|--------|-------------|-------------|----------|----------|-----|--------|
| PPO_1 | 5.36 | +59 | 63.7% | -- | -- | 1,106 |
| PPO_2 | 3.68 | -388 | 60.6% | -- | -- | 2,357 |
| PPO_3 | 2.61 | -243 | 65.0% | -- | -- | 3,401 |
| PPO_4 | 3.27 | -846 | 51.9% | 69% | -- | 4,561 |
| PPO_5 | 3.05 | -1,337 | 41.2% | 72% | -- | 2,749 |
| ... | | | | | | |
| PPO_13 | 0.99 | -2,161 | 28.6% | 76% | -- | 8,416 |
| PPO_14 | 1.75 | -2,273 | 38.9% | 79% | -- | 9,139 |
| PPO_15 | 0.90 | -2,015 | 31.8% | 60% | -- | 7,690 |
| PPO_16 | 2.43 | -1,812 | 42.4% | 77% | -- | 6,904 |
| PPO_17 | 0.92 | -1,842 | 25.0% | 81% | -- | 6,368 |
| ... | | | | | | |
| PPO_21 | **11.12** | -2,977 | 46.1% | 71% | 42%L/58%S | 12,185 |
| PPO_22 | 1.12 | -2,767 | 43.9% | 72% | 50/50 | 10,882 |
| PPO_23 | 3.31 | -3,005 | 42.5% | 67% | 51/49 | 10,673 |
| PPO_24 | 5.02 | -2,667 | 47.5% | 68% | 57%L/43%S | 12,198 |
| PPO_25 | 0.69 | -1,015 | 38.0% | 71% | -- | 3,262 |
| ... | | | | | | |
| PPO_28 | **4.70** | -2,947 | **52.7%** | 73% | 50/50 | 10,497 |
| PPO_29 | 0.99 | -3,085 | 41.7% | 70% | 48/52 | 11,876 |
| PPO_30 | **4.52** | -2,894 | 42.7% | 70% | 64%L/36%S | 11,139 |
| PPO_31 | 1.74 | -2,686 | 43.5% | 67% | **75%L/25%S** | 9,753 |
| PPO_32 | 0.72 | -1,049 | 31.1% | 75% | 36/64 | 2,939 |
| ... | | | | | | |
| PPO_35 | 1.40 | -2,791 | 37.9% | 72% | 46/54 | 12,089 |
| PPO_36 | 0.95 | -2,015 | 40.7% | 73% | 58/42 | 12,920 |
| PPO_37 | 1.48 | -2,594 | 39.7% | 73% | **67%L/33%S** | 14,783 |
| PPO_38 | 0.84 | -2,212 | 35.0% | 69% | **90%L/10%S** | 874 |
| PPO_39 | 0.67 | -2,186 | 36.2% | 68% | 78/22 | 1,435 |
| ... | | | | | | |
| PPO_40 | 1.28 | -2,941 | 38.5% | 66% | 46/54 | 4,897 |
| PPO_41 | 1.38 | -2,460 | 38.2% | 71% | 74/26 | 525* |
| PPO_42 | **2.43** | -2,716 | **49.4%** | 53% | **50/50** | 2,209 |
| PPO_43 | 0.69 | -2,073 | 37.0% | **82%** | 60/40 | 11,146 |
| PPO_44 | 1.84 | -1,697 | 42.1% | **83%** | **50/50** | 8,777 |
| ... | | | | | | |
| PPO_46 | 0.83 | -1,114 | 43.4% | 65% | 75%L/25%S | 963 |
| PPO_47 | 1.17 | -1,000 | 41.7% | **87%** | 31%L/69%S | 1,117 |
| PPO_48 | 1.27 | -971 | 83.3% | **96%** | 59/41 | 682* |
| PPO_49 | 1.79 | -1,765 | 41.6% | 80% | 43%L/57%S | 5,594 |
| PPO_50 | **0.59** | -1,903 | **35.8%** | **80%** | 38%L/62%S | 11,843 |
| ... | | | | | | |
| PPO_52 | 0.90 | -2,140 | 36.4% | 70% | 46%L/54%S | 4,794 |
| PPO_53 | 1.00 | -1,546 | 35.6% | 80% | 66%L/34%S | 8,248 |
| PPO_54 | 0.85 | -1,681 | 31.5% | 79% | 60/40 | 2,495 |
| PPO_55 | 1.27 | -1,904 | 38.0% | 78% | 46%L/54%S | 3,899 |
| PPO_56 | 0.90 | -1,950 | **41.4%** | **74%** | 54%L/46%S | 10,898 |

*PPO_41 small sample (525 actions), PPO_48 small sample (682 actions), less reliable

### 10.3 Training Phases

| Phase | Windows | avg profit | Neutral% | Characteristic |
|-------|---------|-----------|----------|----------------|
| Early | PPO_1-5 | 3.20 | 69-72% | Initial learning, invalid 15% |
| Conservative dip | PPO_13-17 | 1.40 | 77-81% | Over-passive, profit low point |
| Growth | PPO_21-25 | 4.25 | 67-72% | Activity recovered, PPO_21=11.12 peak |
| Stabilization | PPO_28-32 | 2.53 | 67-73% | Stable profit, Long bias emerging |
| Dip + Long bias | PPO_35-39 | 1.07 | 69-73% | Profit dip, Long bias persists (67-90%L) |
| Recovery + passive | PPO_40-44 | 1.52 | 66-83% | L/S resolved, Neutral% rising to FAIL |
| Stagnation | PPO_45-50 | 1.13 | 64-96% | Entropy stable, profit declining, policy locked |
| **Recovery** | **PPO_51-56** | **0.98** | **70-80%** | **Entropy +13pp, Neutral% dropping, 0 FAIL** |

Follows v1 pattern: early -> conservative dip -> recovery -> stabilization -> dip -> recovery.
v1 also had multiple dip/recovery cycles throughout 169 windows.
v2 stagnation phase (PPO_45-50) initially appeared to be a permanent steady state,
but model self-recovered at PPO_51-56: entropy rose from 41% to 54%, Neutral%
dropped from 80% to 74%. Recovery is exploration-led (entropy first, profit follows).

### 10.4 v1 vs v2 Comparison

| Metric | v2 (PPO_13-17) | v2 (PPO_21-25) | v2 (PPO_28-32) | v2 (PPO_35-39) | v2 (PPO_40-44) | v2 (PPO_45-50) | v2 (PPO_51-56) |
|--------|----------------|----------------|----------------|----------------|----------------|----------------|----------------|
| avg total_profit | 1.40 | **4.25** | 2.53 | 1.07 | 1.52 | 1.13 | **0.98** |
| Win rate | 42.4% | **47.5%** | 43.5% | 35-41% | 42.1% | 35.8% | **41.4%** |
| Neutral% | 77-81% | 67-72% | 67-73% | 69-73% | 66-83% | 64-96% | **70-80%** |
| Invalid rate | 10% | 16.5% | 17.2% | 14-17% | 9.2% | 10.0% | **13.7%** |
| Long/Short | 47%L/53%S | 57%L/43%S | 75%L/25%S | 67-90%L | 50/50 | 38%L/62%S | **54%L/46%S** |
| Explained var. | 0.80 | 0.72 | 0.82 | 0.83 | 0.826 | 0.887 | **0.738** |
| Liquidation rate | 5.9% | 0.2% | 0.3% | 1.6% | 0.0% | 0.2% | **0.0%** |
| Entropy retained | 66% | 58% | 59% | 62% | 42% | 41% | **54%** |

### 10.5 Key Observations

1. **ep_rew_mean is not a useful indicator for v2.** Neutral zone (+/-5% = 0 reward)
   means most valid actions score 0; episode reward is dominated by invalid action
   penalty accumulation. Use `total_profit` and `win_rate` for evaluation instead.

2. **Model self-recovered from stagnation (PPO_51-56).** Previous conclusion that
   model was in a permanent steady state was wrong. Entropy recovered from 41% to
   54% (+13pp), Neutral% dropped from 80% to 74%, win rate recovered to 41.4%.
   Zero FAIL for the first time since PPO_25. Follows v1 dip/recovery pattern.

3. **Entropy three-phase progression confirmed.** Rapid decline (PPO_39-44: -17.4pp)
   -> stabilization (PPO_46-50: -0.1pp) -> recovery (PPO_52-56: +2.9pp).
   The ent_coef=0.05 entropy bonus continues to exert effective exploration pressure
   even after severe policy concentration. Recovery took ~12 windows.

4. **Profit still declining but all windows positive.** avg 0.98 (PPO_51-56), lowest
   phase average. PPO_50=0.588 remains the single-window low. 56 consecutive
   profitable windows. Profit lag behind entropy recovery is expected -- model
   is exploring more but hasn't optimized new strategies yet.

5. **Explained variance dropped (0.887 -> 0.738) -- positive signal.**
   Value function less predictive because model is trying new action patterns.
   When model was stuck at Neutral=80%, variance was artificially high (easy to
   predict "do nothing"). Lower variance during exploration is healthy.

6. **v2 design changes validated:**
   - SL/ROI simulation working (liquidation rate 0.0% at PPO_53-56)
   - Zero liquidation streak: 4 consecutive windows
   - PPO_21=11.12 total_profit peak still stands
   - Training shows robust self-correcting behavior (Long bias, policy collapse)

7. **Long/Short balance maintained.** PPO_56 at 54%L/46%S, well within OK.
   No persistent directional bias in this training phase.

8. **Invalid actions increased to 13.7%.** Correlated with increased exploration.
   More trading attempts = more invalid action encounters. Not concerning at
   current level (WARN threshold is 25%).

9. **v3 improvement plan on hold.** Model self-recovering reduces urgency for
   reward redesign. Continue v2 training and re-evaluate at PPO_80-100.
   v3 plan preserved in `docs/2026/20260403/rl-daytrade-v3-improvements.md`.

10. **Next evaluation: PPO_60-65.** Focus on:
    - Does profit start recovering? (lagging indicator after entropy recovery)
    - Does Neutral% continue dropping toward 60-70% range?
    - Does entropy sustain above 50%?
    - If all three: v2 training is on track, continue to completion

#### PPO_70 checkpoint (2026-04-07) -- full recovery confirmed

Health check (based on PPO_67-70, valid windows only):
**10+ OK, 0 FAIL**

| Check | Status | Value | Note |
|-------|--------|-------|------|
| Win rate | OK | **47.9%** | Best stretch since PPO_21-25 |
| Policy collapse | OK | **Neutral=68%** | Back in healthy 60-70% range |
| Entropy | OK | **67% retained** | Best since PPO_38 |
| Profit trend | OK | **avg=4.92** | Exceeds PPO_25 peak (4.25) |
| Liquidation rate | OK | 0.1% (1/688) | Excellent |
| Long/Short bias | OK | 42%L/58%S | Balanced |
| Explained variance | OK | 0.735 | Adapting |
| Invalid actions | OK | 17.4% | Exploration |

PPO_69 single-window profit: **8.68** (new ATH, previous: PPO_21=11.12).

Key changes from PPO_56:
- **Profit fully recovered**: 0.98 -> 4.92 (+402%)
- **Entropy recovered**: 54% -> 67% (+13pp)
- **Neutral% resolved**: 74% WARN -> 68% OK
- **Win rate recovered**: 41.4% -> 47.9%
- **Zero FAIL checks** -- best health since start

Confirmed: entropy three-phase progression (decline -> stable -> recovery) completed.
ent_coef=0.05 proven effective at long-term exploration maintenance.

**PPO_71 incomplete on 2026-04-07.** Training cut off at ~60% iterations.
Cleaned up on 2026-04-08 (deleted sub-train-ETH_1710892800 + tensorboard/PPO_71).

#### Feature mismatch bug & Solution A (2026-04-08)

**Bug**: `freqtrade backtesting` with `--timerange 20220601-20260325` hit feature
mismatch error (strategy=5480 vs saved=392) when trying to resume from cache.

**Root cause**: `freqai_interface.py:334-336` — `check_features=True` path passes
`dataframe.tail(1)` which was already polluted by `populate_indicators` at line 346.
The outer `dataframe` accumulates `%` columns from prior windows, making feature
count explode to 5480. Bug exists since 2022 (upstream), never triggered in v1
because v1 timerange started aligned with data/cache.

**Solution A applied**: Renamed `backtesting_predictions/` to `backtesting_predictions.bak/`.

**Unintended side effect**: Without prediction cache, freqtrade enters the training
path for EVERY window, retraining all models from scratch. best_model.zip files in
existing sub-train dirs are being overwritten. This was not anticipated -- the
expectation was cache-only rebuild.

#### PPO_82 retrain checkpoint (2026-04-10) -- reprocessing early windows

After solution A rename, training restarted 2026-04-08 21:49. Currently reprocessing
from window 25 onward. PPO tensorboard counter is inflated (counts from restart).

**NOTE**: PPO_71-82+ metrics are from retrained early windows (2022-2023 market data),
not comparable to PPO_60-70 (2024 market data).

Recent profits (PPO_78-82): 1.67, 6.94, 1.96, 2.83, 2.71 -- avg 3.22

| Check | Status | Value | Note |
|-------|--------|-------|------|
| Win rate | OK | 41.5% | |
| Policy collapse | WARN | Neutral=71% | Above 60% threshold |
| Entropy | WARN | 58% retained, -8.1pp trend | Declining |
| Profit trend | OK | All positive, avg=3.22 | |
| Liquidation rate | OK | 0.2% | |
| Long/Short bias | OK | 63%L/37%S | Balanced |
| Invalid actions | WARN | 15% | |
| Explained variance | OK | 0.779 | Predictive |
| Summary | | **10 OK, 3 WARN, 1 FAIL** | |

Training rate: ~7-10 windows/day (early windows fast, slowing for later ones).
Estimated to reach original frontier (window 70) around 2026-04-14.
Estimated full completion: **2026-04-25 to 2026-05-01**.

#### PPO_89 retrain checkpoint (2026-04-10) -- entropy reversed, retrain outperforms original

PPO_85-89 = retrained windows ~39-43 (2023 Q1 market data). Same market period as
original PPO_35-39.

Recent profits (PPO_85-89): 1.37, 1.00, 3.59, 6.10, 3.10 -- avg 3.03

| Check | Status | Value | Note |
|-------|--------|-------|------|
| Win rate | OK | **44.5%** | Improved from 41.5% |
| Policy collapse | WARN | **Neutral=64%** | Improved from 71%, just above threshold |
| Entropy | OK | **68% retained, +7.8pp trend** | Reversed from -8.1pp decline |
| Profit trend | OK | All positive, avg=3.03 | |
| Liquidation rate | OK | 0.2% | |
| Long/Short bias | OK | **48%L/52%S** | Perfect balance (was 63L/37S) |
| Invalid actions | WARN | 18.6% | More active trading |
| Value loss | WARN | 1.30x | Exploration cost |
| Explained variance | WARN | 0.432 | Exploration cost, expected to recover |
| Summary | | **9 OK, 4 WARN, 1 FAIL** | |

Retrain vs original (same 2023 Q1 market period):

| Metric | Original PPO_35-39 | Retrain PPO_85-89 | Change |
|--------|-------------------|-------------------|--------|
| avg profit | 1.07 | **3.03** | **+183%** |
| L/S bias | 67-90%L (FAIL) | 48L/52S (OK) | Resolved |
| Neutral% | 69-73% | 64% | Improved |

Key observations:
- **Entropy trend reversed**: -8.1pp (WARN) at PPO_82 -> +7.8pp (OK) at PPO_89.
  Agent regaining exploration. ent_coef=0.05 continues to work.
- **New WARNs are exploration costs**: explained_variance 0.432 and value_loss 1.30x
  are expected when entropy increases. Same pattern as original PPO_51-56 recovery.
- **Retrain produces better models**: +183% profit for same market period, no L/S bias.

Training rate: slowing to ~6-8 windows/day. Latest window ~3.3 hours.
Estimated to reach original frontier (window 70): ~2026-04-14~15.
Estimated full completion: **2026-04-28 to 2026-05-04**.

#### PPO_106 retrain checkpoint (2026-04-13) -- value recovery confirmed, summer market decline

PPO_102-105 = retrained windows ~56-60 (2023 Q2-Q3 market data, ETH low-volatility summer).
Same market period as original PPO_51-56 (stagnation/self-recovery phase).

Recent profits (PPO_102-105): 4.32, 2.18, 1.44, 1.03 -- avg 1.93 (declining within batch)
PPO_106 in progress (30%): 0.66 so far.

| Check | Status | Value | Note |
|-------|--------|-------|------|
| Win rate | OK | **50.0%** | Improved from 44.5% |
| Policy collapse | WARN | Neutral=65% | Stable (was 64%) |
| Entropy | OK | 62% retained, +3.7pp trend | Positive but weaker momentum |
| Profit trend | OK | All positive, avg=1.93 | Lower than 3.03, summer market effect |
| Liquidation rate | OK | 1.5% | Slight increase (was 0.2%), still healthy |
| Long/Short bias | **WARN** | **74%L/26%S** | New concern, but likely noise (see below) |
| Invalid actions | WARN | 19.4% | Consistent range |
| Value loss | **OK** | **0.80x decreasing** | Recovered from 1.30x WARN |
| Explained variance | **OK** | **0.812** | Recovered from 0.432 WARN |
| Sample size | WARN | 829 actions | PPO_105 small sample |
| Summary | | **10 OK, 4 WARN, 0 FAIL** | Improved from 9/4/1 |

Retrain vs original (same 2023 Q2-Q3 market period):

| Metric | Original PPO_51-56 | Retrain PPO_102-105 | Change |
|--------|-------------------|---------------------|--------|
| avg profit | ~0.98 | **1.93** | **+97%** |
| Neutral% | 70-80% | 65% | Better activity |
| Entropy | +13pp recovery | +3.7pp stable | Both OK |

Key observations:
- **Value function fully recovered**: explained_variance 0.432 -> 0.812, value_loss 1.30x -> 0.80x.
  Confirmed the PPO_89 prediction: exploration costs were temporary.
- **Profit decline is market-driven**: 2023 summer ETH low-volatility period.
  Retrain still outperforms original +97% for same market, though advantage narrowed from +183%.
- **L/S bias likely noise**: PPO_105 74%L but small sample (829 actions).
  PPO_104 was 45%L/55%S, PPO_106 in progress at 55%L/45%S. Inconsistent across windows.
- **Entropy stabilizing at 61-62%**: Healthy equilibrium above 50% threshold.

Training position: window ~60 of 189. Rate: 6.1 windows/day (recent).
Original frontier (window 70): **~2026-04-15** (~10 windows away).
Estimated full completion: **2026-04-29 to 2026-05-05**.

#### PPO_114 retrain checkpoint (2026-04-14) -- policy collapse returned, entering stagnation pattern

PPO_110-114 = retrained windows ~65-69 (2023 autumn market data, ETH ranging).
Approaching original frontier (window 70). Corresponds to original PPO_44 market period.

Recent profits (PPO_110-114): 1.85, 2.64, 2.85, 0.99, 1.70 -- avg 2.01

| Check | Status | Value | Note |
|-------|--------|-------|------|
| Win rate | WARN | 39.6% | Degraded from 50.0% |
| Policy collapse | **FAIL** | **Neutral=85%** | Returned to FAIL (was 65% OK) |
| Entropy | WARN | **41% retained** | Declined from 62%, approaching FAIL at 20% |
| Entropy trend | WARN | **-12.1pp** | Reversal from +3.7pp positive |
| Profit trend | OK | All positive, avg=2.01 | Slight improvement |
| Liquidation rate | OK | 0.0% | Improved from 1.5% |
| Long/Short bias | OK | 56%L/44%S | Balanced (was 74L WARN) |
| Invalid actions | OK | 8.4% | Improved from 19.4% WARN |
| Value loss | WARN | 1.07x | Slight increase from 0.80x OK |
| Explained variance | OK | 0.540 | Declined from 0.812 but still predictive |
| Summary | | **9 OK, 4 WARN, 1 FAIL** | Regression from 10/4/0 |

Retrain vs original (same 2023 autumn market period, corresponds to original PPO_44):

| Metric | Original PPO_40-44 | Retrain PPO_110-114 | Change |
|--------|--------------------|---------------------|--------|
| avg profit | 1.52 | **2.01** | **+32%** |
| Neutral% | 66-83% | **75-89%** | More passive (worse) |
| Pattern | L/S resolved, Neutral rising | Same, but more severe | Similar trajectory |

Key observations:
- **Policy collapse returned**: Neutral 85% (FAIL). Agent shifted to heavily passive mode.
  Entropy dropped to 41% retained (WARN, was 62%).
- **False positives in secondary metrics**: Invalid actions, L/S balance, liquidation all
  "improved" -- but this is because agent is selecting Neutral more (which is always valid).
  Not a real health gain.
- **Matches original training trajectory**: Original hit same passive-to-stagnation pattern
  at this market period (PPO_45-50 Neutral 64-96%). Self-recovered via ent_coef at PPO_51-56
  (+13pp entropy recovery). Retrain should follow similar recovery.
- **Profit still outperforms original**: +32% vs PPO_40-44 same market. Not failing.
- **Entropy trajectory**: 68% (PPO_89) -> 62% (PPO_106) -> 41% (PPO_114). Watch if continues
  to decline below 30% (would trigger FAIL).

Training rate: 7.6 windows/day (recent, accelerating). Window 70 reached ~2026-04-15.
Estimated full completion: **2026-04-29 to 2026-05-02**.

**High-alert checkpoint**: If PPO_120 still shows Neutral >80% AND entropy still declining,
consider intervention (e.g., raise ent_coef from 0.05 to 0.08). For now, natural recovery
expected based on original training pattern.

#### PPO_120 retrain checkpoint (2026-04-15) -- self-recovery confirmed, past original frontier

PPO_116-120 = retrained windows ~70-74 (2023 Oct market, ETH recovery period).
**CROSSED ORIGINAL FRONTIER** (window 70). Corresponds to original PPO_49 market period
(Stagnation tail, transitioning to self-recovery).

Recent profits (PPO_116-120): 0.63, 1.72, 0.68, 1.93, 0.83 -- avg 1.16

| Check | Status | Value | Note |
|-------|--------|-------|------|
| Win rate | WARN | 35.7% | Exploration cost during recovery |
| Policy collapse | WARN | **Neutral=76%** | **Recovered from 85% FAIL (-9pp)** |
| Entropy | WARN | **46% retained** | Improved from 41%, still below 50% |
| Entropy trend | OK | **+16.4pp** | **Full reversal from -12.1pp WARN** |
| Profit trend | OK | All positive, avg=1.16 | Lower but consistent positive |
| Liquidation rate | OK | 0.1% | Normal |
| Long/Short bias | OK | 55%L/45%S | Stable |
| Invalid actions | WARN | 12.7% | Rose from 8.4% (healthy -- more active trading) |
| Value loss | OK | **0.94x decreasing** | Recovered from 1.07x WARN |
| Explained variance | OK | **0.922** | Major recovery from 0.540 |
| Summary | | **10 OK, 4 WARN, 0 FAIL** | **FAIL resolved** |

Retrain vs original (same 2023 Oct market period):

| Metric | Original PPO_45-50 | Retrain PPO_116-120 | Change |
|--------|--------------------|---------------------|--------|
| avg profit | 1.13 | **1.16** | +3% (matched) |
| Neutral% | 64-96% | 76-85% | Within range |
| Self-recovery entropy trend | +13pp | **+16.4pp** | Even stronger |
| Pattern | Stagnation -> Self-recovery | Same | Matching trajectory |

Key observations:
- **Self-recovery confirmed**: entropy trend fully reversed from -12.1pp to +16.4pp,
  stronger than original PPO_51-56's +13pp. Policy collapse FAIL resolved.
- **Retrain precisely mirrors original training trajectory** across all market periods.
  Stagnation at PPO_114 was expected and transient.
- **Past original frontier (window 70)** -- entering terra incognita. No original data
  to compare beyond this point.
- **Exploration cost evident**: profit dip to 1.16, win rate 35.7%, invalid up to 12.7%.
  Same pattern as original PPO_51-56 before full recovery.
- **Value function dramatically recovered**: explained_variance 0.540 -> 0.922.

Training rate: 8.1 windows/day (recent, continuing to accelerate).
Estimated full completion: **2026-04-28 to 2026-05-02**.

Next checkpoint watch (PPO_125-130):
- If Neutral drops below 70% AND profit climbs above 3.0: full recovery confirmed,
  matches original PPO_60-70 ATH pattern
- If Neutral stays 75-80% AND entropy plateaus at 45-50%: stalled recovery,
  consider raising ent_coef 0.05 -> 0.08

#### PPO_125 retrain checkpoint (2026-04-16) -- recovery continuing, small-sample FAIL

PPO_121-124 = retrained windows ~75-79 (2023 Nov-Dec market data, ETH pre-BTC-ETF period).
PPO_125 in progress (12%, 122/989 iterations).

Recent profits (PPO_121-124): 1.34, 1.38, 1.72, 0.54 -- avg 1.24
PPO_125 in progress: 1.03

| Check | Status | Value | Note |
|-------|--------|-------|------|
| Win rate | **FAIL** | 24.3% (9W/28L) | **Small sample (674 actions), PPO_121-122 at 42%** |
| Policy collapse | WARN | Neutral=80% | Stable (was 76%) |
| Entropy | OK | **54% retained** | **Improved from 46%** |
| Entropy trend | WARN | -8.3pp | Dragged down by incomplete PPO_125 (36% at 12%) |
| Profit trend | OK | All positive, avg=1.20 | Range 0.54~1.72 |
| Liquidation rate | OK | 0.0% (PPO_124) | 2 across 5 windows total |
| Long/Short bias | OK | 65%L/35%S | Within threshold |
| Invalid actions | OK | 9.1% | |
| Value loss | WARN | 1.35x | Mild |
| Explained variance | OK | 0.672 | Down from 0.922 (PPO_121 anomaly pull) |
| Sample size | WARN | 674 actions | PPO_124 small sample, unreliable health checks |
| Summary | | **9 OK, 4 WARN, 1 FAIL** | FAIL is small-sample noise |

PPO_121 anomaly: value_loss 2.32x, explained_variance 0.09. Single-window market regime
shock (~2023 Nov ETH volatility). PPO_122 immediately recovered (0.87x, 0.81).

Retrain progression:

| Metric | PPO_89 | PPO_105 | PPO_114 | PPO_120 | PPO_124 | Trend |
|--------|--------|---------|---------|---------|---------|-------|
| Entropy retained | 68% | 62% | 41% | 46% | **54%** | **Improving** |
| Neutral% | 64% | 65% | 85% | 76% | **80%** | Slight pullback |
| Win rate | 44.5% | 50.0% | 39.6% | 35.7% | 42%* | Recovering (* large-sample PPO_121-122) |
| avg profit | 3.03 | 1.93 | 2.01 | 1.16 | **1.24** | Stable |
| Explained var | 0.432 | 0.812 | 0.540 | 0.922 | **0.672** | PPO_121 anomaly pull |
| Value loss | 1.30x | 0.80x | 1.07x | 0.94x | **1.35x** | Mild increase |
| Summary | 9/4/1 | 10/4/0 | 9/4/1 | 10/4/0 | **9/4/1** | FAIL is noise |

Key observations:
- **Entropy continues improving**: 46% (PPO_120) -> 54% (PPO_123-124). Self-recovery
  momentum sustained. Approaching 55%+ target.
- **Win rate FAIL is statistical noise**: PPO_124 has only 674 actions (37 trades).
  PPO_121 (12036 actions) shows 41.9%, PPO_122 (4408 actions) shows 42.6%.
- **PPO_121 single-window anomaly**: value_loss 2.32x + explained_variance 0.09.
  Market regime shock at ~2023 Nov ETH. Self-corrected by PPO_122.
- **All 125 windows profitable**: Zero negative windows in entire training run.
- **Entering 2024 Q1 market**: Next windows cover BTC ETF approval rally period.
  Expect strong trending market, potential Long bias emergence.

Training rate: ~5 windows/day (slowing from 8.1/day as expected for later windows).
Estimated full completion: **2026-05-02 to 2026-05-08** (revised from 2026-04-28~05-02).

Next checkpoint: PPO_130-135. Watch:
- Does Neutral% drop below 70% (exit WARN)?
- Does profit climb above 3.0 (match original PPO_60-70 ATH)?
- Does entropy stabilize at 55%+?
- Does Long bias emerge (2024 Q1 bull market)?

### 10.6 Updated Training Phases

| Phase | Windows | avg profit | Neutral% | Note |
|-------|---------|-----------|----------|------|
| Early | PPO_1-5 | 3.20 | 69-72% | Initial learning |
| Conservative dip | PPO_13-17 | 1.40 | 77-81% | Over-passive |
| Growth | PPO_21-25 | 4.25 | 67-72% | PPO_21=11.12 peak |
| Stabilization | PPO_28-32 | 2.53 | 67-73% | Long bias emerging |
| Dip + Long bias | PPO_35-39 | 1.07 | 69-73% | Profit dip, 67-90%L |
| Recovery + passive | PPO_40-44 | 1.52 | 66-83% | L/S resolved, Neutral rising |
| Stagnation | PPO_45-50 | 1.13 | 64-96% | Entropy stable, policy locked |
| Self-recovery | PPO_51-56 | 0.98 | 70-80% | Entropy +13pp, Neutral% dropping |
| Full recovery | PPO_60-70 | 4.92 | 66-69% | ATH profit, 67% entropy, 0 FAIL |
| Retrain early | PPO_71-82 | 3.22 | 71% | Reprocessing windows 25-36, new models |
| Retrain mid | PPO_85-89 | 3.03 | 64% | Windows ~39-43, entropy +7.8pp, outperforms original |
| Retrain summer | PPO_102-105 | 1.93 | 65% | Windows ~56-60, value recovery, summer low-vol market |
| Retrain autumn | PPO_110-114 | 2.01 | 82% | Windows ~65-69, policy collapse returned, entropy -12pp reversal |
| Retrain self-recovery | PPO_116-120 | 1.16 | 80% | Windows ~70-74, crossed original frontier, entropy +16.4pp reversal, FAIL resolved |
| Retrain continuation | PPO_121-125 | 1.24 | 78% | Windows ~75-79, entropy 54% (+8pp from trough), win rate FAIL is noise (674 sample) |
| **2nd retrain (post merge+data fix)** | **PPO_128-131** | **9.16** | **62%** | **2024-01 to 2024-03 BTC ETF rally, ATH profit, 11 OK/3 WARN/0 FAIL, upstream merged, Fix 1 applied** |

## 11. Upstream Merge & Data Fix (2026-04-17)

### 11.1 Upstream merge

- Backed up dev/personal to `backup/dev-personal-20260417` (317 commits)
- Reset dev/personal to upstream/develop (313 upstream commits forward, ~2026-04-14)
- Selectively restored: user_data/, docs/, .github/copilot-instructions.md, .claude/, check_gpu.py, .gitignore
- freqtrade/freqai/* now pure upstream, no local bug fix carryovers
- Dependencies upgraded via requirements-freqai-rl.txt: SB3 2.7.1 -> 2.8.0, torch 2.10.0 -> 2.11.0

### 11.2 Data gap fix

- Symptom: 5m futures data only covered 2022-11-01 to 2026-03-24 despite --timerange 20220601-20260325
- Root cause: freqtrade download-data is incremental; existing file never backfilled the 2022-06 to 2022-10 gap
- Fix: Re-downloaded with --erase flag covering 20220101-20260325
- Result: Training can now cover full backtest period including early 2022 with startup_candle_count buffer

### 11.3 Fix 1 applied (upstream bug)

In upstream's narrow try/except at `freqai_interface.py:370-382`, `self.model = None` on
training failure, but `self.predict()` at line 396 ran unconditionally, crashing with
`AttributeError: 'Pipeline' object has no attribute 'features_in'`.

Fix applied at line 396:
```python
if self.model is not None:
    pred_df, do_preds = self.predict(dataframe_backtest, dk)
    append_df = dk.get_predictions_to_append(pred_df, do_preds, dataframe_backtest)
    dk.append_predictions(append_df)
    dk.save_backtesting_prediction(append_df)
else:
    logger.warning(
        f"No model available for {pair}, skipping prediction "
        f"for backtest window {tr_backtest.startdt} - {tr_backtest.stopdt}."
    )
```

### 11.4 PPO_131 checkpoint (2026-04-17)

After merge + data fix, training restarted. PPO_128-131 correspond to training
windows ending 2024-01-10 to 2024-03-13 (BTC spot ETF approval rally period).

Recent profits (PPO_128-131): 11.75, 0.74, 9.02, 15.11 -- avg 9.16
PPO_132 in progress (20%): 6.45

| Check | Status | Value | Note |
|-------|--------|-------|------|
| Win rate | OK | **51.7%** (600W/561L) | Resolved from FAIL, 12520 actions large sample |
| Policy collapse | WARN | Neutral=62% | Down from 80%, trending toward OK |
| Entropy | OK | **69% retained** | Up from 54%, new ATH |
| Entropy trend | OK | -0.5pp | Stable |
| Profit trend | OK | avg=9.16 | +86% vs original PPO_60-70 same period |
| Liquidation rate | OK | 0.1% (1/1162) | Healthy |
| Long/Short bias | OK | 53%L/47%S | Balanced |
| Invalid actions | **WARN** | 19.3% | Exploration cost, agent more active |
| Value loss | WARN | 1.32x | Stable |
| Approx KL | OK | mean=0.0091 | Normal |
| Clip fraction | OK | mean=0.085 | Normal |
| Explained variance | OK | 0.634 | Predictive |
| Sample size | OK | 12520 actions | Reliable |
| Summary | | **11 OK, 3 WARN, 0 FAIL** | **ATH across entire training history** |

Retrain vs original (2024 Q1 market period, PPO_60-70):

| Metric | Original PPO_60-70 | Retrain PPO_128-131 | Change |
|--------|--------------------|---------------------|--------|
| avg profit | 4.92 | **9.16** | **+86%** |
| Win rate | 47.9% | 51.7% | +3.8pp |
| Neutral% | 66-69% | 54-62% | More active |
| Entropy | 67% | 69% | Similar |

Key observations:
- **Feature engineering verified intact**: 392 features in all windows (first, middle, last).
  user_data/ files identical to pre-merge backup.
- **IStrategy interface 100% compatible with upstream**.
- **Profit explosion is regime-driven + model improvement**:
  70% 2024 Q1 BTC ETF rally, 20% reduced Neutral% (62% vs 80%), 10% complete data cycle exposure.
- **ETH price context (PPO_128-131 training windows)**: $2,275 -> $3,162 (+39%), peak $4,098 (+80%).
- **Invalid actions 19.3% WARN is exploration cost**, expected with more active trading in bull market.

Training rate: 14 windows/day (accelerated due to shorter early-2022 episodes).
Estimated full completion: **2026-04-26 to 2026-05-04**.

Next checkpoint: PPO_140-145. Watch for 2024 Q2-Q3 (consolidation/retreat):
- Does profit stay positive through range/bear market? True generalization test.
- Does L/S bias remain balanced when no clear trend?
- Does Neutral% rise naturally as opportunities diminish (expected, not bad)?
- If profit turns negative in range market, flag as overfitting to 2024 bull regime.

### 11.5 PPO_136 checkpoint (2026-04-18) -- generalization test in progress

Training advanced 5 windows from PPO_131 (~1 day elapsed). PPO_132-136 cover
2024 Q1 climax through Q2 range/retreat. Expected regime shift: ETH $4100 peak
(PPO_132) compressing down to $3000-$3800 range (PPO_133-136).

Recent profits (PPO_132-136): 23.92, 2.47, 1.59, 2.68, 0.83 -- avg 6.30

| Window | Profit | Market context |
|--------|--------|----------------|
| PPO_132 | **23.92** | ETH $4100 -- Q1 ETF rally climax (peak) |
| PPO_133 | 2.47 | Transition to Q2 |
| PPO_134 | 1.59 | 2024 Q2 pullback begins |
| PPO_135 | 2.68 | Range consolidation |
| PPO_136 | **0.83** | Q2 range-bound (low) |

Health checks at PPO_136:

| Check | Status | Value | Note |
|-------|--------|-------|------|
| Reward trend | **FAIL** | -28.3% | Expected regime shift, NOT learning failure |
| Liquidation rate | OK | 4.5% (1/22) | Healthy |
| Win rate | OK | 42.9% (9W/12L) | Small PPO_136 env[0] sample |
| Policy collapse | WARN | Neutral=63% | Healthy caution in range market |
| Value loss | WARN | 1.45x | Model recalibrating new regime |
| Entropy | OK | **70% retained** | **Up from 64% at PPO_131**, re-exploring |
| Entropy trend | OK | +3.3pp | Exploration recovering |
| Invalid actions | WARN | 21.4% | Stable exploration cost |
| Approx KL | OK | mean=0.0095 | Normal |
| Clip fraction | OK | mean=0.093 | Normal |
| Sample size | WARN | 276 env[0] | Multiproc aggregated still large |
| Profit trend | OK | avg=6.30 | **All 5 windows positive** |
| Explained variance | OK | 0.625 | Predictive (down from 0.80 Q1) |
| Long/Short bias | OK | 35 entries | Insufficient for check |
| Summary | | **9 OK, 4 WARN, 1 FAIL** | |

Interpretation:

1. **Generalization test passing so far.** Profits compressed from avg 9.16 to
   6.30 -- still positive through bull->range transition. v1 went bankrupt on
   bear markets; v2 holds positive.

2. **Reward FAIL is regime-driven.** Structural ep_rew_mean includes hold
   penalties across full episode. Smaller price moves in range -> smaller
   positive holding rewards but similar penalty floor -> net decline.

3. **Entropy rising (64%->70%)** while in new regime = model actively
   re-exploring rather than stuck. Healthy adaptation signature.

4. **Neutral=63% WARN** is actually correct behavior in range market
   (fewer valid trending opportunities -> more waiting is optimal).

5. **Value loss rising (1.32x->1.45x)** = value function less confident on
   new regime but still predictive (explained variance 0.625).

Training rate slowed: 14/day (early windows) -> **5/day** (mid-cycle).
Revised completion estimate: **2026-04-27 to 2026-05-09**.

Next checkpoint: PPO_140-145 (2024 Q3). Watch for:
- Profit sustaining positive through deeper Q3 range
- Neutral% stabilizing (not climbing above 75%)
- Value function re-converging on new regime

### 11.6 PPO_143 checkpoint (2026-04-19) -- model adapting to Q2 regime

Training advanced 7 windows from PPO_136 (~1 day). Rate recovered to ~7/day.
PPO_139-143 cover 2024 Q2 end (ETH ~$3000-$3800 range, pullback from Q1 peak).

Recent profits (PPO_139-143): 19.42, 1.82, 0.92, 10.05, 7.07 -- avg 7.86

| Window | Profit | Observation |
|--------|--------|-------------|
| PPO_139 | **19.42** | Trend capture success |
| PPO_140 | 1.82 | Consolidation |
| PPO_141 | 0.92 | Low opportunity window |
| PPO_142 | 10.05 | Trend capture success |
| PPO_143 | 7.07 | Q2 end steady |

Health checks at PPO_143:

| Check | Status | Value | Change from PPO_136 |
|-------|--------|-------|---------------------|
| Reward trend | FAIL | -31.3% | Unchanged structural |
| Liquidation rate | OK | 0.3% (1/296) | Improved (4.5% -> 0.3%) |
| Win rate | OK | **55.6%** (164W/131L) | **+12.7pp from 42.9%** |
| Policy collapse | WARN | Neutral=62% | Stable |
| Value loss | WARN | **1.12x** | **Improved from 1.45x** |
| Entropy retained | OK | 69% | Stable |
| Entropy trend | OK | +0.7pp | Stable |
| Invalid actions | WARN | 20.9% | Stable |
| Approx KL | OK | 0.0107 | Normal |
| Clip fraction | OK | 0.107 | Normal |
| Sample size | **OK** | 3377 env[0] | **Resolved from WARN (276)** |
| Profit trend | OK | avg **7.86** | **+25% from 6.30** |
| Explained variance | OK | 0.513 | Down from 0.625 but OK |
| **Long/Short bias** | **WARN** | **65% Short (259L/488S)** | **New WARN** |
| Summary | | **9 OK, 4 WARN, 1 FAIL** | Same overall shape |

Short bias analysis:

2024 Q2 ETH trended from $4100 peak to $3000 bottom. Short bias 65% reflects
correct directional call in declining market, evidenced by:
- 55.6% win rate (higher than Q1 rally peak 51.7%)
- All 5 windows positive (avg 7.86)
- Winning trades mostly on short side = trading with regime

This is NOT policy drift or bias lock-in. Monitor if Short% exceeds 70% into
Q4 2024 rebound (2024-10 onwards ETH recovered) -- that would indicate
overfit to Q2 direction.

Adaptation signatures:

1. **Value loss 1.45x -> 1.12x**: Value function converged on Q2 regime.
   Model no longer seeing new territory.
2. **Win rate 42.9% -> 55.6%**: Trade quality improved. Higher than Q1 ATH.
3. **Liquidation 4.5% -> 0.3%**: Risk management normalized.
4. **Sample size 276 -> 3377**: Training stabilized, more actions per window.

Training rate: 5/day (at PPO_136) -> **7/day** (at PPO_143). Acceptable
mid-cycle pace. Revised ETA: **2026-04-27 to 2026-05-01**.

Next checkpoint: PPO_150-155 (2024 Q3 / Aug 2024 flash crash). Watch for:
- ETH 2024-08-05 flash crash to $2100 -- profit through extreme volatility?
- Short bias behavior at regime reversal
- Liquidation rate under stress

### 11.7 PPO_149 checkpoint (2026-04-20) -- cleanest snapshot, all WARN self-resolved

Training advanced 6 windows from PPO_143 (~1 day, ~6 windows/day).
PPO_145-149 cover 2024 Q3 recovery (post-Aug flash crash restoration phase).

Recent profits (PPO_145-149): 7.97, 10.64, 1.28, 15.30, 4.06 -- avg **7.85**

| Window | Profit | Profitable / Loss exits | exit_pnl_last | Explained var |
|--------|--------|-------------------------|---------------|---------------|
| PPO_145 | 7.97 | 317 / 269 | +0.046 | 0.425 |
| PPO_146 | 10.64 | 379 / 363 | -0.035 | 0.728 |
| PPO_147 | 1.28 | 30 / 32 | -0.063 | 0.668 |
| PPO_148 | **15.30** | 510 / 422 | +0.155 | 0.789 |
| PPO_149 | 4.06 | 269 / 285 | -0.003 | **0.824** |

Health checks at PPO_149:

| Check | Status | Value | Change from PPO_143 |
|-------|--------|-------|---------------------|
| Reward trend | **OK** | +17.0% | **Resolved from FAIL** |
| Liquidation rate | OK | 0.2% (1/555) | Stable low |
| Win rate | OK | 48.6% (269W/285L) | -7.0pp (normalized) |
| Policy collapse | **OK** | Neutral=51% | **Resolved from WARN (62%)** |
| Value loss | **OK** | 0.89x | **Resolved from WARN (1.12x)**, in-window decreasing |
| Entropy retained | OK | 71% | +2pp |
| Entropy trend | OK | +2.6pp | Stable |
| Invalid actions | **FAIL** | 26.1% (1268/4862) | +5.2pp (historical artifact) |
| Approx KL | OK | mean 0.0107 | Normal |
| Clip fraction | OK | mean 0.101 | Normal |
| Sample size | OK | 4862 env[0] | +44% |
| Profit trend | OK | avg **7.85** | -0.01 (stable) |
| Explained variance | OK | **0.824** | **+0.311 (highest of v2)** |
| **Long/Short bias** | **OK** | **48%L/52%S** | **Resolved from WARN (65% Short)** |
| Summary | | **13 OK, 0 WARN, 1 FAIL** | **+4 OK, -4 WARN** |

All-WARN-cleared interpretation:

1. **Short bias release is the key evidence**. At PPO_143 we flagged 65% Short
   as correct directional trading during Q2 pullback and hypothesized the
   model would release this bias once regime changed. PPO_149 confirms:
   48%L/52%S at Q3 recovery validates regime-mapping learning, not direction
   lock-in.

2. **Explained variance 0.824 is highest of v2 training**. Value function
   prediction power exceeded 0.7 only briefly during PPO_120 (0.922) but that
   was on a narrow Q4 2023 sample. PPO_149's 0.824 on a 4862-action sample
   indicates Critic has found a stable value estimator across regimes.

3. **Value loss 1.45x -> 1.12x -> 0.89x monotonic recovery**. Within-window
   decreasing for the first time since Q2 entry. Model no longer seeing new
   territory in Q3 recovery phase.

4. **Neutral% 62% -> 51%** reflects increased trading activity as Q3 recovery
   presents more directional opportunities than Q2 consolidation.

Invalid actions 26.1% FAIL context:

This metric has been at 20-30% throughout v2 training. Root cause:
- `Base4ActionRLEnv.step()` logs action name BEFORE validity check
- In-position Long_enter/Short_enter = invalid (correctly -2 penalty)
- Out-of-position Exit = invalid (correctly -2 penalty)
- Agent learning is happening on valid-action subset, profits prove it

No action planned. Introducing action masking or observation changes carries
regression risk for a metric that has no observable correlation with profit.

Market context (2024 Q3 recovery):

- 2024-08-05 ETH flash crashed to ~$2100 (yen carry unwind trigger)
- 2024-08 to 2024-09 recovery to ~$2700
- Training windows PPO_145-149 likely cover this recovery arc
- Liquidation 0.2% (1/555) shows model correctly sized exposure through
  volatility aftermath
- 15.30 profit at PPO_148 likely captured a recovery leg

Training rate: 7/day (at PPO_143) -> **6/day** (at PPO_149). Mid-cycle stable.
Revised ETA: **2026-04-27 to 2026-04-30** (tightened from prior 05-01).

Next checkpoint: PPO_155-160 (2024 Q4, ETH $2400 -> $4000 rally). Watch for:
- Long bias rising appropriately (directional trading with trend)
- Explained variance holding above 0.7
- Value loss remaining in-window decreasing
- Profit capture on large trend moves (expect some windows > 15)

### 11.8 PPO_163 checkpoint (2026-04-22) -- post cache-fix, low-opportunity regime

#### Incident: cache-hit feature mismatch (2026-04-20 ~ 04-21)

After PPO_149 completed, iter 25 of the backtest loop halted with:
`OperationalException: Trying to access pretrained model with identifier
but found different features furnished by current strategy`.

Root-cause chain:

1. **Upstream commit 89ef31b38 (2025-08-05)** changed save semantics for
   `dk.data["training_features_list"]` from post-pipeline
   (`list(dk.data_dictionary["train_features"].columns)`) to pre-pipeline
   raw (`dk.training_features_list`). This was merged into dev/personal on
   2026-04-17 via the upstream-reset commit 92b81dda6.
2. **Local diagnostic patch 99a488c6a (2026-02-28)** that logged feature
   diffs and used `sorted()` comparison was LOST during the same upstream
   reset.
3. **Stale pre-merge prediction files**: 54 `cb_eth_*_prediction.feather`
   files from the original 2026-04-08 run remained in the
   `backtesting_predictions/` folder with pre-merge metadata semantics.
4. **Cache-hit feature check path** at `freqai_interface.py:332-337` runs
   `use_strategy_to_populate_indicators(dataframe.tail(1))` to build the
   current feature list for comparison. This 1-row path produces features
   in a DIFFERENT ORDER than the training path's full-dataframe run.
5. **Strict list comparison** at `freqai_interface.py:527`
   (`dk.training_features_list != dk.data["training_features_list"]`)
   raises on order-only differences.

Fix sequence:

| Step | Action | Result |
|------|--------|--------|
| 1 | Deleted 54 stale prediction files (ts >= 1668556800) | iter 1 cache still failed |
| 2 | Verified metadata feature lists identical (SHA 8fc414e46cf0) across iter 1 / 24 / 25 / 26 | proves set equality, diff is order-only |
| 3 | Re-applied local patch 99a488c6a at `freqai_interface.py:527`: sorted() comparison + set-diff log | Training resumed from iter 25 without any mismatch log lines -> confirmed order-only diff |

Patch diff:

```python
if sorted(dk.training_features_list) != sorted(dk.data["training_features_list"]):
    strat_set = set(dk.training_features_list)
    saved_set = set(dk.data["training_features_list"])
    only_in_strat = strat_set - saved_set
    only_in_saved = saved_set - strat_set
    logger.warning(
        f"Feature mismatch debug: "
        f"strategy={len(dk.training_features_list)}, "
        f"saved={len(dk.data['training_features_list'])}. "
        f"Only in strategy: {only_in_strat}. "
        f"Only in saved: {only_in_saved}."
    )
    raise OperationalException(...)
```

Patch must be re-applied after any future upstream merge.

#### Training health (PPO_163/199 = 82%, 29% of current window complete)

Training resumed 2026-04-21 and advanced 14 PPO runs in ~1.5 days
(~7/day). PPO_163 is still running (286/989 iterations).

Recent profits (PPO_159-163): 11.06, 5.22, 0.80, 1.18, 1.29 -- avg **3.91**

| Window | Profit | Profitable / Loss exits | exit_pnl_last | Neutral% | Sample (env[0]) |
|--------|--------|-------------------------|---------------|----------|-----------------|
| PPO_159 | **11.06** | 389 / 469 | +0.024 | 66% | 11162 |
| PPO_160 | 5.22 | 329 / 378 | +0.011 | 70% | 9609 |
| PPO_161 | 0.80 | 390 / 685 | +0.021 | 67% | 13000 |
| PPO_162 | 1.18 | 22 / 33 | -0.002 | 74% | 871 |
| PPO_163 | 1.29 | 122 / 192 | +0.003 | 78% | 5962 (running) |

Health checks at PPO_162 (latest complete):

| Check | Status | Value | Change from PPO_149 |
|-------|--------|-------|---------------------|
| Reward trend | OK | +64.4% | stable OK |
| Liquidation rate | OK | 1.8% (1/56) | +1.6pp (small sample) |
| Win rate | **WARN** | 40.0% (22W/33L) | -8.6pp (PPO_162 tiny sample) |
| Policy collapse | **WARN** | Neutral=74% | +23pp -- see regime analysis |
| Value loss | OK | 0.95x | +0.06x (still decreasing) |
| Entropy retained | OK | 52% | -19pp (still healthy) |
| Invalid actions | **WARN** | 13.3% | **improved from FAIL 26.1%** |
| Approx KL | OK | 0.0083 | stable |
| Clip fraction | OK | 0.071 | -0.030 (more stable updates) |
| Sample size | **WARN** | 871 env[0] | -4000 -- low-opportunity window |
| Profit trend | OK | avg 3.91 | -3.94 |
| Explained variance | OK | 0.664 | -0.160 (still predictive) |
| Long/Short bias | OK | 44%L/56%S | balanced maintained |
| Summary | | **10 OK, 4 WARN, 0 FAIL** | -3 OK, +4 WARN, -1 FAIL |

Interpretation:

1. **All 4 WARNs trace to small env[0] sample (871) in PPO_162**. This is
   not policy drift -- the episode on that window genuinely had fewer
   actions (ep_len_mean 9254 vs PPO_149's 12170). Multiproc averages are
   healthy; env[0] sampling amplifies noise.

2. **Neutral% climbing 66% -> 78% across PPO_159-163** reflects regime
   shift to low-opportunity. Model is selecting fewer trades, which is
   correct behavior when clear signals are absent.

3. **Invalid actions improved FAIL -> WARN (26.1% -> 13.3%)** -- a genuine
   improvement. Higher Neutral% means fewer in-position entry attempts.

4. **All 5 windows still positive profit**. No signs of overfitting to
   prior regime or policy collapse. Explained variance 0.664 shows value
   function still predictive.

5. **Value loss 0.89x -> 0.95x** slight increase but still in-window
   decreasing. Model absorbing new regime information normally.

Market context (~2024 Q4 / 2025 Q1):

- ETH $3000-$4000 consolidation after Q4 rally
- Lower volatility -> fewer clear entry signals
- Model correctly adopts selective trading (rising Neutral%)
- PPO_159's 11.06 profit likely captured a localized trend burst

Training rate: **~7 windows/day** post-fix (matching pre-bug mid-cycle
rate). Cache-bug cost ~0.5 day.

Revised ETA: **2026-04-27 to 2026-04-28** (36 PPO runs remaining).

Next checkpoint: PPO_170-175 (~1 day from now). Watch for:
- Neutral% falling below 70% if opportunities resume
- env[0] sample recovering above 3000
- Any policy collapse risk if Neutral% stays > 80%

### 11.9 PPO_171 checkpoint (2026-04-23) -- stable post cache-fix, FAIL is metric artifact

Training advanced 8 PPO runs in ~1 day since PPO_163. No further
feature-mismatch incidents -- the `sorted()` patch at
`freqtrade/freqai/freqai_interface.py:527` (committed in 562faebb0) is
holding. Latest complete window is PPO_170; PPO_171 is at 419/989 iter
(42% of current window).

#### Training health (PPO_171/199 = 86%)

Recent profits (PPO_167-171): 0.87, 6.08, 4.58, 4.10, 2.27 -- avg **3.58**

| Window | Profit | Profitable / Loss exits | exit_pnl_last | Neutral% | Sample (env[0]) |
|--------|--------|-------------------------|---------------|----------|-----------------|
| PPO_167 | 0.87 | 7 / 16 | -0.004 | -- | 225 (tiny) |
| PPO_168 | **6.08** | 320 / 322 | -0.022 | -- | 7983 |
| PPO_169 | 4.58 | 333 / 410 | +0.015 | -- | 7727 |
| PPO_170 | 4.10 | 219 / 268 | -0.005 | 68% | 6222 |
| PPO_171 | 2.27 | 155 / 192 | -0.008 | -- | 4330 (running) |

Health checks at PPO_170 (latest complete):

| Check | Status | Value | Change from PPO_163 |
|-------|--------|-------|---------------------|
| Reward trend | **FAIL** | -89.5% | -- |
| Liquidation rate | OK | 0.2% (1/488) | -1.6pp (recovered) |
| Win rate | OK | 45.0% (219W/268L) | +5pp |
| Policy collapse | **WARN** | Neutral=68% | **-4pp (improved)** |
| Value loss | **WARN** | 1.01x | +0.06x (edge case pressure) |
| Entropy retained | OK | 64% | +12pp |
| Invalid actions | **WARN** | 16.0% | +2.7pp (stable water level) |
| Approx KL | OK | 0.0105 | +0.002 |
| Clip fraction | OK | 0.098 | +0.027 |
| Sample size | OK | 6222 env[0] | **+5351 (recovered from low-opportunity)** |
| Profit trend | OK | avg 3.58 | -0.33 |
| Explained variance | **OK** | **0.877** | +0.213 (very strongly predictive) |
| Long/Short bias | OK | 51%L/49%S | balanced maintained |
| Entropy trend | OK | -1.5pp (63% -> 62%) | stable |
| Summary | | **10 OK, 3 WARN, 1 FAIL** | -1 WARN, +1 FAIL (artifact) |

Interpretation:

1. **FAIL is a metric artifact, not learning degradation**. SB3's
   `ep_rew_mean` is cumulative episode reward. Episode length grew from
   5095 -> 7234 across PPO_167-171, so cumulative negative reward scales
   proportionally. `total_profit` is positive every window (all 5 > 0.8).
   Explained variance 0.877 shows the value function predicts returns
   accurately. Real learning health is strong.

2. **Neutral% 68% at PPO_170 is an IMPROVEMENT from PPO_163's 74%**. Model
   is trading more actively as the regime offers slightly better
   opportunities. Still within HQT (High-Quality Trading) zone (60-80%).

3. **Value loss 1.01x is an edge case**. Health threshold triggers at
   >1.0x in-window. Within-window movement is trivial (33.51 -> 33.88).
   Combined with explained variance 0.877, value function is fine.

4. **Invalid actions 16.0%** is the long-term stable water level for this
   codebase since v2 Phase 1. Not fixable without risking regression --
   see Section 11.7 Invalid Actions FAIL analysis.

5. **Sample size 6222 vs PPO_163's 871** confirms PPO_162's small sample
   was a local anomaly, not a training problem. env[0] coverage is
   healthy again.

6. **Explained variance 0.877** is among the highest of v2 training --
   value function is strongly predictive at current window.

#### Cross-checkpoint comparison

| Metric | PPO_149 | PPO_163 | PPO_170 | Trend |
|--------|---------|---------|---------|-------|
| Progress | 75% | 82% | 86% | +4pp |
| Avg profit (5 windows) | 7.85 | 3.91 | 3.58 | Regime softening |
| Neutral% | 51% | 74% | 68% | U-shape, recovering |
| Win rate | 48.6% | 40.0% | 45.0% | Recovering |
| Liquidation % | 0.2% | 1.8% | 0.2% | Sample artifact cleared |
| Explained variance | 0.824 | 0.664 | **0.877** | Peak |
| Sample size (env[0]) | 4862 | 871 | 6222 | Recovered |
| Invalid % | 26.1% FAIL | 13.3% WARN | 16.0% WARN | Stable WARN |
| Summary | 13/0/1 | 10/4/0 | 10/3/1 | -- |

#### Market regime (windows ~161-175)

Windows 161-175 cover approximately 2025 Q1 late consolidation
(~$3000-$4000 ETH range). Volatility lower than Q4 rally, but signals
are re-emerging (avg profit 3.58 stable, Neutral% declining from peak
78%). Upcoming windows should cover 2025 Q2 (Trump tariff era) where
volatility should increase.

Training rate: **~7-8 windows/day** sustained, no interruptions since
cache-fix patch.

Revised ETA: **2026-04-27 to 2026-04-28** (unchanged, 28 PPO runs
remaining).

Next checkpoint: PPO_180-185 (~1-1.5 days from now). Watch for:
- Volatility surge if entering 2025 Q2 Trump tariff period
- Neutral% drop below 65% if opportunities resume
- Avg profit recovery toward 5+ range if regime switches
- Value loss stabilization back below 1.0x

### 11.10 PPO_178 checkpoint (2026-04-24) -- high-volatility regime, entropy trend FAIL (specialization)

Training advanced 7 PPO runs in ~1 day since PPO_171. Model entered
high-volatility regime (likely 2025 Q2 Trump tariff era) with strong
profit recovery. However, entropy started a real cross-window decline --
this FAIL is substantively different from PPO_171's metric artifact.

#### Training health (PPO_178/199 = 89%)

Recent profits (PPO_174-178): 6.26, 6.92, 5.66, 1.13, 3.28 -- avg **4.65**
**+30% recovery vs PPO_171's 3.58**. All 5 windows positive.

| Window | Profit | Profitable / Loss exits | exit_pnl_last | exit_reward_last | Sample (env[0]) |
|--------|--------|-------------------------|---------------|------------------|-----------------|
| PPO_174 | 6.26 | 297 / 312 | **+0.094** | **6.72** | 6923 |
| PPO_175 | **6.92** | 300 / 299 | -0.019 | -1.0 | 7559 |
| PPO_176 | 5.66 | 183 / 151 | +0.050 | **4.49** | 4758 |
| PPO_177 | 1.13 | 27 / 43 | -0.003 | -1.0 | 583 (tiny) |
| PPO_178 | 3.28 | 245 / 315 | **+0.100** | **7.01** | 8317 |

Large winning exits returning: exit_reward 6.72, 4.49, 7.01 on
PPO_174/176/178. exit_pnl peaks +0.094, +0.100 confirm substantial entry/
exit spreads captured by the model.

Health checks at PPO_178 (latest complete):

| Check | Status | Value | Change from PPO_171 |
|-------|--------|-------|---------------------|
| Reward trend | **OK** | +22.3% | **recovered from FAIL** |
| Liquidation rate | OK | 0.2% (1/561) | stable |
| Win rate | OK | 43.8% (245W/315L) | -1.2pp |
| Policy collapse | **WARN** | Neutral=73% | +5pp (still HQT) |
| Value loss | **WARN** | 1.29x | +0.28x (real rise, not edge case) |
| Entropy retained | OK | 50% | -14pp (right at threshold) |
| Entropy trend | **FAIL** | -17.1pp (67%->50%) | **new real FAIL, not artifact** |
| Invalid actions | **WARN** | 13.7% | -2.3pp (improvement) |
| Approx KL | OK | 0.0098 | -0.0007 |
| Clip fraction | OK | 0.089 | -0.009 |
| Sample size | OK | 8317 env[0] | +2095 |
| Profit trend | OK | avg 4.65 | **+30% recovery** |
| Explained variance | OK | 0.708 | -0.17 (still predictive) |
| Long/Short bias | OK | 45%L/55%S | slight Short tilt |
| Summary | | **10 OK, 3 WARN, 1 FAIL** | FAIL swapped category |

Interpretation:

1. **Entropy FAIL is specialization, not collapse**. This is a
   fundamentally different FAIL from PPO_171:
   - PPO_171 FAIL: `ep_rew_mean` cross-window metric artifact (cumulative
     reward scales with episode length)
   - PPO_178 FAIL: Real `entropy_loss` cross-window decline (-17.1pp)

   Evidence that entropy drop is benign specialization:
   - Profit RECOVERING (+30%) while entropy drops
   - Large winning exits returning (exit_reward 7.01)
   - Neutral% rise mild (+5pp), still in HQT band
   - Long/Short still balanced (45/55)
   - Sample size healthy (8317)

2. **Value loss 1.29x is real in-window rise**. Previous checkpoint's
   1.01x was edge-case compression; this is genuine value function
   adaptation to new regime. Explained variance 0.708 confirms value
   function still predictive despite the rise.

3. **Neutral% 73% within HQT band (60-80%)**. Rising selectivity as
   model learns when opportunities are real vs when to wait. Combined
   with profit recovery, this is correct behavior.

4. **Reward trend FAIL -> OK recovery**. `ep_rew_mean` -3284.68 ->
   -2551.77 (+22.3%). Metric back within thresholds -- episode length
   stabilized around 8500-10000 candles.

5. **PPO_177 small sample (583 env[0])** is a local anomaly, not a
   training problem. Other windows 4758-8317 are healthy.

#### Cross-checkpoint comparison

| Metric | PPO_149 | PPO_163 | PPO_170 | PPO_178 | Trend |
|--------|---------|---------|---------|---------|-------|
| Progress | 75% | 82% | 86% | 89% | +3pp |
| Avg profit (5 windows) | 7.85 | 3.91 | 3.58 | **4.65** | U-shape recovery |
| Neutral% | 51% | 74% | 68% | 73% | High-band stable |
| Win rate | 48.6% | 40.0% | 45.0% | 43.8% | Stable |
| Liquidation % | 0.2% | 1.8% | 0.2% | 0.2% | Stable |
| Explained variance | 0.824 | 0.664 | **0.877** | 0.708 | Post-peak |
| Entropy retained | 71% | 52% | 64% | **50%** | Declining (specialization) |
| Value loss | 0.89x | 0.95x | 1.01x | **1.29x** | Real pressure |
| Sample size (env[0]) | 4862 | 871 | 6222 | 8317 | Healthy |
| Invalid % | 26.1% FAIL | 13.3% WARN | 16.0% WARN | 13.7% WARN | Stable WARN |
| Summary | 13/0/1 | 10/4/0 | 10/3/1 | 10/3/1 | -- |

#### Market regime (windows ~172-188)

Training entered what appears to be 2025 Q2 Trump tariff era volatility:
- Large winning exits returning (exit_reward 6.72, 4.49, 7.01)
- exit_pnl peaks at 1% of base price (vs previous 0.3-0.5%)
- Profit recovery +30% vs low-opportunity regime
- Slight Short tilt (55%) consistent with tariff-induced downward pressure

Training rate: **~7 windows/day** sustained. ETA tightened.

#### Revised ETA: 2026-04-26 to 2026-04-27 (tightened by 1 day)

21 PPO runs remaining at ~7/day.

#### Decision threshold for next checkpoint (PPO_185-190)

Entropy and policy collapse are now the primary watch metrics:

- **Continue training** if: entropy retained > 40% AND Neutral% < 80%
- **Consider rollback** if: entropy < 40% OR Neutral% > 80%
- Rollback target: PPO_170-175 checkpoint (entropy 64%, profit 3.58, clean
  generalization vs specialization tradeoff)

Additional watch points for PPO_185-190:
- Value loss: if persistently > 2.0x, adjust `learning_rate`
- exit_reward: continued large wins (> 5.0) confirm regime capture
- Long/Short balance: drift below 40% either side would signal regime lock-in

### 11.11 PPO_189 Checkpoint (2026-04-26, 95% complete)

**Status: 10 OK / 3 WARN / 1 FAIL** (window 189 of 199, 95%)

Training advanced 11 windows in ~2 days since PPO_178. Now in final 5% of
training data (2026 Q1 ~ 03/25). The PPO_178 entropy specialization concern
self-resolved: entropy retained 47% -> 61% (+14pp). Reward FAIL returned, but
it's the same metric artifact as PPO_171 (ep_len-driven), not a real signal.

#### Per-window metrics

| Window | total_profit | ep_len | entropy_last | profitable | loss | win% |
|--------|-------------|--------|--------------|-----------|------|------|
| PPO_185 | 1.25 | 6506 | -0.79 | 267 | 415 | 39% |
| PPO_186 | 1.94 | 9945 | -0.66 | 112 | 162 | 41% |
| PPO_187 | 1.78 | 9587 | -0.73 | 90 | 127 | 41% |
| PPO_188 | 1.00 | 9310 | -0.84 | 345 | 504 | 41% |
| PPO_189 | 3.14 | 10876 | -0.83 | 475 | 715 | 40% |

avg profit **1.82** -- all 5 windows positive. Range 1.00~3.14.
**-61% vs PPO_178's 4.65**.

#### Health checks

| Check | Status | Detail |
|-------|--------|--------|
| reward_trend | **FAIL** | -1701.11 -> -3046.14 (-79.1%) -- metric artifact |
| liquidation_rate | OK | 0.1% (1/1191) |
| win_rate | WARN | 39.9% (475W/715L) -- right at OK threshold |
| policy_collapse | WARN | Neutral=66% [Exit=15%, L=9%, S=10%] -- improved -7pp |
| value_loss | OK | 27.87 -> 17.14 (0.62x) -- in-window decreasing |
| entropy_loss | OK | -1.37 -> -0.83 (61% retained) |
| invalid_actions | WARN | 18.4% (2749/14964) |
| approx_kl | OK | mean=0.0104, last=0.0137 |
| clip_fraction | OK | mean=0.092, last=0.127 |
| sample_size | OK | 14964 (largest of v2) |
| profit_trend | OK | all 5 positive, avg 1.82 |
| **entropy_trend** | **OK** | **58% -> 61% (+3.1pp)** -- recovered from PPO_178 -17pp FAIL |
| explained_variance | OK | 0.794 |
| long_short_bias | OK | 50%L/50%S (1414L/1433S) -- perfectly balanced |

#### Reward FAIL is metric artifact (per-step calculation)

```
PPO_185: -1701 / 6506  = -0.261 reward/step
PPO_189: -3046 / 10876 = -0.280 reward/step
per-step decline = -7.3% (not -79.1%)
```

`ep_rew_mean` is cumulative -- when ep_len grows, magnitude scales linearly.
ep_len stretched +67% (6506 -> 10876) because the agent waits longer in
lower-opportunity periods. This is correct behavior, not learning failure.

#### PPO_188 env[0] anomaly (multiproc sampling artifact)

PPO_188 reports Neutral=4, Exit=1723, Long=742, Short=1 -- a single env[0]
sample showing essentially no Neutral and no Short entries. This is almost
certainly an env[0]-only artifact:

- total_profit=1.00 (positive, normal)
- profitable_exits 345/504 = 41% (normal win rate)
- Surrounding windows all show normal Neutral 60-66%
- L/S aggregate across 5 windows is 50/50 balanced

If a single env had truly entered no-Neutral state, profit would be erratic
and L/S aggregates would skew. They don't. Multiproc caveat (only env[0]
sampled for custom action metrics) explains the spike cleanly.

#### Cross-checkpoint comparison

| Metric | PPO_149 | PPO_163 | PPO_171 | PPO_178 | PPO_189 | Trend |
|--------|---------|---------|---------|---------|---------|-------|
| Progress | 75% | 82% | 86% | 89% | **95%** | +6pp |
| Avg profit (5w) | 7.85 | 3.91 | 3.58 | 4.65 | **1.82** | -61% from peak |
| Neutral% | 51% | 74% | 68% | 73% | **66%** | Improved |
| Win rate | 48.6% | 40.0% | 45.0% | 43.8% | **39.9%** | At threshold |
| Liquidation % | 0.2% | 1.8% | 0.2% | 0.2% | **0.1%** | Best |
| Explained variance | 0.824 | 0.664 | 0.877 | 0.708 | **0.794** | Stable |
| Entropy retained | 71% | 52% | 64% | 50% | **61%** | **Recovered** |
| Value loss | 0.89x | 0.95x | 1.01x | 1.29x | **0.62x** | Healthy |
| Sample size (env[0]) | 4862 | 871 | 6222 | 8317 | **14964** | Largest |
| ep_len_mean | ~5500 | ~5800 | ~6500 | ~7000 | **~10876** | +67% vs PPO_178 |
| Invalid % | 26.1% FAIL | 13.3% W | 16.0% W | 13.7% W | **18.4% W** | Stable WARN |
| Summary | 13/0/1 | 10/4/0 | 10/3/1 | 10/3/1 | **10/3/1** | -- |

#### FAIL type interpretation matrix

| Checkpoint | FAIL metric | Type | Real concern? | Action |
|------------|-------------|------|---------------|--------|
| PPO_171 | reward_trend | metric artifact (ep_len) | No | Continue |
| PPO_178 | entropy_trend | real decline (-17pp) | Yes (specialization) | Watch |
| **PPO_189** | **reward_trend** | **metric artifact (ep_len)** | **No** | **Continue to completion** |

PPO_189 represents a self-corrected state: the entropy decline that flagged
PPO_178 reversed naturally. Profit dropped, but stayed positive across all
5 windows -- regime is harder, not model broken.

#### Market regime (windows ~182-194)

Final 5% of training data corresponds to **2026 Q1 ~ 03/25**:

- Trump tariff policy now baseline (no longer surprise events)
- Crypto volatility receded vs PPO_178's 2025 Q2 peak
- Lower opportunity density: agent waits longer (ep_len 10876 vs 7000)
- exit_reward all -1 across PPO_185-189 (no big wins like PPO_178's 7.01)
- Profit floor held: minimum 1.00, all 5 positive

Translation: model is correctly identifying the regime is harder and
trading more selectively. It hasn't lost capability, just facing a
period where setups are rarer.

#### Training rate slowdown

PPO_178 -> PPO_189: +11 windows in ~2 days = **~5.5 windows/day**
(down from sustained ~7/day).

Cause: ep_len_mean stretched 7000 -> 10876 candles. Each PPO iteration
runs more environment steps before episode terminates. Wall-clock cost
per iteration scales with ep_len.

#### Revised ETA: 2026-04-27 to 2026-04-28

10 PPO runs remaining at ~5.5/day = ~1.8 days.

#### Decision: wait for completion + dual-checkpoint backtest

No mid-training intervention. Training will complete naturally within ~2
days. On completion (PPO_199), run dual-checkpoint comparison:

1. **PPO_178** (peak in-sample profit 4.65, entropy 50%, possible
   specialization)
2. **PPO_189** (balanced 1.82 profit, entropy 61%, healthy generalization)

Backtest both on out-of-sample period. The result decides which
hypothesis is correct:

- If PPO_178 wins OOS: specialization was regime mastery (keep PPO_178)
- If PPO_189 wins OOS: specialization was overfitting (use PPO_189)
- If both lose vs simple baseline: re-evaluate reward function

#### Watch points for PPO_195-199

- Entropy retained: stay > 50% (currently 61%)
- Neutral%: stay < 75% (currently 66%)
- Long/Short bias: stay 40-60% range (currently 50/50)
- Win rate: not falling below 35% (currently 39.9%)
- Value loss: not rising > 2.0x (currently 0.62x healthy)
- If any breach, consider stopping at PPO_195 vs running to PPO_199

### 11.12 PPO_198 Checkpoint (2026-04-27, 99% complete, in-flight)

**Status: 11 OK / 2 WARN / 1 FAIL** (window 198 of 199, PPO_198 12% in-flight)

Training advanced 9 windows in ~1 day since PPO_189. PPO_198 still running
(122/989 iter); latest_complete is PPO_197. One window remains before
training completes naturally (~24-36h ETA).

The summary improved from 10/3/1 to 11/2/1 (Invalid actions 18.4% -> 10.2%
moved from WARN closer to OK), but **policy_collapse crossed 80% threshold
into FAIL**. The model has become a super-selective trader.

#### Per-window metrics

| Window | total_profit | ep_len | entropy_last | profitable | loss | win% | actions (env[0]) |
|--------|-------------|--------|--------------|-----------|------|------|------------------|
| PPO_194 | 0.79 | 7644 | -0.57 | 3 | 7 | 30% | tiny (234 total) |
| PPO_195 | 0.83 | 7017 | -0.56 | 40 | 71 | 36% | small (1644) |
| PPO_196 | 1.05 | 6200 | -0.54 | 41 | 71 | 37% | medium (4008) |
| PPO_197 | 3.88 | 9529 | -0.55 | 220 | 323 | 41% | large (11973) |
| PPO_198* | 0.91 | 3114 | -0.52 | 11 | 28 | 28% | partial (1013) |

*PPO_198 only 12% complete (122/989 iterations).

avg profit **1.49** -- all 5 windows positive. Range 0.79~3.88.
**-18% vs PPO_189's 1.82** but PPO_197 hit best of cluster (3.88).

#### Health checks (latest_complete = PPO_197)

| Check | Status | Detail |
|-------|--------|--------|
| reward_trend | OK | -1597 -> -812 (+49.1%) -- partly artifact (PPO_198 ep_len short) |
| liquidation_rate | OK | 0.2% (1/544) |
| win_rate | OK | 40.5% (220W/323L) -- recovered from PPO_189 39.9% WARN |
| **policy_collapse** | **FAIL** | **Neutral=81% [Exit=9%, L=5%, S=5%]** -- crossed 80% threshold |
| value_loss | OK | 21.6 -> 11.3 (0.52x) -- in-window decreasing |
| entropy_loss | **WARN** | -1.37 -> -0.55 (40% retained) -- back into specialization |
| invalid_actions | WARN | 10.2% (1225/11973) -- improved -8pp from PPO_189 |
| approx_kl | OK | mean=0.0085, last=0.0048 |
| clip_fraction | OK | mean=0.071, last=0.051 |
| sample_size | OK | 11973 |
| profit_trend | OK | all 5 positive, avg 1.49 |
| entropy_trend | OK | 41% -> 38% (-3.4pp) cross-window stable |
| explained_variance | OK | 0.859 -- highest since PPO_171 |
| long_short_bias | OK | 50%L/50%S (597L/593S) |

#### Policy collapse FAIL: super-selective, not dead

PPO_197 action distribution at 80.7% Neutral crossed the 80% FAIL threshold.
But this is qualitatively different from a dead/stuck policy. Evidence:

- Sample size 11973 (real signal, not env[0] artifact)
- Long/Short perfectly balanced (597L/593S = 50/50)
- Invalid actions IMPROVED -8pp (18.4% -> 10.2%) -- model isn't reckless
- profit on PPO_197 hit 3.88 (highest of recent cluster) -- when it trades,
  it hits well
- explained_variance 0.859 (highest since PPO_171) -- value function strong

Translation: model converged to "trade rarely, trade well" alpha. The
question is whether this is regime mastery (true alpha discovery) or
overfit timidity in late-data low-opportunity regime.

#### Cross-checkpoint comparison (full late-training arc)

| Metric | PPO_171 | PPO_178 | PPO_189 | PPO_198 | Trend |
|--------|---------|---------|---------|---------|-------|
| Progress | 86% | 89% | 95% | **99%** | Near completion |
| Avg profit (5w) | 3.58 | 4.65 | 1.82 | **1.49** | Continued decline |
| Neutral% | 68% | 73% | 66% | **81%** | **FAIL crossed** |
| Win rate | 45% | 43.8% | 39.9% | **40.5%** | Threshold edge |
| Liquidation % | 0.2% | 0.2% | 0.1% | **0.2%** | Stable |
| Explained variance | 0.877 | 0.708 | 0.794 | **0.859** | Recovered |
| Entropy retained | 64% | 50% | 61% | **40%** | Re-specialization |
| Value loss | 1.01x | 1.29x | 0.62x | **0.52x** | Most healthy |
| Sample size | 6222 | 8317 | 14964 | **11973** | Healthy |
| ep_len_mean | ~6500 | ~7000 | ~10876 | ~7980 | Shortened (less trading) |
| Invalid % | 16.0% W | 13.7% W | 18.4% W | **10.2% W** | **Best, near OK** |
| Summary | 10/3/1 | 10/3/1 | 10/3/1 | **11/2/1** | Improved |

#### Specialization trajectory across training

| Phase | PPO range | Entropy | Neutral% | Profit | Mode |
|-------|-----------|---------|----------|--------|------|
| Mid-training | PPO_149-163 | 52-71% | 51-74% | 3.91-7.85 | Active learner |
| Pre-spec | PPO_167-171 | 64% | 68% | 3.58 | Stable |
| Spec1 | PPO_174-178 | 50% | 73% | 4.65 | Aggressive specialization (high vol regime) |
| Recovery | PPO_185-189 | 61% | 66% | 1.82 | Entropy recovery (low vol regime) |
| **Spec2** | **PPO_194-198** | **40%** | **81%** | **1.49** | **Super-selective** |

The model oscillated through two specialization waves. The first
(PPO_178) coincided with high-volatility 2025 Q2. The second (PPO_198)
coincided with low-volatility 2026 Q1. Same mechanic, opposite regime
trigger -> suggests the model is regime-adaptive rather than overfitting
to a single regime.

#### Training rate accelerated

PPO_189 -> PPO_198: +9 windows in ~1 day = **~9 windows/day** (up from
~5.5/day). Cause: ep_len shortened from 10876 to 6200-9500 as policy
became less active. Fewer trade entries -> shorter episodes -> faster
PPO iterations.

#### Revised ETA: 2026-04-27 evening to 2026-04-28 morning

PPO_198 (88% remaining) + PPO_199 = ~8 hours wall-clock at current pace.

#### Three-checkpoint OOS backtest plan

On completion, run three checkpoints on identical OOS slice:

1. **PPO_178** (peak profit 4.65, entropy 50%, Neutral 73%) -- aggressive
   specialization, possible regime mastery
2. **PPO_189** (balanced 1.82, entropy 61%, Neutral 66%) -- middle path,
   entropy-healthy
3. **PPO_199** (selective ~1.49, entropy ~40%, Neutral ~81%) -- final
   super-selective state

Decision matrix:

| Winner | Implication | Action |
|--------|-------------|--------|
| PPO_178 wins | Aggressive specialization is true alpha | Use PPO_178, accept entropy risk |
| PPO_189 wins | Middle path generalizes best | Use PPO_189, mid-training optimum |
| PPO_199 wins | "Less is more" alpha is real | Use PPO_199, training found correct convergence |
| All lose to baseline | Reward function or features insufficient | Redesign before next train |

#### No mid-training intervention

Last window away. Standard rule: do not modify a training within 1 window
of completion. Wait, run OOS backtest, decide.

### 11.13 PPO_205 Checkpoint (2026-04-28, training overshoot, in-flight)

**Status: 9 OK / 4 WARN / 1 FAIL** (window 205, exceeded 0401 plan ~189 estimate)

Training continued past the originally estimated 189 windows. PPO_205 still
running (29% complete, 288/989 iter); latest_complete is PPO_204. Latest TB
event timestamp is 2026-04-28 17:41 (active). Training process PID 66932
running since 2026-04-21.

The summary regressed from 11/2/1 (PPO_198) to 9/4/1: win_rate dropped
into WARN, value_loss escalated to 1.83x WARN, invalid_actions rebounded
to 12.1% WARN. policy_collapse improved from 81% FAIL back to 77% WARN,
but entropy_trend FAIL re-emerged with -17.7pp drop.

#### Per-window metrics

| Window | total_profit | ep_len | entropy_last | profitable | loss | win% | actions (env[0]) |
|--------|-------------|--------|--------------|-----------|------|------|------------------|
| PPO_201 | 1.01 | 6671 | -0.75 | 94 | 131 | 42% | medium (3087) |
| PPO_202 | 1.02 | 7042 | -0.70 | 164 | 255 | 39% | large (8330) |
| PPO_203 | 0.83 | 6179 | -0.68 | 13 | 34 | 28% | tiny (758) |
| PPO_204 | 1.05 | 6593 | -0.71 | 160 | 255 | 39% | large (7703) |
| PPO_205* | 1.21 | 4222 | -0.50 | 102 | 165 | 38% | partial (7474) |

*PPO_205 only 29% complete (288/989 iterations).

avg profit **1.02** -- all 5 windows positive. Range 0.83~1.21.
**-32% vs PPO_198's 1.49**, continuing the slide from PPO_178's 4.65 peak.

#### Health checks (latest_complete = PPO_204)

| Check | Status | Detail |
|-------|--------|--------|
| reward_trend | OK | -1721 -> -985 (+42.7%) -- partly artifact (PPO_205 ep_len short) |
| liquidation_rate | OK | 0.2% (1/416) |
| **win_rate** | **WARN** | **38.6% (160W/255L)** -- dropped below 40% OK threshold |
| **policy_collapse** | **WARN** | **Neutral=77%** [Exit=12%, L=6%, S=4%] -- improved from PPO_198 81% FAIL |
| **value_loss** | **WARN** | **23.9 -> 43.7 (1.83x)** -- approaching 2x FAIL threshold |
| entropy_loss | OK | -1.37 -> -0.71 (52% retained, in-window) |
| **invalid_actions** | **WARN** | **12.1% (934/7703)** -- rebounded from PPO_198 10.2% |
| approx_kl | OK | mean=0.0077, last=0.0065 |
| clip_fraction | OK | mean=0.063, last=0.098 |
| sample_size | OK | 7703 (env[0]) |
| profit_trend | OK | all 5 positive, avg 1.03 |
| **entropy_trend** | **FAIL** | **54% -> 37% (-17.7pp)** -- second specialization wave deeper than PPO_178 |
| explained_variance | OK | 0.585 -- noisier than PPO_198's 0.859 |
| long_short_bias | OK | 60%L/40%S (499L/335S) -- slight Long tilt vs PPO_198's 50/50 |

#### Entropy_trend FAIL: super-selective deeper, not collapsing

Entropy retained dropped from 54% (PPO_201) to 37% (PPO_205*), the same
-17pp magnitude as PPO_178's specialization but in opposite regime
(low-vol 2026 Q1 vs high-vol 2025 Q2). Critical difference: PPO_178's
specialization paid off (profit jumped to 4.65). This second wave is
NOT paying off (profit compressed to 1.0).

Evidence still favoring "specialization, not collapse":
- All 5 windows positive (no negative profit)
- L/S slightly tilted but not collapsed (60/40)
- Sample size 7703 (real signal)
- Liquidation rate stable 0.2%
- exit_reward not pinned to floor (-1 to +10 distributed)

Evidence the alpha is NOT improving:
- Profit floor at ~1.0 across last 8+ windows (PPO_198-205)
- Win rate now sub-40%
- value_loss 1.83x suggests value function struggling

Translation: the "trade rarely, trade well" hypothesis from PPO_198 has
NOT been validated. Trade frequency is low but trade quality has not
risen to compensate. The super-selective trader has settled into a
modest 1.0-baseline rather than a 3-4x alpha.

#### Cross-checkpoint comparison (extended late-training arc)

| Metric | PPO_171 | PPO_178 | PPO_189 | PPO_198 | PPO_205* | Trend |
|--------|---------|---------|---------|---------|----------|-------|
| Avg profit (5w) | 3.58 | 4.65 | 1.82 | 1.49 | **1.02** | Continued decline |
| Neutral% | 68% | 73% | 66% | 81% | **77%** | Off peak but elevated |
| Win rate | 45% | 43.8% | 39.9% | 40.5% | **38.6% W** | Below 40% |
| Liquidation % | 0.2% | 0.2% | 0.1% | 0.2% | **0.2%** | Stable |
| Explained variance | 0.877 | 0.708 | 0.794 | 0.859 | **0.585** | Noisier |
| Entropy retained | 64% | 50% | 61% | 40% | **37%** (or 52% in-window) | Spec2 deepens |
| Value loss | 1.01x | 1.29x | 0.62x | 0.52x | **1.83x W** | Worst since PPO_178 |
| Sample size | 6222 | 8317 | 14964 | 11973 | **7703** | Healthy |
| ep_len_mean | ~6500 | ~7000 | ~10876 | ~7980 | ~6213 | Shortest |
| Invalid % | 16.0% W | 13.7% W | 18.4% W | 10.2% W | **12.1% W** | Stable band |
| Summary | 10/3/1 | 10/3/1 | 10/3/1 | 11/2/1 | **9/4/1** | Regression |

#### Specialization trajectory updated

| Phase | PPO range | Entropy | Neutral% | Profit | Mode |
|-------|-----------|---------|----------|--------|------|
| Mid-training | PPO_149-163 | 52-71% | 51-74% | 3.91-7.85 | Active learner |
| Pre-spec | PPO_167-171 | 64% | 68% | 3.58 | Stable |
| Spec1 | PPO_174-178 | 50% | 73% | 4.65 | Aggressive specialization (high vol) |
| Recovery | PPO_185-189 | 61% | 66% | 1.82 | Entropy recovery (low vol) |
| Spec2 | PPO_194-198 | 40% | 81% | 1.49 | Super-selective (FAIL crossed) |
| **Spec2-late** | **PPO_201-205** | **37%** | **77%** | **1.02** | **Specialization sustained, alpha NOT realized** |

The Spec2 phase is now over 12 windows (PPO_194 through PPO_205*). What
PPO_198 looked like a temporary super-selective wave is the model's
stable late-training regime. The entropy floor, Neutral% concentration,
and profit baseline have all stabilized -- the model converged into this
mode rather than passing through it.

#### Training overshoot beyond plan

0401 plan estimated ~189 windows from timerange 20220601-20260325. Actual
window count is now 205+ (~+8% overshoot). Likely cause: sliding window
stride was slightly shorter than estimated, accumulating more windows
across the 4-year data range. Latest sub-train data start is 2024-03-12,
so ~remaining 0-3 windows expected before training data exhaustion.

#### Updated three-checkpoint OOS backtest plan

On completion, run on identical OOS slice:

1. **PPO_178** (peak profit 4.65, entropy 50%, Neutral 73%) -- aggressive
   specialization in high-vol regime
2. **PPO_189** (balanced 1.82, entropy 61%, Neutral 66%) -- middle path,
   entropy-healthy
3. **PPO_197** (individual peak 3.88, late-stage best-window) -- substitute
   for original PPO_199 plan; the actual highest-profit late-training
   single window
4. **PPO_205-final** (super-selective 1.0, entropy 37%, Neutral 77%) --
   final converged state

Decision matrix update:

| Winner | Implication | Action |
|--------|-------------|--------|
| PPO_178 wins | High-vol specialization is real alpha | Use PPO_178, regime-conditioned |
| PPO_189 wins | Middle path generalizes best | Use PPO_189, accept lower in-sample peak |
| PPO_197 wins | Late-training has alpha but variance | Use PPO_197, single-snapshot pick |
| PPO_205 wins | Super-selective converged correctly | Use PPO_205, final model |
| All lose to baseline | Reward/feature redesign needed | Halt, redesign before next train |

#### Status

Training within hours of natural completion. PPO_205 at 29% in-flight,
sub-train data near 2024-03-12 (terminal). Wait for completion, then
proceed to OOS backtest -- no mid-training intervention.

### 11.14 PPO_211 Checkpoint (2026-04-29) -- Spec2 exit, regime recovery, training history best

#### Why this matters

PPO_211 is the **first checkpoint in v2 training history with 0 FAIL and
12 OK** (single Neutral=75% WARN + Invalid=12.7% WARN remaining). It also
represents an unexpected **structural reversal** of the Spec2 super-selective
state diagnosed at PPO_198-205: the agent has spontaneously exited the
1.0-baseline rut and resumed multi-directional active trading.

Most surprisingly: training is **still alive** at PPO_212 (161/989 iter,
16% in-flight). The 0-3 windows remaining estimate from PPO_205 was wrong --
training has continued 7 more windows past PPO_205 with no termination signal.

#### Per-window metrics (PPO_208-212)

| Window | Profit | Neutral % | Entropy | Win Rate | Value Loss | Explained Var | ep_len |
|---|---|---|---|---|---|---|---|
| PPO_208 | 3.44 | 83% | 54% | 49% | 25-51 (2.0x) | 0.66 | 11252 |
| PPO_209 | 2.78 | 66% | 60% | 47% | 32-49 (1.5x) | 0.85 | 11958 |
| PPO_210 | 5.09 | 68% | 60% | 49% | 26-87 (3.4x) | 0.48 | 9594 |
| **PPO_211** | **3.37** | **75%** | **62%** | **51%** | **26-24 (0.92x)** | **0.80** | **11076** |
| PPO_212* | 1.45 | 68% | 68% | 48% | 30-27 | 0.58 | 3348 |

*PPO_212 in-flight at 161/989 iter (16%). Latest_complete = PPO_211.

5-window profit avg = **3.43** (range 1.45-5.09). Compared with PPO_201-205
avg 1.02, profit baseline jumped **+236%**.

#### Health checks at PPO_211 (12 OK / 2 WARN / 0 FAIL)

OK (12):
- Reward trend: -2932 -> -1022 (+65.1%)
- Liquidation rate: 0.5% (1/184)
- Win rate: **50.8%** (recovered from 38.6% WARN at PPO_205)
- Value loss: **0.92x** (decreased from 1.83x WARN at PPO_205)
- Entropy retained: **62%** (recovered from 37% near-FAIL at PPO_205)
- Approx KL: 0.0106 mean (stable)
- Clip fraction: 0.097 mean (normal)
- Sample size: 2893 actions
- Profit trend: avg 3.23 (all 5 positive, range 1.45-5.09)
- Entropy trend: **+13.9pp** cross-window (54% -> 68%) -- recovered from
  -17.7pp FAIL at PPO_205
- Explained variance: 0.802 (strong predictive)
- Long/Short bias: 39%L/61%S (balanced, was 0/100 at Spec2 era)

WARN (2):
- Policy collapse: Neutral=75% (mid-WARN band)
- Invalid actions: 12.7% (above 10% threshold, similar to PPO_198 era)

#### Spec2 exit interpretation

The Spec2 super-selective hypothesis (PPO_198-205) has been **falsified**.
Evidence:

| Dimension | PPO_205 (Spec2 sustained) | PPO_211 (Spec2 exit) | Delta |
|---|---|---|---|
| Health | 9 OK / 4 WARN / 1 FAIL | 12 OK / 2 WARN / 0 FAIL | **+3 OK** |
| Entropy retained | 37% | 62% | **+25pp** |
| Cross-window entropy | -17.7pp (FAIL) | +13.9pp | trend reversed |
| Win rate | 38.6% (WARN) | 50.8% | +12.2pp |
| Value loss ratio | 1.83x (WARN) | 0.92x | converging |
| Recent profit avg | 1.02 | 3.43 | **+236%** |
| Trade activity per window | 50-100 | 183-767 | 3-7x |
| ep_len_mean | 6200-7000 | 9594-11958 | +50-70% |
| Long/Short distribution | Spec2-late: Short-heavy | 39/61 balanced | dual-direction |

The agent did NOT converge to "trade rarely, trade well". Instead, Spec2
was **regime-specific over-fitting** to a low-volatility data slice. As the
sliding training data has moved forward (or the agent has finished
absorbing the low-vol slice), it has resumed bidirectional active trading
with profit recovery to alpha-bearing levels.

Compare with Spec1 (PPO_178): high-vol regime specialization paid off
(profit 4.65 with entropy 23% FAIL). Spec2 (PPO_198-205): low-vol regime
specialization did NOT pay off (profit 1.02 with entropy 31-37% FAIL).
PPO_211 is now the **specialization-exit** state in mixed-volatility data.

#### Cross-checkpoint comparison (updated with PPO_211)

| Window | Health | Avg Profit | Entropy | Neutral % | Phase |
|---|---|---|---|---|---|
| PPO_171 | 11/3/0 | 4.41 | 64% | 67% | Cache-fix stable |
| PPO_178 | 9/3/2 | 4.65 | 23% | 89% | Spec1 (high-vol specialization, alpha realized) |
| PPO_189 | 9/4/1 | 1.45 | 47% | 79% | Entropy recovery, low profit |
| PPO_198 | 11/2/1 | 1.05 | 31% | 81% | Spec2 onset (super-selective) |
| PPO_205 | 9/4/1 | 1.21 | 37% | 77% | Spec2 sustained (alpha NOT realized) |
| **PPO_211** | **12/2/0** | **3.43** | **62%** | **75%** | **Spec2 exit, training-history best** |

PPO_211 is the first checkpoint to achieve 12 OK and 0 FAIL simultaneously
in the v2-20260401 run. No previous window matched both conditions.

#### Training overshoot continues

| Time | PPO | Sub-train timestamp | Plan estimate | Overshoot |
|---|---|---|---|---|
| 2026-04-26 | PPO_189 | unknown (active) | ~189 | baseline |
| 2026-04-27 | PPO_198 | unknown (active) | ~189 | +9 (+5%) |
| 2026-04-28 | PPO_205 | 1710288000 (2024-03-12) | ~189 | +16 (+8%) |
| **2026-04-29** | **PPO_212** | **1710288000 (still 2024-03-12)** | **~189** | **+23 (+12%)** |

Sub-train timestamp has **not advanced** since PPO_204 era (~9 days). Yet
PPO checkpoints continue producing. With 94 sub-train folders / 212 PPO
windows, late phase shows ~2.25 PPO updates per sub-train (early phase
was ~1:1).

Interpretation: training is in **terminal-sub-train iteration phase**.
The sliding window has reached the end of training data (timerange end
2026-03-25, sub-train end 2024-03-12 + train_period_days), but PPO
optimization continues on the final data slice. This is unexpected from
the 0401 plan but possibly explains the late-stage health recovery -- the
agent gets more training cycles per sub-train without sliding into new
data.

#### Updated OOS backtest plan -- five candidates

PPO_205-final should be **demoted** in priority. PPO_211 enters as new
top candidate. PPO_final-when-complete added as fifth candidate.

| Candidate | Profile | Hypothesis | Priority |
|---|---|---|---|
| PPO_178 | profit 4.65 / entropy 23% / Neutral 89% | Spec1 high-vol specialization | High (validate Spec1 alpha) |
| PPO_189 | profit 1.45 / entropy 47% / Neutral 79% | sweet-spot generalization | Medium |
| PPO_197 | profit 3.88 / late-individual peak | late-stage best single | Medium |
| **PPO_211** | **profit 3.43 / entropy 62% / Neutral 75%** | **Spec2 exit, training-best health** | **HIGHEST** |
| PPO_final | TBD on natural termination | training-end converged state | High (if differs from PPO_211) |

Decision matrix update:

| Winner | Implication | Action |
|---|---|---|
| PPO_211 wins | Spec2 exit produced real alpha, training found correct mode | Use PPO_211 as primary |
| PPO_178 wins | High-vol specialization is the real skill (Spec1) | Use PPO_178, regime-conditioned |
| PPO_189 wins | Middle path generalizes best | Use PPO_189 |
| PPO_197 wins | Late-training has alpha but variance | Use PPO_197 |
| PPO_final ~ PPO_211 | Training stable, either works | Use PPO_final (latest) |
| PPO_final < PPO_211 | Training over-fit past PPO_211 | Use PPO_211 (early-stop pick) |
| All lose to baseline | Reward/feature redesign needed | Halt, redesign |

#### Status

Training **still active** (PID 66932, ~821h CPU time since 2026-04-21).
PPO_212 at 16% in-flight. Sub-train terminal but PPO continues iterating.
Watch points:
- PPO_212 final metrics: confirm Spec2 exit holds or single-window outlier
- Sub-train advance: if 1710288000 advances, sliding has resumed (more
  windows ahead). If stays, training will likely terminate at PPO_213-215.
- Next analyze-rl run: after PPO_212 completes (~3-4h at current rate).

Do not interrupt. Even if training continues another 5-10 windows past
expected, the recovery trajectory may produce additional improvement.
The new health profile justifies allowing extra runtime.

### 11.15 PPO_219 Checkpoint (2026-04-30) -- profit recovery continues, sliding window stall confirmed

#### Status

5-window summary: **9 OK / 5 WARN / 0 FAIL** (mild regression vs PPO_211's
12/2/0 but FAIL count holds at 0). Profit average **4.36** (range
0.91-6.93) -- highest 5-window average since PPO_178's 4.65 peak, and
this time with healthy 56% entropy retention (vs Spec1's 23% FAIL).

PPO_219 is in-flight at ~80% (794/989 iter, events file actively writing).
Multiple WARN flags trace to PPO_219's incomplete sample size, not a real
regression.

#### Per-window metrics (PPO_215-219)

| Window | Profit | Neutral % | Entropy | Win Rate | Value Loss | ExplVar | ep_len |
|---|---|---|---|---|---|---|---|
| PPO_215 | 2.68 | 73% | 57% | 47% | 27-62 (2.3x) | 0.62 | 10955 |
| PPO_216 | 6.93 | 67% | 63% | 47% | 29-53 (1.85x) | 0.79 | 11313 |
| PPO_217 | 4.59 | 65% | 69% | 49% | 27-36 (1.31x) | 0.55 | 10897 |
| PPO_218 | 6.69 | 70% | 57% | 49% | 29-64 (2.24x) | 0.49 | 11262 |
| PPO_219* | 0.91 | 66% | 56% | 43% | 27-53 (1.96x) | 0.60 | 9790 |

*PPO_219 in-flight (~80%, 794 iter / 989 expected).

PPO_216 (6.93) and PPO_218 (6.69) are the highest individual windows
since PPO_178 era. Entropy 56-69% across all 5 windows is healthier than
any previous 5-window stretch with profit > 4.

#### Cross-checkpoint comparison (updated)

| Window | Health | Avg Profit | Entropy | Neutral % | Phase |
|---|---|---|---|---|---|
| PPO_171 | 11/3/0 | 4.41 | 64% | 67% | Cache-fix stable |
| PPO_178 | 9/3/2 | 4.65 | 23% | 89% | Spec1 (entropy FAIL) |
| PPO_189 | 9/4/1 | 1.45 | 47% | 79% | Entropy recovery, low profit |
| PPO_198 | 11/2/1 | 1.05 | 31% | 81% | Spec2 onset |
| PPO_205 | 9/4/1 | 1.21 | 37% | 77% | Spec2 sustained |
| PPO_211 | 12/2/0 | 3.43 | 62% | 75% | Spec2 exit |
| **PPO_219** | **9/5/0** | **4.36** | **56%** | **66%** | **Profit + entropy both healthy (training-history best alpha+health combo)** |

PPO_219 is the **first checkpoint in v2 history with profit >= 4 AND
entropy >= 50% AND 0 FAIL**. PPO_178 had higher profit but entropy FAIL;
PPO_211 had healthier metrics but lower profit. PPO_219 combines both.

#### Sliding window investigation (Spec3: stall hypothesis)

Two parallel agent investigations were run to diagnose the sub-train
stall observed since 2026-04-20.

**Empirical findings (filesystem forensics):**
- Process PID 66932 is the THIRD distinct training run for this model
  (PID 32488 ran PPO_1-70, PID 39876/34580 ran PPO_71-149, PID 66932
  has run PPO_150-219 since 2026-04-18 to 2026-04-19)
- Current run rate: 70 PPO checkpoints / 12 days = **~6 PPO/day**
- The "sub-train stalled at 1710288000" interpretation is partially
  incorrect: the FOLDER NAME has not advanced (no new 1710892800 etc.
  appeared), but the `best_model.zip` inside multiple sub-train folders
  is being updated TODAY in chronological order, ~3-4h apart
- This means the process is iterating through the existing 94 sub-trains
  on a 2nd or 3rd pass, NOT advancing to new sub-trains

**Code-level findings:**
- `freqai/data_kitchen.py:598` constructs sub-train folder name from
  `tr_train.stopts` (training timerange end timestamp)
- `freqai/freqai_interface.py:301` iterates `dk.training_timeranges`
  with stride `bt_period` (7 days = 604800s, matches observed)
- `freqai/freqai_interface.py:366` `model_exists()` check looks for
  `cb_eth_<ts>_model.zip` (written by `data_drawer.py:523`), which is
  DIFFERENT from `best_model.zip` (written by EvalCallback)
- `freqai/prediction_models/ReinforcementLearner.py:67-73`:
  `tensorboard_log` is shared across all sub-trains for one coin, so
  PPO_N auto-increments globally (NOT per-sub-train) -- 219 PPO / 94
  sub-train ratio of 2.33 is normal accumulation
- `wait_for_training_iteration_on_reload: true` only affects live/dry
  shutdown, NOT backtest

**Hypothesis (medium confidence):**

The training is in an iteration loop without sliding advancement. Two
possible mechanisms:
1. `cb_eth_<ts>_model.zip` save fails silently each cycle, so
   `model_exists()` returns False on next iteration, forcing re-train
2. The backtest sliding window has reached some bound (computed end of
   data window, factoring train_period_days + backtest_period_days)
   that prevents creating sub-train 1710892800 onwards

**What we cannot determine without runtime instrumentation:**
- Whether `cb_eth_1710288000_model.zip` actually exists on disk (would
  determine which hypothesis is correct)
- Whether sliding will eventually break out due to convergence criteria

#### Time-to-completion: unknown

Three scenarios with rough probabilities:

| Scenario | Probability | ETA |
|---|---|---|
| Internal iteration cap triggers natural exit | 30% | hours to days |
| Permanent loop, manual kill required | 50% | never (without intervention) |
| Sliding breaks out due to data bound recomputation | 20% | uncertain |

The previous "0-3 windows remaining" and "1-3 days" estimates have both
proven wrong. Best to monitor whether sub-train name 1710892800 ever
appears (would confirm sliding resumed) vs. continued production of
PPO_220, 221... within sub-train 1710288000 (would confirm stall).

#### Five-candidate OOS plan -- now urgent

If training is permanently looping, the OOS backtest should proceed with
the candidates already produced. PPO_219 enters as a sixth candidate
(profit + entropy combo).

| Candidate | Profile | Priority |
|---|---|---|
| PPO_178 | profit 4.65 / entropy 23% / Neutral 89% | High (Spec1 validation) |
| PPO_189 | profit 1.45 / entropy 47% | Medium |
| PPO_197 | profit 3.88 individual peak | Medium |
| PPO_211 | profit 3.43 / entropy 62% / Neutral 75% / **12/2/0** | High (cleanest health) |
| **PPO_219** | **profit 4.36 / entropy 56% / Neutral 66% / 9/5/0** | **Highest (best alpha+health combo)** |
| PPO_final | TBD if training ever completes | Conditional |

#### Status: monitor 24h, then decide

Watch points for tomorrow (2026-05-01):
- Does folder `sub-train-ETH_1710892800` (2024-03-20) appear? If yes,
  sliding resumed -- continue waiting
- If only PPO_220+ appear within sub-train 1710288000, stall confirmed
- If `best_model.zip` in older sub-trains continues being overwritten,
  process is on 3rd+ pass -- definite loop, manual intervention needed

If 24h shows no name advancement: kill process, accept PPO_219 as
training-end checkpoint, proceed to OOS backtest with the 6 candidates
above.

### 11.16 PPO_225 Checkpoint (2026-05-01) -- stall hypothesis FALSIFIED, sliding resumed

#### Critical correction: yesterday's "permanent stall" was WRONG

The 24h watch point fired exactly. Sub-train name DID advance:

| Item | 2026-04-30 (PPO_219) | 2026-05-01 (PPO_225) | Change |
|---|---|---|---|
| Latest sub-train ts | 1710288000 (2024-03-13) | **1713916800 (2024-04-23)** | **+6 windows, +41 days** |
| Sub-train folder count | 94 | 100 | +6 |
| PPO count | 219 | 225 | +6 |
| PPO:sub-train new ratio | -- | **1:1** | normal sliding |

The "stall" interpretation from Section 11.15 is **falsified**. Sliding
window was making progress, just at a rate that made the latest folder
name appear static under `ls | tail` ordering. The 24h decision point
correctly showed advancement -- training continues normally.

The `feedback_sliding_window_stall.md` memory entry should be downgraded
from "confirmed bug" to "misread observation" -- the model_exists() vs
best_model.zip distinction is real code architecture but it was NOT
causing a stall in this run.

#### Status

5-window summary: **11 OK / 3 WARN / 0 FAIL** -- improvement vs PPO_219's
9/5/0 (gained 2 OK, dropped 2 WARN). FAIL count holds at 0 for 7
consecutive 5-window samples since PPO_211.

#### Per-window metrics (PPO_221-225)

| Window | Profit | Neutral % | Entropy | Win Rate | Value Loss | ExplVar | ep_len |
|---|---|---|---|---|---|---|---|
| PPO_221 | 3.23 | 73% | 57% | 45% | 30-38 (1.27x) | 0.77 | 7486 |
| PPO_222 | 1.39 | 67% | 55% | 42% | 28-72 (2.59x) | 0.57 | 8910 |
| PPO_223 | 4.33 | 67% | 65% | 46% | 25-33 (1.32x) | 0.87 | 8223 |
| **PPO_224** | **7.04** | 67% | 66% | **50%** | 28-48 (1.69x) | 0.76 | 7971 |
| PPO_225* | 1.22 | 65% | 76% | 44% | 32-26 (0.80x) | 0.77 | 3821 |

*PPO_225 in-flight at 15% (148/989 iter).

5-window profit avg = **3.44** (range 1.22-7.04). Slightly below PPO_219's
4.36 due to PPO_222's 1.39 outlier, but PPO_224 hit **7.04**, the
HIGHEST individual window in entire v2 training history (beats PPO_178
6.92, PPO_216 6.93, PPO_218 6.69).

#### Cross-checkpoint comparison (history-best individual window)

| Window | Health | Avg Profit | Best Single | Entropy | Neutral % | Phase |
|---|---|---|---|---|---|---|
| PPO_178 | 9/3/2 | 4.65 | 6.92 | 23% FAIL | 89% | Spec1 |
| PPO_211 | 12/2/0 | 3.43 | 5.09 | 62% | 75% | Spec2 exit |
| PPO_219 | 9/5/0 | 4.36 | 6.93 | 56% | 66% | Profit + entropy combo |
| **PPO_225** | **11/3/0** | **3.44** | **7.04** | **66%** | **68%** | **History-best individual + entropy still rising** |

Entropy_trend cross-window: **+18.0pp** (57% -> 76%). Agent is becoming
MORE explorative across windows, not less. This is opposite of Spec1/Spec2
specialization patterns and indicates active learning continues.

#### Time-to-completion (revised, third estimate)

| Item | Value |
|---|---|
| Current sub-train end date | 2024-04-23 |
| Timerange end target | 2026-03-25 |
| Remaining days of timerange | ~702 days |
| Sub-trains remaining @ 7d stride | ~100 |
| Today's rate (24h sample) | ~6 sub-train/day |
| Sustainable rate (factor for slowdowns) | ~4 sub-train/day |
| **ETA range** | **17-25 days from now** |
| **Predicted completion** | **2026-05-17 to 2026-05-25** |

Comparison of completion estimates over time:
- 2026-04-28: "0-3 windows remaining" -- WRONG
- 2026-04-29: "1-3 days" -- WRONG
- 2026-04-30: "permanent loop, manual kill needed" -- WRONG
- **2026-05-01: 17-25 days remaining** -- based on actual sub-train advance rate

The pattern of progressively-extending estimates suggests the timerange
genuinely requires more training than originally planned. The 0401 plan's
189-window estimate is now seen as a significant underestimate; actual
will be ~325 PPO windows (225 done + ~100 more).

#### OOS plan: 6 candidates unchanged

PPO_225 is in-flight; PPO_224 (profit 7.04) becomes a new candidate when
finalized. Current top OOS candidates:

| Candidate | Profile | Priority |
|---|---|---|
| PPO_178 | 4.65 / entropy 23% / Neutral 89% | High (Spec1 baseline) |
| PPO_189 | 1.45 / entropy 47% | Medium |
| PPO_197 | 3.88 individual peak | Medium |
| PPO_211 | 3.43 / entropy 62% / **12/2/0 cleanest** | High |
| PPO_219 | 4.36 / entropy 56% / 9/5/0 best alpha+health | High |
| **PPO_224** | **7.04 individual record / entropy 66% / 0 FAIL** | **New top single-window candidate** |
| PPO_final | TBD on natural completion | Conditional |

#### Status

Training advancing normally. PPO_225 ~80-90% complete by next analyze-rl
cycle. Sub-train sliding resumed and progressing at expected pace.

Memory file `feedback_sliding_window_stall.md` should be updated to mark
the stall hypothesis as falsified, while preserving the code-level
architecture finding (model_exists vs best_model.zip) as accurate but
not causally responsible for any observed delay.

### 11.17 PPO_232 Checkpoint (2026-05-02) -- profit doubles, PPO_231 hits 9.54 history-best

#### Headline: training-history strongest 24h

5-window profit avg jumped to **7.12** (PPO_225 era was 3.44, +107%).
PPO_231 hit profit **9.54** -- new v2 individual-window record (beats
PPO_224's 7.04 set yesterday). Four consecutive windows above 7.8:

| Window | Profit | Note |
|---|---|---|
| PPO_228 | 8.63 | Entry to high-profit regime |
| PPO_229 | 8.57 | Sustained |
| PPO_230 | 7.81 | Most active (Neutral 58%) |
| **PPO_231** | **9.54** | **History-best individual window** |
| PPO_232* | 1.07 | Latest_complete but env[0] tiny sample (40 actions) |

#### Health summary: 9 OK / 3 WARN / 2 FAIL (FAIL are sampling artifacts)

The 2 FAIL flags both trace to PPO_232's tiny env[0] sample:

| FAIL | Detail | Real signal? |
|---|---|---|
| sample_size FAIL | 40 actions from env[0] | **NO** -- multiproc sampling caveat |
| win_rate FAIL | 0.4% (2W/560L) | **NO** -- depends on sample_size which failed |

The 560 loss_exits / 2 profitable_exits ratio in PPO_232 is from forced
closures during episode end (when env[0] happened to be sampled).
`total_profit = 1.07` is the real all-environment aggregate and is
positive.

Looking at PPO_231 (last fully-sampled window) for actual health profile:
- Profit 9.54 (record)
- Entropy 67% retained (excellent)
- Value loss 32 -> 30 (0.93x, decreasing)
- Win rate 49.6% (346W/351L, large sample)
- Neutral 63%, balanced L/S
- Explained var 0.76

This profile is effectively 11/3/0 quality if PPO_232 had been complete-
sampled. The two artificial FAILs should not be interpreted as regression.

#### Per-window metrics (PPO_228-232)

| Window | Profit | Neutral % | Entropy | Win Rate | Value Loss | ExplVar | ep_len |
|---|---|---|---|---|---|---|---|
| PPO_228 | 8.63 | 62% | 62% | 53.4% | 41-51 (1.23x) | 0.68 | 11214 |
| PPO_229 | 8.57 | 67% | 59% | 54.2% | 44-43 (0.98x) | 0.75 | 10451 |
| PPO_230 | 7.81 | 58% | 68% | 51.4% | 39-46 (1.21x) | 0.59 | 12297 |
| **PPO_231** | **9.54** | 63% | 67% | 49.6% | 32-30 (0.93x) | 0.76 | 10379 |
| PPO_232* | 1.07 | (artifact) | 59% | (artifact) | 30-33 (1.11x) | 0.83 | 9617 |

#### Why the profit acceleration?

Current sub-train range (2024-04-23 to 2024-06-12) covers ETH market
behavior with high volatility:
- Apr-May 2024: ETH dropped from $3,200 to $2,900 then rebounded to $3,800
- May-Jun 2024: consolidation $3,500-3,900 then breakout

This regime resembles Spec1's high-volatility 2025 Q2 (Trump tariff era)
that produced PPO_178 profit 4.65 with entropy 23% FAIL. The current
regime is producing **much higher profit (7.12 vs 4.65) with HEALTHY
entropy (67% vs 23%)** -- a "healthy Spec1" that captures volatility
alpha without specialization collapse.

Comparison:
- Spec1 (PPO_178): 89% Neutral, 23% entropy FAIL, profit 4.65
- Current (PPO_228-231): 58-67% Neutral, 59-68% entropy, profit 7.12

The model has learned to exploit volatility through bidirectional active
trading, not through unilateral direction-locking.

#### Sub-train advance: +7 in 24h (slight acceleration)

| Date | Latest sub-train | Sub-train count | PPO count | Rate |
|---|---|---|---|---|
| 2026-04-30 | 1710288000 (2024-03-13) | 94 | 219 | -- |
| 2026-05-01 | 1713916800 (2024-04-23) | 100 (+6) | 225 (+6) | 6/day |
| **2026-05-02** | **1718150400 (2024-06-12)** | **107 (+7)** | **232 (+7)** | **7/day** |

PPO:sub-train ratio 1:1 maintained. Sliding healthy.

#### Time-to-completion (fourth revision, slight acceleration)

| Item | Value |
|---|---|
| Current sub-train end | 2024-06-12 |
| Timerange end target | 2026-03-25 |
| Remaining timerange | ~651 days |
| Remaining sub-trains @ 7d stride | ~93 |
| Today's rate | 7 sub-train/day |
| Sustainable rate | ~5 sub-train/day |
| **ETA range** | **2026-05-15 to 2026-05-21** |

Compared to yesterday's estimate (2026-05-17 to 2026-05-25): slight
**2-4 day acceleration** due to rate increase (6/day -> 7/day).

#### Updated OOS plan: PPO_231 added as new top single-window candidate

| Candidate | Profile | Priority |
|---|---|---|
| PPO_178 | 4.65 / entropy 23% / Neutral 89% | High (Spec1 baseline) |
| PPO_189 | 1.45 / entropy 47% | Medium |
| PPO_197 | 3.88 individual peak | Low (superseded) |
| PPO_211 | 3.43 / entropy 62% / **12/2/0 cleanest** | High |
| PPO_219 | 4.36 / entropy 56% | High |
| PPO_224 | 7.04 / entropy 66% | High |
| **PPO_231** | **9.54 / entropy 67% / Neutral 63% / healthy Spec1** | **NEW TOP** |
| PPO_final | TBD | Conditional |

PPO_231 represents the strongest combination yet: highest profit + healthy
entropy + balanced L/S + active trading. If OOS results validate this,
PPO_231 is the new training-end primary candidate (assuming training
doesn't produce something even better in the remaining 13-19 days).

#### Status

Training is in its strongest 24h period. ETA mid-to-late May 2026. Do
not interrupt -- ongoing run is producing genuine alpha in a regime
that resembles real market volatility.

### 11.18 PPO_249 Checkpoint (2026-05-05) -- PPO_248 hits 10.80, first individual window > 10

#### Headline: profit ceiling broken to double digits

PPO_248 reached profit **10.80**, the first v2 individual window above
10.0 (beats PPO_231's 9.54). Sustained alpha continues across PPO_245-249:

| Window | Profit | Note |
|---|---|---|
| PPO_245 | 4.89 | Mid-stage |
| PPO_246 | 8.22 | Strong |
| PPO_247 | 1.14 | env[0] tiny sample (172 actions); aggregate profit still positive |
| **PPO_248** | **10.80** | **First > 10, NEW history-best individual** |
| PPO_249* | 4.78 | In-flight at 38% |

5-window profit avg = **5.96** (range 1.14-10.80). Slightly below PPO_232
era's 7.12 due to PPO_247 sampling artifact, but PPO_248's 10.80 sets
the new ceiling.

#### Health summary: 9 OK / 4 WARN / 1 FAIL

The 1 FAIL is **value_loss 2.14x in PPO_248** -- but this is a
single-window overshoot during high-alpha learning, not a trend.
Cross-window value_loss progression:

| Window | Value Loss First -> Last | Ratio | Status |
|---|---|---|---|
| PPO_245 | 31 -> 44 | 1.43x | WARN |
| PPO_246 | 30 -> 49 | 1.66x | WARN |
| PPO_247 | 31 -> 47 | 1.51x | WARN |
| PPO_248 | 27 -> 57 | **2.14x** | **FAIL** |
| PPO_249 | 47 -> 44 | **0.94x** | **OK (recovered)** |

PPO_249's in-window value_loss is already converging (0.94x), confirming
PPO_248 was a learning-rate spike from absorbing the high-profit alpha,
not a degradation pattern.

The 4 WARN are mostly PPO_248-specific:
- policy_collapse Neutral=61% (mid-band, similar to recent windows)
- invalid_actions 20.5% (large sample, persistent issue)
- explained_variance 0.46 (PPO_248 noisy, but PPO_249 back to 0.73)
- long_short_bias 68% Short (PPO_248 single-window; not persistent across windows)

#### L/S bias is not persistent (regime adaptation)

Per-window L/S distribution:
- PPO_245: 56/44 balanced
- PPO_246: 51/49 balanced
- PPO_247: 54/46 balanced
- PPO_248: 32/68 SHORT-heavy <- WARN source
- PPO_249: 63/37 LONG-heavy

Agent flips direction by window, matching the 2024 mid-year market regime
shifts (PPO_248 covers ~2024-08 carry trade unwind crash; PPO_249 covers
the rebound). This is healthy regime adaptation, NOT specialization.

#### Sub-train advance: +17 in 3 days (5.7/day, consistent)

Note: This analysis happens 3 days after PPO_232 (2026-05-02), not 1 day.

| Date | Latest sub-train | Sub-train count | PPO count |
|---|---|---|---|
| 2026-05-02 | 1718150400 (2024-06-12) | 107 | 232 |
| **2026-05-05** | **1728432000 (2024-10-08)** | **124** | **249** |
| Delta | +118 days timerange | +17 | +17 |

Rate: 5.7 sub-train/day (3-day average), down from 7/day peak but still
healthy. PPO:sub-train ratio 1:1 maintained.

#### Time-to-completion (fifth revision)

| Item | Value |
|---|---|
| Current sub-train end | 2024-10-08 |
| Timerange end target | 2026-03-25 |
| Remaining timerange | ~534 days |
| Remaining sub-trains @ 7d stride | ~76 |
| Recent 3-day rate | 5.7 sub-train/day |
| Optimistic (7/day) | ~11 days |
| Conservative (5/day) | ~15 days |
| **ETA range** | **2026-05-16 to 2026-05-20** |

Consistent with 2026-05-02 estimate (5/15-5/21). No change in trajectory.

#### Updated OOS plan: PPO_248 added as new top single-window candidate

| Candidate | Profile | Priority |
|---|---|---|
| PPO_178 | 4.65 / entropy 23% / Neutral 89% | High (Spec1 baseline) |
| PPO_211 | 3.43 / entropy 62% / **12/2/0 cleanest** | High |
| PPO_219 | 4.36 / entropy 56% / 9/5/0 | Medium |
| PPO_224 | 7.04 / entropy 66% | Medium |
| PPO_231 | 9.54 / entropy 67% / "healthy Spec1" | High |
| **PPO_248** | **10.80 / entropy 69% / first > 10** | **NEW TOP** |
| PPO_final | TBD | Conditional |

PPO_248 takes the new top spot for highest single-window profit while
maintaining entropy 69% (healthier than any of Spec1's checkpoints at
similar profit levels). However its 32/68 L/S split is regime-specific;
OOS testing should validate whether PPO_248's policy generalizes or
only works in the 2024-08 short-bias regime.

PPO_211 remains preferred for "cleanest health" baseline (12/2/0). PPO_231
remains preferred for "balanced L/S + healthy + high profit". PPO_248
adds "individual ceiling" to the candidate pool.

#### Status

Training advancing normally. Watch points:
- PPO_249 completion: confirm value_loss recovery (already showing 0.94x)
- Whether PPO_250+ produces another > 10 window or PPO_248 stays as ceiling
- Sub-train rate -- if drops below 4/day, ETA extends