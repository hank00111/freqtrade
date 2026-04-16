# RL Elliott Wave Strategy - Analysis Document

> Date: 2026-02-08
> Status: Design Phase
> Target: Elliott Wave-based RL trading strategy for ETH + SOL futures
> Reference: "Elliott Wave Theory" (Prognosis Software Development, 47 pages)

---

## 1. Objectives

| Item | Value |
|------|-------|
| Capital | 100 USDT |
| Leverage | 10x (ETH), consider 5x (SOL) |
| Daily Target | 5~10 USDT net profit (5~10% daily ROI) |
| Style | Wave 3 (or C) capture, trend-following intraday |
| Pairs | ETH/USDT:USDT, SOL/USDT:USDT |
| Main Timeframe | 5m |
| Multi-Timeframe | 5m, 15m, 1h, 4h |
| Corr Pairlist | BTC/USDT:USDT |

---

## 2. Elliott Wave Theory Foundation (from PDF)

### 2.1 Core Principle

Markets move in repetitive fractal patterns:
- Impulse (trend direction): 5 waves (1-2-3-4-5)
- Correction (counter-trend): 3 waves (A-B-C)
- One complete degree = 5 + 3 = 8 waves
- Each wave contains sub-waves of the same structure (fractal)

```
Bull market:                Bear market:
       5                          1
      / \                        / \
    /    \                     /    2
   3      \                   /      \
  / \      \  B              /        3
 /   4      \/  \           A          \   /C
1     \      A   C           \        / \ /
       2                      B     4    5
                                   /
```

### 2.2 Three Inviolable Rules

| Rule | Description | Quantitative Condition |
|------|-------------|----------------------|
| R1 | Wave 2 cannot retrace beyond the origin of Wave 1 | `wave2_retrace < wave1_length * 1.0` |
| R2 | Wave 3 is never the shortest impulse wave | `wave3_length > min(wave1_length, wave5_length)` |
| R3 | Wave 4 cannot overlap Wave 1's price territory | `wave4_low > wave1_high` (bull) |

Source: PDF page 10

### 2.3 Fibonacci Ratios (PDF pages 31-34)

#### Impulse Waves

| Wave | Fibonacci Target | Notes |
|------|-----------------|-------|
| Wave 1 | Stops at base of previous correction (B wave) | Often coincides with 38.2% or 61.8% of previous correction |
| Wave 2 | Retraces 38.2%, 50%, or 61.8% of Wave 1 | Often stops at sub-wave 4 or 2 of Wave 1; >76% is suspicious |
| Wave 3 | 1.618x Wave 1 (most common) | At least equal to Wave 1; never the shortest |
| Wave 4 | Retraces 23.6%~38.2% of Wave 3 | In strong markets, may retrace only 14% |
| Wave 5 | Equal to Wave 1, or 61.8% of Wave 1 | If extended: 1.618x Wave 3 |

#### Corrective Waves

| Wave | Fibonacci Target | Notes |
|------|-----------------|-------|
| Wave A | In Triangle/B/4: 38.2% of entire previous impulse; In Zigzag: 61.8% of Wave 5 | Often retraces into previous 4th wave territory |
| Wave B | Zigzag: 38.2%~61.8% of A; Flat: approx equal to A | Expanded Flat: 123.6% or 138.2% of A |
| Wave C | Equal to A, or 1.618x A | At least 61.8% of A; if shorter = failure |
| Wave D | In contracting triangle: 61.8% of B | |
| Wave E | In contracting triangle: 61.8% of C | Cannot be longer than C |

### 2.4 Wave Patterns (PDF pages 8-27)

#### Trend Patterns

| Pattern | Structure | Key Characteristics |
|---------|-----------|-------------------|
| Impulse | 5-3-5-3-5 | No overlap between Wave 4 and 1; Wave 3 strongest momentum; alternation between Wave 2 and 4 corrective types |
| Extension | 5/9/13/17 waves | One of waves 1/3/5 is much longer; normally Wave 3 |
| Diagonal Type 1 | 3-3-3-3-3 | Occurs in Wave 5 or C (rarely Wave 1 per Prognosis PDF); converging channel; followed by violent reversal |
| Diagonal Type 2 | 5-3-5-3-5 | Occurs in Wave 1 or A; converging channel |
| Failure/Truncated 5th | 5 waves | Wave 5 fails to exceed Wave 3; indicates weak trend |

#### Correction Patterns

| Pattern | Structure | Key Characteristics |
|---------|-----------|-------------------|
| Zigzag | 5-3-5 | Sharp reversal; B retraces <= 61.8% of A; C >= A |
| Double Zigzag | WXY (5-3-5-x-5-3-5) | Two zigzags connected by X wave |
| Flat | 3-3-5 | Sideways; B retraces > 61.8% of A; C approx A |
| Expanded Flat | 3-3-5 | B extends beyond previous impulse end; strong C wave |
| Triangle | 3-3-3-3-3 | Contracting/expanding; only in B, X, 4 (never 2 or A) |
| Combination (WXY) | Mixed | Combines different correction types |
| Running Flat | 3-3-5 | Rare; B extends far; C very short |

### 2.5 Channeling (PDF pages 27-31)

Parallel channels serve two purposes:
1. Determine which sub-waves belong together (same degree)
2. Project price targets for the next wave

```
Channel for Wave 3 target:
  Connect: origin of Wave 1 (0) and end of Wave 2
  Parallel from: top of Wave 1
  Guideline: Wave 3 should break through upper line (if not, probably a C wave)
  Base line (0 to Wave 2) serves as stop

Channel for Wave 4 target:
  Connect: end of Wave 1 and end of Wave 3
  Parallel from: end of Wave 2
  Wave 4 normally slightly breaks the base line

Channel for Wave 5 target (Method 1):
  Connect: end of Wave 2 and end of Wave 4
  Parallel from: end of Wave 3
  Wave 5 usually fails to reach upper line (except extensions)

Channel for Wave 5 target (Method 2, for strong Wave 3):
  Connect: end of Wave 2 and end of Wave 4
  Parallel from: end of Wave 1 (cuts through Wave 3)
  More accurate when Wave 3 is near-vertical
```

### 2.6 Confirmation Tools (PDF pages 37, 29)

The PDF mentions two non-pattern confirmation tools:

1. **Momentum indicators** (generic, no specific indicator named)
   - "Use momentum indicators and volume to support your wave labeling"
   - "Wave 3 should have the highest momentum"
   - "the third wave shows the greatest momentum, except when the fifth
     is the extended wave"

2. **Volume**
   - "indicated by high volume and momentum indicators"
   - "Wave 3 should have the highest momentum and volume
     (if it is the longest wave)"

No other technical indicators (RSI, EMA, MACD, Bollinger, ATR, Stochastic)
are mentioned in the PDF.

---

## 3. Trading Strategy (PDF pages 35-45)

### 3.1 Core Strategy: Trade Wave 3

PDF page 43: "Simple, but effective trading strategy"

> Since all patterns or their sub patterns are either 3 wave or 5 wave
> structures, it follows that at the minimum always three waves will
> occur, no matter what happens. Therefore if you concentrate on the
> 3rd wave, which will be a wave 3 in an impulse or wave C in a
> correction, you have a strong probability of making a profit.

```
Step 1: Identify Wave 1 (or A) completion
  - A directional move after a correction/reversal

Step 2: Identify Wave 2 (or B) completion
  - Retracement into Fibonacci 38.2%~61.8% zone
  - Price holds above Wave 1 origin (if breaks, count invalid)

Step 3: Enter on Wave 3 (or C) start
  - Entry: after Wave 2 completion is confirmed
  - Confirmation: price moves beyond Wave 1 end in trend direction
  - Stop: at Wave 1 origin

Step 4: Hold through Wave 3
  - Wave 3 is normally the strongest and fastest wave
  - Target: Wave 1 length * 1.618 extension

Step 5: Exit at target or invalidation
  - Target: Fibonacci extension 1.618 of Wave 1
  - Alternative: channel upper boundary
  - Exit also if momentum diverges (Wave 5 signal)
```

### 3.2 Entry/Exit Decision Flow (from PDF pages 38-42)

```
Observe rising price after a decline:
  |
  +--> Wave 1 or A identified (directional move)
  |
  +--> Pullback occurs (Wave 2 or B)
  |      |
  |      +--> Retraces 38.2~61.8%? --> Likely Wave 2 completion zone
  |      |
  |      +--> Retraces > 100%? --> Wave count INVALID, exit/no trade
  |
  +--> Price resumes trend direction (Wave 3 or C starting)
  |      |
  |      +--> Momentum accelerating? --> CONFIRMED, enter trade
  |      |
  |      +--> Volume increasing? --> Additional confirmation
  |
  +--> During Wave 3:
  |      |
  |      +--> Wave 4 overlaps Wave 1? --> NOT an impulse, likely correction
  |      |     --> May still profit from C wave, but re-evaluate
  |      |
  |      +--> Reaching 1.618 extension? --> Consider exit
  |
  +--> Stop: below Wave 1 origin (entire count invalidated)
```

### 3.3 Multi-Degree Confirmation (PDF page 37)

```
4h  -> "What degree is the larger wave structure?"
1h  -> "Is the current move a Wave 3 at this degree?"
15m -> "Is the sub-wave structure correct (5-3-5-3-5)?"
5m  -> "Is the entry timing right (momentum turning)?"

The agent sees all timeframes simultaneously.
Fibonacci and swing features at each TF encode different wave degrees.
```

### 3.4 Alternative Scenario Approach (PDF page 37)

The PDF emphasizes designing multiple alternative scenarios:

```
Scenario A: Impulse (1-2-3-4-5) → Trade Wave 3
Scenario B: Correction (A-B-C)  → Trade Wave C
Scenario C: Triangle forming     → Stay neutral

Key: if multiple scenarios all predict same direction → high probability
```

For RL: the agent implicitly handles alternative scenarios through feature
combinations. It does not need explicit scenario labeling.

---

## 4. Feature Engineering

### 4.1 Design Principle

Only use analysis tools mentioned in the PDF:
- Fibonacci ratios (core)
- Channeling (core)
- Wave structure/rules (core)
- Momentum (confirmation, generic)
- Volume (confirmation)

Do NOT include indicators not in the PDF:
- No EMA, RSI, ATR, MACD, Bollinger Bands, Stochastic

### 4.2 ZigZag Swing Detection (Foundation)

All Elliott Wave features depend on identifying swing highs/lows.

```
Method: ATR-adaptive ZigZag with confirmation delay

Parameters:
  zigzag_pct:    Minimum % move to count as a new swing leg
                 (e.g., 0.5% for 5m, auto-scales per timeframe)
  confirm_bars:  Number of bars after turning point before confirmed
                 (e.g., 3 bars, prevents repainting)

Process:
  1. Identify candidate swing high: highest point before price drops by zigzag_pct
  2. Wait confirm_bars bars to confirm (price doesn't make new high)
  3. Mark as confirmed swing high
  4. Same logic for swing lows
  5. Maintain rolling list of last N confirmed swings

Output: ordered list of (price, index, type) for recent swings
  e.g., [(3050, 100, 'high'), (2980, 120, 'low'), (3100, 160, 'high'), ...]
```

Repaint prevention:
- Only confirmed swings are used for feature calculation
- The "current unconfirmed leg" is captured via fib_retrace_pct
  (which naturally updates as price moves)

### 4.3 feature_engineering_expand_basic

Features that expand across: include_timeframes x include_shifted_candles
x include_corr_pairlist (NOT indicator_periods).

```python
# === Fibonacci Features (5) ===

"%-fib_retrace_pct"
  # Current retracement as fraction of last completed swing
  # = (swing_high - current_price) / (swing_high - swing_low) for downswing
  # = (current_price - swing_low) / (swing_high - swing_low) for upswing
  # Range: 0.0 (no retrace) to 1.0+ (full retrace)
  # Purpose: detect Wave 2 completion zone (0.382, 0.50, 0.618)

"%-dist_to_fib_382"
  # (price - fib_382_level) / price, normalized
  # Negative = below level, positive = above level
  # Purpose: proximity to 38.2% retracement (common Wave 2 target)

"%-dist_to_fib_500"
  # (price - fib_500_level) / price, normalized
  # Purpose: proximity to 50% retracement (common Wave 2 target per Prechter)

"%-dist_to_fib_618"
  # (price - fib_618_level) / price, normalized
  # Purpose: proximity to 61.8% retracement (deep Wave 2 / Wave 4 target)

"%-dist_to_fib_ext_1618"
  # (fib_ext_1618_level - price) / price, normalized
  # Purpose: proximity to 1.618 extension (Wave 3 target)

# === Channeling Features (2) ===

"%-channel_position"
  # Price position within parallel channel drawn from recent swings
  # 0.0 = at channel bottom, 1.0 = at channel top
  # Uses the last 3 confirmed swing points to draw channel
  # Purpose: target projection, wave degree confirmation

"%-channel_width"
  # Channel width / price, normalized
  # Decreasing width = converging channel (possible triangle/diagonal)
  # Purpose: detect triangles, diagonals (channel convergence)

# === Swing Structure Features (4) ===

"%-swing_ratio"
  # Length of current (unconfirmed) swing / length of last confirmed swing
  # Purpose: wave proportionality
  #   Wave 3 target: ratio approaching 1.618
  #   Wave 5 target: ratio approaching 1.0 or 0.618
  #   Wave C target: ratio approaching 1.0 or 1.618

"%-wave_overlap"
  # Does current swing territory overlap the swing two positions back?
  # 0 = no overlap (valid impulse), 1 = overlap (correction or diagonal)
  # Purpose: Rule R3 verification (Wave 4 must not overlap Wave 1)

"%-swing_count"
  # Number of confirmed swings since last major reversal, normalized
  # Normalized by dividing by 10 (reasonable max for one wave degree)
  # Purpose: implicit wave count
  #   2 swings after reversal = possibly at end of Wave 2
  #   4 swings = possibly at end of Wave 4
  #   5 swings = possibly completing impulse

"%-swing_acceleration"
  # Rate of price change (slope) of current swing vs previous swing
  # = (current_swing_slope / prev_swing_slope)
  # Purpose: Wave 3 has highest acceleration;
  #   if current swing accelerates vs previous = Wave 3 characteristic
  #   if decelerates = possibly Wave 5 or correction
```

### 4.4 feature_engineering_expand_all

Features that expand across: indicator_periods_candles x include_timeframes
x include_shifted_candles x include_corr_pairlist.

```python
# === Momentum (1 indicator, PDF: "momentum indicators") ===

"%-momentum-period"
  # Rate of Change (ROC): (close - close[period]) / close[period]
  # The most direct measure of momentum without using named indicators
  # Purpose: Wave 3 should have the highest momentum (PDF page 10, 37)
  # With periods [10, 20]: captures short and medium-term momentum
```

Rationale for ROC instead of RSI/MACD:
- PDF says "momentum indicators" without naming a specific one
- ROC is the purest momentum measure (raw price change rate)
- It avoids adding indicator-specific assumptions not in the PDF

### 4.5 feature_engineering_standard

Features that do NOT expand (base timeframe only):

```python
# Required for RL environment:
"%-raw_close"    # Close price
"%-raw_open"     # Open price
"%-raw_high"     # High price
"%-raw_low"      # Low price

# Time features:
"%-hour_sin"     # Hour of day (cyclical encoding)
"%-hour_cos"
"%-dow_sin"      # Day of week (cyclical encoding)
"%-dow_cos"

# Volume (PDF: "volume to support wave labeling"):
"%-volume_norm"  # volume / rolling_mean(volume, 20)
                 # Purpose: Wave 3 should have highest volume
                 # Normalized to be scale-invariant
```

### 4.6 Feature Count Summary

```
expand_all:     1 indicator x 2 periods x 4 TF x 3 shifts x 2 pairs =  48
expand_basic:  11 indicators x 4 TF x 3 shifts x 2 pairs             = 264
standard:       9 features                                             =   9
──────────────────────────────────────────────────────────────────────────
Total:        ~321 features
```

Compared to RLDayTradeStrategy (~392 features), this is leaner due to
the absence of RSI, ATR, EMA, BB, MACD indicators.

---

## 5. Reward Function

### 5.1 Reuse HQT Reward Model

The Elliott Wave trading concept aligns with the High-Quality Trading (HQT)
reward model from `RLDayTrader_multiproc.py`:

| Elliott Wave Concept | HQT Reward Mapping | Alignment |
|---------------------|-------------------|-----------|
| Wait for Wave 2 completion, don't trade blindly | Neutral = 0, Entry = 0 | Aligned |
| Wave 3 is fast and strong | Quick profit bonus +2 | Aligned |
| Hold through Wave 3 to target | Profitable hold +0~+2 | Aligned |
| Stop at Wave 1 origin (accept small loss) | Small loss exit = -1 | Aligned |
| Large loss = invalid wave count | Loss gradient -1~-10 | Aligned |
| Never get liquidated | Liquidation = -10 | Aligned |
| "Let your profits run" (PDF page 38) | Holding reward scales with PNL | Aligned |
| "Limit your losses" (PDF page 38) | Asymmetric loss penalty (2.5x) | Aligned |

No new reward function needed. Use `RLDayTrader_multiproc.py` as-is.

### 5.2 Reward Structure Reference

```
Reward
+10 -- Fast profitable exit (Wave 3 captured quickly)
 +8 -- Profitable exit cap
 +2 -- Profitable hold cap (riding Wave 3)
  0 -- Neutral / Entry (waiting for setup)
 -1 -- Small loss exit (stop at Wave 1 origin)
 -2 -- Invalid action (learning cost)
 -5 -- Large loss hold / medium loss exit
-10 -- Liquidation / extreme loss
```

---

## 6. Architecture

### 6.1 File Structure

```
user_data/
  strategies/
    RLElliottWaveStrategy.py       # Strategy (Elliott Wave features)
  freqaimodels/
    RLDayTrader_multiproc.py       # Reused (HQT reward, no changes)
  config_elliott.json              # New config (identifier: rl-elliott-v1)
```

### 6.2 Inheritance

```
Strategy:
  IStrategy
    -> RLElliottWaveStrategy (Elliott Wave features)

Model (reused):
  ReinforcementLearner_multiproc
    -> RLDayTrader_multiproc
       -> MyRLEnv (Base4ActionRLEnv, HQT reward, unchanged)
```

### 6.3 Execution Command

```bash
freqtrade backtesting --strategy RLElliottWaveStrategy \
  --config user_data/config_elliott.json \
  --freqaimodel RLDayTrader_multiproc \
  --timerange 20230101-20260101 --export trades
```

---

## 7. Training Configuration

### 7.1 Config Differences from rl-daytrade-v1

| Parameter | rl-daytrade-v1 | rl-elliott-v1 | Reason |
|-----------|---------------|---------------|--------|
| identifier | rl-daytrade-v1 | rl-elliott-v1 | Separate model storage |
| strategy | RLDayTradeStrategy | RLElliottWaveStrategy | Different features |
| conv_width | 10 | 10 | Same temporal window |
| All other rl_config | Same | Same | Same reward, same PPO params |

### 7.2 Config Template

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
    "identifier": "rl-elliott-v1",
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
      "principal_component_analysis": false,
      "feature_flags": {
        "expand_all": {
          "momentum": true
        },
        "expand_basic": {
          "fib_retrace": true,
          "fib_dist": true,
          "fib_extension": true,
          "channel": true,
          "swing_structure": true
        },
        "standard": {
          "raw_ohlc": true,
          "time_cyclical": true,
          "volume_norm": true
        }
      }
    },
    "data_split_parameters": {
      "test_size": 0.25,
      "random_state": 1
    },
    "rl_config": {
      "train_cycles": 500,
      "max_trade_duration_candles": 48,
      "model_type": "PPO",
      "policy_type": "MlpPolicy",
      "max_training_drawdown_pct": 0.50,
      "cpu_count": 8,
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
    },
    "model_training_parameters": {
      "learning_rate": 0.0003,
      "gamma": 0.99,
      "batch_size": 1024,
      "n_steps": 2048,
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
}
```

---

## 8. Comparison: S/R Flip vs Elliott Wave

### 8.1 Feature Philosophy

| Aspect | S/R Flip (RLDayTradeStrategy) | Elliott Wave (RLElliottWaveStrategy) |
|--------|------------------------------|--------------------------------------|
| Core concept | Horizontal price levels | Swing structure and proportionality |
| Entry signal | Level flip (S becomes R) | Wave 2 completion (Fibonacci zone) |
| Exit signal | Level rejection | Wave 3 target (1.618 extension) |
| Feature type | Distance to price levels | Fibonacci ratios between swings |
| Best for | Range-bound, level retests | Trending markets, Wave 3 capture |
| Indicator dependency | RSI, ATR, EMA, BB, MACD | None (pure price action + Fibonacci) |
| Feature count | ~392 | ~321 |

### 8.2 Complementary Nature

The two strategies are complementary, not competing:
- S/R Flip excels in sideways/ranging markets (support holds, resistance breaks)
- Elliott Wave excels in trending markets (impulse wave structure)
- The same market may present both setups at different times
- Backtest comparison will reveal which approach performs better on ETH/SOL

### 8.3 Evaluation Criteria

After both backtests complete, compare using HQT metrics:

| Metric | Target | What It Tells |
|--------|--------|--------------|
| Trade count / window | Lower = more selective | Strategy selectivity |
| Win rate | > 40% with good avg win/loss | Trade quality |
| Avg profit per winning trade | > 2x avg loss | Reward asymmetry |
| Profit factor | > 1.5 | Gross profit / gross loss |
| Max drawdown | < 20% | Risk control |
| Neutral ratio | 50~70% | Appropriate selectivity |

---

## 9. Implementation Challenges

### 9.1 ZigZag Repaint Problem

The ZigZag indicator's last leg can change direction as new data arrives.

**Mitigation:**
- Use confirmation delay (N bars after swing point)
- Only compute features from confirmed swings
- Current unconfirmed movement is captured naturally by fib_retrace_pct
- Similar approach to fractal detection in S/R Flip (fractal_n parameter)

### 9.2 Swing Sensitivity

Too sensitive = too many swings (noise), too insensitive = miss real swings.

**Mitigation:**
- ATR-adaptive threshold: `zigzag_pct = ATR(14) * multiplier / price`
- The multiplier is a tunable parameter
- Multi-timeframe expansion naturally provides different sensitivity levels
  (5m swings are more sensitive than 4h swings)

### 9.3 Channel Calculation

Drawing parallel channels from 3 swing points requires:
- At least 3 confirmed swings to draw a valid channel
- Insufficient data at start of training window

**Mitigation:**
- Return neutral values (0.5 for channel_position, 0 for channel_width)
  when insufficient swings are available
- Same approach used for S/R features in RLDayTradeStrategy

### 9.4 Elliott Wave Subjectivity

The PDF acknowledges: "sometimes it is not totally clear if the internal
structure of a wave is a 3 wave or a 5 wave."

**Mitigation:**
- RL agent does not need explicit wave labels
- Features encode the structural properties (ratios, overlap, count)
- Agent learns which combinations predict profitable moves
- The PDF's alternative scenario approach is implicitly handled by the
  agent's probabilistic policy

---

## 10. Implementation Checklist

- [ ] Create config_elliott.json
  - [ ] Separate identifier (rl-elliott-v1)
  - [ ] feature_flags for Elliott Wave features
  - [ ] All other params same as config_daytrade.json
- [ ] Create RLElliottWaveStrategy.py
  - [ ] ZigZag swing detection function
  - [ ] Fibonacci retracement features (fib_retrace_pct, dist_382, dist_500, dist_618)
  - [ ] Fibonacci extension feature (dist_ext_1618)
  - [ ] Channeling features (channel_position, channel_width)
  - [ ] Swing structure features (swing_ratio, wave_overlap, swing_count, swing_acceleration)
  - [ ] Momentum feature (ROC) in expand_all
  - [ ] Volume feature in standard
  - [ ] Raw OHLCV and time features in standard
  - [ ] set_freqai_targets (&-action = 0)
  - [ ] populate_entry_trend (4-action: Long=2, Short=3)
  - [ ] populate_exit_trend (unified Exit=1)
  - [ ] leverage() callback (ETH=10.0, SOL=5.0)
  - [ ] custom_stoploss() callback (ETH=-0.03, SOL=-0.02)
- [ ] Download data (if not already available)
- [ ] Wait for rl-daytrade-v1 backtest to complete (baseline comparison)
- [ ] Run rl-elliott-v1 backtest
- [ ] Compare results against rl-daytrade-v1

---

## 11. References

### 11.1 PDF Sources

- Elliott Wave Theory (Prognosis Software Development, 47 pages)
  - Saved at: `docs/elliott_wave_2.pdf`
  - Key sections: Basic Theory (p4-7), Patterns (p8-27), Channeling (p27-31),
    Fibonacci Ratios (p31-34), Trading (p35-45)
- The Basics of the Elliott Wave Principle (Robert R. Prechter Jr., 46 pages)
  - Saved at: `docs/2026/20260208/ew-basics.pdf`
  - Cross-referenced for: Three Rules, Fibonacci retracements (50% level),
    corrective wave multiples, channeling technique, alternation guideline
- Mastering Elliott Wave (Rich Swannell, 168 pages, scanned)
  - Saved at: `docs/2026/20260208/master-elliott-wave.pdf`
  - Note: Image-based PDF, text not extractable; used for visual reference only

### 11.2 Existing Strategy (for comparison)

- `docs/2026/20260203/rl-daytrade-strategy-analysis.md` - S/R Flip strategy design doc
- `user_data/strategies/RLDayTradeStrategy.py` - S/R Flip strategy implementation
- `user_data/freqaimodels/RLDayTrader_multiproc.py` - HQT reward model (reused)

### 11.3 FreqAI Framework

- `freqtrade/freqai/RL/Base4ActionRLEnv.py` - 4-action environment
- `freqtrade/freqai/RL/BaseEnvironment.py` - Base gym.Env
- `docs/freqai-reinforcement-learning.md` - Official RL documentation
- `docs/freqai-feature-engineering.md` - Feature engineering documentation
