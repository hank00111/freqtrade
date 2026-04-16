# RL Day Trade Strategy - Feature Documentation

> Date: 2026-02-05
> Strategy: RLDayTradeStrategy
> Model: RLDayTrader
> Config: config_daytrade.json

---

## 1. Overview

### 1.1 Observation Shape

```
Observation: (conv_width, total_features) = (10, 488)
             ↓
Flattened:   4,880 dimensions per step
```

### 1.2 Feature Expansion Config

| Parameter | Value | Description |
|-----------|-------|-------------|
| `include_timeframes` | `["5m", "15m", "1h", "4h"]` | 4 timeframes |
| `include_corr_pairlist` | `["BTC/USDT:USDT"]` | 1 correlation pair |
| `indicator_periods_candles` | `[10, 20]` | 2 indicator periods |
| `include_shifted_candles` | `2` | 3 shifts (0, 1, 2) |
| `conv_width` | `10` | 10 candles of history |
| `drop_ohlc_from_features` | `false` | Raw OHLC included |

### 1.3 Feature Count Summary

| Category | Base Features | Expansion | Total |
|----------|---------------|-----------|-------|
| `expand_all` | 4 | × 2 periods × 4 TF × 3 shifts × 2 pairs | **192** |
| `expand_basic` | 12 | × 4 TF × 3 shifts × 2 pairs | **288** |
| `standard` | 8 | (no expansion) | **8** |
| **Total** | | | **488** |

---

## 2. Feature Engineering: expand_all

**Expansion**: `indicator_periods × timeframes × shifted × pairs`
= 2 × 4 × 3 × 2 = **48× per indicator**

### 2.1 Indicators (4)

| Feature Name | Formula | Range | Description |
|--------------|---------|-------|-------------|
| `%-rsi-period` | `ta.RSI(df, timeperiod=period)` | 0-100 | Relative Strength Index |
| `%-adx-period` | `ta.ADX(df, timeperiod=period)` | 0-100 | Average Directional Index (trend strength) |
| `%-atr_norm-period` | `ATR / close` | 0+ | Normalized Average True Range (volatility) |
| `%-rel_vol-period` | `volume / volume.rolling(period).mean()` | 0+ | Relative Volume |

### 2.2 Generated Feature Names (192 total)

Pattern: `%-{pair}{indicator}-period_{period}_{timeframe}_shift{shift}`

**Example for ETH/USDT (main pair):**
```
%-ETHUSDT:USDTrsi-period_10_5m
%-ETHUSDT:USDTrsi-period_10_5m_shift1
%-ETHUSDT:USDTrsi-period_10_5m_shift2
%-ETHUSDT:USDTrsi-period_10_15m
...
%-ETHUSDT:USDTrsi-period_20_4h_shift2
```

**Example for BTC/USDT (corr pair):**
```
%-BTCUSDT:USDTrsi-period_10_5m
...
```

---

## 3. Feature Engineering: expand_basic

**Expansion**: `timeframes × shifted × pairs`
= 4 × 3 × 2 = **24× per indicator**

### 3.1 Trend Indicators (2)

| Feature Name | Formula | Range | Description |
|--------------|---------|-------|-------------|
| `%-ema_20` | `ta.EMA(df, timeperiod=20)` | Price | 20-period Exponential Moving Average |
| `%-ema_50` | `ta.EMA(df, timeperiod=50)` | Price | 50-period Exponential Moving Average |

### 3.2 Bollinger Bands (2)

| Feature Name | Formula | Range | Description |
|--------------|---------|-------|-------------|
| `%-bb_percent` | `(close - BB_lower) / (BB_upper - BB_lower)` | 0-1 | Price position within band |
| `%-bb_width` | `(BB_upper - BB_lower) / BB_mid` | 0+ | Normalized bandwidth (volatility) |

**Bollinger Parameters**: window=21, stds=2.0

### 3.3 MACD (2)

| Feature Name | Formula | Range | Description |
|--------------|---------|-------|-------------|
| `%-macd_norm` | `MACD / close` | -/+ | Normalized MACD line |
| `%-macd_hist_norm` | `MACD_histogram / close` | -/+ | Normalized MACD histogram |

**MACD Parameters**: fastperiod=12, slowperiod=26, signalperiod=9

### 3.4 S/R Flip Features (6)

| Feature Name | Formula | Range | Description |
|--------------|---------|-------|-------------|
| `%-dist_to_support` | `(close - support) / close` | 0+ | Distance to nearest support (normalized) |
| `%-dist_to_resistance` | `(resistance - close) / close` | 0+ | Distance to nearest resistance (normalized) |
| `%-price_pos_in_sr` | `(close - support) / (resistance - support)` | 0-1 | Position within S/R range |
| `%-sr_strength` | `touch_count / max_touches` | 0-1 | Strength of nearest level |
| `%-sr_flip_bullish` | Binary | 0 or 1 | Bullish flip confirmed |
| `%-sr_flip_bearish` | Binary | 0 or 1 | Bearish flip confirmed |

**S/R Algorithm Parameters**:
- `fractal_n`: 5 (local extreme detection window)
- `cluster_atr_mult`: 0.5 (clustering distance = ATR × 0.5)
- `retest_pct`: 0.2% (flip confirmation threshold)
- `stale_break`: 20 candles (break expiration)

### 3.5 Generated Feature Names (288 total)

Pattern: `%-{pair}{indicator}_{timeframe}_shift{shift}`

**Example:**
```
%-ETHUSDT:USDTema_20_5m
%-ETHUSDT:USDTema_20_5m_shift1
%-ETHUSDT:USDTema_20_5m_shift2
%-ETHUSDT:USDTema_20_15m
...
%-BTCUSDT:USDTsr_flip_bearish_4h_shift2
```

---

## 4. Feature Engineering: standard

**No expansion** - base timeframe only, main pair only.

### 4.1 Raw OHLC (4)

| Feature Name | Source | Range | Description |
|--------------|--------|-------|-------------|
| `%-raw_close` | `df["close"]` | Price | Closing price (required for RL env) |
| `%-raw_open` | `df["open"]` | Price | Opening price |
| `%-raw_high` | `df["high"]` | Price | High price |
| `%-raw_low` | `df["low"]` | Price | Low price |

**Note**: Raw OHLC is used internally by RL environment for PNL calculation. When `drop_ohlc_from_features: false`, these are also included in the observation.

### 4.2 Time Features (4)

| Feature Name | Formula | Range | Description |
|--------------|---------|-------|-------------|
| `%-hour_sin` | `sin(2π × hour / 24)` | -1 to 1 | Hour of day (cyclical, sin) |
| `%-hour_cos` | `cos(2π × hour / 24)` | -1 to 1 | Hour of day (cyclical, cos) |
| `%-dow_sin` | `sin(2π × dow / 7)` | -1 to 1 | Day of week (cyclical, sin) |
| `%-dow_cos` | `cos(2π × dow / 7)` | -1 to 1 | Day of week (cyclical, cos) |

**Cyclical Encoding Rationale**: Prevents discontinuity at day/week boundaries (e.g., 23:59 → 00:00).

---

## 5. Feature Categories by Purpose

### 5.1 Trend Detection

| Feature | Timeframes | Purpose |
|---------|------------|---------|
| `ema_20`, `ema_50` | 5m, 15m, 1h, 4h | Short/medium trend direction |
| `adx` | 5m, 15m, 1h, 4h | Trend strength |
| `macd_norm`, `macd_hist_norm` | 5m, 15m, 1h, 4h | Momentum and trend change |

### 5.2 Volatility

| Feature | Timeframes | Purpose |
|---------|------------|---------|
| `atr_norm` | 5m, 15m, 1h, 4h | Normalized volatility |
| `bb_width` | 5m, 15m, 1h, 4h | Bollinger-based volatility |

### 5.3 Overbought/Oversold

| Feature | Timeframes | Purpose |
|---------|------------|---------|
| `rsi` | 5m, 15m, 1h, 4h | Momentum oscillator |
| `bb_percent` | 5m, 15m, 1h, 4h | Price position in range |

### 5.4 Support/Resistance

| Feature | Timeframes | Purpose |
|---------|------------|---------|
| `dist_to_support` | 5m, 15m, 1h, 4h | Distance to support |
| `dist_to_resistance` | 5m, 15m, 1h, 4h | Distance to resistance |
| `price_pos_in_sr` | 5m, 15m, 1h, 4h | Position in S/R range |
| `sr_strength` | 5m, 15m, 1h, 4h | Level strength |
| `sr_flip_bullish` | 5m, 15m, 1h, 4h | Bullish flip signal |
| `sr_flip_bearish` | 5m, 15m, 1h, 4h | Bearish flip signal |

### 5.5 Volume

| Feature | Timeframes | Purpose |
|---------|------------|---------|
| `rel_vol` | 5m, 15m, 1h, 4h | Relative volume (activity) |

### 5.6 Time Context

| Feature | Purpose |
|---------|---------|
| `hour_sin`, `hour_cos` | Intraday session patterns |
| `dow_sin`, `dow_cos` | Weekly patterns |

### 5.7 Price Data

| Feature | Purpose |
|---------|---------|
| `raw_open`, `raw_high`, `raw_low`, `raw_close` | Raw price for RL env + potential pattern learning |

---

## 6. Missing Features (Potential Improvements)

### 6.1 Candlestick Structure Features

Currently **NOT** implemented:

| Feature | Formula | Purpose |
|---------|---------|---------|
| `body_ratio` | `(close - open) / (high - low)` | Candle body vs range |
| `upper_wick_ratio` | `(high - max(open, close)) / (high - low)` | Upper shadow ratio |
| `lower_wick_ratio` | `(min(open, close) - low) / (high - low)` | Lower shadow ratio |
| `is_bullish` | `close > open` | Bullish/bearish candle |

### 6.2 Candlestick Pattern Recognition

Currently **NOT** implemented (available via TA-Lib):

| Pattern | Function | Signal |
|---------|----------|--------|
| Hammer | `CDLHAMMER` | Bullish reversal |
| Engulfing | `CDLENGULFING` | Reversal |
| Doji | `CDLDOJI` | Indecision |
| Morning/Evening Star | `CDLMORNINGSTAR` | Reversal |

### 6.3 Order Flow / Market Microstructure

Not available in current setup:

| Feature | Description |
|---------|-------------|
| Order book imbalance | Bid/ask volume ratio |
| Trade flow | Buy vs sell volume |
| Spread | Bid-ask spread |

---

## 7. Data Flow Summary

```
┌─────────────────────────────────────────────────────────────────┐
│ Raw OHLCV Data (75 days × 288 candles/day = ~21,600 candles)    │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│ Feature Engineering                                              │
│                                                                  │
│   expand_all:   4 indicators × 48 expansion = 192 features       │
│   expand_basic: 12 indicators × 24 expansion = 288 features      │
│   standard:     8 features (no expansion)                        │
│                                                                  │
│   Total: 488 features per candle                                 │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│ Data Split (shuffle=False, chronological)                        │
│                                                                  │
│   Train: 75% (~16,200 candles)                                   │
│   Test:  25% (~5,400 candles)                                    │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│ Normalization (RobustScaler)                                     │
│                                                                  │
│   - Fit on train data only                                       │
│   - Transform both train and test                                │
│   - Range: approximately [-1, 1] after scaling                   │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│ RL Environment Observation                                       │
│                                                                  │
│   Shape: (conv_width=10, features=488)                           │
│   Flattened: 4,880 dimensions                                    │
│                                                                  │
│   Agent sees: Last 10 candles × 488 features each                │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│ MlpPolicy Neural Network                                         │
│                                                                  │
│   Input:  4,880 dimensions (flattened)                           │
│   Hidden: 256 → 256 (ReLU)                                       │
│   Output: 4 actions (Neutral, Exit, Long_enter, Short_enter)     │
└─────────────────────────────────────────────────────────────────┘
```

---

## 8. References

- Strategy: `user_data/strategies/RLDayTradeStrategy.py`
- Config: `user_data/config_daytrade.json`
- Model: `user_data/freqaimodels/RLDayTrader.py`
- Analysis: `docs/2026/20260203/rl-daytrade-strategy-analysis.md`
