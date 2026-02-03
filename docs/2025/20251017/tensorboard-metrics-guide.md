# FreqAI Reinforcement Learning Tensorboard 指標完全指南

**日期**: 2025-10-17  
**版本**: 1.0  
**適用**: Freqtrade FreqAI + Stable-Baselines3

---

## 目錄

1. [概述](#概述)
2. [Tensorboard 視圖類型](#tensorboard-視圖類型)
3. [SCALARS 視圖指標](#scalars-視圖指標)
   - 3.1 [actions 區塊 - 動作分佈](#31-actions-區塊---動作分佈)
   - 3.2 [eval 區塊 - 評估指標](#32-eval-區塊---評估指標)
   - 3.3 [rollout 區塊 - 訓練指標](#33-rollout-區塊---訓練指標)
   - 3.4 [train 區塊 - 訓練損失](#34-train-區塊---訓練損失)
   - 3.5 [info 區塊 - 環境狀態](#35-info-區塊---環境狀態)
4. [HPARAMS 視圖](#hparams-視圖)
5. [實際案例分析](#實際案例分析)
6. [常見問題與優化](#常見問題與優化)
7. [代碼示例](#代碼示例)

---

## 概述

FreqAI 的 Reinforcement Learning 模組使用 Stable-Baselines3 進行訓練，並透過 Tensorboard 提供即時監控。本指南詳細說明所有 Tensorboard 指標的含義、解讀方式及優化建議。

### 啟動 Tensorboard

```bash
# 在專案根目錄執行
tensorboard --logdir user_data/models/YOUR_IDENTIFIER

# 例如
tensorboard --logdir user_data/models/DQN_GPU_RTX2070
```

訪問: http://localhost:6006

---

## Tensorboard 視圖類型

| 視圖名稱 | 用途 | 說明 |
|---------|------|------|
| **TIME SERIES** | 時間序列監控 | 查看指標隨訓練時間的變化 |
| **SCALARS** | 標量指標 | 所有數值指標的主要視圖 |
| **HPARAMS** | 超參數對比 | 比較不同配置的訓練效果 |
| **IMAGES** | 圖像記錄 | 環境渲染圖(FreqAI 未使用) |
| **TEXT** | 文字記錄 | 自訂文字日誌 |

---

## SCALARS 視圖指標

### 3.1 actions 區塊 - 動作分佈

**來源**: `Base5ActionRLEnv.step()` 透過 `self.tensorboard_log()` 記錄

#### 📊 指標列表

| 指標名稱 | 動作編號 | 含義 | 理想趨勢 |
|---------|---------|------|---------|
| **Long_enter** | 1 | 做多進場次數 | 穩定在合理頻率 |
| **Long_exit** | 2 | 做多出場次數 | ≈ Long_enter (配對) |
| **Neutral** | 0 | 維持當前狀態次數 | 最多(避免過度交易) |
| **Short_enter** | 3 | 做空進場次數 | 與市場條件相符 |
| **Short_exit** | 4 | 做空出場次數 | ≈ Short_enter (配對) |
| **Invalid** | N/A | 無效動作次數 | 應趨近於 0 |

#### 🔍 解讀方式

**1. 動作頻率變化**
- **初期高峰**: 探索階段(exploration)，Agent 隨機嘗試各種動作
- **快速下降**: 策略收斂，Agent 學會避免無效動作
- **趨於穩定**: 策略成熟，動作選擇變得有目的性

**2. 進出配對**
```
Long_enter ≈ Long_exit  ✅ 正常
Short_enter ≈ Short_exit ✅ 正常

Long_enter >> Long_exit  ⚠️ 可能持續持多單未平倉
```

**3. Neutral 占比**
- **> 50%**: 保守策略，等待明確信號 ✅
- **< 20%**: 過度交易(overtrading) ⚠️
- **> 80%**: 過度保守，錯失機會 ⚠️

**4. Invalid 動作**
- **含義**: Agent 嘗試執行不合法的操作
- **範例**:
  - 已持多單時再次 `Long_enter`
  - 空倉時執行 `Long_exit`
  - 做多時嘗試 `Short_enter`
- **懲罰**: 通常返回 `-2` 獎勵
- **目標**: 應逐漸趨近於 0

#### 📈 實際案例

**觀察到的數據** (2025-10-17):
```
時間: 04:37 PM → 04:45 PM

Long_enter:  210 → 100 (下降 52%)
Long_exit:   200 → 110 (下降 45%)
Neutral:     800 → 500 (下降 38%)
Short_enter: 200 → 80  (下降 60%)
Short_exit:  210 → 110 (下降 48%)
Invalid:     420 → 180 (下降 57%)
```

**分析**:
- ✅ 所有動作頻率下降 → 策略收斂正常
- ✅ Neutral 占比最高 → 避免過度交易
- ⚠️ Invalid 仍有 180 次 → 需要改善環境設計或加大懲罰

---

### 3.2 eval 區塊 - 評估指標

**來源**: 定期評估階段(如果啟用)

#### 📊 指標列表

| 指標名稱 | 含義 | 單位 | 理想值 |
|---------|------|------|--------|
| **eval/mean_reward** | 評估時的平均 episode 總獎勵 | 標量 | 越高越好 |
| **eval/mean_ep_length** | 評估時的平均 episode 長度 | Candles | 取決於策略 |

#### 🔍 解讀方式

**1. mean_reward**
- **用途**: 衡量模型在測試環境中的表現
- **與訓練獎勵的區別**:
  - `eval/mean_reward`: 使用固定策略(不探索)
  - `rollout/ep_rew_mean`: 訓練時包含探索噪音

**2. mean_ep_length**
- **觀察到的值**: 約 2,150 candles
- **換算**: 2,150 × 5分鐘 = 10,750分鐘 ≈ **7.5 天**
- **解讀**: 與 config 的 `backtest_period_days: 7` 相符 ✅

#### 📈 實際案例

**觀察到的數據**:
```
mean_ep_length: 2.15e+3 (2,150 candles)
mean_reward: -2e+3 → 接近 0 (改善中)
```

**分析**:
- ✅ Episode 長度穩定
- ⚠️ Reward 從負轉正，但尚未盈利
- 📊 需要更多訓練週期

---

### 3.3 rollout 區塊 - 訓練指標

**來源**: Stable-Baselines3 自動記錄

#### 📊 指標列表

| 指標名稱 | 含義 | 理想趨勢 |
|---------|------|---------|
| **rollout/ep_rew_mean** | 訓練時的平均 episode 獎勵 | 穩定上升 |
| **rollout/ep_len_mean** | 訓練時的平均 episode 長度 | 穩定或適中 |

#### 🔍 解讀方式

**1. ep_rew_mean (Episode Reward Mean)**
- **含義**: 最近幾個 episode 的平均總獎勵
- **趨勢分析**:
  - 📈 **持續上升**: 學習有效 ✅
  - 📊 **波動但上升**: 正常(RL 固有特性)
  - 📉 **持續下降**: 可能過擬合或 reward function 問題 ⚠️
  - ➡️ **平穩**: 已收斂或陷入局部最優 ⚠️

**2. ep_len_mean (Episode Length Mean)**
- **短期交易** (< 100 candles): 頻繁進出
- **中期交易** (100-500 candles): 平衡策略
- **長期交易** (> 500 candles): 持倉為主或觸發 `max_trade_duration_candles`

---

### 3.4 train 區塊 - 訓練損失

**來源**: Stable-Baselines3 訓練過程

#### 📊 指標列表 (DQN)

| 指標名稱 | 含義 | 理想趨勢 |
|---------|------|---------|
| **train/loss** | Q-value 預測誤差 | 逐漸下降並穩定 |
| **train/learning_rate** | 當前學習率 | 根據 scheduler 調整 |

#### 📊 指標列表 (PPO)

| 指標名稱 | 含義 | 理想趨勢 |
|---------|------|---------|
| **train/policy_loss** | 策略網路損失 | 小幅波動 |
| **train/value_loss** | 價值網路損失 | 逐漸下降 |
| **train/explained_variance** | 價值函數解釋的方差比例 | 接近 1.0 |
| **train/approx_kl** | 近似 KL 散度 | < 0.05 |
| **train/clip_fraction** | 被裁剪的動作比例 | 0.1-0.3 |
| **train/entropy_loss** | 策略熵損失 | 逐漸下降 |

#### 🔍 解讀方式

**1. value_loss / loss**
- **含義**: 價值網路(或 Q 網路)對未來回報的預測誤差
- **理想曲線**:
  ```
  初期: 高值(模型隨機初始化)
  中期: 快速下降(學習中)
  後期: 穩定在低值(收斂)
  ```
- **異常情況**:
  - 持續高值 → 學習率過高或網路容量不足
  - 震蕩不穩 → Batch size 過小或數據不穩定

**2. explained_variance**
- **含義**: 價值函數解釋的回報方差比例
- **計算公式**:
  ```
  explained_var = 1 - Var(returns - values) / Var(returns)
  ```
- **解讀**:
  - **1.0**: 完美預測 ✅
  - **0.9-1.0**: 非常好 ✅
  - **0.5-0.9**: 良好 ✓
  - **< 0.5**: 模型欠擬合 ⚠️
  - **< 0**: 預測比均值還差 ❌

---

### 3.5 info 區塊 - 環境狀態

**來源**: `Base5ActionRLEnv.step()` 返回的 `info` dict

#### 📊 指標列表

| 指標名稱 | 含義 | 單位 | 來源代碼 |
|---------|------|------|---------|
| **TimeLimit.truncated** | Episode 是否被截斷 | 0/1 | Gymnasium wrapper |
| **action** | 當前選擇的動作編號 | 0-4 | `info['action']` |
| **current_profit_pct** | 當前持倉未實現盈虧 | % | `self.get_unrealized_profit()` |
| **position** | 當前持倉狀態 | 0/1/2 | `self._position.value` |
| **tick** | 當前時間步索引 | int | `self._current_tick` |
| **total_profit** | 累積已實現利潤 | 倍數 | `self._total_profit` |
| **total_reward** | 累積獎勵總和 | 標量 | `self.total_reward` |
| **trade_duration** | 當前交易持續時間 | Candles | `self.get_trade_duration()` |

#### 🔍 詳細說明

#### 1️⃣ **TimeLimit.truncated**

**含義**: Episode 是否因達到時間限制而截斷

**值**:
- `0`: 正常結束(達到數據終點或觸發終止條件)
- `0.8`: 達到 `max_trade_duration_candles` 被強制終止

**圖表解讀**:
```
持續在 0 附近 → 大部分 episode 正常結束 ✅
頻繁出現尖峰 → 經常觸發時間限制 ⚠️
```

---

#### 2️⃣ **action**

**含義**: Agent 當前選擇的動作編號

**值範圍**: 0-4
- `0`: Neutral
- `1`: Long_enter
- `2`: Long_exit
- `3`: Short_enter
- `4`: Short_exit

**實際案例**:
```
觀察值: 2.0 → 1.1
解讀: 
- 初期偏向 Long_exit (2)
- 後期趨向 Long_enter (1) 和 Neutral 之間
- 策略逐漸偏向做多
```

**異常情況**:
- 值超出 0-4 → 環境錯誤 ❌
- 持續某單一值 → 策略過於簡單 ⚠️

---

#### 3️⃣ **current_profit_pct** ⚠️

**含義**: **當前持倉**的未實現盈虧百分比

**計算方式** (BaseEnvironment.py):
```python
if self._position == Positions.Long:
    current_price = self.add_exit_fee(self.prices.iloc[self._current_tick].open)
    last_trade_price = self.add_entry_fee(self.prices.iloc[self._last_trade_tick].open)
    return (current_price - last_trade_price) / last_trade_price

if self._position == Positions.Short:
    current_price = self.add_entry_fee(self.prices.iloc[self._current_tick].open)
    last_trade_price = self.add_exit_fee(self.prices.iloc[self._last_trade_tick].open)
    return (last_trade_price - current_price) / last_trade_price
```

**實際案例**:
```
觀察值: -3e-3 到 -7e-3 (-0.3% 到 -0.7%)
問題: 持續負值表示當前持倉虧損 ❌
```

**可能原因**:
1. Agent 進場時機不佳
2. 止損未觸發,虧損擴大
3. Reward function 進場獎勵設計不當

**優化方向**:
- 增加進場條件(如 RSI < 30 才給獎勵)
- 加入動態止損機制
- 調整 `model_reward_parameters.profit_aim`

---

#### 4️⃣ **position**

**含義**: 當前持倉狀態

**值編碼** (BaseEnvironment.py):
```python
class Positions(Enum):
    Neutral = 0  # 空倉
    Long = 1     # 做多
    Short = 2    # 做空
```

**實際案例**:
```
觀察值: 0.44 → 0.48
解讀:
- 平均約 48% 時間處於持倉狀態
- 52% 時間空倉觀望
- 相對保守的策略 ✅
```

**理想範圍**:
- **30-50%**: 保守策略,適合波動市場
- **50-70%**: 積極策略,適合趨勢市場
- **> 80%**: 過度持倉,風險高 ⚠️

---

#### 5️⃣ **tick**

**含義**: 當前環境時間步(candle 索引)

**實際案例**:
```
觀察值: 750 → 1250
計算: 1250 × 5分鐘 = 6,250分鐘 ≈ 4.3 天
```

**用途**:
- 追蹤訓練進度
- 計算訓練速度(ticks/second)
- 偵錯環境重置問題

---

#### 6️⃣ **total_profit** ✅

**含義**: **累積已實現**利潤(倍數,起始值 1.0)

**計算方式** (BaseEnvironment.py):
```python
def _update_total_profit(self):
    pnl = self.get_unrealized_profit()
    if self.compound_trades:
        # 複利模式
        self._total_profit = self._total_profit * (1 + pnl)
    else:
        # 單利模式
        self._total_profit += pnl
```

**實際案例**:
```
觀察值: 0.988 → 1.01+
解讀:
- 初期: 虧損 1.2% (0.988)
- 後期: 盈利 1%+ (1.01)
- 策略正在改善 ✅
```

**解讀標準**:
- `> 1.0`: 盈利 ✅
- `= 1.0`: 損益平衡
- `< 1.0`: 虧損 ⚠️
- `< 0.95`: 嚴重虧損 ❌

---

#### 7️⃣ **total_reward** ⚠️

**含義**: 累積獎勵總和(每步 `step_reward` 累加)

**計算方式**:
```python
self.total_reward += step_reward
```

**實際案例**:
```
觀察值: 2.4e+3 → 400 (下降 83%)
矛盾: total_profit 上升,但 total_reward 下降 ❌
```

**問題分析**:

典型 reward function 設計缺陷:
```python
def calculate_reward(self, action: int) -> float:
    if action in (Actions.Long_enter, Actions.Short_enter):
        return 25 * factor  # 🚫 固定進場獎勵過高
    
    if action in (Actions.Long_exit, Actions.Short_exit):
        pnl = self.get_unrealized_profit()
        return float(pnl * factor)  # ✅ 根據 PnL 獎勵
```

**為什麼 total_reward 下降?**
1. **初期**: 頻繁進場 → 大量進場獎勵 (25 × 次數)
2. **後期**: 策略收斂 → 進場次數減少 → total_reward 下降
3. **但**: 交易品質提升 → total_profit 上升

**解決方案**: Reward function 應關注 **實際盈利** 而非動作頻率(見下方代碼示例)

---

#### 8️⃣ **trade_duration**

**含義**: 當前交易持續的 candle 數量

**計算方式** (BaseEnvironment.py):
```python
def get_trade_duration(self):
    if self._last_trade_tick is None:
        return 0
    else:
        return self._current_tick - self._last_trade_tick
```

**實際案例**:
```
觀察值: 0 → 20
換算: 20 × 5分鐘 = 100分鐘 ≈ 1.7小時
對比: max_trade_duration_candles = 300 (25小時)
結論: 屬於短線交易策略 ✅
```

**策略類型判斷**:
- **< 50 candles**: 超短線(scalping)
- **50-200 candles**: 短線
- **200-500 candles**: 中線
- **> 500 candles**: 長線或觸發 max 限制

---

## HPARAMS 視圖

### 4.1 概述

**用途**: 比較不同超參數配置的訓練效果

**視圖組成**:
1. **TABLE VIEW**: 表格列出所有訓練運行
2. **PARALLEL COORDINATES VIEW**: 視覺化超參數與指標的關係
3. **SCATTER PLOT MATRIX VIEW**: 散點圖矩陣

### 4.2 啟用方式

**問題**: FreqAI 的 ReinforcementLearner **預設不記錄 HPARAMS**

**原因**: FreqAI 只記錄 SCALARS,需要手動添加 `HParamCallback`

**解決方案**: 見下方代碼示例

### 4.3 實際案例

**觀察到的狀態**:
```
Trial ID: tensorboard\BTC\DQN_1, tensorboard\BTC\DQN_2
Hyperparameters: algorithm=DQN, learning_rate=0.00010000
Metrics: 全部空白 ❌
```

**替代方案** (推薦):
1. 使用 identifier 命名包含超參數:
   ```json
   "identifier": "DQN_LR0001_BS512_ARCH512x3"
   ```

2. 在 TIME SERIES 視圖手動選擇多個運行對比

3. 手動記錄結果表格

---

## 實際案例分析

### 案例 1: 動作分佈分析 (2025-10-17)

**訓練時間**: 04:37 PM - 04:45 PM (8分鐘)

**觀察結果**:

| 動作 | 初期 | 後期 | 變化 | 分析 |
|-----|------|------|------|------|
| Long_enter | 210 | 100 | -52% | 進場更謹慎 ✅ |
| Long_exit | 200 | 110 | -45% | 與進場配對 ✅ |
| Neutral | 800 | 500 | -38% | 仍占最多 ✅ |
| Short_enter | 200 | 80 | -60% | 減少做空 ⚠️ |
| Short_exit | 210 | 110 | -48% | 配對正常 ✅ |
| Invalid | 420 | 180 | -57% | 仍需改善 ⚠️ |

**結論**:
1. ✅ 策略收斂正常(所有動作頻率下降)
2. ✅ Neutral 占比最高(避免過度交易)
3. ⚠️ 做空明顯少於做多(可能訓練期為牛市)
4. ⚠️ Invalid 動作仍有 180 次(需優化)

---

### 案例 2: 狀態信息分析

**關鍵發現**:

```
total_profit: 0.988 → 1.01+  ✅ 策略改善,開始獲利
total_reward: 2400 → 400     ❌ 獎勵下降(設計缺陷)
current_profit_pct: -0.3% → -0.7%  ❌ 當前持倉虧損
position: 0.44 → 0.48        ✅ 持倉比例合理
trade_duration: 0 → 20       ✅ 短線交易
```

**矛盾分析**:
- **total_profit 上升**: 已實現交易獲利 ✅
- **total_reward 下降**: Reward function 過度獎勵進場動作 ❌
- **current_profit_pct 負值**: 當前持倉虧損,未止損 ⚠️

**優化方向**:
1. 修改 reward function(移除固定進場獎勵)
2. 加入止損機制
3. 增加訓練週期至 100+

---

## 常見問題與優化

### 問題 1: Invalid 動作過多

**症狀**: `actions/Invalid` 指標持續高值

**原因**:
1. 環境狀態轉移邏輯有誤
2. Agent 未學會動作約束
3. 懲罰力度不足

**解決方案**:
```python
def calculate_reward(self, action: int) -> float:
    if not self._is_valid(action):
        self.tensorboard_log("invalid")
        return -5  # 增加懲罰(原本 -2)
```

---

### 問題 2: total_reward 與 total_profit 不一致

**症狀**: Profit 上升但 Reward 下降

**原因**: Reward function 過度獎勵動作頻率而非實際盈利

**解決方案**: 見下方代碼示例

---

### 問題 3: current_profit_pct 持續負值

**症狀**: 當前持倉長期虧損

**原因**:
1. 進場時機不佳
2. 缺乏止損機制
3. 持倉時間過長

**解決方案**:
```python
def calculate_reward(self, action: int) -> float:
    # 動態止損
    if self._position != Positions.Neutral:
        current_profit = self.get_unrealized_profit()
        if current_profit < -0.02:  # 虧損超過 2%
            return -50  # 強力懲罰
```

---

### 問題 4: 策略偏向做多/做空

**症狀**: Long_enter >> Short_enter 或相反

**原因**: 訓練數據集的市場方向偏向

**解決方案**:
1. 使用包含牛市+熊市的訓練數據
2. 在 reward function 中平衡多空激勵
3. 添加市場狀態特徵(趨勢方向)

---

## 代碼示例

### 示例 1: 優化的 Reward Function

```python
from freqtrade.freqai.RL.Base5ActionRLEnv import Actions, Base5ActionRLEnv, Positions


class OptimizedRLEnv(Base5ActionRLEnv):
    """
    優化的 RL 環境,改善 reward function 設計
    """
    
    def calculate_reward(self, action: int) -> float:
        """
        基於實際盈利的 reward function
        移除固定進場獎勵,專注於交易結果
        """
        
        # 1. 嚴格懲罰無效動作
        if not self._is_valid(action):
            self.tensorboard_log("invalid")
            return -5  # 從 -2 增加到 -5
        
        # 2. 輕微懲罰過度中立(鼓勵適度交易)
        if action == Actions.Neutral.value and self._position == Positions.Neutral:
            return -0.1
        
        # 3. 根據持倉時間調整獎勵
        trade_duration = self.get_trade_duration()
        max_duration = self.rl_config.get('max_trade_duration_candles', 300)
        
        # 持倉時間過長的懲罰
        if trade_duration > max_duration * 0.8:  # 超過 80% 上限
            return -0.5
        
        # 4. 出場時根據實際 PnL 給獎勵
        if action in (Actions.Long_exit.value, Actions.Short_exit.value):
            pnl = self.get_unrealized_profit()
            
            # 獲利交易
            if pnl > 0:
                # 根據獲利幅度和持倉時間調整獎勵
                time_factor = 1.5 if trade_duration < max_duration * 0.5 else 1.0
                reward = pnl * 100 * time_factor
                
                # 超額獲利額外獎勵
                profit_aim = self.rl_config['model_reward_parameters'].get('profit_aim', 0.025)
                if pnl > profit_aim:
                    win_factor = self.rl_config['model_reward_parameters'].get('win_reward_factor', 2)
                    reward *= win_factor
                
                return reward
            
            # 虧損交易
            else:
                # 小虧損適度懲罰
                if pnl > -0.01:  # 虧損 < 1%
                    return pnl * 50
                # 大虧損嚴重懲罰
                else:
                    return pnl * 100
        
        # 5. 進場時不給固定獎勵,只檢查條件
        if action in (Actions.Long_enter.value, Actions.Short_enter.value):
            # 可選: 根據技術指標給予小獎勵
            # rsi = self.raw_features[f"%-rsi-period_10_shift-1_{pair}_{tf}"].iloc[self._current_tick]
            # if (action == Actions.Long_enter.value and rsi < 30) or \
            #    (action == Actions.Short_enter.value and rsi > 70):
            #     return 1  # 小獎勵鼓勵好的進場時機
            return 0  # 進場本身不獎勵
        
        # 6. 持倉中保持中立給予小獎勵(鼓勵持倉耐心)
        if action == Actions.Neutral.value and self._position != Positions.Neutral:
            current_profit = self.get_unrealized_profit()
            if current_profit > 0:
                return 0.1  # 獲利中持倉給小獎勵
            else:
                return -0.1  # 虧損中持倉給小懲罰
        
        return 0
```

---

### 示例 2: 啟用 HPARAMS 記錄 (進階)

```python
from freqtrade.freqai.prediction_models.ReinforcementLearner import ReinforcementLearner
from stable_baselines3.common.logger import HParam
from stable_baselines3.common.callbacks import BaseCallback


class HParamCallback(BaseCallback):
    """
    記錄超參數到 Tensorboard HPARAMS 視圖
    """
    
    def _on_training_start(self) -> None:
        """在訓練開始時記錄超參數"""
        
        # 定義要追蹤的超參數
        hparam_dict = {
            "algorithm": self.model.__class__.__name__,
            "learning_rate": float(self.model.learning_rate),
            "batch_size": getattr(self.model, 'batch_size', 'N/A'),
            "buffer_size": getattr(self.model, 'buffer_size', 'N/A'),
            "gamma": getattr(self.model, 'gamma', 'N/A'),
            "train_cycles": self.model.env.get_attr('train_cycles')[0] if hasattr(self.model.env, 'get_attr') else 'N/A',
        }
        
        # 定義要追蹤的指標
        metric_dict = {
            "eval/mean_reward": 0.0,
            "rollout/ep_rew_mean": 0.0,
            "rollout/ep_len_mean": 0.0,
            "train/value_loss": 0.0,
            "train/explained_variance": 0.0,
        }
        
        # 記錄到 Tensorboard
        self.logger.record(
            "hparams",
            HParam(hparam_dict, metric_dict),
            exclude=("stdout", "log", "json", "csv"),
        )
    
    def _on_step(self) -> bool:
        return True


class CustomRLModel(ReinforcementLearner):
    """
    自訂 RL 模型,啟用 HPARAMS 記錄
    
    使用方式:
    freqtrade backtesting --freqaimodel CustomRLModel --config config.json --strategy RLStrategy
    """
    
    # 使用上面定義的 OptimizedRLEnv
    class MyRLEnv(OptimizedRLEnv):
        pass
```

---

### 示例 3: 動態止損機制

```python
class StopLossRLEnv(Base5ActionRLEnv):
    """
    帶有動態止損的 RL 環境
    """
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.stop_loss_pct = 0.02  # 2% 止損
        self.trailing_stop = True   # 啟用移動止損
        self.max_profit_seen = 0    # 追蹤最高利潤
    
    def reset(self, seed=None):
        self.max_profit_seen = 0
        return super().reset(seed)
    
    def calculate_reward(self, action: int) -> float:
        """帶止損邏輯的 reward function"""
        
        if not self._is_valid(action):
            return -5
        
        # 檢查止損
        if self._position != Positions.Neutral:
            current_profit = self.get_unrealized_profit()
            
            # 更新最高利潤
            if current_profit > self.max_profit_seen:
                self.max_profit_seen = current_profit
            
            # 固定止損
            if current_profit < -self.stop_loss_pct:
                self.tensorboard_log("stop_loss_triggered", category="events")
                # 強制平倉
                if self._position == Positions.Long:
                    action = Actions.Long_exit.value
                elif self._position == Positions.Short:
                    action = Actions.Short_exit.value
                return current_profit * 100  # 執行止損的獎勵
            
            # 移動止損
            if self.trailing_stop and self.max_profit_seen > 0.01:  # 利潤超過 1%
                trailing_stop_level = self.max_profit_seen * 0.5  # 回撤 50% 止盈
                if current_profit < trailing_stop_level:
                    self.tensorboard_log("trailing_stop_triggered", category="events")
                    # 執行止盈
                    if self._position == Positions.Long:
                        action = Actions.Long_exit.value
                    elif self._position == Positions.Short:
                        action = Actions.Short_exit.value
                    return current_profit * 150  # 止盈獲得更高獎勵
        
        # 其他邏輯與 OptimizedRLEnv 相同
        # ...
        return 0
```

---

## 總結

### 關鍵指標速查表

| 指標 | 位置 | 主要用途 | 理想值 |
|------|------|---------|--------|
| `actions/Long_enter` | SCALARS | 進場次數 | 與市場機會相符 |
| `actions/Neutral` | SCALARS | 觀望次數 | 應為最多 |
| `actions/Invalid` | SCALARS | 無效動作 | 趨近於 0 |
| `eval/mean_reward` | SCALARS | 評估表現 | 持續上升 |
| `rollout/ep_rew_mean` | SCALARS | 訓練表現 | 穩定上升 |
| `train/value_loss` | SCALARS | 價值預測誤差 | 下降並穩定 |
| `train/explained_variance` | SCALARS | 預測品質 | > 0.9 |
| `info/total_profit` | SCALARS | 累積利潤 | > 1.0 |
| `info/current_profit_pct` | SCALARS | 當前持倉盈虧 | 正值或小負值 |
| `info/position` | SCALARS | 持倉比例 | 30-70% |
| `info/trade_duration` | SCALARS | 平均持倉時間 | 取決於策略 |

---

### 優化檢查清單

訓練開始前:
- [ ] 下載足夠的歷史數據(訓練期 + startup_candles)
- [ ] 設定合理的 `train_cycles` (建議 100+)
- [ ] 檢查 `max_trade_duration_candles` 與策略相符
- [ ] 確認 `profit_aim` 與市場波動率匹配

訓練過程中:
- [ ] `rollout/ep_rew_mean` 穩定上升
- [ ] `actions/Invalid` 逐漸減少
- [ ] `train/explained_variance` > 0.5
- [ ] `info/total_profit` 趨向 > 1.0

訓練結束後:
- [ ] Backtest 驗證結果
- [ ] 檢查 Sharpe Ratio 和 Max Drawdown
- [ ] 對比不同配置的訓練運行
- [ ] 記錄最佳超參數組合

---

**文檔版本**: 1.0  
**最後更新**: 2025-10-17  
**作者**: FreqAI RL 訓練團隊  
**參考**: 
- Freqtrade Documentation: https://www.freqtrade.io/en/stable/freqai-reinforcement-learning/
- Stable-Baselines3 Documentation: https://stable-baselines3.readthedocs.io/
- Tensorboard Guide: https://www.tensorflow.org/tensorboard/get_started
