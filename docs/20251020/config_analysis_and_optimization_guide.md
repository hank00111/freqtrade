# FreqAI DQN 配置分析與優化指南

**建立日期：** 2025-10-20  
**配置檔案：** `user_data/config_dqn_gpu.json`  
**模型識別碼：** DQN_GPU_RTX2070  
**訓練策略：** RLStrategy4Action + ReinforcementLearner4Action

---

## 📊 當前配置特性分析

### 1️⃣ 交易模式配置

#### 基本設定
| 參數 | 數值 | 特性 | 風險等級 |
|------|------|------|----------|
| `trading_mode` | futures | 期貨交易（支援做多做空） | 🟡 中高 |
| `margin_mode` | isolated | 隔離保證金 | 🟢 較安全 |
| `max_open_trades` | 8 | 同時最多 8 個倉位 | 🟢 適中 |
| `stake_amount` | unlimited | 無限制投入 | 🔴 極高 |
| `tradable_balance_ratio` | 0.99 | 使用 99% 可用資金 | 🔴 極高 |
| `timeframe` | 5m | 5 分鐘 K 線 | 🟡 高頻 |

**風險評估：**
- ✅ **隔離保證金**：單個倉位爆倉不會影響其他倉位
- ⚠️ **極度激進**：99% 資金 + unlimited stake = 資金利用率接近 100%
- ⚠️ **高頻交易**：5 分鐘週期，每天最多 288 根 K 線

---

### 2️⃣ FreqAI 訓練配置

#### 訓練週期
| 參數 | 數值 | 相較預設 | 影響 |
|------|------|----------|------|
| `train_period_days` | 45 天 | +50% (預設 30) | 訓練資料更多，模型更穩定 |
| `backtest_period_days` | 7 天 | 標準 | 回測視窗適中 |
| `train_cycles` | 50 | +67% (預設 30) | 訓練更徹底，但時間更長 |
| `purge_old_models` | 3 | 標準 | 保留最近 3 個模型 |

**時間成本計算：**
```
146 訓練視窗 × 50 週期 × 2 交易對 = 約 14,600 次訓練迭代
預估時間：40-60 小時（使用 RTX 2070）
```

#### 特徵工程
| 參數 | 數值 | 說明 |
|------|------|------|
| `include_shifted_candles` | 5 | 包含前 5 根 K 線的移位特徵 |
| `label_period_candles` | 20 | 標籤週期 20 根 K 線 |
| `indicator_periods_candles` | [10, 20] | 使用 10 和 20 週期指標 |
| `principal_component_analysis` | false | 不使用 PCA 降維 |

**特徵數量：112 features**

---

### 3️⃣ DQN 模型配置

#### 核心參數
| 參數 | 數值 | 相較預設 | 特性 |
|------|------|----------|------|
| `model_type` | DQN | - | Deep Q-Network（深度 Q 學習） |
| `learning_rate` | 0.00025 | +150% (預設 0.0001) | **更快學習，但可能不穩定** |
| `batch_size` | 512 | 標準/偏大 | 訓練穩定，但記憶體需求高 |
| `buffer_size` | 1,000,000 | 極大 | **超大回放記憶體，樣本多樣性高** |
| `learning_starts` | 10,000 | 標準 | 收集 10k 樣本後開始學習 |
| `device` | cuda | GPU 加速 | **使用 RTX 2070 GPU** |

#### 神經網路架構
```python
net_arch: [512, 512, 256]
```
- **3 層深度網路**
- 輸入層 → 512 神經元 → 512 神經元 → 256 神經元 → 輸出層
- **高模型容量**：適合複雜模式學習

#### 探索策略
| 參數 | 數值 | 說明 |
|------|------|------|
| `exploration_initial_eps` | 1.0 | 初始 100% 隨機探索 |
| `exploration_final_eps` | 0.05 | 最終 5% 隨機探索 |
| `exploration_fraction` | 0.3 | 前 30% 訓練進行探索衰減 |

**探索-利用平衡：**
```
訓練開始 → 100% 隨機
30% 訓練後 → 5% 隨機（主要使用學習策略）
```

---

### 4️⃣ 強化學習獎勵機制

#### 獎勵參數
| 參數 | 數值 | 影響 |
|------|------|------|
| `profit_aim` | 0.025 | 目標獲利 **2.5%** |
| `win_reward_factor` | 2.0 | 獲利交易獎勵 **× 2** |
| `rr` | 1 | 風險報酬比 1:1 |
| `max_training_drawdown_pct` | 0.02 | 訓練時最大回撤 **2%** |

**獎勵特性：**
- ✅ **偏好獲利交易**：勝率獎勵加倍
- ✅ **保守回撤限制**：2% 回撤上限避免過度冒險
- ⚠️ **2.5% 獲利目標**：對 5 分鐘週期來說較高

---

### 5️⃣ 硬體資源使用

#### GPU 設定
```json
"device": "cuda"
"model_type": "DQN"
```

**RTX 2070 (8GB) 資源評估：**
| 項目 | 預估使用 | 狀態 |
|------|---------|------|
| GPU 記憶體 | 4-6 GB | ✅ 充足 |
| Batch Size | 512 | ✅ 適中 |
| Network Size | [512,512,256] | ✅ 可負載 |
| Buffer Size | 1M | 🟡 主要在 RAM |

---

### 6️⃣ 配置特性總結

#### 🎯 整體定位
```
激進型交易 + 保守型訓練
```

#### ✅ 優點
1. **GPU 加速訓練**：充分利用硬體
2. **大型回放記憶體**：樣本多樣性高，避免過擬合
3. **深度網路**：高模型容量，能學習複雜模式
4. **較長訓練週期**：45 天訓練窗口，模型更穩定
5. **回撤保護**：2% 訓練回撤限制防止學習過激策略

#### ⚠️ 風險點
1. **資金利用率極高**：99% + unlimited = 幾乎無緩衝
2. **高頻交易**：5 分鐘週期，交易頻繁，手續費成本高
3. **高學習率**：0.00025 可能導致訓練不穩定
4. **訓練時間長**：50 週期 × 146 視窗 = 40-60 小時
5. **無持續學習**：`continual_learning: false`，每次重新訓練

---

## 🔧 訓練效果不佳時的參數調整指南

### 📊 步驟 1：診斷問題類型

#### 使用 Tensorboard 監控關鍵指標

```powershell
tensorboard --logdir user_data/models/DQN_GPU_RTX2070
```

**重點觀察指標：**
| 指標 | 正常表現 | 異常表現 |
|------|---------|---------|
| `ep_rew_mean` | 逐漸上升 | 持平或下降 |
| `loss` | 逐漸下降並穩定 | 劇烈震盪或不變 |
| `value_loss` | 逐漸下降 | 爆炸性增長 |
| `ep_len_mean` | 適度增長 | 過短或過長 |

---

### 🎯 常見問題與解決方案

#### 問題 1️⃣：訓練不穩定（Loss 劇烈震盪）

**症狀：**
- Tensorboard 顯示 loss 波動極大
- 獎勵曲線上下跳動
- 訓練日誌出現 NaN

**原因：** 學習率過高 + 梯度更新過於頻繁

**調整方案：**
```json
{
  "model_training_parameters": {
    "learning_rate": 0.0001,           // 從 0.00025 降至 0.0001
    "batch_size": 1024,                // 從 512 增至 1024
    "target_update_interval": 1000,    // 從 500 增至 1000
    "gradient_steps": 2,               // 從 4 降至 2
    "train_freq": 8                    // 從 4 增至 8
  }
}
```

**效果：** 訓練更穩定，但學習速度稍慢

---

#### 問題 2️⃣：模型不學習（Loss 和獎勵持平）

**症狀：**
- `ep_rew_mean` 長時間不變
- Loss 曲線平坦
- Backtest 結果接近隨機

**原因：** 學習率過低 / 探索不足 / 訓練週期太少

**調整方案：**
```json
{
  "model_training_parameters": {
    "learning_rate": 0.0005,           // 從 0.00025 增至 0.0005
    "batch_size": 256,                 // 從 512 降至 256（更頻繁更新）
    "exploration_fraction": 0.5,       // 從 0.3 增至 0.5（更長探索期）
    "exploration_final_eps": 0.1       // 從 0.05 增至 0.1（更多探索）
  },
  "rl_config": {
    "train_cycles": 100                // 從 50 增至 100
  }
}
```

**效果：** 學習更快，但可能不穩定

---

#### 問題 3️⃣：過擬合（訓練好，回測差）

**症狀：**
- 訓練獎勵很高
- Backtest 結果差
- 驗證集表現遠低於訓練集

**原因：** 模型過於複雜 / 訓練週期過多 / 訓練資料太少

**調整方案：**
```json
{
  "freqai": {
    "train_period_days": 60,           // 從 45 增至 60（更多訓練資料）
    "feature_parameters": {
      "include_shifted_candles": 2,    // 從 5 降至 2（減少特徵）
      "shuffle_after_split": true      // 啟用打亂（增加泛化）
    }
  },
  "rl_config": {
    "train_cycles": 30,                // 從 50 降至 30
    "net_arch": [256, 256, 128],       // 從 [512,512,256] 簡化
    "max_training_drawdown_pct": 0.05  // 從 0.02 放寬至 0.05
  }
}
```

**效果：** 泛化能力更強

---

#### 問題 4️⃣：欠擬合（訓練和回測都差）

**症狀：**
- 訓練獎勵很低
- Backtest 虧損
- 模型表現不如簡單策略

**原因：** 模型容量不足 / 特徵不足 / 學習時間太短

**調整方案：**
```json
{
  "freqai": {
    "feature_parameters": {
      "include_shifted_candles": 10,   // 從 5 增至 10
      "include_timeframes": ["5m", "15m"], // 增加時間框架
      "indicator_periods_candles": [10, 20, 50] // 增加週期
    }
  },
  "rl_config": {
    "train_cycles": 100,               // 從 50 增至 100
    "net_arch": [1024, 512, 256],      // 從 [512,512,256] 增大
    "learning_rate": 0.0005            // 提高學習率
  }
}
```

**效果：** 模型容量更大，學習更充分

---

#### 問題 5️⃣：獎勵機制問題

##### 症狀（太保守）：
- 很少開倉
- `ep_len_mean` 很短
- 大部分時間都是 Neutral

**調整方案：**
```json
{
  "rl_config": {
    "model_reward_parameters": {
      "profit_aim": 0.015,             // 從 0.025 降至 0.015（更容易達成）
      "win_reward_factor": 3.0,        // 從 2.0 增至 3.0（更獎勵交易）
      "rr": 0.5                        // 降低風險要求
    },
    "max_training_drawdown_pct": 0.05  // 從 0.02 放寬
  }
}
```

##### 症狀（太激進）：
- 頻繁開倉
- 大量虧損交易
- 回撤過大

**調整方案：**
```json
{
  "rl_config": {
    "model_reward_parameters": {
      "profit_aim": 0.03,              // 從 0.025 增至 0.03
      "win_reward_factor": 1.5,        // 從 2.0 降至 1.5
      "rr": 2.0                        // 提高風險要求
    },
    "max_training_drawdown_pct": 0.01, // 從 0.02 收緊
    "max_trade_duration_candles": 150  // 從 300 減半
  }
}
```

---

#### 問題 6️⃣：訓練時間過長

**症狀：**
- 40+ 小時還沒完成
- 需要快速迭代測試

**快速測試配置：**
```json
{
  "freqai": {
    "train_period_days": 30,           // 從 45 降至 30
    "backtest_period_days": 5          // 從 7 降至 5
  },
  "rl_config": {
    "train_cycles": 15,                // 從 50 降至 15
    "net_arch": [256, 128]             // 簡化網路
  },
  "model_training_parameters": {
    "batch_size": 256,                 // 從 512 降至 256
    "buffer_size": 100000              // 從 1000000 降至 100000
  }
}
```

**效果：** 訓練時間降至 8-12 小時，但效果可能較差

---

### 🔄 系統化調整流程

#### 階段 1：快速診斷（3-5 小時）
```json
// 使用最小配置快速測試
{
  "train_period_days": 15,
  "train_cycles": 10,
  "net_arch": [128, 64]
}
```
**目的：** 驗證資料、策略、環境是否正確

#### 階段 2：基準測試（12-24 小時）
```json
// 使用標準配置
{
  "train_period_days": 30,
  "train_cycles": 25,
  "learning_rate": 0.0001,
  "net_arch": [256, 256]
}
```
**目的：** 建立基準效能

#### 階段 3：精細調整（40+ 小時）
```json
// 使用當前配置或優化版
{
  "train_period_days": 45,
  "train_cycles": 50,
  "learning_rate": 0.00025,  // 根據階段 2 結果調整
  "net_arch": [512, 512, 256]
}
```
**目的：** 最終模型訓練

---

### 📈 參數調整優先順序

#### 第一優先（影響最大）：
1. `learning_rate` - 學習率
2. `train_cycles` - 訓練週期數
3. `profit_aim` - 獲利目標

#### 第二優先（穩定性）：
4. `batch_size` - 批次大小
5. `target_update_interval` - 目標網路更新頻率
6. `max_training_drawdown_pct` - 最大回撤

#### 第三優先（效能優化）：
7. `net_arch` - 網路架構
8. `train_period_days` - 訓練週期長度
9. `exploration_fraction` - 探索比例

---

## 💡 實用調整範本

### 保守穩定型（適合新手）
```json
{
  "model_training_parameters": {
    "learning_rate": 0.0001,
    "batch_size": 1024
  },
  "rl_config": {
    "train_cycles": 30,
    "net_arch": [256, 256],
    "max_training_drawdown_pct": 0.01,
    "model_reward_parameters": {
      "profit_aim": 0.02,
      "win_reward_factor": 1.5
    }
  }
}
```

### 激進學習型（快速迭代）
```json
{
  "model_training_parameters": {
    "learning_rate": 0.0005,
    "batch_size": 256
  },
  "rl_config": {
    "train_cycles": 100,
    "net_arch": [1024, 512, 256],
    "model_reward_parameters": {
      "profit_aim": 0.03,
      "win_reward_factor": 3.0
    }
  }
}
```

### 平衡型（推薦起點）
```json
{
  "model_training_parameters": {
    "learning_rate": 0.00015,
    "batch_size": 512,
    "target_update_interval": 750
  },
  "rl_config": {
    "train_cycles": 40,
    "net_arch": [384, 256, 128],
    "max_training_drawdown_pct": 0.03,
    "model_reward_parameters": {
      "profit_aim": 0.02,
      "win_reward_factor": 2.0
    }
  }
}
```

---

## 🎯 調整建議總結表

| 問題 | 主要調整 | 次要調整 |
|------|---------|---------|
| 訓練不穩定 | ↓ learning_rate | ↑ batch_size, ↑ target_update |
| 不學習 | ↑ learning_rate, ↑ train_cycles | ↓ batch_size |
| 過擬合 | ↓ net_arch, ↑ train_period | ↓ train_cycles |
| 欠擬合 | ↑ net_arch, ↑ train_cycles | ↑ features |
| 太保守 | ↓ profit_aim, ↑ win_reward | ↑ max_drawdown |
| 太激進 | ↑ profit_aim, ↓ win_reward | ↓ max_drawdown |

---

## 📝 重要提醒

1. **每次只調整 1-2 個參數**，觀察效果後再繼續調整
2. **使用 Tensorboard 監控訓練過程**：`tensorboard --logdir user_data/models/DQN_GPU_RTX2070`
3. **保存每次調整的配置和結果**，建立調整日誌
4. **實盤前務必降低 `tradable_balance_ratio`**，建議 0.3-0.5
5. **優先使用保守配置**，確認穩定後再逐步優化

---

## 📚 相關文件

- 配置檔案：`user_data/config_dqn_gpu.json`
- 策略檔案：`user_data/strategies/RLStrategy4Action.py`
- 模型檔案：`user_data/freqaimodels/ReinforcementLearner4Action.py`
- 訓練日誌：`user_data/logs/freqtrade.log`
- Tensorboard：`user_data/models/DQN_GPU_RTX2070/tensorboard/`

---

**最後更新：** 2025-10-20  
**訓練狀態：** 進行中（預計 40-60 小時）  
**資料範圍：** 2022-11-01 至 2025-10-19
