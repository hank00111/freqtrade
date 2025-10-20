# FreqAI 模型評估指南：`continual_learning: false` 的驗證方法

**建立日期：** 2025-10-20  
**配置檔案：** `user_data/config_dqn_gpu.json`  
**訓練模式：** Walk-Forward Validation (146 獨立視窗)  
**適用場景：** DQN 模型訓練完成後的品質評估

---

## 🎯 理解 `continual_learning: false` 的優勢

### 為什麼不開啟持續學習？

| 特性 | continual_learning: false | continual_learning: true |
|------|---------------------------|--------------------------|
| 訓練方式 | 每個視窗獨立訓練 | 基於前一個模型繼續訓練 |
| 模型品質 | 每個窗口都是全新學習 | 可能累積錯誤 |
| 驗證可靠性 | ✅ **更可靠**（真正的 walk-forward） | 可能有資訊洩漏 |
| 適應性 | 對市場變化更敏感 | 可能過度適應舊模式 |
| 災難性遺忘 | ✅ 不會發生 | 可能發生 |
| 評估難度 | 容易（每個視窗獨立） | 較難（需考慮累積效應）|

### 關鍵優勢

**你的配置（146 個訓練視窗）實際上是 146 次獨立驗證！**

```
每個視窗：
1. 使用 45 天歷史資料訓練全新模型
2. 在接下來 7 天進行回測驗證
3. 模型間完全獨立，無資訊洩漏
4. 真實模擬實盤部署情境
```

---

## 📊 評估階段 1：訓練期間監控（即時）

### 1. Tensorboard 指標監控

```powershell
# 開啟 Tensorboard（如果還沒開）
tensorboard --logdir user_data/models/DQN_GPU_RTX2070

# 瀏覽器訪問
http://localhost:6006
```

#### 關鍵指標檢查清單

| 指標 | 健康狀態 | 警告信號 |
|------|---------|---------|
| **ep_rew_mean** | 逐漸上升並穩定 | 持續下降或完全平坦 |
| **loss** | 下降後穩定在低值 | 劇烈震盪或爆炸 |
| **value_loss** | 逐漸下降 | 持續上升或 NaN |
| **ep_len_mean** | 穩定在合理範圍 | 接近 0 或最大值 300 |
| **learning_rate** | 按計劃遞減 | 突然變化或 NaN |
| **exploration_rate** | 從 1.0 降至 0.05 | 不變或跳變 |

#### 健康訓練的典型曲線模式

```
ep_rew_mean:  📈 ╱───  初期快速上升，後期穩定（允許小波動）
loss:         📉 ╲___  初期快速下降，後期平穩低值
value_loss:   📉 ╲___  平穩下降並穩定
ep_len_mean:  📊 ───   穩定在 50-200 之間
```

#### 問題訓練的警告信號

```
ep_rew_mean:  📊 ━━━━━  完全平坦（模型沒學習）
              📉 ╲╲╲╲╲  持續下降（學習錯誤策略）
loss:         📊 ╱╲╱╲╱  劇烈震盪（訓練不穩定）
              📈 ╱╱╱╱╱  持續上升（發散）
value_loss:   📈 🚨     持續上升或出現 NaN
ep_len_mean:  📊 ___    接近 0（模型過於保守）
              📊 ▔▔▔    固定在 300（達到最大限制）
```

---

### 2. 訓練日誌監控

```powershell
# 即時查看訓練日誌
Get-Content user_data/logs/freqtrade.log -Tail 50 -Wait
```

#### 正常日誌範例

```log
INFO - Training 146 timeranges
INFO - Training BTC/USDT:USDT, 1/2 pairs from 2022-11-17 to 2023-01-01, 1/146 trains
INFO - Training model on 112 features and 9717 data points
INFO - BTC/USDT:USDT: dropped 4 training points due to NaNs in populated dataset 12960
INFO - --------------------Starting training BTC/USDT:USDT --------------------
INFO - Saving model to disk
INFO - Training complete, saving predictions
```

#### 警告/錯誤日誌

| 日誌訊息 | 嚴重程度 | 意義 |
|---------|---------|------|
| `All training data dropped due to NaNs` | 🔴 嚴重 | 資料不足或特徵計算錯誤 |
| `CUDA out of memory` | 🔴 嚴重 | GPU 記憶體不足 |
| `Training reward is negative` | 🟡 警告 | 策略虧損，需檢查獎勵函數 |
| `Model diverged` | 🔴 嚴重 | 訓練發散，需降低學習率 |
| `Dropped X training points due to NaNs` | 🟢 正常 | 少量 NaN 可接受（<1%） |

---

## 📈 評估階段 2：訓練完成後分析

### 1. Backtest 結果分析

訓練完成後，FreqTrade 會生成詳細的回測報告。

#### 查看完整結果

```powershell
# 查看最終結果（訓練日誌末尾）
Get-Content user_data/logs/freqtrade.log -Tail 200

# 或查看 HTML 報告（如果生成）
Start-Process "user_data/backtest-results/backtest-result-*.html"
```

#### 關鍵績效指標（KPI）評分標準

| 指標 | 優秀 | 良好 | 可接受 | 需改進 |
|------|------|------|--------|--------|
| **Total Profit** | >50% | 20-50% | 5-20% | <5% |
| **Win Rate** | >60% | 50-60% | 45-50% | <45% |
| **Profit Factor** | >2.0 | 1.5-2.0 | 1.2-1.5 | <1.2 |
| **Sharpe Ratio** | >2.0 | 1.0-2.0 | 0.5-1.0 | <0.5 |
| **Max Drawdown** | <10% | 10-20% | 20-30% | >30% |
| **Sortino Ratio** | >3.0 | 2.0-3.0 | 1.0-2.0 | <1.0 |
| **Calmar Ratio** | >2.0 | 1.0-2.0 | 0.5-1.0 | <0.5 |
| **Trade Count** | 100-500 | 50-100 | 20-50 | <20 或 >1000 |
| **Avg Trade Duration** | 1-5 小時 | 5-12 小時 | 12-24 小時 | >24 小時 |

#### 額外檢查指標

```python
# 重要比率計算
Risk/Reward Ratio = Average Win / Average Loss
應該 ≥ 1.5

Expectancy = (Win% × Avg Win) - (Loss% × Avg Loss)
應該 > 0

Win/Loss Ratio = Total Wins / Total Losses
應該 > 1.0

Profit per Trade = Total Profit / Total Trades
應該 > 交易成本的 3 倍
```

---

### 2. Walk-Forward 驗證分析

**你的配置自動執行了 146 次 Walk-Forward 驗證！**

#### Walk-Forward 視窗結構

```
視窗 1:   訓練 [2022-11-17 to 2023-01-01 (45天)]  →  回測 [2023-01-01 to 2023-01-08 (7天)]
視窗 2:   訓練 [2022-12-24 to 2023-02-07 (45天)]  →  回測 [2023-02-07 to 2023-02-14 (7天)]
視窗 3:   訓練 [2023-01-31 to 2023-03-17 (45天)]  →  回測 [2023-03-17 to 2023-03-24 (7天)]
...
視窗 146: 訓練 [2025-09-04 to 2025-10-19 (45天)]  →  回測 [2025-10-19 to 2025-10-26 (7天)]
```

#### 如何評估 Walk-Forward 結果

**一致性檢查（最重要）：**

```
✅ 大部分視窗都獲利（>60% 視窗為正報酬）
✅ 虧損視窗的虧損幅度小（平均虧損 < 平均獲利）
✅ 沒有極端異常值（單一視窗暴賺 >50% 或暴虧 >30%）
✅ 不同市況都能適應：
   - 牛市（2023 Q1, 2024 Q1）
   - 熊市（2023 Q3, 2024 Q3）
   - 盤整（2023 Q4, 2024 Q2）
✅ 最近視窗表現與整體相近（沒有驟降）
```

**警告信號：**

```
❌ 只有少數視窗獲利（<40%）
❌ 獲利集中在某幾個視窗（lucky trades）
   例如：只有 3 個視窗貢獻了 80% 獲利
❌ 最近 20 個視窗表現驟降
❌ 特定市況下完全失效
   例如：所有熊市視窗都虧損 >10%
❌ 獲利視窗和虧損視窗呈現規律性
   例如：獲利-虧損-獲利-虧損交替出現
```

#### 視窗獲利率分佈分析

```python
# 理想分佈（正偏態）
獲利視窗: ████████████████████ 70%
虧損視窗: ████████ 30%
平均獲利: +8%
平均虧損: -3%

# 警告分佈（雙峰或負偏態）
獲利視窗: ██████████ 40%
虧損視窗: ███████████████ 60%
```

---

### 3. 視覺化檢查

#### 繪製權益曲線

```powershell
# 使用 FreqTrade 的繪圖工具
freqtrade plot-dataframe --strategy RLStrategy4Action --config user_data/config_dqn_gpu.json --timerange 20230101-20230301 --pairs BTC/USDT:USDT

# 生成完整回測圖表
freqtrade plot-profit --strategy RLStrategy4Action --config user_data/config_dqn_gpu.json
```

#### 健康的權益曲線特徵

```
📈 整體趨勢
   - 穩定向右上方傾斜
   - 沒有長期（>30天）平台期
   - 新高不斷創建

📊 回撤特徵
   - 回撤深度 <20%
   - 回撤恢復時間 <30 天
   - 回撤後能創新高

🔄 一致性
   - 不同時期表現相似
   - 沒有明顯的"運氣"時段
   - 上升斜率相對穩定
```

#### 過擬合的典型警告

```
❌ 訓練期完美，驗證期崩潰
   - 訓練窗口內：+50%
   - 驗證窗口：-20%

❌ 在某些特定日期暴賺
   - 2023-03-15: +30% （單日）
   - 其他日期: 接近 0

❌ 回測結果太完美
   - 年化報酬 >200%
   - 最大回撤 <3%
   - 勝率 >90%
   → 極可能過擬合

❌ 極少虧損交易
   - 總交易 500 筆
   - 虧損僅 5 筆
   → 可能未學到止損
```

---

## 🔍 評估階段 3：額外驗證測試

### 1. 未見數據測試（Out-of-Sample）

這是最關鍵的測試！

```powershell
# 1. 下載最新數據（訓練範圍之外）
freqtrade download-data --exchange binance --pairs BTC/USDT:USDT ETH/USDT:USDT --timeframe 5m --timerange 20251019-20251030 --trading-mode futures

# 2. 使用最後訓練的模型測試
# 注意：使用最後一個視窗訓練的模型
freqtrade backtesting --strategy RLStrategy4Action --config user_data/config_dqn_gpu.json --timerange 20251020-20251030
```

#### 結果評估

| 未見數據表現 vs 訓練期表現 | 評級 | 解讀 |
|---------------------------|------|------|
| 相近或略好 (±10%) | 🟢 優秀 | 模型泛化能力強 |
| 稍差 (10-30%) | 🟡 良好 | 正常範圍，可接受 |
| 明顯下降 (30-50%) | 🟠 警告 | 可能輕微過擬合 |
| 大幅下降 (>50%) 或虧損 | 🔴 失敗 | 嚴重過擬合，不可使用 |

---

### 2. 穩健性測試

#### A. 參數敏感性測試

測試模型對參數變化的容忍度：

```json
// 測試 1: 稍微改變止損
"stoploss": -0.06  // 原本 -0.05 (改變 20%)

// 測試 2: 改變最大持倉時間
"max_trade_duration_candles": 250  // 原本 300

// 測試 3: 改變最大開倉數
"max_open_trades": 6  // 原本 8
```

**穩健的模型特徵：**
- 參數變化 ±20% 時，效能下降 <30%
- 沒有懸崖式崩潰（cliff effect）
- 不同參數組合下都能獲利

**脆弱模型的警告：**
- 止損從 -0.05 改為 -0.06，報酬從 +30% 變成 -10%
- 極度依賴特定參數組合

---

#### B. 市況適應性測試

手動檢查模型在不同市況的表現：

```powershell
# 測試牛市期間
freqtrade backtesting --timerange 20230101-20230331

# 測試熊市期間  
freqtrade backtesting --timerange 20230801-20231031

# 測試盤整期間
freqtrade backtesting --timerange 20240501-20240731
```

#### 評估標準

| 市況 | 期望表現 | 警告信號 |
|------|---------|---------|
| **牛市** | 獲利 >20% | 虧損或 <5% |
| **熊市** | 不虧或小虧 (<-5%) | 虧損 >-15% |
| **盤整** | 穩定小幅獲利 (5-15%) | 虧損或 <2% |
| **高波動** | 適度獲利但回撤可控 | 回撤 >30% |

**理想模型：** 在不同市況下都能賺錢或保本  
**可接受模型：** 牛市賺，熊市小虧，整體獲利  
**不良模型：** 只在特定市況賺錢，其他時候大虧

---

#### C. 交易對替換測試

測試策略是否過度擬合特定交易對：

```json
// 測試配置
"pair_whitelist": [
    "BNB/USDT:USDT",   // 替代 BTC
    "SOL/USDT:USDT"    // 替代 ETH
]
```

**良好泛化：** 不同交易對上也能獲利  
**過擬合：** 只在 BTC/ETH 上賺錢，其他交易對虧損

---

### 3. 與基準策略比較

#### 建立性能基準

```python
# 基準策略 1: 買入持有（Buy & Hold）
# 計算：期間開始買入，期間結束賣出

# 基準策略 2: 簡單移動平均 (SMA Cross)
# 5m SMA(20) 金叉 SMA(50) 買入
# 死叉賣出

# 基準策略 3: 隨機進出場
# 隨機時間點開倉/平倉
```

#### 比較結果

**好的 RL 模型應該：**
- 總報酬 > Buy & Hold × 1.5
- 夏普比率 > 簡單策略 × 2
- 最大回撤 < Buy & Hold × 0.5
- 明顯優於隨機策略

**如果 RL 模型輸給簡單策略 → 訓練失敗！**

---

## ✅ 模型可用性判斷標準

### 最低要求（及格線 - 僅供測試）

```
✓ 總獲利 > 5%
✓ Win Rate > 45%
✓ Profit Factor > 1.1
✓ Max Drawdown < 30%
✓ Sharpe Ratio > 0.3
✓ 無明顯過擬合跡象
✓ Walk-forward >50% 視窗獲利
✓ Tensorboard 指標健康
```

**結論：** 可繼續優化，不建議實盤

---

### 推薦標準（可實盤 - 小資金測試）

```
✓ 總獲利 > 20%
✓ Win Rate > 50%
✓ Profit Factor > 1.5
✓ Max Drawdown < 20%
✓ Sharpe Ratio > 1.0
✓ Calmar Ratio > 1.0
✓ Walk-forward >65% 視窗獲利
✓ 穩健性測試通過
✓ 未見數據表現良好（下降 <30%）
✓ 不同市況都能適應
```

**結論：** 可用小額資金（總資金 5-10%）實盤測試

---

### 優秀標準（高信心 - 正常實盤）

```
✓ 總獲利 > 50%
✓ Win Rate > 60%
✓ Profit Factor > 2.0
✓ Max Drawdown < 10%
✓ Sharpe Ratio > 2.0
✓ Sortino Ratio > 3.0
✓ Calmar Ratio > 2.0
✓ Walk-forward >80% 視窗獲利
✓ 各種市況都能適應
✓ 不同參數下仍穩定
✓ 優於所有基準策略
✓ 未見數據表現相近（±10%）
```

**結論：** 可正常實盤，但仍需持續監控

---

## 🚨 不可使用的紅線警告

**立即停止使用的信號：**

```
❌ 訓練期間 loss 持續不收斂
❌ Tensorboard ep_rew_mean 持續為負
❌ Profit Factor < 1.0（總體虧損）
❌ Max Drawdown > 40%
❌ Win Rate < 35%
❌ Walk-forward 只有 <30% 視窗獲利
❌ 未見數據表現崩潰（虧損或下降 >80%）
❌ 權益曲線完全平坦或持續下降
❌ 極端依賴單一交易或時段
❌ 完全無法適應熊市（所有熊市視窗虧損 >10%）
❌ 穩健性測試失敗（參數小變化導致崩潰）
```

---

## 📋 實際評估流程 Checklist

### Phase 1: 訓練完成後立即執行（5 分鐘）

- [ ] 1. 檢查訓練日誌無嚴重錯誤
- [ ] 2. 確認 146 個視窗都成功訓練完成
- [ ] 3. 查看 Tensorboard 整體趨勢（ep_rew_mean, loss）
- [ ] 4. 記錄 Backtest 最終結果（Total Profit, Win Rate）
- [ ] 5. 快速判斷：是否達到最低標準

### Phase 2: 深度分析（30 分鐘）

- [ ] 6. 計算所有關鍵指標（Sharpe、Profit Factor、Calmar 等）
- [ ] 7. 分析 Walk-forward 一致性（獲利視窗比例）
- [ ] 8. 繪製權益曲線並視覺檢查
- [ ] 9. 檢查不同市況表現（牛市/熊市/盤整）
- [ ] 10. 識別異常值和異常時段
- [ ] 11. 與基準策略比較（Buy & Hold, SMA Cross）

### Phase 3: 穩健性驗證（1-2 小時）

- [ ] 12. 未見數據測試（如有新數據）
- [ ] 13. 參數敏感性測試（3-5 組參數）
- [ ] 14. 交易對替換測試（2-3 個新交易對）
- [ ] 15. 市況壓力測試（極端市況）
- [ ] 16. 與歷史最佳模型比較

### Phase 4: 風險評估（30 分鐘）

- [ ] 17. 計算 VaR (Value at Risk)
- [ ] 18. 分析最大連續虧損
- [ ] 19. 評估回撤恢復時間
- [ ] 20. 估算實盤所需最小資金

### Phase 5: 最終決策

- [ ] 21. ✅ 符合最低標準 → 繼續優化參數
- [ ] 22. ✅ 符合推薦標準 → 小資金實盤測試
- [ ] 23. ✅ 符合優秀標準 → 正常實盤部署
- [ ] 24. ❌ 不符標準 → 重新訓練/大幅調整參數

---

## 💡 實用工具與技巧

### 1. 自動化評估腳本

```powershell
# evaluate_model.ps1
param(
    [string]$LogFile = "user_data/logs/freqtrade.log",
    [string]$ModelId = "DQN_GPU_RTX2070"
)

# 提取關鍵指標
$totalProfit = Select-String -Path $LogFile -Pattern "Total profit" | Select-Object -Last 1
$winRate = Select-String -Path $LogFile -Pattern "Win Rate" | Select-Object -Last 1
$sharpe = Select-String -Path $LogFile -Pattern "Sharpe" | Select-Object -Last 1
$maxDD = Select-String -Path $LogFile -Pattern "Max Drawdown" | Select-Object -Last 1

# 生成評估報告
@"
=== Model Evaluation Report ===
Model: $ModelId
Date: $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')

Profitability:
$totalProfit
$winRate

Risk Metrics:
$maxDD
$sharpe

Status: $(if ($sharpe -match '\d+\.\d+' -and [double]$matches[0] -gt 1.0) {'✅ PASS'} else {'❌ FAIL'})
"@ | Out-File "evaluation_report_$(Get-Date -Format 'yyyyMMdd').txt"
```

---

### 2. 訓練記錄模板

```markdown
## 訓練記錄

### 2025-10-20 訓練 #001

**配置：**
- Strategy: RLStrategy4Action
- Model: ReinforcementLearner4Action
- Train Period: 45 days
- Train Cycles: 50
- Learning Rate: 0.00025

**結果：**
- Total Profit: 32.5%
- Win Rate: 58.3%
- Profit Factor: 1.82
- Sharpe Ratio: 1.45
- Max Drawdown: 12.3%
- Walk-forward: 72% 視窗獲利

**評估：**
- ✅ 符合推薦標準
- ✅ Walk-forward 一致性良好
- ⚠️ 熊市表現稍弱（-3.2%）

**決策：** 小資金實盤測試

**備註：**
- Tensorboard 顯示訓練穩定
- 未見數據測試：+18.5%（良好）
```

---

### 3. A/B 測試框架

同時訓練多個配置進行比較：

| 配置 | Learning Rate | Train Cycles | Net Arch | 結果 |
|------|---------------|--------------|----------|------|
| A (當前) | 0.00025 | 50 | [512,512,256] | +32.5% |
| B (保守) | 0.0001 | 30 | [256,256] | +28.1% |
| C (激進) | 0.0005 | 100 | [1024,512,256] | +45.2% ⚠️ |

**結論：** 配置 A 最佳（平衡性能與穩定性）

---

### 4. 視覺化儀表板

使用 Tensorboard 建立自訂儀表板：

```python
# custom_metrics.py
from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter('user_data/models/DQN_GPU_RTX2070/custom_metrics')

# 記錄自訂指標
writer.add_scalar('Walk_Forward/Win_Rate', win_rate_per_window)
writer.add_scalar('Risk/Drawdown', current_drawdown)
writer.add_scalar('Performance/Sharpe_Ratio', sharpe_ratio)
```

---

## 🎓 總結與關鍵要點

### `continual_learning: false` 的核心價值

**146 次獨立驗證 = 最嚴格的測試！**

1. **無資訊洩漏**  
   每個視窗的模型完全獨立，真實模擬實盤部署

2. **抗災難性遺忘**  
   不會因為舊模式影響新模式的學習

3. **更真實的效能預測**  
   Walk-forward 結果直接反映實盤預期表現

4. **易於診斷問題**  
   可以找出哪些時段/市況表現不佳

---

### 判斷模型好壞的黃金法則

#### 核心指標（必須滿足）

1. **Walk-forward 一致性** → >65% 視窗獲利
2. **風險調整報酬** → Sharpe > 1.0
3. **最大回撤可控** → Max DD < 20%
4. **未見數據表現** → 與訓練期相近（±30%）

#### 次要指標（加分項）

5. 不同市況適應性
6. 參數穩健性
7. 優於基準策略
8. 合理的交易頻率

---

### 記住這三個原則

1. **完美的回測不存在**  
   過於完美（>200% 年化，<3% 回撤）反而是過擬合的信號

2. **穩定比暴利重要**  
   年化 30% 且穩定 > 年化 100% 但波動巨大

3. **實盤是唯一真相**  
   再好的回測也要經過實盤小額驗證

---

## 📚 相關文件

- 配置分析：`docs/20251020/config_analysis_and_optimization_guide.md`
- 配置檔案：`user_data/config_dqn_gpu.json`
- 策略檔案：`user_data/strategies/RLStrategy4Action.py`
- 模型檔案：`user_data/freqaimodels/ReinforcementLearner4Action.py`
- 訓練日誌：`user_data/logs/freqtrade.log`
- Tensorboard：`user_data/models/DQN_GPU_RTX2070/tensorboard/`

---

**最後更新：** 2025-10-20  
**訓練狀態：** 進行中（預計 40-60 小時）  
**下一步：** 訓練完成後執行本指南的評估流程
