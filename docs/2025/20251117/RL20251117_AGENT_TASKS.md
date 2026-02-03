# RL20251117 Agent 分析任務清單
**專案**: FreqAI 強化學習交易系統  
**建立日期**: 2025年11月17日  
**優先級**: 高 - 訓練前必須完成

---

## 📋 任務總覽

本文件列出需要交付給其他 Agent（或執行額外分析）的具體任務，以完成 `RL20251117_ANALYSIS.md` 中標記為「待補充」的章節。

---

## 任務 1: FreqAI 官方 RL 實作分析

### 目標
理解 FreqAI 的強化學習實作架構，以便正確建立 `RL20251117_ReinforcementLearner.py`。

### 需要分析的檔案
1. `docs/freqai-reinforcement-learning.md` - 官方 RL 教學文件
2. `freqtrade/freqai/prediction_models/ReinforcementLearner.py` - 基類原始碼（如果存在）
3. `freqtrade/freqai/rl/` - RL 相關模組（如果存在）
4. 任何範例 RL 模型檔案（搜尋 `user_data/freqaimodels/` 或官方範例）

### 預期產出

#### 1.1 ReinforcementLearner 類別結構
```python
# 需要提供完整的類別框架，包括：
class RL20251117_ReinforcementLearner(BaseReinforcementLearningModel):
    """
    必須覆寫的方法：
    - __init__()
    - calculate_reward()  # 核心！整合獎懲機制
    - _set_action_space()  # 定義動作空間
    - _set_observation_space()  # 定義狀態空間
    - 其他必要方法...
    """
```

#### 1.2 獎懲函數整合位置
- 確認 `calculate_reward()` 的參數簽章
- 說明如何訪問交易結果（profit/loss）
- 說明如何訪問帳戶狀態（balance, liquidation status）
- 提供整合 `RL20251117_ANALYSIS.md` 第 3.2 節獎懲邏輯的具體代碼

#### 1.3 動作空間與狀態空間定義
- 如何使用 `gym.spaces` 定義混合動作空間（離散 + 連續）
- 狀態空間應包含哪些維度
- 如何從 DataFrame 中提取特徵作為狀態

#### 1.4 多線程訓練配置
- `cpu_count: 16` 參數如何影響訓練
- 是否需要額外配置（如 `n_envs`）
- 如何確保每個線程的隨機性

### 輸出格式
- Markdown 文件，包含程式碼範例與詳細註解
- 檔案名稱: `RL20251117_FREQAI_RL_ARCHITECTURE.md`

---

## 任務 2: config_rl_10x_v2_old.json 完整配置分析

### 目標
提取並理解參考配置檔案的所有參數，作為 `config_RL20251117.json` 的基礎。

### 需要分析的檔案
1. `config_rl_10x_v2_old.json` - 參考配置（需要先定位此檔案）
2. `docs/freqai-configuration.md` - 配置參數說明
3. `docs/configuration.md` - 通用配置參數

### 預期產出

#### 2.1 完整配置檔案內容
```json
{
  // 需要提供完整的 JSON 內容
  // 並對每個參數加上註解說明
}
```

#### 2.2 關鍵參數說明表

| 參數路徑 | 數值 | 說明 | 是否需要修改 |
|---------|------|------|-------------|
| `freqai.rl_config.train_cycles` | 800 | 訓練週期數 | ✅ 保持 |
| `freqai.rl_config.model_type` | "PPO" | RL 算法 | ✅ 保持 |
| `freqai.rl_config.learning_rate` | ? | 學習率 | ❓ 待確認 |
| ... | ... | ... | ... |

#### 2.3 與本專案需求的差異對比
- 哪些參數需要調整以符合 1000 USDT 本金
- 哪些參數需要調整以符合 10% 風險限制
- 建議的修改項目清單

### 輸出格式
- Markdown 文件
- 檔案名稱: `RL20251117_CONFIG_ANALYSIS.md`

---

## 任務 3: 技術指標特徵工程實作

### 目標
提供可直接使用的技術指標計算代碼，整合到 `populate_indicators()` 方法。

### 需要分析的檔案
1. `docs/freqai-feature-engineering.md` - 官方特徵工程指南
2. 現有 strategy 檔案中的 `populate_indicators()` 範例
3. FreqAI 如何處理特徵標準化
4. TA-Lib 或 pandas-ta 的相關函數

### 預期產出

#### 3.1 斐波那契回撤計算

```python
def calculate_fibonacci_retracement(dataframe, lookback_period=20):
    """
    計算斐波那契 50% 和 61.8% 回撤位
    
    Returns:
        dataframe with new columns:
        - fib_high: 波段高點
        - fib_low: 波段低點
        - fib_0.5: 50% 回撤位
        - fib_0.618: 61.8% 回撤位
        - distance_to_fib_0.5: 當前價格距離 0.5 位的標準化距離
        - distance_to_fib_0.618: 當前價格距離 0.618 位的標準化距離
    """
    # 需要提供完整實作
```

#### 3.2 公允價值缺口 (FVG) 檢測

```python
def detect_fair_value_gap(dataframe):
    """
    檢測三 K 線形態的 FVG
    
    條件:
    - K1 和 K3 的影線在 K2 處無重疊
    
    Returns:
        dataframe with new columns:
        - fvg_bullish: 是否存在看漲 FVG (0/1)
        - fvg_bearish: 是否存在看跌 FVG (0/1)
        - fvg_midpoint: FVG 的中點價格
        - distance_to_fvg: 當前價格距離 FVG 中點的距離
    """
    # 需要提供完整實作
```

#### 3.3 趨勢識別

```python
def identify_trend(dataframe, lookback=50):
    """
    識別當前趨勢方向
    
    使用移動平均、支撐阻力等組合判斷
    
    Returns:
        dataframe with new columns:
        - trend_direction: -1 (下降), 0 (橫盤), 1 (上升)
        - support_level: 當前支撐位
        - resistance_level: 當前阻力位
        - distance_to_support: 標準化距離
        - distance_to_resistance: 標準化距離
    """
    # 需要提供完整實作
```

#### 3.4 艾略特波浪計數（選做）

如果有可靠的實作方法，提供波浪計數的邏輯。如果過於複雜，可以先略過，使用簡化的趨勢識別替代。

#### 3.5 完整的 populate_indicators() 整合

```python
def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
    """
    整合所有技術指標
    """
    # 基礎指標
    dataframe = self.calculate_fibonacci_retracement(dataframe)
    dataframe = self.detect_fair_value_gap(dataframe)
    dataframe = self.identify_trend(dataframe)
    
    # 標準化所有特徵到 [0, 1] 或 [-1, 1] 範圍
    # ...
    
    return dataframe
```

### 輸出格式
- Python 程式碼檔案（可直接複製使用）
- 附帶說明文件（Markdown）
- 檔案名稱: 
  - `RL20251117_INDICATORS.py`
  - `RL20251117_INDICATORS_GUIDE.md`

---

## 任務 4: Strategy 檔案架構分析

### 目標
提供 `RL20251117_Strategy.py` 的完整框架，整合風險管理與 RL 模型。

### 需要分析的檔案
1. FreqAI 的 strategy 範例（官方或社群）
2. `docs/freqai-running.md` - 執行指南
3. `docs/strategy-customization.md` - 策略自定義

### 預期產出

#### 4.1 Strategy 類別完整框架

```python
from freqtrade.strategy import IStrategy
from pandas import DataFrame

class RL20251117_Strategy(IStrategy):
    """
    完整的 strategy 框架，包括：
    - 必要的類別屬性設定
    - populate_indicators() - 呼叫任務 3 的指標
    - populate_entry_trend() - RL 模型進場邏輯
    - populate_exit_trend() - RL 模型出場邏輯
    - custom_stoploss() - 動態停損
    - custom_stake_amount() - 倉位大小計算（整合風險管理）
    - 其他必要方法
    """
```

#### 4.2 風險管理邏輯嵌入

展示如何在 strategy 中實作：
- 固定 100 USDT 風險的倉位計算
- 停損位的動態設定
- 強制平倉的預防機制
- 與 RL 模型的交互方式

#### 4.3 RL 模型的調用方式

```python
def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
    """
    如何從 FreqAI 的 RL 模型獲取進場信號
    如何傳遞技術指標作為狀態
    """
    # 需要提供具體代碼
```

### 輸出格式
- Python 程式碼檔案（完整可運行的 strategy）
- 附帶說明文件
- 檔案名稱:
  - `RL20251117_Strategy.py`
  - `RL20251117_STRATEGY_GUIDE.md`

---

## 任務 5: 訓練數據準備指南

### 目標
確認訓練所需的歷史數據規格與下載方法。

### 需要分析的內容
1. FreqAI 對歷史數據的要求（時間範圍、頻率）
2. 如何下載適合 RL 訓練的數據
3. 數據品質檢查方法

### 預期產出

#### 5.1 數據下載指令

```bash
# 需要提供完整的 freqtrade download-data 指令
# 包括：
# - 推薦的交易對
# - 時間範圍
# - 時間週期（1h, 4h 等）
# - 交易所選擇
```

#### 5.2 數據品質檢查

- 如何檢查數據完整性（無缺失）
- 如何處理異常值
- 推薦的數據量（幾個月？）

### 輸出格式
- Markdown 文件
- 檔案名稱: `RL20251117_DATA_PREPARATION.md`

---

## 任務 6: 整合測試與除錯指南

### 目標
提供完整的測試流程，確保所有組件正確整合。

### 預期產出

#### 6.1 單元測試建議

- 如何測試獎懲函數（不同情境的返回值）
- 如何測試倉位計算邏輯
- 如何測試技術指標的正確性

#### 6.2 常見錯誤與解決方案

基於 FreqAI RL 訓練的常見問題：
- 訓練不收斂
- 內存溢出
- 獎勵值異常
- 模型總是選擇「持有」不動

#### 6.3 除錯工具使用

- TensorBoard 整合
- FreqAI 日誌解讀
- 如何提取訓練過程中的獎勵曲線

### 輸出格式
- Markdown 文件
- 檔案名稱: `RL20251117_TESTING_DEBUG.md`

---

## 📊 任務優先級與依賴關係

```
優先級 P0 (立即執行):
├─ 任務 1: FreqAI RL 架構分析
├─ 任務 2: 配置檔案分析
└─ 任務 5: 數據準備

優先級 P1 (任務 1 完成後):
├─ 任務 3: 技術指標實作
└─ 任務 4: Strategy 檔案

優先級 P2 (所有代碼完成後):
└─ 任務 6: 測試與除錯
```

---

## 🎯 執行建議

### 方案 A: 使用 Sub-agent
將每個任務交付給獨立的 sub-agent，並行執行以節省時間。

### 方案 B: 順序執行
按優先級順序，逐一完成每個任務。

### 方案 C: 混合模式
- P0 任務優先完成（阻塞性任務）
- P1 任務可並行執行（獨立任務）
- P2 任務在所有代碼就緒後執行

---

## 📝 任務完成標準

每個任務完成後應：
1. ✅ 產出指定格式的文件或程式碼
2. ✅ 通過基本的語法檢查（對於 Python 代碼）
3. ✅ 提供使用範例或測試案例
4. ✅ 更新 `RL20251117_ANALYSIS.md` 中的對應章節

---

## 📌 注意事項

1. **保持一致性**: 所有代碼應使用 `RL20251117_` 前綴
2. **文件組織**: 所有產出檔案應放在 `docs/20251117/` 目錄
3. **版本控制**: 重大修改應註明版本號與修改日期
4. **交叉引用**: 不同文件之間應使用相對路徑引用

---

**建立日期**: 2025年11月17日  
**狀態**: 待執行  
**預計完成時間**: 2-4 小時（取決於執行方式）
