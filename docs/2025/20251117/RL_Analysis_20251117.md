# FreqAI RL 策略分析與實作指南 (RL20251117)

本文件基於 `20251117.md` 的需求、官方 FreqAI RL 文件以及 `config_rl_10x_v2_old.json` 參考配置，提供完整的實作分析與指南。

## 1. 專案目標 (Project Goal)

建立一個基於強化學習 (RL) 的自動交易策略，代號 **RL20251117**。
該策略需具備以下核心特徵：
*   **資金管理**：本金 1000 USDT，嚴格控制單筆風險 (10%)。
*   **技術指標**：結合趨勢 (Trend)、斐波那契回撤 (Fibonacci)、公允價值缺口 (FVG)。
*   **獎勵機制**：針對「強制平倉」與「大額虧損 (>10%)」實施加重懲罰。

## 2. 系統架構與配置 (Architecture & Configuration)

### 2.1 設定檔 (Configuration)
基於 `config_rl_10x_v2_old.json` 進行修改。

*   **檔案位置**: `user_data/config_RL20251117.json`
*   **關鍵參數修改**:
    *   `stake_amount`: 設定為 `100` (本金 1000 的 10%) 或 `unlimited` (由策略控制倉位)。
    *   `max_open_trades`: 建議設為 `1` 或 `2`，以便集中資金進行風險控管。
    *   `freqai.identifier`: `RL20251117`。
    *   `freqai.rl_config.train_cycles`: `800`。
    *   `freqai.rl_config.cpu_count`: `16`。
    *   `freqai.rl_config.model_type`: `PPO`。
    *   `freqai.rl_config.policy_type`: `MlpPolicy`。
    *   `freqai.rl_config.max_training_drawdown_pct`: 建議設為 `0.15` 或更高，避免在訓練初期因觸發 10% 虧損而過早終止 Episode，讓模型有機會學習到懲罰。

### 2.2 策略檔案 (Strategy)
*   **檔案位置**: `user_data/strategies/RL20251117_Strategy.py`
*   **職責**:
    1.  定義 `feature_engineering_*` 函數，計算技術指標 (Trend, Fib, FVG)。
    2.  將這些指標轉換為模型可讀的特徵。
    3.  設定 `populate_entry_trend` 和 `populate_exit_trend` 來接收 RL Agent 的動作。

### 2.3 模型檔案 (Model)
*   **檔案位置**: `user_data/freqaimodels/RL20251117_Model.py`
*   **職責**:
    1.  繼承 `ReinforcementLearner`。
    2.  定義 `MyRLEnv` 類別。
    3.  **核心**: 實作 `calculate_reward` 函數，包含自定義的懲罰邏輯。

---

## 3. 詳細實作邏輯 (Implementation Details)

### 3.1 特徵工程 (Feature Engineering) - 策略端

AI 需要「看見」市場狀態，因此我們必須在策略中計算以下指標並餵給它：

```python
# 虛擬碼範例 (User Strategy)
import talib.abstract as ta

def feature_engineering_standard(self, dataframe: DataFrame, **kwargs) -> DataFrame:
    # 1. 趨勢 (Trend) - 使用 EMA 交叉或價格位置
    dataframe['ema_20'] = ta.EMA(dataframe, timeperiod=20)
    dataframe['ema_50'] = ta.EMA(dataframe, timeperiod=50)
    dataframe['trend_up'] = (dataframe['ema_20'] > dataframe['ema_50']).astype(int)

    # 2. 斐波那契 (Fibonacci) - 計算近期高低點的回撤位
    # 假設使用過去 100 根 K 線的高低點
    rolling_high = dataframe['high'].rolling(100).max()
    rolling_low = dataframe['low'].rolling(100).min()
    diff = rolling_high - rolling_low
    # 避免除以零
    diff = diff.replace(0, 0.0000001)
    
    dataframe['fib_0.618'] = rolling_high - (diff * 0.618)
    dataframe['fib_0.5'] = rolling_high - (diff * 0.5)
    
    # 特徵化：當前價格相對於 Fib 0.618 的距離 (正規化)
    dataframe['dist_to_fib_618'] = (dataframe['close'] - dataframe['fib_0.618']) / dataframe['close']

    # 3. 公允價值缺口 (FVG)
    # FVG 定義: (High[n-2] < Low[n]) for Bullish FVG
    dataframe['fvg_bull'] = (
        (dataframe['high'].shift(2) < dataframe['low']) & 
        (dataframe['close'].shift(1) > dataframe['open'].shift(1)) # 中間是紅K(上漲)
    ).astype(int)
    
    # 必須包含原始 OHLCV 供 RL 環境使用
    dataframe["%-raw_close"] = dataframe["close"]
    dataframe["%-raw_open"] = dataframe["open"]
    dataframe["%-raw_high"] = dataframe["high"]
    dataframe["%-raw_low"] = dataframe["low"]

    return dataframe
```

### 3.2 獎勵函數 (Reward Function) - 模型端

這是本專案最關鍵的部分，需嚴格執行 `20251117.md` 中的懲罰規則。

```python
# 虛擬碼範例 (RL Model)
from freqtrade.freqai.RL.Base5ActionRLEnv import Actions, Base5ActionRLEnv, Positions

class MyRLEnv(Base5ActionRLEnv):
    def calculate_reward(self, action: int) -> float:
        # 1. 基礎檢查
        if not self._is_valid(action):
            return -2.0
            
        pnl = self.get_unrealized_profit()
        
        # 2. 取得當前帳戶資訊 (模擬)
        # 注意: 在訓練環境中，我們通常使用相對 PnL (百分比)
        # 假設本金 1000U，10% 虧損即 PnL < -0.10
        
        reward = 0.0
        
        # 3. 懲罰邏輯實作
        
        # A. 強制平倉 (Liquidation) 或 極大虧損
        # 假設 -0.8 (80% 虧損) 視為接近爆倉
        if pnl < -0.8: 
            return -200.0 # 極重懲罰
            
        # B. 虧損超過 10% (PnL < -0.10)
        if pnl < -0.10:
            # 加重懲罰：除了原本的 PnL 虧損外，額外扣分
            # 例如：虧損 -15%，獎勵為 -0.15 * 10 (放大係數) - 20 (固定罰分)
            reward = (pnl * 20.0) - 20.0
            
        # C. 虧損在 10% 以內 (PnL >= -0.10 且 PnL < 0)
        elif pnl < 0:
            # 正常懲罰：直接反映 PnL
            reward = pnl * 10.0 # 係數可調整
            
        # D. 獲利 (PnL > 0)
        elif pnl > 0:
            # 獎勵獲利，可加入 R:R 的考量
            reward = pnl * 10.0
            
        # E. 持倉時間懲罰 (Time Penalty)
        # 避免 AI 死抱虧損單
        trade_duration = self._current_tick - self._last_trade_tick
        if trade_duration > 100:
            reward -= 0.1 * (trade_duration / 100)

        return reward
```

## 4. 執行步驟 (Action Plan)

1.  **建立目錄結構**: 確保 `user_data/freqaimodels` 和 `user_data/strategies` 存在。
2.  **建立設定檔**: 複製並修改 `config_rl_10x_v2_old.json` 為 `config_RL20251117.json`。
3.  **建立策略檔**: 撰寫 `user_data/strategies/RL20251117_Strategy.py`，實作上述特徵工程。
4.  **建立模型檔**: 撰寫 `user_data/freqaimodels/RL20251117_Model.py`，實作自定義獎勵函數。
5.  **下載歷史數據 (Data Download)**:
    *   為了進行 20230101 開始的 Backtesting，需要下載包含「訓練緩衝期」的數據。建議下載 2022 年以後的數據。
    ```bash
    freqtrade download-data --config user_data/config_RL20251117.json --timerange 20220101-20251119
    ```
6.  **開始訓練 (Backtesting)**:
    *   使用指定的 Backtesting 區間 (20230101-20251119)。
    *   FreqAI 會自動利用 20230101 之前的數據進行初始模型訓練。
    ```bash
    freqtrade backtesting --freqai --freqaimodel RL20251117_Model --strategy RL20251117_Strategy --config user_data/config_RL20251117.json --timerange 20230101-20251119
    ```
    此指令會模擬從 2023 年初到 2025 年底的交易，並在過程中持續訓練/更新模型。

7.  **Trade 模式 (Dry-run/Live)**:
    ```bash
    freqtrade trade --freqaimodel RL20251117_Model --strategy RL20251117_Strategy --config user_data/config_RL20251117.json
    ```
    此指令會啟動 Dry-run，並隨著新數據進來持續訓練模型 (若 `live_retrain_hours` > 0)。

## 5. 補充說明

*   **R:R (風險回報比)**: 在 RL 中，這通常是透過獎勵函數隱式學習的。如果我們給予大賠極大的懲罰，並給予大賺線性的獎勵，模型會自然傾向於尋找高 R:R 的交易。
*   **槓桿 (Leverage)**: 設定檔中的 `leverage` 參數應配合 `stake_amount` 使用。若本金 1000U，單筆風險 100U，且止損距離為 1%，則倉位大小應為 10000U (10倍槓桿)。這需要在策略邏輯中動態計算，或由 RL Agent 學習控制倉位大小 (進階)。

此分析文件提供了實作 `RL20251117` 所需的完整藍圖。
