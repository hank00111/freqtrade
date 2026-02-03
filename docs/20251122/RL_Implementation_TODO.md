# RL20251117 實作待辦事項清單 (Implementation TODO List)

本文件基於 `docs/20251117/RL_Analysis_20251117.md` 規劃實作步驟，並包含詳細的程式碼規格，適合交付給開發 Agent 執行。

## 1. 環境與設定 (Environment & Configuration)

- [x] **建立目錄結構**
    - [x] 確認 `user_data/freqaimodels` 存在
    - [x] 確認 `user_data/strategies` 存在

- [x] **建立設定檔 (`user_data/config_RL20251117.json`)**
    - [x] 複製參考設定檔 (若無則使用標準 FreqAI 範本)
    - [x] 設定 `stake_amount`: `100` (本金 1000 的 10%)
    - [x] 設定 `max_open_trades`: `1`
    - [x] 設定 `freqai.identifier`: `"RL20251117"`
    - [x] 設定 `freqai.rl_config` 相關參數

### 實作規格 (Configuration Spec)

```json
{
    "stake_amount": 100,
    "max_open_trades": 1,
    "freqai": {
        "enabled": true,
        "identifier": "RL20251117",
        "rl_config": {
            "train_cycles": 800,
            "cpu_count": 16,
            "model_type": "PPO",
            "policy_type": "MlpPolicy",
            "max_training_drawdown_pct": 0.15
        }
    }
}
```

## 2. 策略開發 (Strategy Development)

- [x] **建立策略檔案 (`user_data/strategies/RL20251117_Strategy.py`)**
    - [x] 繼承 `FreqaiExampleStrategy` 或標準 `IStrategy`
    - [x] 實作 `feature_engineering_standard` 函數
        - [x] **趨勢指標 (Trend)**: 計算 EMA 20 與 EMA 50 及其交叉狀態
        - [x] **斐波那契 (Fibonacci)**:
            - [x] 計算 Rolling Max/Min (過去 100 根)
            - [x] 計算 0.618 回撤位
            - [x] 計算價格與 0.618 的距離特徵
        - [x] **公允價值缺口 (FVG)**:
            - [x] 實作 Bullish FVG 判斷邏輯 (High[n-2] < Low[n])
        - [x] **原始數據**: 加入 `%-raw_close`, `%-raw_open` 等欄位

### 實作規格 (Strategy Code Spec)

```python
import talib.abstract as ta
from pandas import DataFrame

def feature_engineering_standard(self, dataframe: DataFrame, metadata: dict, **kwargs) -> DataFrame:
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

## 3. 模型開發 (Model Development)

- [x] **建立模型檔案 (`user_data/freqaimodels/RL20251117_Model.py`)**
    - [x] 繼承 `ReinforcementLearner`
    - [x] 定義 `MyRLEnv` 類別 (繼承 `Base5ActionRLEnv`)
    - [x] **實作 `calculate_reward` 函數**:
        - [x] **強制平倉懲罰**: PnL < -0.8 時回傳 `-200`
        - [x] **大額虧損懲罰**: PnL < -0.10 時使用加重懲罰公式 (例如 `(pnl * 20) - 20`)
        - [x] **一般虧損**: 線性懲罰
        - [x] **獲利獎勵**: 線性獎勵
        - [x] **持倉時間懲罰**: 避免無效持倉

### 實作規格 (Model Code Spec)

```python
from freqtrade.freqai.RL.Base5ActionRLEnv import Actions, Base5ActionRLEnv
from freqtrade.freqai.RL.BaseEnvironment import Positions

class MyRLEnv(Base5ActionRLEnv):
    def calculate_reward(self, action: int) -> float:
        # 1. 基礎檢查
        if not self._is_valid(action):
            return -2.0
            
        pnl = self.get_unrealized_profit()
        
        # 2. 懲罰邏輯實作
        reward = 0.0
        
        # A. 強制平倉 (Liquidation) 或 極大虧損 (<-80%)
        if pnl < -0.8: 
            return -200.0 
            
        # B. 虧損超過 10% (PnL < -0.10)
        if pnl < -0.10:
            # 加重懲罰：(PnL * 20) - 20
            # 例: -0.15 -> (-3) - 20 = -23
            reward = (pnl * 20.0) - 20.0
            
        # C. 虧損在 10% 以內 (PnL >= -0.10 且 PnL < 0)
        elif pnl < 0:
            reward = pnl * 10.0 
            
        # D. 獲利 (PnL > 0)
        elif pnl > 0:
            reward = pnl * 10.0
            
        # E. 持倉時間懲罰 (Time Penalty)
        trade_duration = self._current_tick - self._last_trade_tick
        if trade_duration > 100:
            reward -= 0.1 * (trade_duration / 100)

        return reward
```

## 4. 數據與執行 (Data & Execution)

- [ ] **下載歷史數據**
    - [ ] 執行指令: `freqtrade download-data --config user_data/config_RL20251117.json --timerange 20220101-20251119`
    - [ ] 確認數據包含 2022 年 (作為訓練緩衝)

- [ ] **執行回測訓練 (Backtesting)**
    - [ ] 執行指令: `freqtrade backtesting --freqai --freqaimodel RL20251117_Model --strategy RL20251117_Strategy --config user_data/config_RL20251117.json --timerange 20230101-20251119`
    - [ ] 檢查 Log 確認是否有 "Division by zero" 或其他錯誤
    - [ ] 確認模型檔案已生成於 `user_data/models/RL20251117`

### 執行指令規格 (Execution Spec)

```bash
# 1. 下載數據 (含緩衝期)
freqtrade download-data --config user_data/config_RL20251117.json --timerange 20220101-20251119

# 2. 執行訓練與回測
freqtrade backtesting --freqai --freqaimodel RL20251117_Model --strategy RL20251117_Strategy --config user_data/config_RL20251117.json --timerange 20230101-20251119
```

## 5. 驗證與優化 (Verification)

- [ ] **檢查訓練結果**
    - [ ] 觀察 Tensorboard (若有開啟) 或 Log 中的 Reward 變化
    - [ ] 確認是否有觸發大額虧損懲罰的紀錄
- [ ] **調整參數 (若需要)**
    - [ ] 若訓練效果不佳，調整 `train_cycles` 或 Reward 權重
