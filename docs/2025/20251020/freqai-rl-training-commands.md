# FreqAI RL 訓練指令參考手冊

> **日期**: 2025-10-20  
> **配置文件**: config_dqn_gpu.json (DQN + GPU)  
> **硬體**: NVIDIA RTX 2070 8GB  
> **訓練數據**: BTC/USDT:USDT, ETH/USDT:USDT (2023年10月 - 2024年2月, 5分鐘)

## 目錄

1. [訓練前驗證](#1-訓練前驗證)
2. [訓練執行](#2-訓練執行)
3. [即時監控](#3-即時監控)
4. [訓練控制](#4-訓練控制)
5. [訓練後驗證](#5-訓練後驗證)
6. [故障排除](#6-故障排除)
7. [完整訓練腳本](#7-完整訓練腳本)
8. [快速參考](#8-快速參考)

---

## ⚠️ 重要提示：虛擬環境

**所有 FreqTrade 指令都必須在虛擬環境中執行！**

### 啟動虛擬環境

在執行任何指令前，請先啟動虛擬環境：

```powershell
# 方法 1: 先啟動，然後執行指令
.\.venv\Scripts\Activate.ps1
freqtrade <command>

# 方法 2: 單行指令（每次都啟動）
.\.venv\Scripts\Activate.ps1; freqtrade <command>
```

### 驗證虛擬環境

確認您在虛擬環境中：

```powershell
# 檢查 Python 路徑（應該指向 .venv）
.\.venv\Scripts\Activate.ps1; python -c "import sys; print(sys.executable)"

# 預期輸出: D:\Code\freqtrade\.venv\Scripts\python.exe
```

---

## 1. 訓練前驗證

> **提醒**: 以下所有指令都需要在虛擬環境中執行

### 1.1 檢查已下載的數據

```powershell
# 啟動虛擬環境
.\.venv\Scripts\Activate.ps1

# 列出所有已下載的數據
freqtrade list-data --exchange binance --trading-mode futures

# 顯示特定交易對的時間範圍
freqtrade list-data --exchange binance --trading-mode futures --show-timerange
```

**預期輸出**:
```
Found 2 pair / timeframe combinations.
BTC/USDT:USDT - 5m: 2023-10-01 00:00:00 -> 2024-02-29 23:55:00
ETH/USDT:USDT - 5m: 2023-10-01 00:00:00 -> 2024-02-29 23:55:00
```

### 1.2 驗證配置文件

```powershell
# 啟動虛擬環境
.\.venv\Scripts\Activate.ps1

# 檢查配置語法
freqtrade show-config --config user_data/config_dqn_gpu.json

# 驗證策略相容性
freqtrade test-pairlist --config user_data/config_dqn_gpu.json
```

### 1.3 GPU 可用性檢查

```powershell
# 啟動虛擬環境並檢查 CUDA
.\.venv\Scripts\Activate.ps1; python -c "import torch; print(f'CUDA Available: {torch.cuda.is_available()}'); print(f'CUDA Version: {torch.version.cuda}'); print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}')"
```

**預期輸出**:
```
CUDA Available: True
CUDA Version: 11.8
GPU: NVIDIA GeForce RTX 2070
```

### 1.4 策略驗證

```powershell
# 啟動虛擬環境
.\.venv\Scripts\Activate.ps1

# 驗證策略可用性
freqtrade list-strategies --config user_data/config_dqn_gpu.json

# 快速回測測試（10天）
freqtrade backtesting --strategy RLStrategy --config user_data/config_dqn_gpu.json --timerange 20231201-20231211
```

---

## 2. 訓練執行

> **提醒**: 確保已啟動虛擬環境 `.\.venv\Scripts\Activate.ps1`

### 2.1 基本訓練指令

```powershell
# 啟動虛擬環境
.\.venv\Scripts\Activate.ps1

# 最小化訓練指令
freqtrade backtesting `
  --strategy RLStrategy `
  --config user_data/config_dqn_gpu.json `
  --freqaimodel ReinforcementLearner `
  --timerange 20231201-20240130
```

### 2.2 帶交易導出的訓練

```powershell
# 啟動虛擬環境
.\.venv\Scripts\Activate.ps1

# 訓練並導出交易記錄
freqtrade backtesting `
  --strategy RLStrategy `
  --config user_data/config_dqn_gpu.json `
  --freqaimodel ReinforcementLearner `
  --timerange 20231201-20240130 `
  --export trades
```

### 2.3 自定義時間範圍訓練

```powershell
# 啟動虛擬環境
.\.venv\Scripts\Activate.ps1

# 不同的訓練週期
# 短期測試：1個月
freqtrade backtesting --strategy RLStrategy --config user_data/config_dqn_gpu.json --freqaimodel ReinforcementLearner --timerange 20231201-20231231

# 中期訓練：2個月
freqtrade backtesting --strategy RLStrategy --config user_data/config_dqn_gpu.json --freqaimodel ReinforcementLearner --timerange 20231201-20240130

# 完整訓練：5個月
freqtrade backtesting --strategy RLStrategy --config user_data/config_dqn_gpu.json --freqaimodel ReinforcementLearner --timerange 20231001-20240229
```

### 2.4 持續學習（恢復訓練）

```powershell
# 啟動虛擬環境
.\.venv\Scripts\Activate.ps1

# 第一次訓練會話
freqtrade backtesting --strategy RLStrategy --config user_data/config_dqn_gpu.json --freqaimodel ReinforcementLearner --timerange 20231201-20240130

# 從檢查點恢復（需要配置文件中設置 continual_learning: true）
freqtrade backtesting --strategy RLStrategy --config user_data/config_dqn_gpu.json --freqaimodel ReinforcementLearner --timerange 20240201-20240229
```

---

## 3. 即時監控

### 3.1 啟動 Tensorboard

```powershell
# 在獨立視窗啟動 Tensorboard（自動啟動 venv）
Start-Process pwsh -ArgumentList "-Command", ".\.venv\Scripts\Activate.ps1; tensorboard --logdir user_data/models/DQN_GPU_RTX2070"

# 或在當前終端機啟動 Tensorboard（會佔用終端機）
.\.venv\Scripts\Activate.ps1
tensorboard --logdir user_data/models/DQN_GPU_RTX2070

# 自定義端口
.\.venv\Scripts\Activate.ps1
tensorboard --logdir user_data/models/DQN_GPU_RTX2070 --port 6007
```

**訪問地址**: http://localhost:6006

### 3.2 監控訓練日誌

```powershell
# 即時監看訓練日誌（終端機2）
Get-Content user_data/logs/freqtrade.log -Wait -Tail 50

# 過濾特定信息
Get-Content user_data/logs/freqtrade.log -Wait -Tail 50 | Select-String "train_cycle"
Get-Content user_data/logs/freqtrade.log -Wait -Tail 50 | Select-String "reward"
Get-Content user_data/logs/freqtrade.log -Wait -Tail 50 | Select-String "best_model"
```

### 3.3 GPU 監控

```powershell
# 即時監控 GPU 使用情況（終端機3）
nvidia-smi -l 2

# 或使用循環進行詳細監控
while ($true) { 
    Clear-Host
    nvidia-smi
    Start-Sleep -Seconds 2
}
```

---

## 4. 訓練控制

### 4.1 暫停訓練

```powershell
# 方法1：鍵盤中斷
# 在訓練終端機按 Ctrl+C

# 方法2：優雅關閉（如果支援）
# 建立停止信號文件
New-Item -Path "user_data/stop_signal" -ItemType File
```

### 4.2 恢復訓練

```powershell
# 啟動虛擬環境
.\.venv\Scripts\Activate.ps1

# 從最後的檢查點恢復（需要 continual_learning: true）
freqtrade backtesting `
  --strategy RLStrategy `
  --config user_data/config_dqn_gpu.json `
  --freqaimodel ReinforcementLearner `
  --timerange 20231201-20240130
```

### 4.3 強制停止

```powershell
# 尋找 FreqTrade 進程
Get-Process | Where-Object { $_.ProcessName -like "*python*" }

# 終止特定進程
Stop-Process -Name python -Force

# 或根據 PID 終止
Stop-Process -Id <PID> -Force
```

---

## 5. 訓練後驗證

### 5.1 檢查已保存的模型

```powershell
# 列出所有已保存的模型
Get-ChildItem -Path "user_data/models/DQN_GPU_RTX2070" -Recurse -Filter "*.zip"

# 特別檢查最佳模型
Get-ChildItem -Path "user_data/models/DQN_GPU_RTX2070" -Recurse -Filter "best_model.zip"

# 顯示模型詳細信息
Get-ChildItem -Path "user_data/models/DQN_GPU_RTX2070" -Recurse -Filter "*.zip" | Format-Table Name, Length, LastWriteTime
```

### 5.2 查看訓練統計數據

```powershell
# 啟動虛擬環境
.\.venv\Scripts\Activate.ps1

# 顯示回測結果摘要
freqtrade backtesting-show

# 顯示特定配置的詳細結果
freqtrade backtesting-show --config user_data/config_dqn_gpu.json

# 導出可讀格式的結果
freqtrade backtesting-analysis --config user_data/config_dqn_gpu.json
```

### 5.3 驗證 Tensorboard 指標

```powershell
# 啟動虛擬環境並啟動 Tensorboard 檢視結果
.\.venv\Scripts\Activate.ps1
tensorboard --logdir user_data/models/DQN_GPU_RTX2070
```

**關鍵指標檢查**:
- `rollout/ep_rew_mean`: 應隨時間增加
- `rollout/ep_len_mean`: 回合長度穩定性
- `train/loss`: 應隨時間減少
- `train/exploration_rate`: 應從 1.0 衰減到 0.05
- `eval/mean_reward`: 評估性能

### 5.4 測試模型性能

```powershell
# 啟動虛擬環境
.\.venv\Scripts\Activate.ps1

# 在未見過的數據上測試（2024年3月）
freqtrade backtesting `
  --strategy RLStrategy `
  --config user_data/config_dqn_gpu.json `
  --freqaimodel ReinforcementLearner `
  --timerange 20240301-20240331 `
  --export trades

# 生成圖表
freqtrade plot-profit `
  --config user_data/config_dqn_gpu.json `
  --strategy RLStrategy `
  --export-filename user_data/backtest_results/backtest-result.json
```

---

## 6. 故障排除

### 6.1 訓練卡住或非常慢

```powershell
# 檢查 GPU 使用率
nvidia-smi

# 檢查 CPU 使用情況
Get-Process python | Select-Object CPU, PM

# 如果 GPU 記憶體滿了，減少 batch_size
# 編輯配置："batch_size": 256（而不是 512）

# 如果系統 RAM 滿了，減少 buffer_size
# 編輯配置："buffer_size": 500000（而不是 1000000）
```

### 6.2 CUDA 記憶體不足

```powershell
# 檢查可用的 GPU 記憶體
nvidia-smi

# 解決方案：
# 1. 在配置中減少 batch_size
"model_training_parameters": {
    "batch_size": 256  # 從 512 減少
}

# 2. 訓練前清除 GPU 快取
.\.venv\Scripts\Activate.ps1; python -c "import torch; torch.cuda.empty_cache()"

# 3. 重新啟動訓練進程
```

### 6.3 訓練未使用 GPU

```powershell
# 啟動虛擬環境
.\.venv\Scripts\Activate.ps1

# 驗證 CUDA 安裝
python -c "import torch; print(torch.cuda.is_available())"

# 檢查配置中的設備設定
freqtrade show-config --config user_data/config_dqn_gpu.json | Select-String "device"

# 應該顯示："device": "cuda"

# 如果正在使用 CPU，驗證 PyTorch CUDA 版本
python -c "import torch; print(torch.version.cuda); print(torch.cuda.get_device_name(0))"
```

### 6.4 最佳模型未保存

```powershell
# 檢查 eval_callback 是否啟用（應該是自動的）
# 驗證模型路徑是否存在
Get-ChildItem -Path "user_data/models/DQN_GPU_RTX2070" -Recurse

# 在訓練日誌中搜尋 "best_model" 提及
Select-String -Path "user_data/logs/freqtrade.log" -Pattern "best_model"

# 手動定位最佳模型
Get-ChildItem -Path "user_data/models" -Recurse -Filter "best_model.zip"
```

---

## 7. 完整訓練腳本

### 7.1 完整訓練工作流程 (train.ps1)

```powershell
# 文件：train.ps1
# 完整的 FreqAI RL 訓練工作流程

Write-Host "=== FreqAI RL 訓練工作流程 ===" -ForegroundColor Cyan

# 1. 啟動虛擬環境
Write-Host "`n[1/7] 正在啟動虛擬環境..." -ForegroundColor Yellow
.\.venv\Scripts\Activate.ps1

# 2. 驗證 GPU 可用性
Write-Host "`n[2/7] 正在檢查 GPU 可用性..." -ForegroundColor Yellow
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}')"

# 3. 驗證數據可用性
Write-Host "`n[3/7] 正在檢查訓練數據..." -ForegroundColor Yellow
freqtrade list-data --exchange binance --trading-mode futures --show-timerange

# 4. 在獨立視窗啟動 Tensorboard
Write-Host "`n[4/7] 正在啟動 Tensorboard..." -ForegroundColor Yellow
Start-Process pwsh -ArgumentList "-Command", "tensorboard --logdir user_data/models/DQN_GPU_RTX2070"
Write-Host "Tensorboard 已啟動於 http://localhost:6006" -ForegroundColor Green

# 5. 在獨立視窗啟動 GPU 監控
Write-Host "`n[5/7] 正在啟動 GPU 監控..." -ForegroundColor Yellow
Start-Process pwsh -ArgumentList "-Command", "while (`$true) { Clear-Host; nvidia-smi; Start-Sleep -Seconds 2 }"
Write-Host "GPU 監控已啟動" -ForegroundColor Green

# 6. 等待用戶確認
Write-Host "`n[6/7] 準備開始訓練" -ForegroundColor Yellow
Write-Host "訓練配置：" -ForegroundColor Cyan
Write-Host "  - 演算法：DQN" -ForegroundColor White
Write-Host "  - 設備：CUDA (RTX 2070)" -ForegroundColor White
Write-Host "  - 訓練周期：30" -ForegroundColor White
Write-Host "  - 交易對：BTC/USDT:USDT, ETH/USDT:USDT" -ForegroundColor White
Write-Host "  - 時間範圍：2023-12-01 至 2024-01-30" -ForegroundColor White
Write-Host "  - 預計時長：約 10.5 小時" -ForegroundColor White
Read-Host "`n按 Enter 鍵開始訓練"

# 7. 執行訓練
Write-Host "`n[7/7] 正在開始訓練..." -ForegroundColor Yellow
freqtrade backtesting `
  --strategy RLStrategy `
  --config user_data/config_dqn_gpu.json `
  --freqaimodel ReinforcementLearner `
  --timerange 20231201-20240130 `
  --export trades

# 訓練完成
Write-Host "`n=== 訓練完成 ===" -ForegroundColor Green
Write-Host "在 Tensorboard 中檢查結果：http://localhost:6006" -ForegroundColor Cyan
Write-Host "模型已保存到：user_data/models/DQN_GPU_RTX2070" -ForegroundColor Cyan
```

### 7.2 快速測試腳本 (quick_test.ps1)

```powershell
# 文件：quick_test.ps1
# 快速 10 天訓練測試

Write-Host "=== 快速訓練測試 (10 天) ===" -ForegroundColor Cyan

# 啟動虛擬環境
.\.venv\Scripts\Activate.ps1

# 快速測試
freqtrade backtesting `
  --strategy RLStrategy `
  --config user_data/config_dqn_gpu.json `
  --freqaimodel ReinforcementLearner `
  --timerange 20231201-20231211 `
  --export trades

Write-Host "`n測試完成！" -ForegroundColor Green
```

---

## 8. Quick Reference

### 8.1 Core Commands Summary

| Phase | Command | Purpose |
|-------|---------|---------|
| **Pre-Training** | `.\.venv\Scripts\Activate.ps1; freqtrade list-data --exchange binance --trading-mode futures --show-timerange` | Verify data |
| | `.\.venv\Scripts\Activate.ps1; python -c "import torch; print(torch.cuda.is_available())"` | Check GPU |
| | `.\.venv\Scripts\Activate.ps1; freqtrade list-strategies --config user_data/config_dqn_gpu.json` | Validate strategy |
| **Training** | `.\.venv\Scripts\Activate.ps1; freqtrade backtesting --strategy RLStrategy --config user_data/config_dqn_gpu.json --freqaimodel ReinforcementLearner --timerange 20231201-20240130` | Start training |
| **Monitoring** | `.\.venv\Scripts\Activate.ps1; tensorboard --logdir user_data/models/DQN_GPU_RTX2070` | Launch Tensorboard |
| | `Get-Content user_data/logs/freqtrade.log -Wait -Tail 50` | Watch logs |
| | `nvidia-smi -l 2` | Monitor GPU |
| **Control** | `Ctrl+C` | Pause training |
| | `Stop-Process -Name python -Force` | Force stop |
| **Validation** | `.\.venv\Scripts\Activate.ps1; freqtrade backtesting-show` | View results |
| | `.\.venv\Scripts\Activate.ps1; freqtrade plot-profit --config user_data/config_dqn_gpu.json --strategy RLStrategy` | Generate plots |

### 8.2 Training Workflow Checklist

#### First-Time Training

- [ ] Activate virtual environment: `.\.venv\Scripts\Activate.ps1`
- [ ] Verify GPU: `.\.venv\Scripts\Activate.ps1; python -c "import torch; print(torch.cuda.is_available())"`
- [ ] Check data: `.\.venv\Scripts\Activate.ps1; freqtrade list-data --exchange binance --trading-mode futures --show-timerange`
- [ ] Validate config: `.\.venv\Scripts\Activate.ps1; freqtrade show-config --config user_data/config_dqn_gpu.json`
- [ ] Start Tensorboard: `Start-Process pwsh -ArgumentList "-Command", ".\.venv\Scripts\Activate.ps1; tensorboard --logdir user_data/models/DQN_GPU_RTX2070"`
- [ ] Start GPU monitor: `nvidia-smi -l 2` (in separate terminal)
- [ ] Execute training: `.\.venv\Scripts\Activate.ps1; freqtrade backtesting --strategy RLStrategy --config user_data/config_dqn_gpu.json --freqaimodel ReinforcementLearner --timerange 20231201-20240130 --export trades`
- [ ] Monitor Tensorboard: http://localhost:6006
- [ ] Review results: `.\.venv\Scripts\Activate.ps1; freqtrade backtesting-show`

#### 後續訓練會話

- [ ] 啟動虛擬環境：`.\.venv\Scripts\Activate.ps1`
- [ ] 啟動 Tensorboard：`.\.venv\Scripts\Activate.ps1; tensorboard --logdir user_data/models/DQN_GPU_RTX2070`
- [ ] 執行訓練：`.\.venv\Scripts\Activate.ps1; freqtrade backtesting --strategy RLStrategy --config user_data/config_dqn_gpu.json --freqaimodel ReinforcementLearner --timerange <new_range>`
- [ ] 在 Tensorboard 中監控指標
- [ ] 與之前的運行比較結果

### 8.3 重要路徑

```
訓練配置:        user_data/config_dqn_gpu.json
策略:               user_data/strategies/RLStrategy.py
模型目錄:       user_data/models/DQN_GPU_RTX2070/
最佳模型:             user_data/models/DQN_GPU_RTX2070/*/data_path/best_model.zip
訓練日誌:          user_data/logs/freqtrade.log
Tensorboard 日誌:       user_data/models/DQN_GPU_RTX2070/*/tensorboard/
回測結果:       user_data/backtest_results/
交易導出:           user_data/backtest_results/*.json
```

### 8.4 常見問題快速修復

| 問題 | 快速修復 |
|-------|----------|
| 訓練未使用 GPU | 檢查：`"device": "cuda"` 在配置中 |
| CUDA 記憶體不足 | 減少 `batch_size` 到 256 |
| 訓練非常慢 | 用 `nvidia-smi` 檢查 GPU 使用率 |
| 最佳模型未保存 | 在 `user_data/models/DQN_GPU_RTX2070/*/data_path/` 中查找 |
| Tensorboard 為空 | 驗證訓練實際運行了（檢查日誌） |
| 無法恢復訓練 | 在配置中啟用 `"continual_learning": true` |

---

## 注意事項

### ⚠️ 虛擬環境（最重要）

**所有指令都必須在虛擬環境中執行！**

- 啟動方式：`.\.venv\Scripts\Activate.ps1`
- 驗證方式：`python -c "import sys; print(sys.executable)"` 應顯示 `.venv` 路徑
- 環境狀態：已驗證包含所有必要套件（PyTorch 2.7.1+cu118, Stable-Baselines3 2.7.0, Gymnasium 0.29.1）
- GPU 支援：已確認 CUDA 11.8 在 venv 中正常工作

### 其他注意事項

- **訓練時長**: 使用 `train_cycles: 30`，預期每個周期約 21 分鐘（總計約 10.5 小時）
- **GPU 記憶體**: RTX 2070 8GB 可以處理 `batch_size: 512` 和 `buffer_size: 1000000`
- **最佳模型**: 如果性能改善，會在每個 `train_cycle` 結束時自動保存
- **Tensorboard**: 在訓練前啟動以從一開始就捕獲所有指標
- **數據需求**: 確保有足夠的歷史數據（建議每個 train_cycle 至少 30 天）
- **Python 版本**: Python 3.13.7（已驗證兼容）

---

**參考文檔**: 
- 主要指南：`docs/20251017/tensorboard-metrics-guide.md`
- 訓練暫停/恢復方法：參見 2025-10-17 對話記錄
- 完整測試工作流程：參見 2025-10-17 對話記錄
