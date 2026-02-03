# RL Day Trade - Environment Setup & Execution Guide

> Date: 2026-02-03
> Platform: Windows (PowerShell)
> Python: >= 3.11 (64-bit)
> Related: [rl-daytrade-strategy-analysis.md](./rl-daytrade-strategy-analysis.md)

---

## 1. Prerequisites

| Item | Requirement | Check Command |
|------|-------------|---------------|
| Python | >= 3.11, 64-bit | `python --version` |
| pip | Latest | `python -m pip --version` |
| Git | Any | `git --version` |
| Visual C++ Build Tools | For TA-Lib compilation | See Section 3 |

---

## 2. Create & Activate venv

```powershell
cd C:\Code\freqtrade

# Create venv
python -m venv .venv

# Activate (PowerShell)
.\.venv\Scripts\Activate.ps1

# Verify activation (should show .venv path)
Get-Command python | Select-Object Source

# Upgrade pip
python -m pip install --upgrade pip
```

Activation for other shells:

| Shell | Command |
|-------|---------|
| PowerShell | `.\.venv\Scripts\Activate.ps1` |
| CMD | `.\.venv\Scripts\activate.bat` |
| Git Bash | `source .venv/Scripts/activate` |

---

## 3. Install TA-Lib (Windows)

TA-Lib requires a C library that does not come pre-built on Windows.
Must be installed **before** `pip install` of Python dependencies.

### Option A: Pre-built wheel (recommended)

1. Go to https://github.com/TA-Lib/ta-lib-python/releases
2. Download the `.whl` matching your Python version, e.g.:
   - `ta_lib-0.6.8-cp312-cp312-win_amd64.whl` (Python 3.12)
   - `ta_lib-0.6.8-cp311-cp311-win_amd64.whl` (Python 3.11)
3. Install:

```powershell
pip install path\to\ta_lib-0.6.8-cpXXX-cpXXX-win_amd64.whl
```

### Option B: Build from source

1. Install Visual Studio Build Tools (C++ workload)
2. Download TA-Lib C source: http://prdownloads.sourceforge.net/ta-lib/ta-lib-0.4.0-msvc.zip
3. Unzip to `C:\ta-lib`
4. Then:

```powershell
pip install ta-lib==0.6.8
```

### Verify

```powershell
python -c "import talib; print(talib.__version__)"
# Expected: 0.6.8
```

---

## 4. Install Dependencies

### 4.1 Base install (CPU)

```powershell
# This single file chains:
#   requirements-freqai-rl.txt
#     -> requirements-freqai.txt (scikit-learn, lightgbm, xgboost, tensorboard)
#       -> requirements.txt (numpy, pandas, ccxt, SQLAlchemy, ...)
#       -> requirements-plot.txt (plotly)
#     -> torch, gymnasium, stable_baselines3, sb3_contrib, tqdm

pip install -r requirements-freqai-rl.txt

# Install freqtrade itself (editable mode)
pip install -e .
```

### 4.2 GPU support (CUDA)

`requirements-freqai-rl.txt` installs CPU-only torch by default.
To enable GPU training, reinstall the PyTorch stack from the official CUDA index.

#### Version compatibility matrix (torch 2.10.0)

| Package | Version | Notes |
|---------|---------|-------|
| torch | 2.10.0 | From requirements-freqai-rl.txt |
| torchvision | 0.25.0 | Must match torch version |
| torchaudio | 2.10.0 | Must match torch version |

Source: [PyTorch Versions Wiki](https://github.com/pytorch/pytorch/wiki/PyTorch-Versions)

#### Available CUDA variants (torch 2.10.0)

| CUDA | index-url | Min driver |
|------|-----------|------------|
| 12.6 | `https://download.pytorch.org/whl/cu126` | >= 560.x |
| 12.8 | `https://download.pytorch.org/whl/cu128` | >= 570.x |
| 13.0 | `https://download.pytorch.org/whl/cu130` | >= 580.x |

Check your driver: `nvidia-smi` (top-right shows max supported CUDA version).

#### Install command (CUDA 12.8 recommended)

```powershell
pip install torch==2.10.0 torchvision==0.25.0 torchaudio==2.10.0 `
  --index-url https://download.pytorch.org/whl/cu128 `
  --force-reinstall --no-deps
```

If `pip` reports "already satisfied" without downloading ~2.9 GB, add
`--force-reinstall --no-deps` to replace the CPU wheel with the CUDA wheel.

#### Enable GPU in config

After installing CUDA torch, update `config_daytrade.json`:

```json
"model_training_parameters": {
    "device": "cuda"
}
```

### 4.3 Alternative: pyproject.toml extras

```powershell
# FreqAI + RL only
pip install -e ".[freqai_rl]"

# Or everything (plot, hyperopt, freqai, rl, jupyter)
pip install -e ".[all]"
```

Note: pyproject.toml extras also install CPU-only torch. Follow Section 4.2
to upgrade to CUDA after install.

### Dependency chain

```
requirements-freqai-rl.txt
 |-- requirements-freqai.txt
 |    |-- requirements.txt          # Core: numpy, pandas, ta-lib, ccxt ...
 |    |-- requirements-plot.txt     # plotly
 |    |-- scikit-learn==1.8.0
 |    |-- lightgbm==4.6.0
 |    |-- xgboost==3.1.3
 |    |-- tensorboard==2.20.0
 |    +-- datasieve==0.1.9
 |-- torch==2.10.0                  # CPU default; see 4.2 for CUDA
 |-- torchvision==0.25.0            # auto-installed with torch
 |-- torchaudio==2.10.0             # auto-installed with torch
 |-- gymnasium==1.2.3
 |-- stable_baselines3==2.7.1
 |-- sb3_contrib>=2.2.1
 +-- tqdm==4.67.1
```

### Verify installation

```powershell
python -c "import freqtrade; print('freqtrade OK')"
python -c "import torch; print(f'torch {torch.__version__}, cuda={torch.cuda.is_available()}')"
python -c "import torchvision; print(f'torchvision {torchvision.__version__}')"
python -c "import torchaudio; print(f'torchaudio {torchaudio.__version__}')"
python -c "import stable_baselines3; print(f'sb3 {stable_baselines3.__version__}')"
python -c "import gymnasium; print(f'gymnasium {gymnasium.__version__}')"
python -c "import talib; print(f'ta-lib {talib.__version__}')"
python -c "import sklearn; print(f'sklearn {sklearn.__version__}')"

# GPU-specific check (only if CUDA installed)
python -c "import torch; print(f'GPU: {torch.cuda.get_device_name(0)}')" 2>$null
```

---

## 5. Download Historical Data

FreqAI backtesting requires historical OHLCV data. The download range must cover:
- `train_period_days` (75 days) **before** the backtest start date
- All timeframes in `include_timeframes`
- All pairs in `pair_whitelist` + `include_corr_pairlist`

### Required data

| Pair | Role | Source |
|------|------|--------|
| ETH/USDT:USDT | Trading pair | config `pair_whitelist` |
| SOL/USDT:USDT | Trading pair | config `pair_whitelist` |
| BTC/USDT:USDT | Correlation pair | config `include_corr_pairlist` |

### Download command

```powershell
freqtrade download-data `
  --config user_data/config_daytrade.json `
  --timerange 20231001-20260201 `
  --timeframe 5m 15m 1h 4h
```

Timerange explanation:
- Backtest target: `20240101-20260101`
- First training window needs 75 days before 20240101 = ~20231018
- Set to 20231001 for safety margin

### Verify data

```powershell
freqtrade list-data `
  --config user_data/config_daytrade.json `
  --data-format-ohlcv feather
```

---

## 6. Run Backtesting

### 6.1 Single-environment (debugging)

Use `RLDayTrader` for initial testing. Tensorboard metrics are reliable
in single-env mode.

```powershell
freqtrade backtesting `
  --strategy RLDayTradeStrategy `
  --config user_data/config_daytrade.json `
  --freqaimodel RLDayTrader `
  --timerange 20240101-20260101 `
  --export trades
```

### 6.2 Multi-process (production training)

Use `RLDayTrader_multiproc` for faster training with `SubprocVecEnv`.
Tensorboard metrics are unreliable in this mode.

```powershell
freqtrade backtesting `
  --strategy RLDayTradeStrategy `
  --config user_data/config_daytrade.json `
  --freqaimodel RLDayTrader_multiproc `
  --timerange 20240101-20260101 `
  --export trades
```

### 6.3 Short test run (quick validation)

Use a shorter timerange to verify everything works before full run.

```powershell
freqtrade backtesting `
  --strategy RLDayTradeStrategy `
  --config user_data/config_daytrade.json `
  --freqaimodel RLDayTrader `
  --timerange 20250101-20250201 `
  --export trades
```

---

## 7. Monitor & Evaluate

### 7.1 Tensorboard (single-env only)

```powershell
tensorboard --logdir user_data/models/rl-daytrade-v1
```

Open http://localhost:6006 in browser.

Key metrics to watch:
- `risk/liquidation` - should decrease over training
- `pnl/exit_pnl` - should trend positive
- `pnl/profitable_exit` vs `pnl/loss_exit` - ratio should improve
- `rewards/exit_reward` - should trend positive
- `actions/invalid` - should decrease

### 7.2 Backtest results

```powershell
# Show last backtest result summary
freqtrade backtesting-show

# Results are also saved to:
#   user_data/backtest_results/
```

### 7.3 Plot results (requires plotly)

```powershell
freqtrade plot-dataframe `
  --strategy RLDayTradeStrategy `
  --config user_data/config_daytrade.json `
  --timerange 20250101-20250201
```

### 7.4 AI-powered training analysis (Claude Code)

Use the `/analyze-rl` skill in Claude Code to get AI-powered health analysis
of TensorBoard training logs. It parses event files, computes health metrics,
and provides actionable recommendations.

```
# Analyze default model (rl-daytrade-v1), last 5 windows
/analyze-rl

# Analyze a specific model with more windows
/analyze-rl rl-10x-v2 10
```

Health checks performed:
- Reward trend (improving / flat / declining)
- Liquidation rate (percentage of episodes ending in liquidation)
- Win rate (profitable exits vs loss exits)
- Policy collapse (action distribution balance)
- Value loss stability (SB3 training metric)
- Invalid action rate

The skill runs `scripts/analyze_rl_training.py` under the hood. You can also
run the script directly:

```powershell
python scripts/analyze_rl_training.py --model-id rl-daytrade-v1 --last-n 5
```

---

## 8. File Reference

### Project files (created)

| File | Path | Description |
|------|------|-------------|
| Config | `user_data/config_daytrade.json` | FreqAI RL configuration |
| Strategy | `user_data/strategies/RLDayTradeStrategy.py` | Feature engineering + S/R Flip + entry/exit |
| Model | `user_data/freqaimodels/RLDayTrader.py` | Reward function + leverage + liquidation |
| Model (MP) | `user_data/freqaimodels/RLDayTrader_multiproc.py` | Multi-process version |

### Generated by training

| Path | Description |
|------|-------------|
| `user_data/models/rl-daytrade-v1/` | Trained model files |
| `user_data/models/rl-daytrade-v1/sub-train-*/` | Per-window training artifacts |
| `user_data/backtest_results/` | Backtest result JSON |

---

## 9. Troubleshooting

### TA-Lib installation fails

```
ERROR: Could not build wheels for ta-lib
```

Solution: Use pre-built wheel (Section 3, Option A).

### torch installation fails or too slow

```powershell
# Install CPU-only version (smaller download, ~114 MB vs ~2.9 GB)
pip install torch==2.10.0 torchvision==0.25.0 torchaudio==2.10.0 `
  --index-url https://download.pytorch.org/whl/cpu
```

### torchaudio/torchvision version conflict after pip install

```
torchaudio X.X.X requires torch==X.X.X, but you have torch Y.Y.Y which is incompatible.
```

This happens when `requirements-freqai-rl.txt` upgrades torch but not
torchaudio/torchvision. Fix by installing matching versions:

```powershell
# CPU
pip install torchaudio==2.10.0 torchvision==0.25.0

# Or with CUDA (see Section 4.2)
pip install torch==2.10.0 torchvision==0.25.0 torchaudio==2.10.0 `
  --index-url https://download.pytorch.org/whl/cu128 `
  --force-reinstall --no-deps
```

### Out of memory during training

Reduce parallel environments:

```json
// config_daytrade.json -> rl_config
"cpu_count": 4  // reduce from 16
```

Or reduce network size:

```json
"net_arch": [128, 128]  // reduce from [256, 256]
```

### Very short training episodes

If episodes end almost immediately, `max_training_drawdown_pct` may be
too aggressive with leverage. See analysis doc Section 8.7.

```json
// Try raising from 0.50 to 0.70
"max_training_drawdown_pct": 0.70
```

### ModuleNotFoundError: No module named 'freqtrade'

```powershell
# Ensure editable install
pip install -e .
```

### "No data found" error

```powershell
# Verify data exists for the required timerange
freqtrade list-data --config user_data/config_daytrade.json

# Re-download if missing
freqtrade download-data `
  --config user_data/config_daytrade.json `
  --timerange 20231001-20260201 `
  --timeframe 5m 15m 1h 4h
```

### cpu_count capped below configured value

`BaseReinforcementLearningModel` caps `cpu_count` to `system_cores / 2`.
`RLDayTrader_multiproc` overrides this cap, but verify with log output:

```
RLDayTrader_multiproc: Overriding max_threads from X to 16
```

If this message does not appear, your system has >= 32 cores and the
override was not needed.

---

## 10. Quick Reference (copy-paste)

Full setup from scratch in one block:

```powershell
cd C:\Code\freqtrade

# 1. Create and activate venv
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip

# 2. Install TA-Lib wheel first (adjust filename for your Python version)
# pip install path\to\ta_lib-0.6.8-cpXXX-cpXXX-win_amd64.whl

# 3. Install all dependencies (CPU)
pip install -r requirements-freqai-rl.txt
pip install -e .

# 4. (Optional) Upgrade to GPU / CUDA 12.8
pip install torch==2.10.0 torchvision==0.25.0 torchaudio==2.10.0 `
  --index-url https://download.pytorch.org/whl/cu128 `
  --force-reinstall --no-deps
# Then set "device": "cuda" in config_daytrade.json

# 5. Download data
freqtrade download-data `
  --config user_data/config_daytrade.json `
  --timerange 20231001-20260201 `
  --timeframe 5m 15m 1h 4h

# 6. Quick validation run (1 month)
freqtrade backtesting `
  --strategy RLDayTradeStrategy `
  --config user_data/config_daytrade.json `
  --freqaimodel RLDayTrader `
  --timerange 20250101-20250201 `
  --export trades

# 7. Full backtest (single-env, with Tensorboard)
freqtrade backtesting `
  --strategy RLDayTradeStrategy `
  --config user_data/config_daytrade.json `
  --freqaimodel RLDayTrader `
  --timerange 20240101-20260101 `
  --export trades

# 8. Full backtest (multi-process, production)
freqtrade backtesting `
  --strategy RLDayTradeStrategy `
  --config user_data/config_daytrade.json `
  --freqaimodel RLDayTrader_multiproc `
  --timerange 20240101-20260101 `
  --export trades
```
