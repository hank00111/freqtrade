# Freqtrade FreqAI Setup Guide for Google Colab

This guide provides step-by-step instructions to set up and run Freqtrade with FreqAI (including Reinforcement Learning) on Google Colab.

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [System Setup](#system-setup)
3. [Installing TA-Lib](#installing-ta-lib)
4. [Cloning Freqtrade Repository](#cloning-freqtrade-repository)
5. [Installing Dependencies](#installing-dependencies)
6. [Configuration Setup](#configuration-setup)
7. [Downloading Training Data](#downloading-training-data)
8. [Running Training](#running-training)
9. [Monitoring Training](#monitoring-training)
10. [Troubleshooting](#troubleshooting)

---

## Prerequisites

### Enable GPU Runtime

1. Go to **Runtime** → **Change runtime type**
2. Select **Hardware accelerator: GPU** (T4 is standard)
3. Click **Save**

### Verify GPU Availability

```python
# Check GPU availability
import torch
print(f"CUDA Available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU Device: {torch.cuda.get_device_name(0)}")
    print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
```

---

## System Setup

### Update System Packages

```bash
# Update package lists
!apt-get update

# Install essential build tools and dependencies
!apt-get install -y \
    build-essential \
    wget \
    curl \
    git \
    python3-dev \
    python3-pip \
    libffi-dev \
    libssl-dev \
    libatlas-base-dev
```

---

## Proxy Configuration (Optional but Recommended)

**⚠️ Important**: Binance and some other exchanges restrict access from certain geographic locations and data center IPs (including Google Colab IPs). If you encounter **HTTP 451 errors** or region restriction errors, you'll need to configure a proxy.

### Check if You Need a Proxy

```python
# Test if you can reach Binance API
import requests

try:
    response = requests.get('https://api.binance.com/api/v3/ping', timeout=10)
    if response.status_code == 200:
        print("✓ Can access Binance API - No proxy needed")
    elif response.status_code == 451:
        print("✗ HTTP 451 - Region restricted! Proxy required")
    else:
        print(f"⚠ Unexpected status: {response.status_code}")
except Exception as e:
    print(f"✗ Connection failed: {e}")
```

### VLESS Proxy Setup (One-Click Installation)

VLESS is a VPN protocol that requires xray-core client. This setup creates a local proxy tunnel that Freqtrade can use.

**Complete Setup Script** - Run this in a Colab cell:

```bash
#!/bin/bash

echo "=== VLESS Proxy Setup for Google Colab ==="

# Step 1: Install xray-core
echo "Installing xray-core..."
bash -c "$(curl -L https://github.com/XTLS/Xray-install/raw/main/install-release.sh)" @ install

# Step 2: Create VLESS configuration
echo "Creating VLESS configuration..."
cat > /usr/local/etc/xray/config.json << 'EOF'
{
  "inbounds": [
    {
      "port": 10808,
      "listen": "127.0.0.1",
      "protocol": "socks",
      "settings": {
        "udp": true
      }
    },
    {
      "port": 10809,
      "listen": "127.0.0.1",
      "protocol": "http"
    }
  ],
  "outbounds": [
    {
      "protocol": "vless",
      "settings": {
        "vnext": [
          {
            "address": "jpanireki.com",
            "port": 443,
            "users": [
              {
                "id": "YOUR_UUID_HERE",
                "encryption": "none"
              }
            ]
          }
        ]
      },
      "streamSettings": {
        "network": "ws",
        "security": "tls",
        "wsSettings": {
          "path": "/YOUR_PATH_HERE"
        }
      }
    }
  ]
}
EOF

# Step 3: Start xray-core service
echo "Starting xray-core..."
systemctl start xray
systemctl enable xray

# Step 4: Wait for service to start
sleep 3

# Step 5: Verify service is running
if systemctl is-active --quiet xray; then
    echo "✓ Xray service started successfully"
    echo "✓ SOCKS5 proxy: 127.0.0.1:10808"
    echo "✓ HTTP proxy: 127.0.0.1:10809"
else
    echo "✗ Failed to start xray service"
    exit 1
fi

echo "=== Setup Complete ==="
```

**⚠️ Important**: Replace `YOUR_UUID_HERE` and `/YOUR_PATH_HERE` with your actual VLESS credentials.

### Configure Environment Variables

After VLESS proxy is running, configure Python to use the local proxy:

```python
import os

# Set proxy to use local xray-core HTTP proxy
os.environ['HTTP_PROXY'] = 'http://127.0.0.1:10809'
os.environ['HTTPS_PROXY'] = 'http://127.0.0.1:10809'

print("✓ Proxy configured to use VLESS tunnel")
print(f"  HTTP_PROXY: {os.environ['HTTP_PROXY']}")
print(f"  HTTPS_PROXY: {os.environ['HTTPS_PROXY']}")
```

### Verify Proxy Connection

```python
import requests

# Test proxy connection to Binance
try:
    response = requests.get('https://api.binance.com/api/v3/ping', timeout=10)
    if response.status_code == 200:
        print("✓ VLESS proxy working! Can access Binance API")
        
        # Check your proxy IP
        ip_response = requests.get('https://api.ipify.org?format=json')
        print(f"✓ Your exit IP: {ip_response.json()['ip']}")
    else:
        print(f"✗ Proxy test failed: Status {response.status_code}")
except Exception as e:
    print(f"✗ Proxy connection error: {e}")
```

### Configure Freqtrade Config File

For training commands, also add proxy to your configuration:

```python
import json

config_path = 'user_data/config_freqai.json'

# Load existing config
with open(config_path, 'r') as f:
    config = json.load(f)

# Add VLESS proxy settings (using local HTTP proxy)
if 'ccxt_config' not in config['exchange']:
    config['exchange']['ccxt_config'] = {}

config['exchange']['ccxt_config']['httpsProxy'] = 'http://127.0.0.1:10809'

# Save updated config
with open(config_path, 'w') as f:
    json.dump(config, f, indent=4)

print("✓ VLESS proxy added to Freqtrade configuration")
```

### Troubleshooting VLESS Proxy

**Check xray-core service status:**

```bash
!systemctl status xray
```

**View xray-core logs:**

```bash
!journalctl -u xray -n 50
```

**Restart xray-core if needed:**

```bash
!systemctl restart xray
```

### Alternative: Use Different Exchanges

If VLESS proxy doesn't work for your region, consider using exchanges without restrictions:

```bash
# Use OKX instead of Binance
!freqtrade download-data --exchange okx --pairs BTC/USDT:USDT --timeframe 5m

# Or use Bybit
!freqtrade download-data --exchange bybit --pairs BTC/USDT:USDT --timeframe 5m
```

**Note**: Different exchanges may have different data availability and pair naming conventions.

---

## Installing TA-Lib

TA-Lib is a critical dependency for Freqtrade. Here's how to install it on Colab:

```bash
# Download and install TA-Lib C library
!wget http://prdownloads.sourceforge.net/ta-lib/ta-lib-0.4.0-src.tar.gz
!tar -xzf ta-lib-0.4.0-src.tar.gz
%cd ta-lib/
!./configure --prefix=/usr
!make
!make install

# Return to root directory
%cd /content

# Install TA-Lib Python wrapper
!pip install TA-Lib
```

### Verify TA-Lib Installation

```python
# Verify TA-Lib is working
import talib
print(f"TA-Lib version: {talib.__version__}")
print("TA-Lib installed successfully!")
```

---

## Cloning Freqtrade Repository

```bash
# Clone the Freqtrade repository (develop branch)
!git clone https://github.com/freqtrade/freqtrade.git
%cd freqtrade

# Verify the clone
!pwd
!ls -la
```

---

## Installing Dependencies

### Install Base Dependencies

```bash
# Upgrade pip
!python -m pip install --upgrade pip

# Install base Freqtrade requirements (this may take 2-3 minutes)
!pip install -r requirements.txt
```

**✓ Verify**: Check that pandas, numpy, ccxt are installed
```python
!pip list | grep -E "(pandas|numpy|ccxt)"
```

### Install FreqAI Dependencies

```bash
# Install FreqAI requirements (this may take 3-5 minutes)
!pip install -r requirements-freqai.txt
```

**✓ Verify**: Check that scikit-learn is installed
```python
!pip list | grep "scikit-learn"
```

### Install Reinforcement Learning Dependencies

```bash
# Install FreqAI-RL requirements (includes PyTorch, Stable-Baselines3, etc.)
# This may take 5-10 minutes
!pip install -r requirements-freqai-rl.txt
```

**✓ Verify**: Check that torch and stable-baselines3 are installed
```python
!pip list | grep -E "(torch|stable-baselines3)"
```

### Install Freqtrade in Editable Mode

**⚠️ Critical Step**: This command installs the `freqtrade` command to your system.

```bash
# Install Freqtrade package
!pip install -e .
```

**Wait for completion** - This may take 1-2 minutes. Look for messages like:
```
Successfully installed freqtrade
```

### Verify Installation

```python
# Verify Freqtrade installation
!freqtrade --version

# Should output something like: freqtrade 2024.x
```

**Troubleshooting**: If you get `command not found`:

```bash
# Check if installed
!pip list | grep freqtrade

# Try module approach
!python -m freqtrade --version

# If still not working, reinstall
!pip install --force-reinstall -e .
```

---

## Configuration Setup

### Create User Data Directory

```bash
# Create user data directory structure
!freqtrade create-userdir --userdir user_data
```

### Create Configuration File

You have two options:

#### Option 1: Interactive Configuration

```bash
# Create config interactively (will prompt for inputs)
!freqtrade new-config --config user_data/config.json
```

#### Option 2: Use Example Configuration

```bash
# Copy example FreqAI config
!cp config_examples/config_freqai.example.json user_data/config_freqai.json
```

### Customize Configuration for Training

Create or modify `user_data/config_freqai.json`:

```python
import json

# Load example config
with open('config_examples/config_freqai.example.json', 'r') as f:
    config = json.load(f)

# Customize for Colab training
config.update({
    "dry_run": True,
    "timeframe": "5m",
    "stake_amount": 100,
    "max_open_trades": 3,
    "exchange": {
        "name": "binance",
        "key": "",
        "secret": "",
        "pair_whitelist": [
            "BTC/USDT:USDT",
            "ETH/USDT:USDT"
        ],
        "pair_blacklist": []
    }
})

# Customize FreqAI settings
config["freqai"] = {
    "enabled": True,
    "purge_old_models": 2,
    "train_period_days": 10,
    "backtest_period_days": 3,
    "identifier": "colab-rl-model",
    "feature_parameters": {
        "include_timeframes": ["5m", "15m", "1h"],
        "include_corr_pairlist": ["BTC/USDT:USDT"],
        "label_period_candles": 24,
        "include_shifted_candles": 2,
        "DI_threshold": 0.9,
        "weight_factor": 0.9,
        "principal_component_analysis": False,
        "use_SVM_to_remove_outliers": True,
        "indicator_periods_candles": [10, 20]
    },
    "data_split_parameters": {
        "test_size": 0.33,
        "random_state": 1
    },
    "model_training_parameters": {}
}

# Add RL config for reinforcement learning
config["freqai"]["rl_config"] = {
    "train_cycles": 25,
    "add_state_info": True,
    "max_trade_duration_candles": 300,
    "max_training_drawdown_pct": 0.02,
    "cpu_count": 2,
    "model_type": "PPO",
    "policy_type": "MlpPolicy",
    "model_reward_parameters": {
        "rr": 1,
        "profit_aim": 0.025
    }
}

# Save customized config
with open('user_data/config_freqai.json', 'w') as f:
    json.dump(config, f, indent=4)

print("Configuration file created: user_data/config_freqai.json")
```

---

## Downloading Training Data

### Download Historical Data

Before training, you need to download historical market data. The data range should be larger than your training range.

**⚠️ Important**: If you get **HTTP 451 errors** (region restriction), configure proxy first (see [Proxy Configuration](#proxy-configuration-optional-but-recommended) section).

```bash
# Download data for backtesting
# Format: YYYYMMDD-YYYYMMDD
# Example: Download 2 months of data

!freqtrade download-data \
    --config user_data/config_freqai.json \
    --timerange 20240101-20240301 \
    --timeframes 5m 15m 1h \
    --exchange binance \
    --trading-mode futures
```

**If you encounter region restriction errors**:

```python
# Option 1: Set proxy in environment (recommended)
import os
os.environ['HTTP_PROXY'] = 'http://your-proxy:port'
os.environ['HTTPS_PROXY'] = 'http://your-proxy:port'

# Then retry download command
```

```bash
# Option 2: Use a different exchange
!freqtrade download-data \
    --config user_data/config_freqai.json \
    --timerange 20240101-20240301 \
    --timeframes 5m 15m 1h \
    --exchange okx \
    --trading-mode futures
```

### Calculate Required Data Range

For FreqAI training, you need additional data before your training period:

```
Required start date = Training start date - train_period_days - (startup_candle_count * timeframe / 24h)
```

**Example:**
- Training range: `20240201-20240301` (1 month)
- `train_period_days`: 10
- `startup_candle_count`: 40
- Max `timeframe`: 1h

Required data: `20240201 - 10 days - (40 * 1h / 24h) = 20240120`

So download from `20240120-20240301`.

### Verify Downloaded Data

```python
# List downloaded data
!ls -lh user_data/data/binance/futures/
```

---

## Running Training

**⚠️ CRITICAL**: Before running any training command, ensure you have configured proxy if you're getting HTTP 451 errors. See [Proxy Configuration](#proxy-configuration-optional-but-recommended) section.

**Why proxy is needed even with local data?**  
Freqtrade backtesting requires connection to exchange API to fetch market metadata (pair info, precision, fees) even when using local data files. This is by design and cannot be disabled.

### Quick Proxy Setup (Before Training)

```python
import os

# Set proxy in the SAME cell where you run backtesting
os.environ['HTTP_PROXY'] = 'http://your-proxy:port'
os.environ['HTTPS_PROXY'] = 'http://your-proxy:port'

# Verify proxy works
import requests
response = requests.get('https://api.binance.com/api/v3/ping', timeout=10)
print(f"Proxy status: {response.status_code}")  # Should be 200
```

### Method 1: Backtesting with FreqAI (Recommended)

This is the primary way to train FreqAI models:

```bash
# Basic FreqAI backtesting (trains models during backtest)
!freqtrade backtesting \
    --strategy FreqaiExampleStrategy \
    --config user_data/config_freqai.json \
    --freqaimodel LightGBMRegressor \
    --timerange 20240201-20240301
```

### Method 2: Reinforcement Learning Training

```bash
# Train with Reinforcement Learning
!freqtrade backtesting \
    --strategy ReinforcementLearningStrategy \
    --config user_data/config_freqai.json \
    --freqaimodel ReinforcementLearner \
    --timerange 20240201-20240301
```

### Method 3: Custom Model Training

```bash
# Use a custom model from user_data/freqaimodels/
!freqtrade backtesting \
    --strategy YourCustomStrategy \
    --config user_data/config_freqai.json \
    --freqaimodel YourCustomModel \
    --timerange 20240201-20240301
```

### Training Parameters Explanation

- `--strategy`: The trading strategy to use (must have FreqAI methods)
- `--config`: Path to your configuration file
- `--freqaimodel`: The ML model to use (LightGBMRegressor, ReinforcementLearner, etc.)
- `--timerange`: Date range for backtesting (format: YYYYMMDD-YYYYMMDD)

### Additional Useful Parameters

```bash
# Enable verbose output
--verbose

# Save trained models for later use
# (automatically saved in user_data/models/unique-id/)

# Use multiple cores
--job-count 2
```

---

## Monitoring Training

### Check Training Progress

```python
# Monitor training in real-time
import time
import os

def monitor_training():
    """Monitor FreqAI training progress"""
    model_dir = "user_data/models/colab-rl-model"
    
    while True:
        if os.path.exists(model_dir):
            print("\nCurrent models:")
            !ls -lh {model_dir}
            print("\nRecent training logs:")
            !tail -20 user_data/logs/freqtrade.log
        time.sleep(30)  # Check every 30 seconds

# Run in background or separate cell
# monitor_training()
```

### View Training Logs

```bash
# View full training log
!cat user_data/logs/freqtrade.log

# View last 50 lines of log
!tail -50 user_data/logs/freqtrade.log

# Follow log in real-time
!tail -f user_data/logs/freqtrade.log
```

### Check Trained Models

```bash
# List all trained models
!ls -lh user_data/models/

# View specific model directory
!ls -lh user_data/models/colab-rl-model/
```

---

## Using Tensorboard (Optional)

FreqAI supports Tensorboard for monitoring training metrics:

### Enable Tensorboard in Config

Make sure `activate_tensorboard` is `True` in your config:

```python
config["freqai"]["activate_tensorboard"] = True
```

### Install Tensorboard Extension for Colab

```bash
!pip install tensorboard
```

### Launch Tensorboard

```python
# Load Tensorboard extension
%load_ext tensorboard

# Launch Tensorboard
%tensorboard --logdir user_data/models/colab-rl-model
```

---

## Analyzing Results

### View Backtest Results

```bash
# Show backtest results
!freqtrade backtesting-show
```

### Generate Backtest Report

```bash
# Generate detailed backtest report
!freqtrade backtesting-analysis \
    --config user_data/config_freqai.json \
    --analysis-groups 0 1 2
```

### Plot Results

```bash
# Install plotting dependencies if not already installed
!pip install -r requirements-plot.txt

# Generate plots
!freqtrade plot-dataframe \
    --config user_data/config_freqai.json \
    --strategy FreqaiExampleStrategy \
    --timerange 20240201-20240210
```

---

## Saving and Downloading Models

### Archive Trained Models

```python
import shutil

# Create archive of trained models
!tar -czf trained_models.tar.gz user_data/models/

# Verify archive
!ls -lh trained_models.tar.gz
```

### Download to Local Machine

```python
from google.colab import files

# Download the archive
files.download('trained_models.tar.gz')

# Or download specific files
files.download('user_data/config_freqai.json')
```

---

## Troubleshooting

### Common Issues and Solutions

#### 0. Command Not Found: freqtrade

**Symptom**: `-bash: freqtrade: command not found`

**Cause**: Freqtrade package not installed or installation failed

**Solution**:
```bash
# Check current directory (must be in /content/freqtrade)
!pwd

# Verify you have the source code
!ls -la | grep pyproject.toml

# Install or reinstall Freqtrade
%cd /content/freqtrade
!pip install -e .

# Verify installation
!freqtrade --version

# If still not working, try module approach
!python -m freqtrade --version
```

**Alternative**: Use `python -m freqtrade` instead of `freqtrade` for all commands

#### 1. Out of Memory (OOM) Errors

```python
# Reduce training parameters in config
config["freqai"]["train_period_days"] = 7  # Reduce from 10
config["freqai"]["rl_config"]["train_cycles"] = 15  # Reduce from 25
```

#### 2. TA-Lib Import Errors

```bash
# Reinstall TA-Lib
!pip uninstall -y TA-Lib
!pip install TA-Lib --no-cache-dir
```

#### 3. CUDA/GPU Errors

```python
# Force CPU training (slower but more stable)
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""
```

#### 4. Region Restriction (HTTP 451 Error)

**Symptom**: 
```
HTTP 451 - Service unavailable from a restricted location
binance GET https://api.binance.com/api/v3/exchangeInfo 451
```

**Cause**: Your IP address is in a restricted region or identified as a data center IP

**Solutions**:

```python
# Solution 1: Configure proxy (recommended)
import os
os.environ['HTTP_PROXY'] = 'http://your-proxy-server:port'
os.environ['HTTPS_PROXY'] = 'http://your-proxy-server:port'

# Verify proxy works
import requests
response = requests.get('https://api.binance.com/api/v3/ping', timeout=10)
print(f"Status: {response.status_code}")  # Should be 200
```

```bash
# Solution 2: Use different exchange (no proxy needed)
!freqtrade download-data \
    --exchange okx \
    --pairs BTC/USDT:USDT ETH/USDT:USDT \
    --timeframe 5m \
    --timerange 20240101-20240301 \
    --trading-mode futures
```

```bash
# Solution 3: Add proxy to config file
# Edit user_data/config_freqai.json and add:
# "exchange": {
#     "ccxt_config": {
#         "httpsProxy": "http://your-proxy:port"
#     }
# }
```

**See also**: [Proxy Configuration](#proxy-configuration-optional-but-recommended) section

#### 5. Data Download Failures

```bash
# Download with smaller timerange
!freqtrade download-data \
    --config user_data/config_freqai.json \
    --timerange 20240215-20240301 \
    --timeframes 5m
```

#### 6. Training Takes Too Long

- Reduce `train_period_days` (e.g., from 15 to 7)
- Reduce `train_cycles` for RL (e.g., from 25 to 10)
- Use fewer pairs in `pair_whitelist`
- Use larger timeframes (e.g., 15m instead of 5m)

#### 7. Session Timeout

Colab sessions timeout after ~12 hours. To prevent losing work:

```python
# Periodically save progress
import time

def keep_alive():
    """Keep Colab session alive"""
    while True:
        print("Session active...")
        time.sleep(300)  # Print every 5 minutes

# Run in background
# keep_alive()
```

---

## Quick Start Example - Complete Workflow

This is a **complete, production-ready workflow** for setting up and training FreqAI on Google Colab. Follow each cell in order.

**⏱️ Total Time**: ~30-45 minutes (depending on download speeds)

---

### 📋 Cell 1: Enable GPU Runtime

**Manual Step** - Do this first:
1. Go to **Runtime** → **Change runtime type**
2. Select **Hardware accelerator: GPU** (T4)
3. Click **Save**

**Verify GPU** (run this cell):

```python
import torch
print(f"CUDA Available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"✓ GPU: {torch.cuda.get_device_name(0)}")
    print(f"✓ Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
else:
    print("⚠️ No GPU detected - check runtime settings")
```

**Expected Output**: Should show `CUDA Available: True` and GPU name (e.g., Tesla T4)

---

### 📦 Cell 2: Install System Dependencies and TA-Lib

```bash
#!/bin/bash
set -e  # Exit on error

echo "=== Installing System Dependencies ==="
apt-get update -qq
apt-get install -y -qq build-essential wget curl git python3-dev python3-pip libffi-dev libssl-dev libatlas-base-dev

echo ""
echo "=== Installing TA-Lib C Library ==="
cd /content
wget -q http://prdownloads.sourceforge.net/ta-lib/ta-lib-0.4.0-src.tar.gz
tar -xzf ta-lib-0.4.0-src.tar.gz
cd ta-lib/
./configure --prefix=/usr --quiet
make -s
make install -s

echo ""
echo "=== Installing TA-Lib Python Wrapper ==="
cd /content
pip install -q TA-Lib

echo ""
echo "✓ TA-Lib installation complete"
python3 -c "import talib; print(f'✓ TA-Lib version: {talib.__version__}')"
```

**Expected Output**: Should end with `✓ TA-Lib version: 0.4.0`

---

### 🚀 Cell 3: Clone and Install Freqtrade

```bash
#!/bin/bash
set -e

echo "=== Cloning Freqtrade Repository ==="
cd /content
git clone -q https://github.com/freqtrade/freqtrade.git
cd freqtrade

echo ""
echo "=== Installing Freqtrade Dependencies ==="
echo "This may take 5-10 minutes..."

# Upgrade pip
python -m pip install --upgrade pip -q

# Install requirements
pip install -q -r requirements.txt
pip install -q -r requirements-freqai.txt
pip install -q -r requirements-freqai-rl.txt

# Install Freqtrade package
pip install -q -e .

echo ""
echo "✓ Freqtrade installation complete"
freqtrade --version
```

**Expected Output**: Should show `freqtrade 2024.x` (or similar version)

---

### 🌐 Cell 4: Setup VLESS Proxy

**⚠️ CRITICAL**: Replace `YOUR_UUID_HERE` and `/YOUR_PATH_HERE` with your actual VLESS credentials before running!

```bash
#!/bin/bash
set -e

echo "=== VLESS Proxy Setup ==="

# Install xray-core
echo "Installing xray-core..."
bash -c "$(curl -L https://github.com/XTLS/Xray-install/raw/main/install-release.sh)" @ install

# Create VLESS configuration
echo "Creating VLESS configuration..."
cat > /usr/local/etc/xray/config.json << 'EOF'
{
  "inbounds": [
    {
      "port": 10808,
      "listen": "127.0.0.1",
      "protocol": "socks",
      "settings": {
        "udp": true
      }
    },
    {
      "port": 10809,
      "listen": "127.0.0.1",
      "protocol": "http"
    }
  ],
  "outbounds": [
    {
      "protocol": "vless",
      "settings": {
        "vnext": [
          {
            "address": "jpanireki.com",
            "port": 443,
            "users": [
              {
                "id": "YOUR_UUID_HERE",
                "encryption": "none"
              }
            ]
          }
        ]
      },
      "streamSettings": {
        "network": "ws",
        "security": "tls",
        "wsSettings": {
          "path": "/YOUR_PATH_HERE"
        }
      }
    }
  ]
}
EOF

# Start xray service
echo "Starting xray-core..."
systemctl start xray
systemctl enable xray
sleep 3

# Verify service
if systemctl is-active --quiet xray; then
    echo ""
    echo "✓ VLESS proxy started successfully"
    echo "✓ SOCKS5: 127.0.0.1:10808"
    echo "✓ HTTP: 127.0.0.1:10809"
else
    echo "✗ Failed to start xray service"
    journalctl -u xray -n 20
    exit 1
fi
```

**Expected Output**: Should show `✓ VLESS proxy started successfully`

---

### 🔧 Cell 5: Configure Python Proxy Environment

```python
import os
import requests

print("=== Configuring Proxy Environment ===")

# Set environment variables to use local VLESS proxy
os.environ['HTTP_PROXY'] = 'http://127.0.0.1:10809'
os.environ['HTTPS_PROXY'] = 'http://127.0.0.1:10809'

print(f"✓ HTTP_PROXY: {os.environ['HTTP_PROXY']}")
print(f"✓ HTTPS_PROXY: {os.environ['HTTPS_PROXY']}")

print("\nVerifying proxy connection...")
try:
    response = requests.get('https://api.binance.com/api/v3/ping', timeout=10)
    if response.status_code == 200:
        print("✓ Proxy working! Can access Binance API")
        
        # Check exit IP
        ip_response = requests.get('https://api.ipify.org?format=json', timeout=10)
        print(f"✓ Exit IP: {ip_response.json()['ip']}")
    else:
        print(f"⚠️ Unexpected status: {response.status_code}")
except Exception as e:
    print(f"✗ Proxy verification failed: {e}")
    print("Check xray-core status: !systemctl status xray")
```

**Expected Output**: Should show `✓ Proxy working! Can access Binance API` and your exit IP address

---

### ⚙️ Cell 6: Create and Configure Freqtrade

```python
import json
import os

print("=== Creating Freqtrade Configuration ===")

# Change to freqtrade directory
os.chdir('/content/freqtrade')

# Create user data directory
!freqtrade create-userdir --userdir user_data

# Load example config
with open('config_examples/config_freqai.example.json', 'r') as f:
    config = json.load(f)

# Customize for Colab training
config.update({
    "dry_run": True,
    "timeframe": "5m",
    "stake_amount": 100,
    "max_open_trades": 3,
    "exchange": {
        "name": "binance",
        "key": "",
        "secret": "",
        "pair_whitelist": [
            "BTC/USDT:USDT",
            "ETH/USDT:USDT"
        ],
        "pair_blacklist": [],
        "ccxt_config": {
            "httpsProxy": "http://127.0.0.1:10809"  # VLESS proxy
        }
    }
})

# FreqAI settings
config["freqai"] = {
    "enabled": True,
    "purge_old_models": 2,
    "train_period_days": 10,
    "backtest_period_days": 3,
    "identifier": "colab-rl-model",
    "feature_parameters": {
        "include_timeframes": ["5m", "15m", "1h"],
        "include_corr_pairlist": ["BTC/USDT:USDT"],
        "label_period_candles": 24,
        "include_shifted_candles": 2,
        "DI_threshold": 0.9,
        "weight_factor": 0.9,
        "principal_component_analysis": False,
        "use_SVM_to_remove_outliers": True,
        "indicator_periods_candles": [10, 20]
    },
    "data_split_parameters": {
        "test_size": 0.33,
        "random_state": 1
    },
    "model_training_parameters": {}
}

# RL config
config["freqai"]["rl_config"] = {
    "train_cycles": 25,
    "add_state_info": True,
    "max_trade_duration_candles": 300,
    "max_training_drawdown_pct": 0.02,
    "cpu_count": 2,
    "model_type": "PPO",
    "policy_type": "MlpPolicy",
    "model_reward_parameters": {
        "rr": 1,
        "profit_aim": 0.025
    }
}

# Save config
with open('user_data/config_freqai.json', 'w') as f:
    json.dump(config, f, indent=4)

print("✓ Configuration file created: user_data/config_freqai.json")
print("✓ Proxy configured in exchange settings")
```

**Expected Output**: Should show `✓ Configuration file created`

---

### 📥 Cell 7: Download Training Data

```python
import os
import subprocess

os.chdir('/content/freqtrade')

print("=== Downloading Historical Data ===")
print("This may take 5-15 minutes depending on timerange...")
print()

# Ensure proxy environment variables are still set
os.environ['HTTP_PROXY'] = 'http://127.0.0.1:10809'
os.environ['HTTPS_PROXY'] = 'http://127.0.0.1:10809'

# Download data with proxy
result = subprocess.run([
    'freqtrade', 'download-data',
    '--config', 'user_data/config_freqai.json',
    '--timerange', '20240101-20240301',
    '--timeframes', '5m', '15m', '1h',
    '--exchange', 'binance',
    '--trading-mode', 'futures'
], capture_output=False, text=True)

if result.returncode == 0:
    print()
    print("=== Verifying Downloaded Data ===")
    !ls -lh user_data/data/binance/futures/
    print()
    print("✓ Data download complete")
else:
    print("✗ Data download failed")
    print("Check proxy status: !systemctl status xray")
```

**Expected Output**: Should show list of downloaded `.feather` files for BTC/USDT and ETH/USDT

**⚠️ If HTTP 451 Error Occurs**: 
1. Verify proxy is running: `!systemctl status xray`
2. Re-run Cell 5 to reset environment variables
3. Check xray logs: `!journalctl -u xray -n 50`

---

### 🎯 Cell 8: Run FreqAI Training

```bash
#!/bin/bash
set -e

cd /content/freqtrade

echo "=== Starting FreqAI Training ==="
echo "This will take 15-30 minutes..."
echo ""

# Run backtesting with FreqAI training
freqtrade backtesting \
    --strategy FreqaiExampleStrategy \
    --config user_data/config_freqai.json \
    --freqaimodel LightGBMRegressor \
    --timerange 20240201-20240301 \
    --verbose

echo ""
echo "✓ Training complete!"
echo ""
echo "=== Trained Models ==="
ls -lh user_data/models/
```

**Expected Output**: Should show training progress, backtest results, and list of trained models

---

### 💾 Cell 9: Save and Download Results

```python
import os
from google.colab import files

os.chdir('/content/freqtrade')

print("=== Creating Archive of Trained Models ===")
!tar -czf trained_models.tar.gz user_data/models/

# Check archive size
import os
archive_size = os.path.getsize('trained_models.tar.gz') / (1024**2)
print(f"✓ Archive created: {archive_size:.2f} MB")

print("\n=== Downloading Files ===")
print("Downloading trained models...")
files.download('trained_models.tar.gz')

print("Downloading configuration...")
files.download('user_data/config_freqai.json')

print("\n✓ All files downloaded!")
print("You can now close the Colab session.")
```

**Expected Output**: Browser should download `trained_models.tar.gz` and `config_freqai.json`

---

### 📊 Optional: View Results in Colab

```bash
# View last backtest results
!freqtrade backtesting-show

# Generate detailed analysis
!freqtrade backtesting-analysis \
    --config user_data/config_freqai.json \
    --analysis-groups 0 1 2
```

---

### ✅ Workflow Complete!

**What You've Accomplished**:
- ✅ GPU environment configured
- ✅ TA-Lib and Freqtrade installed
- ✅ VLESS proxy configured and verified
- ✅ Downloaded 2 months of historical data
- ✅ Trained FreqAI model on BTC/USDT and ETH/USDT
- ✅ Downloaded trained models for deployment

**Next Steps**:
1. Extract `trained_models.tar.gz` on your local machine
2. Use the trained models with Freqtrade in production
3. Refer to the main guide sections for advanced configuration
4. Monitor model performance and retrain periodically

**Troubleshooting**: If any cell fails, refer to the [Troubleshooting](#troubleshooting) section of this guide.

---

## Additional Resources

### Official Documentation

- [FreqAI Documentation](https://www.freqtrade.io/en/latest/freqai/)
- [FreqAI Configuration](https://www.freqtrade.io/en/latest/freqai-configuration/)
- [FreqAI Reinforcement Learning](https://www.freqtrade.io/en/latest/freqai-reinforcement-learning/)
- [Freqtrade Installation](https://www.freqtrade.io/en/latest/installation/)

### Example Strategies

- `freqtrade/templates/FreqaiExampleStrategy.py` - Basic FreqAI strategy
- `freqtrade/templates/FreqaiExampleHybridStrategy.py` - Hybrid strategy

### Key Configuration Files

- `config_examples/config_freqai.example.json` - FreqAI configuration example
- `config_examples/config_full.example.json` - Full configuration reference

---

## Notes

### Important Considerations for Colab

1. **No Virtual Environment**: Colab uses system Python, so no need for venv activation
2. **GPU Memory**: Monitor GPU memory usage, especially for RL training
3. **Session Limits**: Free Colab sessions have time and resource limits
4. **Data Persistence**: Save important data/models before session ends
5. **Network Speed**: Download speeds may vary; large datasets take time

### Best Practices

1. Start with small datasets and short training periods
2. Test with one pair before scaling to multiple pairs
3. Monitor logs regularly for errors or warnings
4. Save models frequently to avoid losing progress
5. Use Tensorboard to visualize training progress

### Performance Tips

- Use GPU runtime for faster training
- Reduce `train_cycles` for RL to speed up training
- Use fewer `include_timeframes` to reduce memory usage
- Consider using Colab Pro for longer sessions and better GPUs

---

## Conclusion

You now have a complete setup for running Freqtrade FreqAI training on Google Colab! Remember to:

- Start with the Quick Start Example
- Customize configurations based on your needs
- Monitor training progress and logs
- Save and download trained models
- Refer to troubleshooting section if issues arise

Happy training! 🚀
