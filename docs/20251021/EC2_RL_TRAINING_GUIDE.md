# EC2 RL Training Complete Guide

## Overview

This guide provides comprehensive steps for setting up and running Reinforcement Learning (RL) training on AWS EC2 using FreqAI.

**Target EC2 Instance:** c6g.8xlarge
- **Architecture:** ARM64 (AWS Graviton2)
- **vCPU:** 32
- **Memory:** 64 GB
- **Instance Type:** Compute Optimized
- **Python Version:** 3.12.3 (FreqAI supports Python 3.11, 3.12, 3.13)

---

## Table of Contents

1. [EC2 Instance Preparation](#1-ec2-instance-preparation)
2. [Initial System Setup](#2-initial-system-setup)
3. [Git Clone Deployment](#3-git-clone-deployment)
4. [Virtual Environment Setup](#4-virtual-environment-setup)
5. [Dependency Installation](#5-dependency-installation)
6. [Configuration Setup](#6-configuration-setup)
7. [Data Download](#7-data-download)
8. [Training Execution](#8-training-execution)
9. [TensorBoard Monitoring](#9-tensorboard-monitoring)
10. [Best Practices for c6g.8xlarge](#10-best-practices-for-c6g8xlarge)
11. [Troubleshooting](#11-troubleshooting)
12. [Performance Optimization](#12-performance-optimization)

---

## 1. EC2 Instance Preparation

### 1.1 Launch EC2 Instance

1. **Choose AMI:**
   - Ubuntu Server 22.04 LTS (ARM64) - Recommended
   - Amazon Linux 2023 (ARM64) - Alternative
   - **Note:** Python 3.12+ is required for FreqAI RL

2. **Instance Type:**
   - c6g.8xlarge (32 vCPU, 64 GB RAM)

3. **Storage:**
   - Root Volume: 100 GB GP3 (minimum)
   - Recommended: 200 GB for training data and models
   - IOPS: 3000 (default)
   - Throughput: 125 MB/s (default)

4. **Security Group:**
   - SSH (22): Your IP
   - TensorBoard (6006): Your IP (optional, for monitoring)

5. **Key Pair:**
   - Create or use existing key pair for SSH access

### 1.2 Connect to EC2 Instance

```bash
# Set correct permissions for key file
chmod 400 your-key.pem

# Connect to EC2
ssh -i your-key.pem ubuntu@<EC2-PUBLIC-IP>
```

For TensorBoard access (optional):
```bash
# SSH with port forwarding
ssh -i your-key.pem -L 6006:localhost:6006 ubuntu@<EC2-PUBLIC-IP>
```

---

## 2. Initial System Setup

### 2.0 Quick Note: Using `python3` vs `python3.12`

**Important:** Throughout this guide, you'll see commands using `python3.12`. However:

✅ **If your `python3 --version` already shows 3.12.x**, you can simply use `python3` instead of `python3.12` in all commands.

✅ **If your `python3 --version` shows an older version** (like 3.10.x), either:
- Use `python3.12` explicitly in commands, OR
- Set `python3` to point to `python3.12` using update-alternatives (shown in section 2.3)

✅ **Inside the virtual environment** (after running `source .venv/bin/activate`), just use `python` - it will automatically use the correct version.

---

### 2.1 Update System Packages

```bash
# Update package lists
sudo apt update

# Upgrade existing packages
sudo apt upgrade -y
```

### 2.2 Install System Dependencies

```bash
# Install essential build tools and dependencies
sudo apt install -y \
    build-essential \
    git \
    curl \
    wget \
    python3.12 \
    python3.12-venv \
    python3.12-dev \
    python3-pip \
    libssl-dev \
    libffi-dev \
    libxml2-dev \
    libxslt1-dev \
    zlib1g-dev \
    libblas-dev \
    liblapack-dev \
    gfortran \
    pkg-config \
    cmake

# Verify Python version (should show Python 3.12.x)
python3.12 --version
```

### 2.3 Verify Python Installation

```bash
# Check Python version
python3 --version

# If it shows "Python 3.12.x", you can use python3 directly!
# If not, check if python3.12 is available:
python3.12 --version

# See what python3 points to:
ls -la /usr/bin/python3*
```

**Option A: If python3 already shows 3.12.x**
```bash
# You can use python3 throughout this guide
# No additional setup needed!
```

**Option B: If python3 shows older version (e.g., 3.10.x)**
```bash
# Option B1: Use python3.12 explicitly (recommended for clarity)
# Just use python3.12 in commands instead of python3

# Option B2: Set python3 to point to python3.12
sudo update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.12 1

# Verify the change
python3 --version  # Should now show 3.12.x
```

### 2.4 Configure Git (Optional)

```bash
# Set git credentials
git config --global user.name "Your Name"
git config --global user.email "your.email@example.com"
```

---

## 3. Git Clone Deployment

**Note:** Ensure Python 3.12+ is installed before proceeding. FreqAI officially supports Python 3.11, 3.12, and 3.13.

### 3.1 Clone Repository

```bash
# Create project directory
mkdir -p ~/trading
cd ~/trading

# Clone your freqtrade repository
git clone https://github.com/hank00111/freqtrade.git

# Navigate to project directory
cd freqtrade

# Checkout develop branch (or your preferred branch)
git checkout develop

# Verify branch
git branch -vv
```

### 3.2 Alternative: Clone with Specific Branch

```bash
# Clone specific branch directly
git clone -b develop https://github.com/hank00111/freqtrade.git
cd freqtrade
```

---

## 4. Virtual Environment Setup

### 4.1 Create Virtual Environment

```bash
# Ensure you're in the freqtrade directory
cd ~/trading/freqtrade

# Create virtual environment
# If your python3 --version shows 3.12.x:
python3 -m venv .venv

# OR, if you need to specify python3.12 explicitly:
# python3.12 -m venv .venv

# Verify creation
ls -la .venv
```

**Note:** Once inside the virtual environment, you'll always use `python` (not `python3` or `python3.12`).

### 4.2 Activate Virtual Environment

```bash
# Activate virtual environment
source .venv/bin/activate

# Verify activation (should show (.venv) in prompt)
which python
python --version
```

**Important:** Always activate the virtual environment before any operations:
```bash
source .venv/bin/activate
```

---

## 5. Dependency Installation

### 5.1 Upgrade pip and Core Tools

```bash
# Ensure virtual environment is activated
source .venv/bin/activate

# Upgrade pip, wheel, and setuptools
python -m pip install --upgrade pip wheel setuptools
```

### 5.2 Install FreqAI RL Dependencies

**For ARM64 (Graviton2) architecture**, PyTorch and related libraries are fully supported.

```bash
# Install FreqAI with RL dependencies (includes PyTorch, stable-baselines3)
pip install -r requirements-freqai-rl.txt

# This includes:
# - torch (PyTorch for ARM64)
# - gymnasium
# - stable_baselines3
# - sb3_contrib
# - All FreqAI dependencies
```

### 5.3 Install Freqtrade

```bash
# Install freqtrade in editable mode
pip install -e .

# Verify installation
freqtrade --version
```

### 5.4 Verify Key Dependencies

```bash
# Verify PyTorch installation and ARM64 support
python -c "import torch; print(f'PyTorch version: {torch.__version__}')"

# Verify stable-baselines3
python -c "import stable_baselines3; print(f'SB3 version: {stable_baselines3.__version__}')"

# Verify FreqAI
python -c "from freqtrade.freqai.prediction_models.ReinforcementLearner import ReinforcementLearner; print('FreqAI RL ready')"
```

Expected output:
```
PyTorch version: 2.9.0
SB3 version: 2.7.0
FreqAI RL ready
```

---

## 6. Configuration Setup

### 6.1 Create User Data Directory Structure

```bash
# Ensure proper directory structure
mkdir -p user_data/models
mkdir -p user_data/data
mkdir -p user_data/strategies
mkdir -p user_data/freqaimodels
```

### 6.2 Copy Optimized Configuration

Copy the `config_v3.json` (see separate file) to your user_data directory:

```bash
# If you have config_v3.json locally, upload it
# scp -i your-key.pem config_v3.json ubuntu@<EC2-IP>:~/trading/freqtrade/user_data/

# Or create it directly on EC2
nano user_data/config_v3.json
```

### 6.3 Configuration Optimization for c6g.8xlarge

Key optimizations in `config_v3.json`:

```json
{
    "freqai": {
        "data_kitchen_thread_count": 16,  // Half of vCPUs for data processing
        "rl_config": {
            "cpu_count": 24,                // 75% of vCPUs for training
            "train_cycles": 100             // Optimized for faster iterations
        },
        "model_training_parameters": {
            "batch_size": 2048,             // Utilizing 64GB RAM
            "n_steps": 8192,                // Larger buffer for better learning
            "n_epochs": 20                  // More epochs per update
        }
    }
}
```

### 6.4 Update Exchange API Keys

```bash
# Edit config_v3.json with your API keys
nano user_data/config_v3.json

# Update the following fields:
# "exchange": {
#     "key": "your-api-key",
#     "secret": "your-api-secret"
# }
```

**Security Note:** Never commit API keys to git. Consider using environment variables:

```bash
# Set environment variables
export BINANCE_KEY="your-api-key"
export BINANCE_SECRET="your-api-secret"

# Or use .env file (not committed to git)
echo "BINANCE_KEY=your-api-key" >> .env
echo "BINANCE_SECRET=your-api-secret" >> .env
```

---

## 7. Data Download

### 7.1 Download Historical Data

```bash
# Ensure virtual environment is activated
source .venv/bin/activate

# Download data for your trading pairs (example: BTC/USDT, ETH/USDT)
freqtrade download-data \
    --config user_data/config_v3.json \
    --timerange 20221101-20251019 \
    --timeframes 15m 1h 4h
freqtrade download-data --exchange binance --pairs BTC/USDT:USDT ETH/USDT:USDT --timeframe 5m 15m 1h 4h --timerange 20221101-20251019 --trading-mode futures --prepend
# Verify downloaded data
ls -lh user_data/data/binance/
```

### 7.2 Download Additional Timeframes (if needed)

```bash
# If your strategy uses multiple timeframes
freqtrade download-data \
    --config user_data/config_v3.json \
    --timerange 20230101-20251019 \
    --timeframes 5m 15m 30m 1h 4h 1d
```

---

## 8. Training Execution

### 8.1 Copy Your Strategy

Ensure your RL strategy is in the user_data directory:

```bash
# Upload your strategy
# scp -i your-key.pem RLStrategy4Action.py ubuntu@<EC2-IP>:~/trading/freqtrade/user_data/strategies/

# Or copy from existing location
cp /path/to/RLStrategy4Action.py user_data/strategies/
```

### 8.2 Dry Run Test

Before full training, test the setup:

```bash
# Test with shorter timerange
freqtrade backtesting \
    --strategy RLStrategy4Action \
    --config user_data/config_v3.json \
    --freqaimodel ReinforcementLearner4Action \
    --timerange 20250101-20250110 \
    --export trades

# Check for errors
```

### 8.3 Start Full Training

```bash
# Ensure virtual environment is activated
source .venv/bin/activate

# Start training with full timerange
freqtrade backtesting \
    --strategy RLStrategy4Action \
    --config user_data/config_v3.json \
    --freqaimodel ReinforcementLearner4Action \
    --timerange 20230101-20251019 \
    --export trades

# Training will start and output progress
```

### 8.4 Run Training in Background (Recommended)

For long training sessions, use `screen` or `tmux`:

**Using screen:**
```bash
# Install screen
sudo apt install screen -y

# Create a new screen session
screen -S freqai_training

# Inside screen, activate venv and start training
source .venv/bin/activate
freqtrade backtesting \
    --strategy RLStrategy4Action \
    --config user_data/config_v3.json \
    --freqaimodel ReinforcementLearner4Action \
    --timerange 20230101-20251019 \
    --export trades

# Detach from screen: Ctrl+A, then D
# Reattach: screen -r freqai_training
# List screens: screen -ls
```

**Using tmux:**
```bash
# Install tmux
sudo apt install tmux -y

# Create a new tmux session
tmux new -s freqai_training

# Inside tmux, activate venv and start training
source .venv/bin/activate
freqtrade backtesting \
    --strategy RLStrategy4Action \
    --config user_data/config_v3.json \
    --freqaimodel ReinforcementLearner4Action \
    --timerange 20230101-20251019 \
    --export trades

# Detach from tmux: Ctrl+B, then D
# Reattach: tmux attach -t freqai_training
# List sessions: tmux ls
```

### 8.5 Monitor Training Progress

```bash
# Check CPU usage
htop

# Check memory usage
free -h

# Monitor training logs in real-time
tail -f user_data/logs/freqtrade.log
```

---

## 9. TensorBoard Monitoring

### 9.1 Start TensorBoard

TensorBoard provides real-time visualization of training metrics.

**In a separate terminal/screen session:**

```bash
# Ensure virtual environment is activated
source .venv/bin/activate

# Start TensorBoard (replace identifier with your config's identifier)
tensorboard --logdir user_data/models/PPO_15m_v3

# TensorBoard will start on http://localhost:6006
```

### 9.2 Access TensorBoard

**Option 1: SSH Port Forwarding (Local Machine)**

If you connected with port forwarding:
```bash
ssh -i your-key.pem -L 6006:localhost:6006 ubuntu@<EC2-PUBLIC-IP>
```

Then access in your browser: `http://localhost:6006`

**Option 2: Security Group Configuration**

1. Add inbound rule to EC2 security group:
   - Type: Custom TCP
   - Port: 6006
   - Source: Your IP address

2. Access directly: `http://<EC2-PUBLIC-IP>:6006`

**Note:** Option 1 (SSH tunneling) is more secure.

### 9.3 TensorBoard Metrics to Monitor

Key metrics to watch:
- **train/loss**: Should decrease over time
- **train/policy_loss**: Policy improvement
- **train/value_loss**: Value function learning
- **rollout/ep_rew_mean**: Average episode reward
- **time/fps**: Training speed (frames per second)

---

## 10. Best Practices for c6g.8xlarge

### 10.1 CPU Utilization

**Optimal Settings:**
- `cpu_count`: 24 (75% of 32 vCPUs)
- `data_kitchen_thread_count`: 16 (50% of vCPUs)
- Leave 8 vCPUs for system operations and data I/O

**Monitoring:**
```bash
# Real-time CPU monitoring
htop

# Average CPU usage
mpstat 1 10
```

### 10.2 Memory Management

**With 64GB RAM:**
- `batch_size`: 2048 (increased from 512)
- `n_steps`: 8192 (increased from 4096)
- Monitor memory usage to avoid OOM

**Memory Monitoring:**
```bash
# Watch memory in real-time
watch -n 1 free -h

# Detailed memory stats
vmstat 1
```

### 10.3 Training Optimization

**Recommended Parameters:**

```json
{
    "train_period_days": 60,        // More data per training
    "backtest_period_days": 10,     // Larger validation window
    "train_cycles": 100,            // Balanced for quality vs. speed
    "n_epochs": 20,                 // More epochs per update
    "batch_size": 2048,             // Utilize RAM
    "n_steps": 8192,                // Larger experience buffer
    "learning_rate": 0.00003        // Slightly lower for stability
}
```

### 10.4 Storage I/O Optimization

**GP3 Volume Settings:**
- IOPS: 3000 (minimum)
- Throughput: 125 MB/s (minimum)
- Consider increasing for larger datasets

**Monitor I/O:**
```bash
# Install iotop
sudo apt install iotop -y

# Monitor I/O
sudo iotop -o
```

### 10.5 ARM64-Specific Considerations

**PyTorch on ARM64:**
- Fully supported since PyTorch 2.0
- Native ARM64 builds provide excellent performance
- No need for Rosetta or emulation

**Performance:**
- Graviton2 provides excellent price/performance
- Expect ~20-30% better cost efficiency vs x86
- Training speed comparable to c5.8xlarge

---

## 11. Troubleshooting

### 11.1 Common Issues

**Issue: Out of Memory (OOM)**

```bash
# Symptoms
# - Training crashes
# - "Killed" message
# - System freeze

# Solutions
# 1. Reduce batch_size
sed -i 's/"batch_size": 2048/"batch_size": 1024/g' user_data/config_v3.json

# 2. Reduce n_steps
sed -i 's/"n_steps": 8192/"n_steps": 4096/g' user_data/config_v3.json

# 3. Monitor swap usage
sudo swapon --show
```

**Issue: Slow Training Speed**

```bash
# Check CPU throttling
cat /proc/cpuinfo | grep MHz

# Check system load
uptime

# Verify all cores are being used
htop  # Press 't' to see all CPUs
```

**Issue: PyTorch Import Error**

```bash
# Verify PyTorch installation
pip list | grep torch

# Reinstall if needed
pip uninstall torch -y
pip install torch==2.9.0
```

**Issue: Data Download Failures**

```bash
# Check network connectivity
ping -c 4 www.binance.com

# Retry with verbose output
freqtrade download-data \
    --config user_data/config_v3.json \
    --timerange 20230101-20251019 \
    --timeframes 15m \
    -v

# Check exchange status
curl -s https://api.binance.com/api/v3/ping
```

### 11.2 Log Analysis

```bash
# View recent errors
grep -i error user_data/logs/freqtrade.log | tail -20

# View training progress
grep -i "training" user_data/logs/freqtrade.log | tail -20

# Monitor log in real-time
tail -f user_data/logs/freqtrade.log
```

### 11.3 Disk Space Issues

```bash
# Check disk usage
df -h

# Find large directories
du -h --max-depth=1 ~/trading/ | sort -rh | head -10

# Clean old models (if needed)
# BE CAREFUL - this deletes data
find user_data/models -type f -mtime +30 -delete
```

---

## 12. Performance Optimization

### 12.1 Training Speed Benchmarks

**Expected Performance on c6g.8xlarge:**

| Configuration | Training Speed | Time for 2-year Dataset |
|--------------|----------------|------------------------|
| Default (8 CPU) | ~100-150 steps/s | ~12-15 hours |
| Optimized (24 CPU) | ~250-350 steps/s | ~5-7 hours |
| Max (32 CPU) | ~300-400 steps/s | ~4-6 hours |

**Note:** Actual speed depends on:
- Strategy complexity
- Number of features
- Network architecture
- Data timeframe

### 12.2 Cost Optimization

**c6g.8xlarge Pricing (us-east-1, on-demand):**
- ~$1.088/hour
- ~$26/day for continuous training

**Recommendations:**
1. Use Spot Instances (60-90% discount)
2. Stop instance when not training
3. Use Savings Plans for long-term usage
4. Consider c6g.4xlarge for smaller datasets

**Spot Instance Setup:**
```bash
# When launching EC2, select "Request Spot Instances"
# Set max price (e.g., $0.50/hour)
# Configure interruption behavior
```

### 12.3 Parallel Training Strategies

**Multi-Pair Training:**

```bash
# Train multiple pairs in parallel using different screens
screen -S btc_training
source .venv/bin/activate
freqtrade backtesting --strategy RLStrategy --config config_btc.json ...

# Detach: Ctrl+A, D

screen -S eth_training
source .venv/bin/activate
freqtrade backtesting --strategy RLStrategy --config config_eth.json ...
```

**Resource Allocation:**
- Split CPU cores between sessions
- Adjust `cpu_count` proportionally
- Monitor total resource usage

### 12.4 Model Checkpointing

FreqAI automatically saves checkpoints. To resume training:

```bash
# Training will automatically resume from last checkpoint
# if identifier matches previous run
freqtrade backtesting \
    --strategy RLStrategy4Action \
    --config user_data/config_v3.json \
    --freqaimodel ReinforcementLearner4Action \
    --timerange 20230101-20251019
```

---

## 13. Post-Training Steps

### 13.1 Export Trained Models

```bash
# Compress trained models
cd ~/trading/freqtrade
tar -czf models_backup_$(date +%Y%m%d).tar.gz user_data/models/

# Download to local machine
# scp -i your-key.pem ubuntu@<EC2-IP>:~/trading/freqtrade/models_backup_*.tar.gz .
```

### 13.2 Analyze Results

```bash
# View backtest results
cat user_data/backtest_results/backtest-result-*.json | jq .

# Check best performing models
ls -lt user_data/models/PPO_15m_v3/
```

### 13.3 Deploy to Live Trading

After successful backtesting:

1. Update config for dry-run mode
2. Test with paper trading
3. Gradually transition to live trading

---

## 14. Cleanup and Shutdown

### 14.1 Stop Training

```bash
# If running in foreground: Ctrl+C

# If running in screen/tmux:
screen -r freqai_training  # Reattach
# Then: Ctrl+C to stop
```

### 14.2 Stop TensorBoard

```bash
# Find TensorBoard process
ps aux | grep tensorboard

# Kill process
kill <PID>
```

### 14.3 Deactivate Virtual Environment

```bash
deactivate
```

### 14.4 Stop EC2 Instance

```bash
# From AWS Console or CLI
aws ec2 stop-instances --instance-ids i-1234567890abcdef0

# Or terminate if no longer needed
aws ec2 terminate-instances --instance-ids i-1234567890abcdef0
```

**Important:** Stopping vs. Terminating:
- **Stop:** Preserves data, can restart later
- **Terminate:** Deletes instance, data lost (unless using EBS volumes)

---

## 15. Quick Reference Commands

### Essential Commands

```bash
# Connect to EC2
ssh -i key.pem ubuntu@<EC2-IP>

# Activate venv
source .venv/bin/activate

# Start training
freqtrade backtesting --strategy RLStrategy4Action --config user_data/config_v3.json --freqaimodel ReinforcementLearner4Action --timerange 20230101-20251019

# Start TensorBoard
tensorboard --logdir user_data/models/PPO_15m_v3

# Monitor resources
htop
free -h
df -h
```

### Screen Commands

```bash
# Create session
screen -S training

# Detach
Ctrl+A, then D

# Reattach
screen -r training

# List sessions
screen -ls

# Kill session
screen -X -S training quit
```

---

## 16. Additional Resources

### Documentation
- FreqAI Documentation: https://www.freqtrade.io/en/stable/freqai/
- Reinforcement Learning Guide: https://www.freqtrade.io/en/stable/freqai-reinforcement-learning/
- AWS Graviton: https://aws.amazon.com/ec2/graviton/

### Support
- Freqtrade Discord: https://discord.gg/freqtrade
- GitHub Issues: https://github.com/freqtrade/freqtrade/issues

### Monitoring Tools
- htop: Process monitoring
- nvidia-smi: GPU monitoring (if using GPU instances)
- TensorBoard: Training visualization

---

## Conclusion

This guide provides a comprehensive workflow for RL training on EC2 c6g.8xlarge. The ARM64-based Graviton2 processors offer excellent performance for FreqAI workloads at competitive pricing.

**Key Takeaways:**
- c6g.8xlarge provides 32 vCPU and 64GB RAM for efficient training
- ARM64 architecture is fully supported by PyTorch and FreqAI
- Optimize `cpu_count`, `batch_size`, and `n_steps` for best performance
- Use screen/tmux for long-running training sessions
- Monitor with TensorBoard for real-time insights
- Consider Spot Instances for cost savings

**Next Steps:**
1. Complete initial setup following this guide
2. Run test training with short timerange
3. Optimize configuration based on resource usage
4. Execute full training on historical data
5. Analyze results and iterate on strategy

Good luck with your RL training!
