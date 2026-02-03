# EC2 RL Training Performance Optimization Guide

## Issue Analysis

### Current Performance
- **Speed**: 321 it/s (target: 2000-5000 it/s)
- **Load Average**: 8.01 (should be ~28)
- **CPU Usage**: 15-24% per core (should be 60-80%)
- **Processes**: ~15 visible (should be ~28)

### Root Causes Identified

#### 1. 🔴 CRITICAL: Insufficient Training Data
```
Configuration:
- train_period_days: 60 days
- Expected data points: 5,760 (60 days × 24h × 4 intervals)
- Actual valid data: 1,932 points (3,183 dropped as NaN)
- Data completeness: 33.5%

Problem:
- 1,932 points ÷ 28 environments = 69 points per environment
- This is too small for efficient parallel training
- Environment switching overhead > training benefit
```

#### 2. 🟡 Code Synchronization Issue
The modified `ReinforcementLearner4Action_multiproc.py` may not be deployed to EC2 yet.

**Check on EC2:**
```bash
# View the current version
cat user_data/freqaimodels/ReinforcementLearner4Action_multiproc.py | grep -A 20 "def __init__"
```

Expected to see:
```python
def __init__(self, **kwargs) -> None:
    """
    Override __init__ to remove the max_threads limitation.
    ...
```

If not present, upload the updated file.

#### 3. 🟠 Data Quality Issues
- 66.5% of data dropped as NaN
- Causes:
  * Multiple timeframes (15m, 1h, 4h) require data alignment
  * `include_shifted_candles: 5` requires 5 previous candles
  * Incomplete data download from exchange

## Solutions (Priority Order)

### Solution 1: Increase Training Period ⭐ MOST IMPORTANT

**Update config_v3.json:**
```json
{
  "freqai": {
    "train_period_days": 180,  // Change from 60 to 180
    "backtest_period_days": 10,
    ...
  }
}
```

**Expected improvement:**
- Data points: 5,760 → 17,280 (3x increase)
- Points per environment: 69 → 207 (3x increase)
- Better parallelization efficiency

**For even better results, use 365 days:**
```json
"train_period_days": 365,  // 1 year of data
```
- Data points: ~34,560
- Points per environment: ~1,234
- Optimal for 28 parallel environments

### Solution 2: Re-download Complete Data

```bash
# On EC2, delete old incomplete data
rm -rf user_data/data/*.json
rm -rf user_data/data/*.feather

# Download fresh complete data
freqtrade download-data \
    --exchange binance \
    --pairs BTC/USDT:USDT ETH/USDT:USDT \
    --timeframes 15m 1h 4h \
    --timerange 20230101-20251019 \
    --trading-mode futures \
    --config user_data/config_v3.json

# Verify data completeness
freqtrade list-data --config user_data/config_v3.json
```

### Solution 3: Deploy Updated Code to EC2

**Option A: Upload modified file (if working on Windows)**
```powershell
# On Windows local machine
scp -i your-key.pem `
    user_data/freqaimodels/ReinforcementLearner4Action_multiproc.py `
    ubuntu@your-ec2-ip:/home/ubuntu/freqtrade/user_data/freqaimodels/
```

**Option B: Direct edit on EC2**
```bash
# SSH into EC2
ssh -i your-key.pem ubuntu@your-ec2-ip

# Edit the file
nano user_data/freqaimodels/ReinforcementLearner4Action_multiproc.py
```

Add this `__init__` method to the class:
```python
class ReinforcementLearner4Action_multiproc(ReinforcementLearner_multiproc):
    
    def __init__(self, **kwargs) -> None:
        """
        Override __init__ to remove the max_threads limitation.
        
        The base class limits max_threads to min(cpu_count, max_system_threads/2).
        For dedicated training servers, we want to use the full configured cpu_count.
        """
        # Call parent __init__ first
        super().__init__(**kwargs)
        
        # Override max_threads to use configured cpu_count directly
        configured_cpu_count = self.freqai_info["rl_config"].get("cpu_count", 1)
        
        # Only override if configured value is higher than current max_threads
        if configured_cpu_count > self.max_threads:
            logger.info(
                f"Overriding max_threads from {self.max_threads} to {configured_cpu_count} "
                f"for better parallelization on dedicated training server"
            )
            self.max_threads = configured_cpu_count
            th.set_num_threads(self.max_threads)
        else:
            logger.info(
                f"Using default max_threads={self.max_threads} "
                f"(configured cpu_count={configured_cpu_count})"
            )
```

### Solution 4: Optimize Configuration for Current Data Size

If you can't increase data immediately, reduce `cpu_count` to match data size:

```json
{
  "rl_config": {
    "cpu_count": 12,  // Reduced from 28
    ...
  }
}
```

**Calculation:**
- 1,932 points ÷ 12 environments = 161 points per environment
- Better balance between parallelism and data size

## Complete Optimized Workflow

### Step 1: Update Configuration on Local Machine

**File: `user_data/config_v3.json`**
```json
{
  "freqai": {
    "train_period_days": 180,
    "backtest_period_days": 10,
    ...
    "rl_config": {
      "cpu_count": 28,
      ...
    }
  }
}
```

### Step 2: Upload to EC2

```powershell
# Upload config
scp -i your-key.pem `
    user_data/config_v3.json `
    ubuntu@your-ec2-ip:/home/ubuntu/freqtrade/user_data/

# Upload modified model
scp -i your-key.pem `
    user_data/freqaimodels/ReinforcementLearner4Action_multiproc.py `
    ubuntu@your-ec2-ip:/home/ubuntu/freqtrade/user_data/freqaimodels/
```

### Step 3: Re-download Data on EC2

```bash
# SSH into EC2
ssh -i your-key.pem ubuntu@your-ec2-ip

# Navigate to freqtrade directory
cd /home/ubuntu/freqtrade

# Activate virtual environment
source .venv/bin/activate

# Clean old data
rm -rf user_data/data/*.json user_data/data/*.feather

# Download fresh complete data
freqtrade download-data \
    --exchange binance \
    --pairs BTC/USDT:USDT ETH/USDT:USDT \
    --timeframes 15m 1h 4h \
    --timerange 20230101-20251019 \
    --trading-mode futures \
    --config user_data/config_v3.json

# Verify
freqtrade list-data --config user_data/config_v3.json
```

### Step 4: Set Environment Variables

```bash
# Optimize thread settings
export NUMEXPR_MAX_THREADS=32
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
```

### Step 5: Start Training

```bash
freqtrade backtesting \
    --strategy RLStrategy4Action \
    --config user_data/config_v3.json \
    --freqaimodel ReinforcementLearner4Action_multiproc \
    --timerange 20230101-20251019 \
    --export trades
```

### Step 6: Monitor Performance

```bash
# In another terminal
htop

# Look for:
# - ~28 Python processes
# - CPU usage 60-80% per core
# - Load average 24-28
# - it/s speed 1500-3000+
```

## Expected Results After Optimization

### With 180-day training period:
| Metric | Before | After | Improvement |
|--------|--------|-------|-------------|
| Training data | 1,932 | ~6,000 | 3.1x |
| Points per env | 69 | ~214 | 3.1x |
| Parallel envs | ~15 | 28 | 1.87x |
| CPU usage | 15-24% | 60-80% | 3-4x |
| Load average | 8 | 24-28 | 3x |
| Training speed | 321 it/s | 1,500-2,000 it/s | 5-6x |

### With 365-day training period:
| Metric | Before | After | Improvement |
|--------|--------|-------|-------------|
| Training data | 1,932 | ~12,000 | 6.2x |
| Points per env | 69 | ~428 | 6.2x |
| Training speed | 321 it/s | 2,500-4,000 it/s | 8-12x |

## Verification Checklist

- [ ] Updated `config_v3.json` with increased `train_period_days`
- [ ] Uploaded modified `ReinforcementLearner4Action_multiproc.py` to EC2
- [ ] Re-downloaded complete historical data
- [ ] Set environment variables for thread optimization
- [ ] Started training with updated configuration
- [ ] Monitored htop showing ~28 processes
- [ ] Verified log shows "Overriding max_threads from 16 to 28"
- [ ] Confirmed training speed > 1500 it/s

## Troubleshooting

### If speed is still low after optimization:

1. **Check actual data completeness:**
   ```bash
   # Look at training logs
   grep "Training on" logs/freqai.log
   grep "dropped as NaN" logs/freqai.log
   ```

2. **Verify environment count:**
   ```bash
   ps aux | grep python | grep multiprocessing | wc -l
   ```
   Should show ~28 processes.

3. **Check memory usage:**
   ```bash
   free -h
   ```
   If swap is being used heavily, reduce `cpu_count`.

4. **Monitor I/O:**
   ```bash
   iotop
   ```
   If I/O wait is high, data reading is the bottleneck.

5. **Check PyTorch threads:**
   Add logging to verify:
   ```python
   import torch as th
   print(f"PyTorch threads: {th.get_num_threads()}")
   ```

## Alternative Approaches

### If 28 environments is too many:

Start with fewer environments and scale up:
```json
"cpu_count": 16,  // Start conservative
```

### If data is still insufficient:

Use shorter backtest periods:
```json
"train_period_days": 90,
"backtest_period_days": 5,
```

### If NaN rate is still high:

Reduce feature complexity:
```json
"feature_parameters": {
    "include_timeframes": ["15m"],  // Remove 1h, 4h
    "include_shifted_candles": 2,   // Reduce from 5
    ...
}
```

## Summary

**The primary bottleneck is insufficient training data, not CPU limits.**

The key fix is:
```json
"train_period_days": 180  // or 365 for best results
```

Combined with:
1. Complete data re-download
2. Updated multiproc code deployment
3. Proper thread environment variables

Expected final performance: **1,500 - 4,000 it/s** (5-12x improvement)
