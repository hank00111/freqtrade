# RL Training Speed Optimization Guide

> Analysis of `config_daytrade.json` training performance and actionable optimizations.
> Current baseline: ~680 it/s on CPU-only Torch.

## 1. Environment

### 1.1 Check Your Environment

Run the following commands to collect your machine's training-relevant specs:

```powershell
# GPU info (model, VRAM, driver version)
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader

# CPU cores and RAM
python -c "import psutil; print(f'CPU: {psutil.cpu_count(logical=False)} cores / {psutil.cpu_count()} threads'); print(f'RAM: {psutil.virtual_memory().total / 1024**3:.1f} GB')"

# Torch version and CUDA status
python -c "import torch; print(f'Torch: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA version: {torch.version.cuda}'); print(f'GPU: {torch.cuda.get_device_name(0)}' if torch.cuda.is_available() else '')"

# SB3 version
python -c "import stable_baselines3; print(f'SB3: {stable_baselines3.__version__}')"
```

### 1.2 Reference Baseline

The following baseline was measured on a specific machine. Use it to estimate
relative performance on your hardware.

| Item | Value |
|------|-------|
| CPU | 16 cores / 24 threads |
| RAM | 63.8 GB |
| GPU | NVIDIA RTX 3080 Ti 12GB |
| Torch | `2.10.0+cpu` (CPU-only build, no CUDA) |
| CUDA Driver | 13.1 |
| SB3 | 2.7.1 |
| Training speed | ~680 it/s (CPU-only, single env) |

### 1.3 Key Hardware Factors

| Factor | Affects | Guideline |
|--------|---------|-----------|
| GPU VRAM | `device: "cuda"` feasibility, max `batch_size` | >= 8 GB recommended. 12 GB+ allows batch_size 1024+ |
| GPU compute | Gradient update speed | Higher-tier GPU = faster forward/backward pass |
| CPU cores | `cpu_count` (multiproc parallelism) | Set `cpu_count` to physical cores or `logical / 2`. More cores = faster data collection |
| RAM | Max subprocess count | Each env uses ~63 MB data + Python overhead. 16 envs needs ~2-3 GB headroom |
| Torch build | GPU utilization | Must be CUDA build (`torch.cuda.is_available() == True`) to use GPU |

**Adjusting config for your hardware**:

- **No GPU**: Keep `"device": "cpu"`. Multiproc still helps (data collection is CPU-bound).
- **GPU < 8 GB VRAM**: Use `"device": "cuda"` with `batch_size: 512`. Monitor with `nvidia-smi`.
- **CPU < 8 cores**: Set `cpu_count` to physical core count. Fewer than 4 cores may not benefit from multiproc.
- **RAM < 16 GB**: Reduce `cpu_count` to 4-8 to limit subprocess memory usage.

## 2. Training Volume Breakdown

Based on `config_daytrade.json`:

| Item | Value |
|------|-------|
| Base features | 488 (expand_all: 192, expand_basic: 288, standard: 8) |
| Observation dim | 488 x conv_width(10) = **4,880** |
| Network params | ~1.3M per network (4880 -> 256 -> 256 -> 4) |
| Train candles/pair | ~16,200 (75 days x 5m x 0.75 train split) |
| total_timesteps | train_cycles(500) x 16,200 = **8,100,000** |
| Rollouts (single env) | 8,100,000 / n_steps(2048) = 3,955 |
| Rollouts (16 envs) | 8,100,000 / (2048 x 16) = 247 |
| Gradient updates/rollout (single) | n_epochs(10) x (2048 / 512) = 40 |
| Gradient updates/rollout (16 envs) | n_epochs(10) x (32,768 / 512) = 640 |
| Time per pair per window | 8.1M / 680 = **198 min (~3.3 hr)** |
| Time for 2 pairs | **~6.6 hr per training window** |

### PPO Training Cycle

```
[Data Collection: env.step() x 2048]  -->  [Gradient Update: 10 epochs x 4 batches]
       ^-- CPU-bound (env logic)                 ^-- GPU-accelerable
       ^-- ~60% of wall time                     ^-- ~40% of wall time
```

## 3. Optimization Plans

### 3.1 Install CUDA Torch + `device: "cuda"` (est. +30-50%)

**Problem**: RTX 3080 Ti 12GB is completely idle. Torch is compiled as CPU-only
(`torch.version.cuda = None`). All matrix operations (forward pass, backward pass,
gradient update) run on CPU.

**Change**:

```powershell
pip install torch torchvision torchaudio --force-reinstall --no-deps --index-url https://download.pytorch.org/whl/cu126
```

Then in `config_daytrade.json`:

```jsonc
"model_training_parameters": {
    "device": "cuda"    // was "cpu"
}
```

**Expected**: Gradient update phase 2-4x faster -> overall +30-50%.

#### Do I need to uninstall CPU Torch first?

**No.** Using `--force-reinstall` is sufficient. Detailed explanation:

Both CPU and CUDA versions of PyTorch use the **same package name** (`torch`).
When you run `pip install torch --index-url ... --force-reinstall`, pip will:

1. Detect that `torch` is already installed
2. Because of `--force-reinstall`, remove the existing version
3. Download and install the CUDA version from the specified index URL

The `--force-reinstall` flag handles the replacement automatically.
A manual `pip uninstall` beforehand is redundant.

**Why `--force-reinstall` is needed**: Without it, pip sees `torch 2.10.0` is already
installed and may skip the install (same version number, different build). The flag
forces a full reinstall regardless.

**Why `--no-deps` is important**: Without it, `--force-reinstall` will also reinstall
all dependencies of torch (numpy, sympy, etc.), which is unnecessary and may cause
version conflicts. `--no-deps` limits the reinstall to only torch/torchvision/torchaudio.

**Why `--index-url` is critical**: PyPI (the default index) only hosts the CPU-only
`torch` wheel. The CUDA wheels are hosted on PyTorch's own index
(`download.pytorch.org/whl/cuXXX`). Without `--index-url`, pip will always fetch
the CPU version.

Available CUDA indexes for Torch 2.10.0:

| CUDA Version | Index URL |
|-------------|-----------|
| CUDA 12.6 | `https://download.pytorch.org/whl/cu126` |
| CUDA 12.8 | `https://download.pytorch.org/whl/cu128` |
| CUDA 13.0 | `https://download.pytorch.org/whl/cu130` |

Recommended: `cu126` (most stable, widest compatibility with CUDA driver 13.1).

**Verify after install**:

```python
import torch
print(torch.cuda.is_available())   # Should be True
print(torch.version.cuda)          # Should show "12.6" or similar
print(torch.cuda.get_device_name(0))  # "NVIDIA GeForce RTX 3080 Ti"
```

**Risk**: Some packages (e.g., freqai dependencies) may later overwrite torch back
to CPU-only. If `torch.cuda.is_available()` becomes False after installing other
packages, re-run the CUDA install command.

### 3.2 Switch to RLDayTrader_multiproc (est. +300-500%)

**Problem**: `RLDayTrader` (single process) uses `DummyVecEnv` with 1 environment.
`cpu_count: 16` in config only sets PyTorch thread count (`th.set_num_threads()`),
it does NOT parallelize environment data collection.

**Change**: Use multiproc model in the training command:

```powershell
freqtrade backtesting --strategy RLDayTradeStrategy `
  --config user_data/config_daytrade.json `
  --freqaimodel RLDayTrader_multiproc `
  --timerange 20240101-20260101
```

No config changes needed. `cpu_count: 16` is already set.

> **Note on `max_threads` cap**: `BaseReinforcementLearningModel` caps
> `max_threads = min(cpu_count, system_cores / 2)`. On this machine (24 threads),
> that yields `min(16, 23) = 16`, so the base class already allows 16.
> `RLDayTrader_multiproc.__init__` has an override as a safety net for machines
> where `system_cores / 2 < cpu_count` (see `RLDayTrader_multiproc.py:37-43`).

**How it works**:

- `SubprocVecEnv` creates 16 independent environment processes
- Each process runs its own `MyRLEnv` instance in parallel
- Data collection: 16 environments step simultaneously
- Evaluation: `eval_freq = len(train_df) // max_threads`

```
Single process (current):
  [env.step()] x 2048  ->  [gradient update]  ->  [env.step()] x 2048  -> ...

Multi-process (16 envs):
  [env1.step()]         \
  [env2.step()]          |
  [env3.step()]          |-- 2048 steps each = 32,768 total  ->  [gradient update]
  ...                    |
  [env16.step()]        /
```

**Expected**: Data collection (60% of time) becomes ~8-12x faster.
Overall: 3-5x improvement.

**Memory impact**: Each subprocess creates a full copy of training data and environment
state. ~63 MB per env x 16 = ~1 GB additional RAM. Ensure sufficient memory.

**Windows note**: Windows uses `spawn` (not `fork`) for multiprocessing. Each subprocess
re-imports all modules on startup. This adds a one-time startup overhead per training
window, but does not affect per-step performance.

**Note**: TensorBoard metrics are unreliable with multiple environments
(counters overlap across processes). Use single-process `RLDayTrader` for reward
function debugging, then switch to `RLDayTrader_multiproc` for production training.

### 3.3 Increase batch_size (est. +10-15%)

**Change** in `config_daytrade.json`:

```jsonc
"model_training_parameters": {
    "batch_size": 1024    // was 512
}
```

**Why**: Larger batches are more GPU-efficient (better utilization of parallel cores)
and reduce the number of gradient updates per rollout.

Gradient updates per rollout = `n_epochs x (buffer_size / batch_size)`:

| Mode | buffer_size | batch=512 | batch=1024 |
|------|-------------|-----------|------------|
| Single env (1) | 2,048 | 40 | 20 |
| Multiproc (16) | 32,768 | 640 | 320 |

**Constraint**: SB3 requires `n_steps x n_envs` to be evenly divisible by `batch_size`.
- Single env: `2048 % batch_size == 0`
- Multiproc (16 envs): `32,768 % batch_size == 0`

### 3.4 Reduce n_epochs (est. +10-15%)

**Change** in `config_daytrade.json`:

```jsonc
"model_training_parameters": {
    "n_epochs": 5    // was 10
}
```

**Trade-off**: Lower sample efficiency (each collected rollout is used fewer times).
But with 16 parallel environments collecting data, the throughput increase compensates.

Gradient updates per rollout (see table in 3.3 for exact numbers):
- Single env: 40 -> 20
- Multiproc (16 envs): 640 -> 320 (or 160 if combined with batch_size=1024)

## 4. Combined Scenarios

| Scenario | Changes | Est. Speed | Time/pair |
|----------|---------|-----------|-----------|
| Baseline | - | 680 it/s | 198 min |
| A: CUDA only | device=cuda | ~1,000 it/s | 135 min |
| B: A + multiproc | + RLDayTrader_multiproc | ~3,500 it/s | 39 min |
| C: B + tuning | + batch=1024, epochs=5 | ~4,500 it/s | **30 min** |

Scenario B is the recommended target: CUDA + multiproc gives the biggest
improvement with minimal risk. Scenario C adds minor tuning on top.

## 5. Implementation Steps

### Phase 1: CUDA Torch

```powershell
# Step 1: Install CUDA Torch (replaces CPU version)
pip install torch torchvision torchaudio --force-reinstall --no-deps --index-url https://download.pytorch.org/whl/cu126

# Step 2: Verify
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, Device: {torch.cuda.get_device_name(0)}')"

# Step 3: Update config
# config_daytrade.json -> "device": "cuda"

# Step 4: Test single window
freqtrade backtesting --strategy RLDayTradeStrategy `
  --config user_data/config_daytrade.json `
  --freqaimodel RLDayTrader `
  --timerange 20250101-20250401
```

- [ ] CUDA Torch installed
- [ ] `torch.cuda.is_available()` returns True
- [ ] config_daytrade.json updated: `"device": "cuda"`
- [ ] Single window training completes without errors
- [ ] Measure new it/s speed

### Phase 2: Multi-process

```powershell
# Switch to multiproc model (no config changes needed)
freqtrade backtesting --strategy RLDayTradeStrategy `
  --config user_data/config_daytrade.json `
  --freqaimodel RLDayTrader_multiproc `
  --timerange 20250101-20250401
```

- [ ] Training completes with RLDayTrader_multiproc
- [ ] Monitor GPU usage: `nvidia-smi -l 1` (should show GPU utilization during gradient updates)
- [ ] Measure new it/s speed
- [ ] Compare backtesting results with single-process (should be similar)

### Phase 3: Batch/Epoch Tuning (optional)

```jsonc
// config_daytrade.json
"model_training_parameters": {
    "batch_size": 1024,   // was 512
    "n_epochs": 5         // was 10
}
```

- [ ] Training completes without degraded results
- [ ] Measure final it/s speed

## 6. Monitoring

Use `/analyze-rl` to compare training quality before and after optimization:

```
/analyze-rl rl-daytrade-v1 5
```

Key metrics to watch:
- `ep_rew_mean` trend should remain similar
- Win rate should not degrade significantly
- Policy collapse should not appear

If multiproc training shows degraded metrics, try reducing `cpu_count` from 16
to 8 (reduces noise from too many parallel environments).
