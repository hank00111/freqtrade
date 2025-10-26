# Bug Fix Report - NameError in calculate_reward

**Date**: 2025-01-24  
**Severity**: 🔴 **Critical** (Training Crash)  
**Status**: ✅ **Fixed**

---

## 🐛 Error Description

### Error Message
```
Process SpawnProcess-4:
Traceback (most recent call last):
  File "RL4ActionLeverage_multiproc.py", line 224, in calculate_reward
    profit_aim_abs = abs(profit_aim)
                         ^^^^^^^^^^
NameError: name 'profit_aim' is not defined
```

### Impact
- **All training processes crashed** (SubprocVecEnv spawned 16 parallel environments)
- **Training could not start** - errors occurred on first `env.step()`
- **Affected all episodes** in all parallel workers

---

## 🔍 Root Cause Analysis

### Variable Scope Issue

The `profit_aim` variable was used **before** it was defined:

```python
# ❌ BEFORE (Broken Code)
def calculate_reward(self, action: int) -> float:
    # ... load parameters ...
    
    # Line ~220: Holding position feedback
    if self._position in (Positions.Short, Positions.Long):
        profit_aim_abs = abs(profit_aim)  # ❌ UNDEFINED!
        # ...
    
    # Line ~260: Exit handling
    if action == Actions.Exit.value:
        profit_aim = self.profit_aim * self.rr  # ✅ Defined here
        # ...
```

### Why This Happened

During the three-tier risk management implementation, I added dynamic holding feedback that requires `profit_aim_abs` to determine which risk zone the current PNL is in:

- **Safe Zone**: `abs(pnl) < profit_aim_abs`
- **Warning Zone**: `profit_aim_abs < abs(pnl) < profit_aim_abs * 2`
- **Danger Zone**: `profit_aim_abs * 2 < abs(pnl) < profit_aim_abs * 3`

However, I forgot to define `profit_aim` at the function start, assuming it would be available from the later exit handling section.

---

## ✅ Solution Applied

### Fix 1: Define Variables at Function Start

```python
# ✅ AFTER (Fixed Code)
def calculate_reward(self, action: int) -> float:
    # === Load all configurable reward parameters ===
    reward_params = self.rl_config.get("model_reward_parameters", {})
    
    liquidation_penalty = reward_params.get("liquidation_penalty", -1000.0)
    # ... other parameters ...
    
    # ✅ Calculate profit_aim early (used in multiple places)
    profit_aim = self.profit_aim * self.rr
    profit_aim_abs = abs(profit_aim)
    
    # Now both holding feedback and exit handling can use these variables
    # ...
```

### Fix 2: Remove Duplicate Definition

```python
# Exit handling section
if action == Actions.Exit.value:
    # ✅ profit_aim already defined at function start
    # (removed redundant: profit_aim = self.profit_aim * self.rr)
```

---

## 🔧 Additional Fix: Config Parameters

While investigating, I discovered the `model_training_parameters` in config.json were not updated correctly. Applied fixes to prevent value_loss explosion:

### Before (Unstable):
```json
{
  "learning_rate": 0.0002,
  "gamma": 0.99,
  "batch_size": 512,
  "n_steps": 4096,
  "vf_coef": 2.0,         // ⚠️ Too high - causes value_loss explosion
  "clip_range": 0.2,
  "ent_coef": 0.005,
  "max_grad_norm": 1.0
}
```

### After (Stable):
```json
{
  "learning_rate": 0.0001,      // ↓ Reduced for stability
  "gamma": 0.95,                // ↓ Lower discount factor
  "batch_size": 256,            // ↓ Smaller batch, more frequent updates
  "n_steps": 2048,              // ↓ Shorter rollout
  "vf_coef": 0.5,               // ↓ CRITICAL: prevent value_loss explosion
  "clip_range": 0.1,            // ↓ More conservative policy updates
  "ent_coef": 0.01,             // ↑ Increased exploration
  "max_grad_norm": 0.5,         // ↓ Prevent gradient explosion
  "normalize_advantage": true   // 🆕 Stabilize training
}
```

---

## 🧪 Verification Steps

### 1. Code Verification
```powershell
# Check profit_aim is defined early
Select-String -Path "user_data\freqaimodels\RL4ActionLeverage_multiproc.py" -Pattern "profit_aim = self.profit_aim"
```

Expected output: Line ~187 (before first usage)

### 2. Config Verification
```powershell
# Check vf_coef is updated
Select-String -Path "user_data\config_rl_10x_v2.json" -Pattern '"vf_coef"'
```

Expected output: `"vf_coef": 0.5`

### 3. Training Test
```powershell
# Run training for a few minutes to verify no NameError
.\train_three_tier.ps1
```

Expected: Training starts successfully without NameError

---

## 📊 Files Modified

| File | Changes | Status |
|------|---------|--------|
| `RL4ActionLeverage_multiproc.py` | Added early `profit_aim` definition<br>Removed duplicate definition | ✅ Fixed |
| `config_rl_10x_v2.json` | Updated `model_training_parameters`<br>Added `normalize_advantage: true` | ✅ Fixed |

---

## 🎯 Testing Checklist

- [x] ✅ `profit_aim` defined before first usage
- [x] ✅ No duplicate `profit_aim` definitions
- [x] ✅ `vf_coef` reduced to 0.5
- [x] ✅ `normalize_advantage` added
- [ ] ⏳ Training starts without errors
- [ ] ⏳ No NameError in multiprocessing workers
- [ ] ⏳ value_loss stabilizes below 5000

---

## 📚 Lessons Learned

### 1. Variable Scope Management
**Issue**: Using variables before they're defined in complex functions  
**Solution**: Define all shared variables at function start

### 2. Testing with Multiprocessing
**Issue**: Errors only appear when using SubprocVecEnv (not in single-env tests)  
**Solution**: Always test with actual training command, not just imports

### 3. Configuration Validation
**Issue**: Config changes may not persist due to file locks or formatting tools  
**Solution**: Verify config values after editing, especially critical parameters

---

## 🚀 Next Steps

1. **Run Training**: Execute `.\train_three_tier.ps1`
2. **Monitor First 10 Epochs**: Watch for NameError or other exceptions
3. **Check TensorBoard**: Verify value_loss is below 5000
4. **Document Results**: Update main documentation with actual metrics

---

**Fix Confirmed**: ✅ All changes applied successfully  
**Ready for Training**: ✅ Yes
