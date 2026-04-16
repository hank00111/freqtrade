# Upstream Merge Plan & Post-Merge Fixes

> Date: 2026-04-17
> Base: dev/personal @ a203f41 (merge base with upstream/develop 2026-03-25)
> Target: upstream/develop (HEAD as of 2026-04-17)
> Upstream ahead: 313 commits (260 non-merge)
> Local ahead: 117 commits

## Goal

Pull upstream/develop into dev/personal, discard all local fixes in `freqtrade/` core,
preserve `user_data/` custom code, then apply minimal post-merge fixes for known bugs
and regressions.

## What Gets Preserved (Zero Conflict)

All files under `user_data/` are local-only and upstream never touches them.
They are preserved automatically by the merge.

- `user_data/strategies/RLDayTradeStrategy.py` — feature engineering, entry/exit signals
- `user_data/freqaimodels/RLDayTrader.py` — single-env RL model
- `user_data/freqaimodels/RLDayTrader_multiproc.py` — multi-env RL model
- `user_data/config_daytrade_v2.json` — training config

## What Gets Discarded

Local fixes in `freqtrade/` core are NOT preserved. Upstream version wins for all
conflicts in these files:

- `freqtrade/freqai/freqai_interface.py` (discard wide try/except scope, discard sorted()
  comparison, discard debug mismatch log, discard gc.collect)
- `freqtrade/freqai/data_drawer.py` (discard device= kwarg in MODELCLASS.load)
- `freqtrade/freqai/RL/BaseReinforcementLearningModel.py` (discard device= kwarg)
- `freqtrade/freqai/prediction_models/ReinforcementLearner.py` (discard device= kwarg)

## Dependency Upgrades

| Package | From | To | Risk |
|---------|------|-----|------|
| stable_baselines3 | 2.7.1 | 2.8.0 | LOW (MaskablePPO tb_log_name fix is improvement) |
| torch | 2.10.0 | 2.11.0 | LOW (verify CUDA after upgrade) |
| sb3_contrib | >=2.2.1 | >=2.8.0 resolved | LOW |
| gymnasium | 1.2.3 | 1.2.3 | No change |

---

## Post-Merge Fixes (MUST APPLY)

After the merge completes, two fixes MUST be applied before running training.

### Fix 1: Empty Data Window Guard (CRITICAL)

**Problem:**
Our timerange `20220601-20260325` includes ~25 early windows where 5m futures data
does not exist (data starts 2022-11-01). Upstream's narrow try/except only wraps
`self.train()`. When training raises on empty data, `self.model = None`, but then
`self.predict()` runs unconditionally with None model → AttributeError → training
crashes.

**Upstream flow that breaks:**

```python
if not self.model_exists(dk):
    dk.find_features(dataframe_train)
    dk.find_labels(dataframe_train)
    try:
        self.model = self.train(dataframe_train, pair, dk)  # raises on empty
    except Exception:
        self.model = None  # narrow scope ends here
    # save model logic...
else:
    self.model = self.dd.load_data(pair, dk)

# This runs with self.model = None:
pred_df, do_preds = self.predict(dataframe_backtest, dk)  # CRASH
```

**Fix location:** `freqtrade/freqai/freqai_interface.py`
**Fix logic:** Wrap predict/append/save in `if self.model is not None:` guard.

**Patch:**

```python
# After the if/else block that sets self.model, change:
pred_df, do_preds = self.predict(dataframe_backtest, dk)
append_df = dk.get_predictions_to_append(pred_df, do_preds, dataframe_backtest)
dk.append_predictions(append_df)
dk.save_backtesting_prediction(append_df)

# To:
if self.model is not None:
    pred_df, do_preds = self.predict(dataframe_backtest, dk)
    append_df = dk.get_predictions_to_append(pred_df, do_preds, dataframe_backtest)
    dk.append_predictions(append_df)
    dk.save_backtesting_prediction(append_df)
else:
    logger.warning(
        f"No model available for {pair}, skipping prediction for "
        f"backtest window {tr_backtest.startdt} - {tr_backtest.stopdt}."
    )
```

**Why this is correct:** Matches the warning message pattern from the original
local wide-try/except commit (`6eebbab07`). Empty data windows skip gracefully
and training continues.

### Fix 2: Feature Mismatch Bug — Raw Dataframe Preservation (CRITICAL)

**Problem:**
In `freqai_interface.py`, the `start_backtesting()` loop has two branches:

1. `else` branch (no prediction cache): calls `use_strategy_to_populate_indicators()`
   on the full `dataframe`, which mutates `dataframe` in place, adding hundreds of
   `%-` feature columns.
2. `if` branch (cache hit): if `check_features == True`, calls
   `use_strategy_to_populate_indicators()` AGAIN on `dataframe.tail(1)`, which is
   already polluted from step 1. The second call duplicates columns, causing feature
   count to explode (392 → 5000+), triggering OperationalException.

**Trigger condition:** Mixed training run where some windows process via `else` branch
(no cache) and then a later window hits `if` branch (has cache). Our partial
`backtesting_predictions/` directory (54 files from previous retrain) creates exactly
this condition.

**Upstream status:** NOT fixed. Line unchanged in upstream/develop.

**Fix location:** `freqtrade/freqai/freqai_interface.py`, around line 291-295 and line 334.

**Patch (Fix B from analysis):**

```python
# Add after `pair = metadata["pair"]` near line 291:
dataframe_raw = dataframe.copy()

# Change at line 334-336 from:
df_fts = self.dk.use_strategy_to_populate_indicators(
    strategy, prediction_dataframe=dataframe.tail(1), pair=pair
)

# To:
df_fts = self.dk.use_strategy_to_populate_indicators(
    strategy, prediction_dataframe=dataframe_raw.tail(1), pair=pair
)
```

**Why this is correct:**
- `dataframe_raw` captured before the loop starts, never mutated
- `check_features` path always receives clean OHLCV rows
- No impact on live trading (live never enters this code path)
- 2-line change, immune to ordering permutations

---

## Execution Order

1. Fetch upstream
2. `git merge upstream/develop -X theirs` (auto-take upstream for conflicts)
3. Verify no user_data/ files were touched
4. Apply Fix 1 (empty data guard)
5. Apply Fix 2 (dataframe_raw)
6. Upgrade dependencies (pip install -r requirements-freqai-rl.txt)
7. Run Tier 1-2 validation (imports + smoke test)
8. Commit merge + fixes
9. Clear backtesting_predictions/ cache
10. Restart training with timerange 20220601-20260325

## Validation Commands

```bash
# Tier 1: Static imports (30 sec)
python -c "
import stable_baselines3; print('SB3', stable_baselines3.__version__)
import torch; print('torch', torch.__version__)
from freqtrade.freqai.freqai_interface import IFreqaiModel
from freqtrade.freqai.RL.BaseReinforcementLearningModel import BaseReinforcementLearningModel
import sys; sys.path.insert(0, 'user_data/freqaimodels')
from RLDayTrader_multiproc import RLDayTrader_multiproc
print('All imports OK')
"

# Tier 2: Smoke test 1 week (2-5 min)
freqtrade backtesting \
  --strategy RLDayTradeStrategy \
  --config user_data/config_daytrade_v2.json \
  --freqaimodel RLDayTrader_multiproc \
  --timerange 20260101-20260108 \
  --export trades \
  --logfile user_data/logs/smoke_test.log

# Success signal: BACKTESTING REPORT in log, .feather files in backtesting_predictions/
# Failure signal: OperationalException, AttributeError on PPO, AttributeError on NoneType
```

## Rollback

Not applicable — user explicitly requested no backup. If merge fails catastrophically,
manual recovery via `git reflog` or re-clone from origin.

## Notes

- Timerange MUST remain 20220601-20260325 (memory: feedback_no_skip_timerange.md)
- All 125 sub-train-ETH_* directories will be retrained from scratch due to prediction
  cache being cleared
- Estimated training completion: 2026-05-10 to 2026-05-15 (revised from 05-02~08,
  accounting for full 199 window retrain)
