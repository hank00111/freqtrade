#!/usr/bin/env python3
"""
Analyze RL training health from TensorBoard logs.

Parses TensorBoard event files for a given model-id, extracts scalar metrics
per training window, computes health checks, and outputs structured JSON.

Usage:
    python scripts/analyze_rl_training.py --model-id rl-daytrade-v1 --last-n 5

Dependencies:
    tensorboard (already installed via requirements-freqai.txt)
"""

import argparse
import json
import math
import os
import re
import sys
import warnings
from pathlib import Path

# Suppress TensorFlow/TensorBoard warnings that pollute stderr
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
warnings.filterwarnings("ignore")

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


# ---------------------------------------------------------------------------
# Health check thresholds
# ---------------------------------------------------------------------------

THRESHOLDS = {
    "reward_trend": {
        "warn": 0.0,       # flat or slight decline
        "fail": -0.20,     # declining > 20%
    },
    "liquidation_rate": {
        "ok": 0.05,        # < 5%
        "warn": 0.15,      # 5-15%
        # > 15% = FAIL
    },
    "win_rate": {
        "ok": 0.40,        # > 40%
        "warn": 0.25,      # 25-40%
        # < 25% = FAIL
    },
    "policy_collapse": {
        "warn": 0.60,      # one action 60-80%
        "fail": 0.80,      # one action > 80%
    },
    "value_loss": {
        "warn": 2.0,       # increasing < 2x
        # > 2x or NaN = FAIL
    },
    "invalid_actions": {
        "ok": 0.10,        # < 10%
        "warn": 0.25,      # 10-25%
        # > 25% = FAIL
    },
    "entropy_loss": {
        "warn": 0.50,      # retained > 50% of initial = OK
        "fail": 0.20,      # retained < 20% of initial = FAIL
        # between 20-50% = WARN
    },
    "approx_kl": {
        "ok": 0.02,        # < 0.02
        "warn": 0.05,      # 0.02-0.05
        # > 0.05 = FAIL
    },
    "clip_fraction": {
        "ok": 0.20,        # < 0.2
        "warn": 0.40,      # 0.2-0.4
        # > 0.4 = FAIL
    },
    "sample_size": {
        "ok": 1000,        # >= 1000 actions from env[0]
        "warn": 200,       # 200-999 actions
        # < 200 = FAIL (health checks unreliable)
    },
    "entropy_trend": {
        "warn": -0.08,     # decline > 8pp across windows
        "fail": -0.15,     # decline > 15pp across windows
    },
    "explained_variance": {
        "ok": 0.50,        # > 0.5 = good predictions
        "warn": 0.0,       # 0-0.5 = noisy
        # < 0 = FAIL (worse than random)
    },
    "long_short_bias": {
        "warn": 0.65,      # dominant side > 65%
        "fail": 0.80,      # dominant side > 80%
    },
}

# Minimum iteration count ratio to consider a window "complete"
# A window with fewer than this fraction of the max iteration count is treated as in-progress
COMPLETE_THRESHOLD_RATIO = 0.50


# ---------------------------------------------------------------------------
# TensorBoard parsing
# ---------------------------------------------------------------------------

def discover_windows(model_dir: Path) -> list[dict]:
    """
    Discover training windows under model_dir/tensorboard/{pair}/PPO_*.
    Returns list of dicts: {pair, window_num, window_name, event_dir}.
    Sorted by window number ascending.
    """
    tb_dir = model_dir / "tensorboard"
    if not tb_dir.exists():
        return []

    windows = []
    for pair_dir in sorted(tb_dir.iterdir()):
        if not pair_dir.is_dir():
            continue
        pair = pair_dir.name
        for ppo_dir in sorted(pair_dir.iterdir()):
            if not ppo_dir.is_dir():
                continue
            match = re.match(r"PPO_(\d+)", ppo_dir.name)
            if not match:
                continue
            # Check that event files exist
            event_files = list(ppo_dir.glob("events.out.tfevents.*"))
            if not event_files:
                continue
            windows.append({
                "pair": pair,
                "window_num": int(match.group(1)),
                "window_name": ppo_dir.name,
                "event_dir": str(ppo_dir),
            })

    windows.sort(key=lambda w: (w["pair"], w["window_num"]))
    return windows


def load_scalars(event_dir: str) -> dict[str, list[dict]]:
    """
    Load all scalar tags from an event directory.
    Returns {tag: [{step, wall_time, value}, ...]}.
    """
    ea = EventAccumulator(event_dir)
    ea.Reload()

    scalars = {}
    for tag in ea.Tags().get("scalars", []):
        events = ea.Scalars(tag)
        scalars[tag] = [
            {"step": e.step, "wall_time": e.wall_time, "value": e.value}
            for e in events
        ]
    return scalars


def find_tag(scalars: dict, base_name: str) -> str | None:
    """
    Find a tag matching base_name, allowing for model-specific suffixes.
    E.g., 'pnl/exit_pnl' matches 'pnl/exit_pnl' or 'pnl/exit_pnl_10x'.

    Only matches suffixes that look like model identifiers (e.g., _10x, _5x)
    to avoid false matches like 'risk/liquidation' -> 'risk/liquidation_distance'.
    """
    if base_name in scalars:
        return base_name
    # Try with numeric suffixes only (e.g., _10x, _5x, _20x)
    suffix_pattern = re.compile(r"_\d+x$")
    for tag in scalars:
        if tag.startswith(base_name):
            suffix = tag[len(base_name):]
            if suffix_pattern.match(suffix):
                return tag
    return None


def get_last_value(scalars: dict, base_name: str, default=None) -> float | None:
    """Get the last value for a tag (with flexible matching)."""
    tag = find_tag(scalars, base_name)
    if tag is None or not scalars[tag]:
        return default
    return scalars[tag][-1]["value"]


def get_values(scalars: dict, base_name: str) -> list[float]:
    """Get all values for a tag (with flexible matching)."""
    tag = find_tag(scalars, base_name)
    if tag is None:
        return []
    return [e["value"] for e in scalars[tag]]


# ---------------------------------------------------------------------------
# Per-window metric extraction
# ---------------------------------------------------------------------------

def extract_window_metrics(scalars: dict) -> dict:
    """Extract key metrics from a single training window's scalars."""
    metrics = {}

    # Reward (SB3 built-in)
    ep_rew_values = get_values(scalars, "rollout/ep_rew_mean")
    if ep_rew_values:
        metrics["ep_rew_mean_first"] = ep_rew_values[0]
        metrics["ep_rew_mean_last"] = ep_rew_values[-1]
        metrics["ep_rew_mean_count"] = len(ep_rew_values)

    # Value loss (SB3 built-in)
    vloss_values = get_values(scalars, "train/value_loss")
    if vloss_values:
        metrics["value_loss_first"] = vloss_values[0]
        metrics["value_loss_last"] = vloss_values[-1]
        metrics["value_loss_count"] = len(vloss_values)

    # Entropy loss (SB3 built-in, logged as negative value)
    entropy_values = get_values(scalars, "train/entropy_loss")
    if entropy_values:
        metrics["entropy_loss_first"] = entropy_values[0]
        metrics["entropy_loss_last"] = entropy_values[-1]
        metrics["entropy_loss_count"] = len(entropy_values)

    # Approx KL divergence (SB3 built-in)
    approx_kl_values = get_values(scalars, "train/approx_kl")
    if approx_kl_values:
        metrics["approx_kl_last"] = approx_kl_values[-1]
        metrics["approx_kl_mean"] = sum(approx_kl_values) / len(approx_kl_values)

    # Clip fraction (SB3 built-in)
    clip_frac_values = get_values(scalars, "train/clip_fraction")
    if clip_frac_values:
        metrics["clip_fraction_last"] = clip_frac_values[-1]
        metrics["clip_fraction_mean"] = sum(clip_frac_values) / len(clip_frac_values)

    # Episode length
    ep_len_values = get_values(scalars, "rollout/ep_len_mean")
    if ep_len_values:
        metrics["ep_len_mean_last"] = ep_len_values[-1]

    # PNL metrics
    metrics["exit_pnl_last"] = get_last_value(scalars, "pnl/exit_pnl")
    metrics["profitable_exits"] = get_last_value(scalars, "pnl/profitable_exit", 0)
    metrics["loss_exits"] = get_last_value(scalars, "pnl/loss_exit", 0)

    # Risk
    metrics["liquidations"] = get_last_value(scalars, "risk/liquidation", 0)

    # Action distribution (last values are cumulative counts)
    action_tags = ["actions/Neutral", "actions/Exit",
                   "actions/Long_enter", "actions/Short_enter"]
    action_counts = {}
    for tag in action_tags:
        val = get_last_value(scalars, tag, 0)
        action_name = tag.split("/")[1] if "/" in tag else tag
        action_counts[action_name] = val
    metrics["action_counts"] = action_counts

    # Invalid actions
    metrics["invalid_actions"] = get_last_value(scalars, "actions/invalid", 0)

    # Exit reward
    metrics["exit_reward_last"] = get_last_value(scalars, "rewards/exit_reward")

    # Total profit
    metrics["total_profit"] = get_last_value(scalars, "info/total_profit")

    # Explained variance (SB3 built-in)
    ev_values = get_values(scalars, "train/explained_variance")
    if ev_values:
        metrics["explained_variance_last"] = ev_values[-1]

    # Default completeness flag; updated by analyze() after all windows are extracted
    metrics["is_complete"] = True

    return metrics


# ---------------------------------------------------------------------------
# Health checks
# ---------------------------------------------------------------------------

def _find_latest_complete(windows_metrics: list[dict]) -> dict:
    """Find the latest complete window, fallback to latest if all incomplete."""
    for wm in reversed(windows_metrics):
        if wm.get("is_complete", True):
            return wm
    return windows_metrics[-1]


def compute_reward_trend(windows_metrics: list[dict]) -> dict:
    """Check if ep_rew_mean is improving across windows."""
    rew_values = []
    for wm in windows_metrics:
        val = wm.get("ep_rew_mean_last")
        if val is not None:
            rew_values.append(val)

    if len(rew_values) < 2:
        return {"status": "OK", "detail": "Insufficient data for trend (1 window)"}

    first = rew_values[0]
    last = rew_values[-1]

    if first == 0:
        if last > 0:
            return {"status": "OK", "detail": f"Reward improving: {first:.2f} -> {last:.2f}"}
        elif last < 0:
            pct = -1.0
        else:
            pct = 0.0
    else:
        pct = (last - first) / abs(first)

    if pct > THRESHOLDS["reward_trend"]["warn"]:
        return {"status": "OK", "detail": f"Reward improving: {first:.2f} -> {last:.2f} ({pct:+.1%})"}
    elif pct > THRESHOLDS["reward_trend"]["fail"]:
        return {"status": "WARN", "detail": f"Reward flat/declining: {first:.2f} -> {last:.2f} ({pct:+.1%})"}
    else:
        return {"status": "FAIL", "detail": f"Reward declining: {first:.2f} -> {last:.2f} ({pct:+.1%})"}


def compute_liquidation_rate(windows_metrics: list[dict]) -> dict:
    """Check liquidation rate in the latest complete window."""
    latest = _find_latest_complete(windows_metrics)
    liquidations = latest.get("liquidations", 0)
    total_exits = latest.get("profitable_exits", 0) + latest.get("loss_exits", 0)

    # Liquidation count is separate from exit counts, add it to total
    total_episodes = total_exits + liquidations
    if total_episodes == 0:
        return {"status": "OK", "detail": "No episodes recorded"}

    note = ""
    if not windows_metrics[-1].get("is_complete", True):
        note = f" (using earlier complete window, latest has {windows_metrics[-1].get('ep_rew_mean_count', 0)} iterations)"

    rate = liquidations / total_episodes
    if rate < THRESHOLDS["liquidation_rate"]["ok"]:
        return {"status": "OK", "detail": f"Liquidation rate: {rate:.1%} ({int(liquidations)}/{int(total_episodes)}){note}"}
    elif rate < THRESHOLDS["liquidation_rate"]["warn"]:
        return {"status": "WARN", "detail": f"Liquidation rate: {rate:.1%} ({int(liquidations)}/{int(total_episodes)}){note}"}
    else:
        return {"status": "FAIL", "detail": f"Liquidation rate: {rate:.1%} ({int(liquidations)}/{int(total_episodes)}){note}"}


def compute_win_rate(windows_metrics: list[dict]) -> dict:
    """Check win rate from profitable/loss exit counts in the latest complete window."""
    latest = _find_latest_complete(windows_metrics)
    wins = latest.get("profitable_exits", 0)
    losses = latest.get("loss_exits", 0)
    total = wins + losses

    if total == 0:
        return {"status": "WARN", "detail": "No exits recorded"}

    note = ""
    if not windows_metrics[-1].get("is_complete", True):
        note = f" (using earlier complete window, latest has {windows_metrics[-1].get('ep_rew_mean_count', 0)} iterations)"

    rate = wins / total
    if rate > THRESHOLDS["win_rate"]["ok"]:
        return {"status": "OK", "detail": f"Win rate: {rate:.1%} ({int(wins)}W/{int(losses)}L){note}"}
    elif rate > THRESHOLDS["win_rate"]["warn"]:
        return {"status": "WARN", "detail": f"Win rate: {rate:.1%} ({int(wins)}W/{int(losses)}L){note}"}
    else:
        return {"status": "FAIL", "detail": f"Win rate: {rate:.1%} ({int(wins)}W/{int(losses)}L){note}"}


def compute_policy_collapse(windows_metrics: list[dict]) -> dict:
    """Check if any single action dominates > 80% of total in the latest complete window."""
    latest = _find_latest_complete(windows_metrics)
    action_counts = latest.get("action_counts", {})

    if not action_counts:
        return {"status": "OK", "detail": "No action data"}

    total = sum(action_counts.values())
    if total == 0:
        return {"status": "OK", "detail": "No actions recorded"}

    max_action = max(action_counts, key=action_counts.get)
    max_ratio = action_counts[max_action] / total

    note = ""
    if not windows_metrics[-1].get("is_complete", True):
        note = f" (using earlier complete window, latest has {windows_metrics[-1].get('ep_rew_mean_count', 0)} iterations)"

    dist_str = ", ".join(
        f"{k}={v / total:.0%}" for k, v in sorted(action_counts.items())
    )

    if max_ratio > THRESHOLDS["policy_collapse"]["fail"]:
        return {"status": "FAIL", "detail": f"Policy collapse: {max_action}={max_ratio:.0%} [{dist_str}]{note}"}
    elif max_ratio > THRESHOLDS["policy_collapse"]["warn"]:
        return {"status": "WARN", "detail": f"Action concentration: {max_action}={max_ratio:.0%} [{dist_str}]{note}"}
    else:
        return {"status": "OK", "detail": f"Action distribution balanced [{dist_str}]{note}"}


def compute_value_loss(windows_metrics: list[dict]) -> dict:
    """Check if value_loss is stable or exploding within the latest complete window.

    Compares value_loss_first vs value_loss_last within a single window,
    because each FreqAI window trains independently and cross-window
    comparison reflects data distribution differences, not training stability.
    """
    latest = _find_latest_complete(windows_metrics)

    first = latest.get("value_loss_first")
    last = latest.get("value_loss_last")

    if first is None or last is None:
        return {"status": "WARN", "detail": "No value_loss data"}

    note = ""
    if not windows_metrics[-1].get("is_complete", True):
        note = f" (using earlier complete window, latest has {windows_metrics[-1].get('ep_rew_mean_count', 0)} iterations)"

    if math.isnan(last) or math.isinf(last):
        return {"status": "FAIL", "detail": f"Value loss is NaN/Inf: {last}{note}"}

    if math.isnan(first) or math.isinf(first):
        return {"status": "FAIL", "detail": f"Value loss started as NaN/Inf: {first}{note}"}

    if first == 0:
        if last == 0:
            return {"status": "OK", "detail": f"Value loss stable at 0{note}"}
        ratio = float("inf")
    else:
        ratio = last / first

    if ratio <= 1.0:
        return {"status": "OK", "detail": f"Value loss decreasing: {first:.4f} -> {last:.4f} ({ratio:.2f}x){note}"}
    elif ratio < THRESHOLDS["value_loss"]["warn"]:
        return {"status": "WARN", "detail": f"Value loss increasing: {first:.4f} -> {last:.4f} ({ratio:.2f}x){note}"}
    else:
        return {"status": "FAIL", "detail": f"Value loss exploding: {first:.4f} -> {last:.4f} ({ratio:.2f}x){note}"}


def compute_entropy_loss(windows_metrics: list[dict]) -> dict:
    """Check if entropy is collapsing within the latest complete window.

    SB3 logs entropy_loss as a negative value (e.g., -1.38 for max entropy
    with 4 actions). We compare abs(last) / abs(first) to see what fraction
    of initial entropy is retained. Low retention = exploration has stopped.
    """
    latest = _find_latest_complete(windows_metrics)

    first = latest.get("entropy_loss_first")
    last = latest.get("entropy_loss_last")

    if first is None or last is None:
        return {"status": "WARN", "detail": "No entropy_loss data"}

    note = ""
    if not windows_metrics[-1].get("is_complete", True):
        note = f" (using earlier complete window, latest has {windows_metrics[-1].get('ep_rew_mean_count', 0)} iterations)"

    abs_first = abs(first)
    abs_last = abs(last)

    if abs_first == 0:
        return {"status": "WARN", "detail": f"Initial entropy is 0{note}"}

    if math.isnan(last) or math.isinf(last):
        return {"status": "FAIL", "detail": f"Entropy loss is NaN/Inf: {last}{note}"}

    retained = abs_last / abs_first

    if retained > THRESHOLDS["entropy_loss"]["warn"]:
        return {"status": "OK", "detail": f"Entropy healthy: {first:.4f} -> {last:.4f} ({retained:.0%} retained){note}"}
    elif retained > THRESHOLDS["entropy_loss"]["fail"]:
        return {"status": "WARN", "detail": f"Entropy declining: {first:.4f} -> {last:.4f} ({retained:.0%} retained){note}"}
    else:
        return {"status": "FAIL", "detail": f"Entropy collapsed: {first:.4f} -> {last:.4f} ({retained:.0%} retained){note}"}


def compute_invalid_actions(windows_metrics: list[dict]) -> dict:
    """Check if invalid action rate is acceptable in the latest complete window."""
    latest = _find_latest_complete(windows_metrics)
    invalid = latest.get("invalid_actions", 0)
    action_counts = latest.get("action_counts", {})
    # action_counts already includes invalid attempts (logged before validity check)
    total_actions = sum(action_counts.values())

    if total_actions == 0:
        return {"status": "OK", "detail": "No actions recorded"}

    note = ""
    if not windows_metrics[-1].get("is_complete", True):
        note = f" (using earlier complete window, latest has {windows_metrics[-1].get('ep_rew_mean_count', 0)} iterations)"

    rate = invalid / total_actions
    if rate < THRESHOLDS["invalid_actions"]["ok"]:
        return {"status": "OK", "detail": f"Invalid action rate: {rate:.1%} ({int(invalid)}/{int(total_actions)}){note}"}
    elif rate < THRESHOLDS["invalid_actions"]["warn"]:
        return {"status": "WARN", "detail": f"Invalid action rate: {rate:.1%} ({int(invalid)}/{int(total_actions)}){note}"}
    else:
        return {"status": "FAIL", "detail": f"Invalid action rate: {rate:.1%} ({int(invalid)}/{int(total_actions)}){note}"}


def compute_approx_kl(windows_metrics: list[dict]) -> dict:
    """Check if policy updates are too aggressive (high KL divergence).

    High approx_kl means the policy is changing too much per update,
    which can destabilize training. Uses the mean across the window
    to avoid noisy single-step spikes.
    """
    latest = _find_latest_complete(windows_metrics)
    kl_mean = latest.get("approx_kl_mean")
    kl_last = latest.get("approx_kl_last")

    if kl_mean is None:
        return {"status": "OK", "detail": "No approx_kl data"}

    note = ""
    if not windows_metrics[-1].get("is_complete", True):
        note = f" (using earlier complete window, latest has {windows_metrics[-1].get('ep_rew_mean_count', 0)} iterations)"

    if kl_mean < THRESHOLDS["approx_kl"]["ok"]:
        return {"status": "OK", "detail": f"KL divergence stable: mean={kl_mean:.4f}, last={kl_last:.4f}{note}"}
    elif kl_mean < THRESHOLDS["approx_kl"]["warn"]:
        return {"status": "WARN", "detail": f"KL divergence elevated: mean={kl_mean:.4f}, last={kl_last:.4f}{note}"}
    else:
        return {"status": "FAIL", "detail": f"KL divergence too high: mean={kl_mean:.4f}, last={kl_last:.4f}{note}"}


def compute_clip_fraction(windows_metrics: list[dict]) -> dict:
    """Check if too many policy ratios are being clipped.

    High clip_fraction means the policy is trying to change more than
    the clipping range allows, suggesting the learning rate may be too high.
    Uses the mean across the window.
    """
    latest = _find_latest_complete(windows_metrics)
    cf_mean = latest.get("clip_fraction_mean")
    cf_last = latest.get("clip_fraction_last")

    if cf_mean is None:
        return {"status": "OK", "detail": "No clip_fraction data"}

    note = ""
    if not windows_metrics[-1].get("is_complete", True):
        note = f" (using earlier complete window, latest has {windows_metrics[-1].get('ep_rew_mean_count', 0)} iterations)"

    if cf_mean < THRESHOLDS["clip_fraction"]["ok"]:
        return {"status": "OK", "detail": f"Clip fraction normal: mean={cf_mean:.3f}, last={cf_last:.3f}{note}"}
    elif cf_mean < THRESHOLDS["clip_fraction"]["warn"]:
        return {"status": "WARN", "detail": f"Clip fraction elevated: mean={cf_mean:.3f}, last={cf_last:.3f}{note}"}
    else:
        return {"status": "FAIL", "detail": f"Clip fraction too high: mean={cf_mean:.3f}, last={cf_last:.3f}{note}"}


def compute_sample_size(windows_metrics: list[dict]) -> dict:
    """Check if the latest window has enough actions from env[0] for reliable metrics.

    Custom metrics (actions, pnl, risk) are sampled from env[0] only in multiproc.
    Small samples produce unreliable health check results.
    """
    latest = _find_latest_complete(windows_metrics)
    action_counts = latest.get("action_counts", {})
    total = sum(action_counts.values())

    note = ""
    if not windows_metrics[-1].get("is_complete", True):
        note = f" (using earlier complete window, latest has {windows_metrics[-1].get('ep_rew_mean_count', 0)} iterations)"

    if total >= THRESHOLDS["sample_size"]["ok"]:
        return {"status": "OK", "detail": f"Sample size adequate: {int(total)} actions from env[0]{note}"}
    elif total >= THRESHOLDS["sample_size"]["warn"]:
        return {"status": "WARN", "detail": f"Small sample from env[0]: {int(total)} actions, health checks may be unreliable{note}"}
    else:
        return {"status": "FAIL", "detail": f"Tiny sample from env[0]: {int(total)} actions, health checks unreliable{note}"}


def compute_profit_trend(windows_metrics: list[dict]) -> dict:
    """Check total_profit trend across windows.

    total_profit is the most actionable performance metric (HQT philosophy).
    Note: values represent env[0]'s last episode, not window-level aggregates.
    """
    profits = []
    for wm in windows_metrics:
        val = wm.get("total_profit")
        if val is not None:
            profits.append(val)

    if not profits:
        return {"status": "OK", "detail": "No total_profit data"}

    avg = sum(profits) / len(profits)
    negatives = sum(1 for p in profits if p < 0)
    min_profit = min(profits)
    max_profit = max(profits)

    if avg < 0:
        return {"status": "FAIL", "detail": f"Average profit negative: {avg:.2f} ({negatives}/{len(profits)} windows negative, range {min_profit:.2f}~{max_profit:.2f})"}
    elif negatives > 0:
        return {"status": "WARN", "detail": f"Some windows negative: {negatives}/{len(profits)}, avg={avg:.2f} (range {min_profit:.2f}~{max_profit:.2f})"}
    else:
        return {"status": "OK", "detail": f"All windows profitable: avg={avg:.2f} (range {min_profit:.2f}~{max_profit:.2f})"}


def compute_entropy_trend(windows_metrics: list[dict]) -> dict:
    """Check if entropy retained is declining across windows.

    A declining cross-window entropy trend indicates the policy is becoming
    increasingly deterministic, which may reduce adaptability.
    """
    entropies = []
    for wm in windows_metrics:
        first = wm.get("entropy_loss_first")
        last = wm.get("entropy_loss_last")
        if first is not None and last is not None and abs(first) > 0:
            retained = abs(last) / abs(first)
            entropies.append(retained)

    if len(entropies) < 2:
        return {"status": "OK", "detail": "Insufficient data for entropy trend (need 2+ windows)"}

    change = entropies[-1] - entropies[0]

    if change > THRESHOLDS["entropy_trend"]["warn"]:
        return {"status": "OK", "detail": f"Entropy stable across windows: {entropies[0]:.0%} -> {entropies[-1]:.0%} ({change * 100:+.1f}pp)"}
    elif change > THRESHOLDS["entropy_trend"]["fail"]:
        return {"status": "WARN", "detail": f"Entropy declining across windows: {entropies[0]:.0%} -> {entropies[-1]:.0%} ({change * 100:+.1f}pp)"}
    else:
        return {"status": "FAIL", "detail": f"Entropy dropping fast: {entropies[0]:.0%} -> {entropies[-1]:.0%} ({change * 100:+.1f}pp)"}


def compute_explained_variance(windows_metrics: list[dict]) -> dict:
    """Check if the value function can predict returns accurately.

    SB3 logs train/explained_variance per rollout iteration.
    > 0.5 good, 0-0.5 noisy, < 0 worse than random.
    """
    latest = _find_latest_complete(windows_metrics)
    ev = latest.get("explained_variance_last")

    if ev is None:
        return {"status": "OK", "detail": "No explained_variance data"}

    note = ""
    if not windows_metrics[-1].get("is_complete", True):
        note = f" (using earlier complete window, latest has {windows_metrics[-1].get('ep_rew_mean_count', 0)} iterations)"

    if ev > THRESHOLDS["explained_variance"]["ok"]:
        return {"status": "OK", "detail": f"Value function predictive: {ev:.3f}{note}"}
    elif ev > THRESHOLDS["explained_variance"]["warn"]:
        return {"status": "WARN", "detail": f"Value function noisy: {ev:.3f}{note}"}
    else:
        return {"status": "FAIL", "detail": f"Value function worse than random: {ev:.3f}{note}"}


def compute_long_short_bias(windows_metrics: list[dict]) -> dict:
    """Check if entries are heavily biased toward Long or Short.

    Some bias can be valid in trending markets, but extreme bias may
    indicate the model has not learned to trade both directions.
    """
    latest = _find_latest_complete(windows_metrics)
    action_counts = latest.get("action_counts", {})
    long_count = action_counts.get("Long_enter", 0)
    short_count = action_counts.get("Short_enter", 0)
    total = long_count + short_count

    if total < 50:
        return {"status": "OK", "detail": f"Insufficient entry data for bias check ({int(total)} entries)"}

    note = ""
    if not windows_metrics[-1].get("is_complete", True):
        note = f" (using earlier complete window, latest has {windows_metrics[-1].get('ep_rew_mean_count', 0)} iterations)"

    dominant = max(long_count, short_count)
    ratio = dominant / total
    direction = "Long" if long_count > short_count else "Short"

    if ratio > THRESHOLDS["long_short_bias"]["fail"]:
        return {"status": "FAIL", "detail": f"Strong {direction} bias: {ratio:.0%} ({int(long_count)}L/{int(short_count)}S){note}"}
    elif ratio > THRESHOLDS["long_short_bias"]["warn"]:
        return {"status": "WARN", "detail": f"Moderate {direction} bias: {ratio:.0%} ({int(long_count)}L/{int(short_count)}S){note}"}
    else:
        return {"status": "OK", "detail": f"Balanced entries: {long_count / total:.0%}L/{short_count / total:.0%}S ({int(long_count)}L/{int(short_count)}S){note}"}


# ---------------------------------------------------------------------------
# Main analysis
# ---------------------------------------------------------------------------

def analyze(model_id: str, last_n: int, base_dir: Path) -> dict:
    """Run full analysis for a model."""
    model_dir = base_dir / "user_data" / "models" / model_id
    if not model_dir.exists():
        return {
            "model_id": model_id,
            "error": f"Model directory not found: {model_dir}",
        }

    all_windows = discover_windows(model_dir)
    if not all_windows:
        return {
            "model_id": model_id,
            "error": f"No TensorBoard event files found in {model_dir / 'tensorboard'}",
        }

    # Group by pair
    pairs = {}
    for w in all_windows:
        pairs.setdefault(w["pair"], []).append(w)

    results = {
        "model_id": model_id,
        "model_dir": str(model_dir),
        "total_windows": len(all_windows),
        "pairs": list(pairs.keys()),
        "windows_analyzed": last_n,
        "per_pair": {},
    }

    all_health_checks = {}

    for pair, pair_windows in pairs.items():
        # Take last N windows for this pair
        selected = pair_windows[-last_n:]
        results["per_pair"][pair] = {
            "windows": [w["window_name"] for w in selected],
            "latest_window": selected[-1]["window_name"],
        }

        # Extract metrics per window
        windows_metrics = []
        per_window_data = {}
        for w in selected:
            scalars = load_scalars(w["event_dir"])
            metrics = extract_window_metrics(scalars)
            windows_metrics.append(metrics)
            per_window_data[w["window_name"]] = metrics

        results["per_pair"][pair]["metrics"] = per_window_data

        # Determine expected iterations from the max count across windows
        counts = [wm.get("ep_rew_mean_count", 0) for wm in windows_metrics]
        max_count = max(counts) if counts else 0
        threshold = max_count * COMPLETE_THRESHOLD_RATIO

        for wm in windows_metrics:
            wm["is_complete"] = wm.get("ep_rew_mean_count", 0) >= threshold

        # Health checks (using the latest pair's data for overall)
        health = {
            "reward_trend": compute_reward_trend(windows_metrics),
            "liquidation_rate": compute_liquidation_rate(windows_metrics),
            "win_rate": compute_win_rate(windows_metrics),
            "policy_collapse": compute_policy_collapse(windows_metrics),
            "value_loss": compute_value_loss(windows_metrics),
            "entropy_loss": compute_entropy_loss(windows_metrics),
            "invalid_actions": compute_invalid_actions(windows_metrics),
            "approx_kl": compute_approx_kl(windows_metrics),
            "clip_fraction": compute_clip_fraction(windows_metrics),
            "sample_size": compute_sample_size(windows_metrics),
            "profit_trend": compute_profit_trend(windows_metrics),
            "entropy_trend": compute_entropy_trend(windows_metrics),
            "explained_variance": compute_explained_variance(windows_metrics),
            "long_short_bias": compute_long_short_bias(windows_metrics),
        }
        results["per_pair"][pair]["health_checks"] = health

        # Record which window was actually used for health checks
        results["per_pair"][pair]["latest_complete_window"] = next(
            (w["window_name"] for w, wm in zip(reversed(selected), reversed(windows_metrics))
             if wm.get("is_complete", True)),
            selected[-1]["window_name"]
        )

        # Merge into overall (take worst status per check)
        for check_name, check_result in health.items():
            existing = all_health_checks.get(check_name)
            if existing is None:
                all_health_checks[check_name] = check_result
            else:
                # FAIL > WARN > OK
                priority = {"FAIL": 2, "WARN": 1, "OK": 0}
                if priority.get(check_result["status"], 0) > priority.get(existing["status"], 0):
                    all_health_checks[check_name] = check_result

    results["health_checks"] = all_health_checks

    # Summary
    statuses = [v["status"] for v in all_health_checks.values()]
    ok_count = statuses.count("OK")
    warn_count = statuses.count("WARN")
    fail_count = statuses.count("FAIL")
    results["summary"] = f"{ok_count} OK, {warn_count} WARN, {fail_count} FAIL"

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Analyze RL training health from TensorBoard logs"
    )
    # Positional args (for /analyze-rl skill invocation)
    parser.add_argument(
        "pos_args",
        nargs="*",
        help="Positional: [model-id] [last-n]",
    )
    # Named args (for direct CLI usage)
    parser.add_argument(
        "--model-id",
        default=None,
        help="Model identifier (directory name under user_data/models/)",
    )
    parser.add_argument(
        "--last-n",
        type=int,
        default=None,
        help="Number of recent training windows to analyze (default: 5)",
    )
    parser.add_argument(
        "--base-dir",
        default=None,
        help="Project base directory (default: auto-detect from script location)",
    )

    args = parser.parse_args()

    # Positional args override named args; fall back to defaults
    model_id = args.model_id or (args.pos_args[0] if len(args.pos_args) > 0 else None) or "rl-daytrade-v1"
    last_n = args.last_n or (int(args.pos_args[1]) if len(args.pos_args) > 1 else None) or 5

    if args.base_dir:
        base_dir = Path(args.base_dir)
    else:
        # Auto-detect: script is in <project>/scripts/
        base_dir = Path(__file__).resolve().parent.parent

    result = analyze(model_id, last_n, base_dir)
    print(json.dumps(result, indent=2, default=str))

    # Only exit non-zero for actual errors (model not found, etc.)
    # Health check FAILs are informational results, not script errors
    if "error" in result:
        sys.exit(1)
    sys.exit(0)


if __name__ == "__main__":
    main()
