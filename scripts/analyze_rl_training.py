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
import re
import sys
from pathlib import Path

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
}


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

    return metrics


# ---------------------------------------------------------------------------
# Health checks
# ---------------------------------------------------------------------------

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
    """Check liquidation rate in the latest window."""
    latest = windows_metrics[-1]
    liquidations = latest.get("liquidations", 0)
    total_exits = latest.get("profitable_exits", 0) + latest.get("loss_exits", 0)

    # Liquidation count is separate from exit counts, add it to total
    total_episodes = total_exits + liquidations
    if total_episodes == 0:
        return {"status": "OK", "detail": "No episodes recorded"}

    rate = liquidations / total_episodes
    if rate < THRESHOLDS["liquidation_rate"]["ok"]:
        return {"status": "OK", "detail": f"Liquidation rate: {rate:.1%} ({int(liquidations)}/{int(total_episodes)})"}
    elif rate < THRESHOLDS["liquidation_rate"]["warn"]:
        return {"status": "WARN", "detail": f"Liquidation rate: {rate:.1%} ({int(liquidations)}/{int(total_episodes)})"}
    else:
        return {"status": "FAIL", "detail": f"Liquidation rate: {rate:.1%} ({int(liquidations)}/{int(total_episodes)})"}


def compute_win_rate(windows_metrics: list[dict]) -> dict:
    """Check win rate from profitable/loss exit counts."""
    latest = windows_metrics[-1]
    wins = latest.get("profitable_exits", 0)
    losses = latest.get("loss_exits", 0)
    total = wins + losses

    if total == 0:
        return {"status": "WARN", "detail": "No exits recorded"}

    rate = wins / total
    if rate > THRESHOLDS["win_rate"]["ok"]:
        return {"status": "OK", "detail": f"Win rate: {rate:.1%} ({int(wins)}W/{int(losses)}L)"}
    elif rate > THRESHOLDS["win_rate"]["warn"]:
        return {"status": "WARN", "detail": f"Win rate: {rate:.1%} ({int(wins)}W/{int(losses)}L)"}
    else:
        return {"status": "FAIL", "detail": f"Win rate: {rate:.1%} ({int(wins)}W/{int(losses)}L)"}


def compute_policy_collapse(windows_metrics: list[dict]) -> dict:
    """Check if any single action dominates > 80% of total."""
    latest = windows_metrics[-1]
    action_counts = latest.get("action_counts", {})

    if not action_counts:
        return {"status": "OK", "detail": "No action data"}

    total = sum(action_counts.values())
    if total == 0:
        return {"status": "OK", "detail": "No actions recorded"}

    max_action = max(action_counts, key=action_counts.get)
    max_ratio = action_counts[max_action] / total

    dist_str = ", ".join(
        f"{k}={v / total:.0%}" for k, v in sorted(action_counts.items())
    )

    if max_ratio > THRESHOLDS["policy_collapse"]["fail"]:
        return {"status": "FAIL", "detail": f"Policy collapse: {max_action}={max_ratio:.0%} [{dist_str}]"}
    elif max_ratio > THRESHOLDS["policy_collapse"]["warn"]:
        return {"status": "WARN", "detail": f"Action concentration: {max_action}={max_ratio:.0%} [{dist_str}]"}
    else:
        return {"status": "OK", "detail": f"Action distribution balanced [{dist_str}]"}


def compute_value_loss(windows_metrics: list[dict]) -> dict:
    """Check if value_loss is stable or exploding."""
    vloss_values = []
    for wm in windows_metrics:
        val = wm.get("value_loss_last")
        if val is not None:
            vloss_values.append(val)

    if not vloss_values:
        return {"status": "WARN", "detail": "No value_loss data"}

    if len(vloss_values) < 2:
        val = vloss_values[0]
        if math.isnan(val) or math.isinf(val):
            return {"status": "FAIL", "detail": f"Value loss is NaN/Inf: {val}"}
        return {"status": "OK", "detail": f"Value loss: {val:.4f} (single window)"}

    first = vloss_values[0]
    last = vloss_values[-1]

    if math.isnan(last) or math.isinf(last):
        return {"status": "FAIL", "detail": f"Value loss is NaN/Inf: {last}"}

    if first == 0:
        if last == 0:
            return {"status": "OK", "detail": "Value loss stable at 0"}
        ratio = float("inf")
    else:
        ratio = last / first

    if ratio <= 1.0:
        return {"status": "OK", "detail": f"Value loss decreasing: {first:.4f} -> {last:.4f} ({ratio:.2f}x)"}
    elif ratio < THRESHOLDS["value_loss"]["warn"]:
        return {"status": "OK", "detail": f"Value loss stable: {first:.4f} -> {last:.4f} ({ratio:.2f}x)"}
    else:
        return {"status": "FAIL", "detail": f"Value loss exploding: {first:.4f} -> {last:.4f} ({ratio:.2f}x)"}


def compute_invalid_actions(windows_metrics: list[dict]) -> dict:
    """Check if invalid action rate is decreasing."""
    latest = windows_metrics[-1]
    invalid = latest.get("invalid_actions", 0)
    action_counts = latest.get("action_counts", {})
    total_actions = sum(action_counts.values()) + invalid

    if total_actions == 0:
        return {"status": "OK", "detail": "No actions recorded"}

    rate = invalid / total_actions
    if rate < THRESHOLDS["invalid_actions"]["ok"]:
        return {"status": "OK", "detail": f"Invalid action rate: {rate:.1%} ({int(invalid)}/{int(total_actions)})"}
    elif rate < THRESHOLDS["invalid_actions"]["warn"]:
        return {"status": "WARN", "detail": f"Invalid action rate: {rate:.1%} ({int(invalid)}/{int(total_actions)})"}
    else:
        return {"status": "FAIL", "detail": f"Invalid action rate: {rate:.1%} ({int(invalid)}/{int(total_actions)})"}


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

        # Health checks (using the latest pair's data for overall)
        health = {
            "reward_trend": compute_reward_trend(windows_metrics),
            "liquidation_rate": compute_liquidation_rate(windows_metrics),
            "win_rate": compute_win_rate(windows_metrics),
            "policy_collapse": compute_policy_collapse(windows_metrics),
            "value_loss": compute_value_loss(windows_metrics),
            "invalid_actions": compute_invalid_actions(windows_metrics),
        }
        results["per_pair"][pair]["health_checks"] = health

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

    # Exit with non-zero if any FAIL
    if "error" in result:
        sys.exit(1)
    if result.get("summary", "").count("FAIL") > 0:
        sys.exit(2)
    sys.exit(0)


if __name__ == "__main__":
    main()
