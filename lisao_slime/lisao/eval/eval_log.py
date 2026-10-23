"""Persist eval metrics to JSON via --custom-eval-rollout-log-function-path."""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from typing import Any


def _pass_k_sort_key(name: str) -> int:
    try:
        return int(str(name).rsplit("@", 1)[-1])
    except ValueError:
        return 0


def _to_jsonable(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): _to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_jsonable(v) for v in obj]
    if hasattr(obj, "item"):
        try:
            return obj.item()
        except Exception:
            pass
    if isinstance(obj, (str, int, float, bool)) or obj is None:
        return obj
    return str(obj)


def log_eval_rollout_data(
    rollout_id: int,
    args: Any,
    data: dict[str, Any],
    extra_metrics: dict[str, Any] | None = None,
) -> bool:
    """Write eval summary JSON; return False so slime keeps default logging."""
    from slime.utils.metric_utils import compute_pass_rate, dict_add_prefix

    log_dict: dict[str, Any] = dict(extra_metrics or {})
    per_dataset: dict[str, Any] = {}

    for key, payload in data.items():
        rewards = [float(r) for r in payload["rewards"]]
        mean_reward = sum(rewards) / len(rewards) if rewards else 0.0
        log_dict[f"eval/{key}"] = mean_reward
        entry: dict[str, Any] = {
            "mean_reward": mean_reward,
            "num_rewards": len(rewards),
            "rewards": rewards,
        }
        if "truncated" in payload:
            truncated = [int(t) for t in payload["truncated"]]
            ratio = sum(truncated) / len(truncated) if truncated else 0.0
            log_dict[f"eval/{key}-truncated_ratio"] = ratio
            entry["truncated"] = truncated
            entry["truncated_ratio"] = ratio
        if getattr(args, "log_passrate", False) and rewards:
            pr = compute_pass_rate(
                flat_rewards=rewards,
                group_size=args.n_samples_per_eval_prompt,
            )
            log_dict |= dict_add_prefix(pr, f"eval/{key}-")
            entry["pass_rate"] = pr
        # Debug snippets: help diagnose all-zero scores (truncation / missing \\boxed).
        samples = payload.get("samples")
        if samples:
            previews = []
            for s in samples[:3]:
                resp = getattr(s, "response", "") or ""
                previews.append(
                    {
                        "reward": getattr(s, "reward", None),
                        "status": str(getattr(s, "status", None)),
                        "response_len": len(resp),
                        "has_boxed": "\\boxed" in resp,
                        "has_think": ("<think>" in resp) or ("</think>" in resp),
                        "response_tail": resp[-400:],
                    }
                )
            entry["response_previews"] = previews
        per_dataset[key] = entry

    out_dir = os.environ.get("EVAL_OUT_DIR") or os.path.join(
        os.environ.get("DATA_ROOT", "/root/data"), "eval"
    )
    os.makedirs(out_dir, exist_ok=True)

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    tag = os.environ.get("EVAL_TAG", "aime2025")
    out_path = os.path.join(out_dir, f"{tag}_rollout{rollout_id}_{stamp}.json")
    latest = os.path.join(out_dir, f"{tag}_latest.json")

    payload = _to_jsonable(
        {
            "rollout_id": rollout_id,
            "timestamp_utc": stamp,
            "load": getattr(args, "load", None),
            "ckpt_step": getattr(args, "ckpt_step", None),
            "ref_load": getattr(args, "ref_load", None),
            "hf_checkpoint": getattr(args, "hf_checkpoint", None),
            "n_samples_per_eval_prompt": getattr(args, "n_samples_per_eval_prompt", None),
            "eval_max_response_len": getattr(args, "eval_max_response_len", None),
            "metrics": log_dict,
            "datasets": per_dataset,
        }
    )
    for path in (out_path, latest):
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        print(f"[lisao.eval_log] wrote {path}", flush=True)

    # Surface pass@k (unbiased, k=1,2,4,...n) plus mean reward.
    for key, entry in per_dataset.items():
        parts: list[str] = []
        pr = entry.get("pass_rate")
        if isinstance(pr, dict) and pr:
            names = sorted(pr, key=_pass_k_sort_key)
            parts.extend(f"{name}={pr[name]}" for name in names)
        parts.append(f"mean_reward={entry.get('mean_reward')}")
        if "truncated_ratio" in entry:
            parts.append(f"truncated={entry['truncated_ratio']}")
        note = (
            "unbiased pass@k from n_samples_per_eval_prompt"
            if isinstance(pr, dict) and pr
            else "mean_reward (= empirical Pass@1 when n_samples=1)"
        )
        print(f"[lisao.eval_log] {key} {' '.join(parts)} ({note})", flush=True)

    # False → also run slime's default logger.
    return False
