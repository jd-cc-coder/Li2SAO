"""DAPO-style overlong reward shaping for LiSAO math rollouts.

Hard truncation is ``--rollout-max-response-len``. This hook then subtracts a
linear penalty in the last ``buffer`` tokens, and applies the full penalty to
any ``TRUNCATED`` sample (hit the response cap or the context window).

``raw_reward`` stays the unshaped math score; training uses the shaped value.
"""

from __future__ import annotations

import math
import os
from collections import defaultdict
from typing import Any


def _status_value(sample: Any) -> str:
    status = getattr(sample, "status", None)
    if status is None:
        return ""
    return str(getattr(status, "value", status))


def overlong_penalty(
    response_length: int,
    max_response_len: int,
    buffer_len: int,
    penalty_factor: float,
    *,
    truncated: bool = False,
) -> float:
    """Return a non-positive penalty.

    Soft window: ``min(0, (max_len - buffer - L) / buffer) * factor``.
    Truncated samples get at least ``-factor`` (the value at the cap).
    """
    if penalty_factor == 0.0:
        return 0.0
    penalty = 0.0
    if buffer_len > 0 and max_response_len > 0:
        penalty = min(0.0, (max_response_len - buffer_len - response_length) / buffer_len) * penalty_factor
    if truncated:
        penalty = min(penalty, -float(penalty_factor))
    return float(penalty)


def _resolve_overlong_cfg(args: Any) -> tuple[int, int, float]:
    max_len = int(getattr(args, "rollout_max_response_len", 0) or 0)
    env_buffer = os.environ.get("OVERLONG_BUFFER_LEN")
    env_factor = os.environ.get("OVERLONG_PENALTY_FACTOR")
    if env_buffer is not None and env_buffer != "":
        buffer_len = int(env_buffer)
    else:
        buffer_len = max(1, max_len // 4) if max_len > 0 else 0
    factor = float(env_factor) if env_factor not in (None, "") else 1.0
    return max_len, buffer_len, factor


def post_process_rewards(args: Any, samples: list) -> tuple[list[float], list[float]]:
    """slime ``--custom-reward-post-process-path``: ``(raw_rewards, shaped_rewards)``."""
    max_len, buffer_len, factor = _resolve_overlong_cfg(args)
    raw_rewards: list[float] = []
    shaped_rewards: list[float] = []
    for sample in samples:
        raw = float(sample.get_reward_value(args))
        penalty = overlong_penalty(
            int(getattr(sample, "response_length", 0) or 0),
            max_len,
            buffer_len,
            factor,
            truncated=_status_value(sample) == "truncated",
        )
        raw_rewards.append(raw)
        shaped_rewards.append(raw + penalty)
    if getattr(args, "advantage_estimator", None) in {"grpo", "gspo", "cispo", "reinforce_plus_plus_baseline"} and getattr(
        args, "rewards_normalization", True
    ):
        groups: dict[int, list[int]] = defaultdict(list)
        group_ids = [getattr(sample, "group_index", None) for sample in samples]
        if any(gid is not None for gid in group_ids):
            if any(gid is None for gid in group_ids):
                raise ValueError("Reward normalization requires a group_index for every sample")
        else:
            n = int(args.n_samples_per_prompt)
            if n < 1 or len(samples) % n:
                raise ValueError("Reward normalization requires complete prompt groups")
            group_ids = [i // n for i in range(len(samples))]
        for i, gid in enumerate(group_ids):
            groups[gid].append(i)
        normalize_std = args.advantage_estimator != "reinforce_plus_plus_baseline" and getattr(
            args, "grpo_std_normalization", True
        )
        for indices in groups.values():
            mean = math.fsum(shaped_rewards[i] for i in indices) / len(indices)
            centered = [shaped_rewards[i] - mean for i in indices]
            std = math.sqrt(math.fsum(x * x for x in centered) / (len(indices) - 1)) if len(indices) > 1 else 0.0
            for i, value in zip(indices, centered, strict=True):
                shaped_rewards[i] = value / (std + 1e-6) if normalize_std else value
    return raw_rewards, shaped_rewards
