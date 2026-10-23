"""Completed-group take policy for inflight-async training.

Default (``--max-policy-lag`` unset): the caller drains the output queue,
sorts, and **discards** anything beyond ``rollout_batch_size``.

With ``--max-policy-lag N``: a completed group is dropped if any sample's
policy lag is greater than ``N``. Lag is ``current - oldest weight_version``
on that trajectory — for cross-policy (abort→resume) samples that is the
policy of the **first tokens**, not the latest resume segment. Extra groups
that still pass the lag bound are returned as leftover for the next
training step instead of being discarded.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from lisao.inflight_async.metrics import sample_policy_lag

__all__ = [
    "QueueTakeResult",
    "group_exceeds_max_policy_lag",
    "group_max_policy_lag",
    "resolve_max_policy_lag",
    "take_groups_from_completed",
]


def _iter_samples(node: Any) -> list[Any]:
    if node is None:
        return []
    if isinstance(node, list):
        out: list[Any] = []
        for item in node:
            out.extend(_iter_samples(item))
        return out
    return [node]


def resolve_max_policy_lag(args: Any) -> int | None:
    """Return ``max_policy_lag`` or ``None`` (legacy take-and-discard)."""
    value = getattr(args, "max_policy_lag", None)
    if value is None:
        return None
    value = int(value)
    if value < 0:
        raise ValueError(f"--max-policy-lag must be >= 0, got {value}")
    return value


def group_max_policy_lag(group: Any, current_weight_version: Any) -> int | None:
    """Largest per-sample lag in ``group`` (each sample uses its oldest version)."""
    lags: list[int] = []
    for sample in _iter_samples(group):
        lag = sample_policy_lag(sample, current_weight_version)
        if lag is not None:
            lags.append(lag)
    return max(lags) if lags else None


def group_exceeds_max_policy_lag(
    group: Any,
    current_weight_version: Any,
    max_policy_lag: int,
) -> bool:
    """True iff any sample's first-token / oldest-version lag exceeds ``max_policy_lag``."""
    lag = group_max_policy_lag(group, current_weight_version)
    return lag is not None and lag > max_policy_lag


@dataclass
class QueueTakeResult:
    accepted: list[tuple[int, list]] = field(default_factory=list)
    leftover: list[tuple[int, list]] = field(default_factory=list)
    dropped_stale: list[tuple[int, list]] = field(default_factory=list)


def take_groups_from_completed(
    items: list[tuple[int, list]],
    *,
    need: int,
    max_policy_lag: int | None,
    current_weight_version: Any = None,
) -> QueueTakeResult:
    """Partition one drain of the completed-group queue.

    * ``max_policy_lag is None``: accept every item (caller sorts and
      discards the overshoot). ``leftover`` / ``dropped_stale`` stay empty.
    * otherwise: drop stale groups, accept up to ``need``, leftover the rest.
    """
    if max_policy_lag is None:
        return QueueTakeResult(accepted=list(items))

    result = QueueTakeResult()
    remaining = max(0, int(need))
    for item in items:
        _gid, group = item
        if group_exceeds_max_policy_lag(group, current_weight_version, max_policy_lag):
            result.dropped_stale.append(item)
            continue
        if remaining > 0:
            result.accepted.append(item)
            remaining -= 1
        else:
            result.leftover.append(item)
    return result
