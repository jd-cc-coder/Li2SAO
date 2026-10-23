"""Li²SAO inflight switch helpers.

``drain`` (default) drops leftover / in-flight groups so n-boost and n-stable
never share a sag. ``flatten`` keeps the worker and splits leftover n-boost
groups into n=1 singletons with unique ``group_index`` values so the actor
uses GAE (``apply_li2sao`` keys off max group size).
"""

from __future__ import annotations

from typing import Any

INFLIGHT_SWITCH_DRAIN = "drain"
INFLIGHT_SWITCH_FLATTEN = "flatten"
INFLIGHT_SWITCH_CHOICES = (INFLIGHT_SWITCH_DRAIN, INFLIGHT_SWITCH_FLATTEN)
DEFAULT_INFLIGHT_SWITCH = INFLIGHT_SWITCH_DRAIN
_FLATTEN_GID_START = 1_000_000_000

__all__ = [
    "DEFAULT_INFLIGHT_SWITCH",
    "INFLIGHT_SWITCH_CHOICES",
    "INFLIGHT_SWITCH_DRAIN",
    "INFLIGHT_SWITCH_FLATTEN",
    "as_singleton_group",
    "flatten_buffer_groups",
    "flatten_queue_items",
    "iter_prompt_members",
    "resolve_inflight_switch",
    "split_group_to_n1",
]


def resolve_inflight_switch(args: Any) -> str:
    raw = getattr(args, "li2sao_inflight_switch", None) or DEFAULT_INFLIGHT_SWITCH
    mode = str(raw).strip().lower()
    if mode not in INFLIGHT_SWITCH_CHOICES:
        raise ValueError(
            f"--li2sao-inflight-switch must be one of {INFLIGHT_SWITCH_CHOICES}, got {raw!r}"
        )
    return mode


def _iter_samples(node: Any) -> list[Any]:
    if node is None:
        return []
    if isinstance(node, list):
        out: list[Any] = []
        for item in node:
            out.extend(_iter_samples(item))
        return out
    return [node]


def iter_prompt_members(group: Any) -> list[Any]:
    """One member is one original n-slot: a Sample, or list[Sample] for fan-out."""
    if group is None:
        return []
    if not isinstance(group, list):
        return [group]
    if not group:
        return []
    head = group[0]
    if isinstance(head, list):
        return list(group)
    return list(group)


def as_singleton_group(member: Any) -> list:
    if member is None:
        return []
    if isinstance(member, list):
        return member
    return [member]


def _assign_unique_group_indices(group: list, group_index_start: int) -> int:
    gi = int(group_index_start)
    for sample in _iter_samples(group):
        if hasattr(sample, "group_index"):
            sample.group_index = gi
            gi += 1
    return gi


def split_group_to_n1(
    gid: int,
    group: Any,
    *,
    next_gid: int,
    next_group_index: int,
) -> tuple[list[tuple[int, list]], int, int]:
    """Split one completed group into n=1 leftover items.

    Already-singleton groups are returned unchanged (same gid / group_index).
    """
    members = iter_prompt_members(group)
    if len(members) <= 1:
        return [(gid, as_singleton_group(members[0]) if members else group or [])], next_gid, next_group_index

    out: list[tuple[int, list]] = []
    gid_i = next_gid
    gi = next_group_index
    for member in members:
        singleton = as_singleton_group(member)
        gi = _assign_unique_group_indices(singleton, gi)
        out.append((gid_i, singleton))
        gid_i += 1
    return out, gid_i, gi


def flatten_queue_items(
    items: list[tuple[int, list]],
    *,
    gid_start: int,
    group_index_start: int,
) -> tuple[list[tuple[int, list]], int, int, int]:
    """Flatten multi-sample queue items. Singletons keep gid and group_index.

    Returns ``(flat_items, next_gid, next_group_index, n_split_members)``.
    ``n_split_members`` counts extra singletons created (0 if nothing to split).
    """
    out: list[tuple[int, list]] = []
    next_gid = int(gid_start)
    next_gi = int(group_index_start)
    n_split = 0
    for gid, group in items:
        members = iter_prompt_members(group)
        if len(members) <= 1:
            out.append((gid, as_singleton_group(members[0]) if members else group or []))
            continue
        pieces, next_gid, next_gi = split_group_to_n1(
            gid,
            group,
            next_gid=next_gid,
            next_group_index=next_gi,
        )
        out.extend(pieces)
        n_split += len(pieces)
    return out, next_gid, next_gi, n_split


def flatten_buffer_groups(data_buffer: Any, *, group_index_start: int) -> tuple[int, int]:
    """Rewrite ``data_buffer.buffer`` n-boost groups into n=1 groups.

    Uses ``group_index_start`` (a flatten-only high range) so ids cannot collide
    with ``data_source.sample_group_index``. Does not bump that counter.
    Returns ``(n_split, next_group_index)``.
    """
    buffer = getattr(data_buffer, "buffer", None)
    next_gi = int(group_index_start)
    if not buffer:
        return 0, next_gi
    new_buf: list[Any] = []
    n_split = 0
    for group in buffer:
        members = iter_prompt_members(group)
        if len(members) <= 1:
            new_buf.append(group)
            continue
        for member in members:
            singleton = as_singleton_group(member)
            next_gi = _assign_unique_group_indices(singleton, next_gi)
            new_buf.append(singleton)
            n_split += 1
    buffer[:] = new_buf
    return n_split, next_gi
