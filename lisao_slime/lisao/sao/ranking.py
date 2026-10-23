"""Global terminal comparisons and their per-trajectory gradient contributions."""

from __future__ import annotations

import torch
import torch.distributed as dist


def data_parallel_group():
    if not (dist.is_available() and dist.is_initialized()):
        return None
    from megatron.core import mpu

    return mpu.get_data_parallel_group()


@torch.no_grad()
def gather_indexed_values(values, indices, size, *, valid=None, process_group=None):
    values = values.detach().float().reshape(-1)
    indices = torch.as_tensor(indices, dtype=torch.long, device=values.device)
    if indices.numel() != values.numel() or indices.unique().numel() != indices.numel():
        raise ValueError("Terminal values require distinct, aligned trajectory indices")
    if indices.numel() and (indices.min() < 0 or indices.max() >= size):
        raise ValueError("Trajectory index is outside the collected batch")
    valid = (
        torch.ones_like(values, dtype=torch.bool)
        if valid is None
        else torch.as_tensor(valid, dtype=torch.bool, device=values.device)
    )
    if valid.shape != values.shape:
        raise ValueError("Terminal validity mask must align with values")
    payload = values.new_zeros((2, size))
    payload[0, indices] = torch.where(valid, values, 0.0)
    payload[1, indices] = valid.float()
    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(payload, group=process_group)
    if (payload[1] > 1).any():
        raise ValueError("A trajectory has more than one owner in the data-parallel group")
    return payload[0], payload[1].bool()


def outcome_pairs(values, group_ids, raw_rewards, valid=None):
    values = values.reshape(-1)
    if len(group_ids) != values.numel() or len(raw_rewards) != values.numel():
        raise ValueError("Group metadata must align with terminal values")
    groups = torch.tensor([0 if g is None else g for g in group_ids], device=values.device)
    rewards = torch.as_tensor(raw_rewards, dtype=values.dtype, device=values.device)
    present = torch.tensor([g is not None for g in group_ids], dtype=torch.bool, device=values.device)
    if valid is not None:
        present = present & valid.to(device=values.device, dtype=torch.bool)
    pairs = (groups[:, None] == groups[None, :]) & (rewards[:, None] > rewards[None, :])
    pairs &= present[:, None] & present[None, :]
    return pairs.nonzero(as_tuple=True)


@torch.no_grad()
def ranking_contributions(values, group_ids, raw_rewards, *, margin, valid=None):
    values = values.detach().reshape(-1)
    pos, neg = outcome_pairs(values, group_ids, raw_rewards, valid)
    gradient = torch.zeros_like(values)
    loss = torch.zeros_like(values)
    pair_counts = torch.zeros_like(values)
    if pos.numel():
        violation = margin - values[pos] + values[neg]
        active = (violation > 0).to(values.dtype) / pos.numel()
        gradient.index_add_(0, pos, -active)
        gradient.index_add_(0, neg, active)
        half_loss = violation.clamp_min(0) / (2 * pos.numel())
        loss.index_add_(0, pos, half_loss)
        loss.index_add_(0, neg, half_loss)
        halves = torch.full_like(half_loss, 0.5)
        pair_counts.index_add_(0, pos, halves)
        pair_counts.index_add_(0, neg, halves)
    return gradient, loss, pair_counts


def terminal_rank_surrogate(values, terminal_mask, gradient, loss):
    terminal = (values.reshape(-1) * terminal_mask.reshape(-1)).sum()
    return gradient * (terminal - terminal.detach()) + loss * terminal_mask.sum()
