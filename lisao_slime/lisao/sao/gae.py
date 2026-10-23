"""SAO GAE helpers: length-adaptive λ and skip-observation token-level GAE."""

from __future__ import annotations

import torch


def terminal_rewards(reward: float, loss_mask: torch.Tensor) -> torch.Tensor:
    rewards = torch.zeros_like(loss_mask, dtype=torch.float32)
    actions = loss_mask.reshape(-1).nonzero(as_tuple=True)[0]
    if actions.numel():
        rewards.reshape(-1)[actions[-1]] = reward
    return rewards


def compute_per_sample_lambd(
    response_lengths: list[int] | torch.Tensor,
    alpha: float = 1.5,
    lambda_min: float = 0.0,
    lambda_max: float = 1.0,
) -> torch.Tensor:
    """λ_i = clamp(1 − 1/(α · L_i), 0, 1). Default α=1.5."""
    if not isinstance(response_lengths, torch.Tensor):
        response_lengths = torch.tensor(response_lengths, dtype=torch.float32)
    lengths = response_lengths.clamp_min(1).float()
    lambd = 1.0 - 1.0 / (alpha * lengths)
    return lambd.clamp(lambda_min, lambda_max)


def _lambda_per_row(
    lambd: float | torch.Tensor,
    batch_size: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    if not torch.is_tensor(lambd):
        return torch.full((batch_size,), float(lambd), device=device, dtype=dtype)
    vec = lambd.to(device=device, dtype=dtype).reshape(-1)
    if vec.numel() == 1:
        return vec.expand(batch_size)
    if vec.numel() != batch_size:
        raise ValueError(f"GAE lambd has {vec.numel()} rows, expected {batch_size}")
    return vec


def skip_obs_gae(
    rewards: torch.Tensor,
    values: torch.Tensor,
    loss_masks: torch.Tensor,
    gamma: float = 1.0,
    lambd: float | torch.Tensor = 1.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Skip-observation token-level GAE (SAO Eq.4–5).

    Action tokens (``loss_masks==1``) link to the next action, skipping
    observation tokens. Observation positions get advantage 0.
    ``lambd`` may be a scalar or a ``[B]`` tensor.
    """
    assert rewards.ndim == 2 and values.ndim == 2 and loss_masks.ndim == 2
    batch_size, seq_len = rewards.shape
    assert values.shape == (batch_size, seq_len) and loss_masks.shape == (batch_size, seq_len)
    device = rewards.device
    dtype = values.dtype
    lambd_b = _lambda_per_row(lambd, batch_size, device, dtype)

    advantages = torch.zeros(batch_size, seq_len, device=device, dtype=dtype)
    next_adv = torch.zeros(batch_size, device=device, dtype=dtype)
    next_val = torch.zeros(batch_size, device=device, dtype=dtype)

    for t in reversed(range(seq_len)):
        is_action = loss_masks[:, t].to(torch.bool)
        delta = torch.where(
            is_action,
            rewards[:, t] + gamma * next_val - values[:, t],
            torch.zeros_like(rewards[:, t]),
        )
        new_adv = delta + gamma * lambd_b * next_adv
        advantages[:, t] = torch.where(is_action, new_adv, torch.zeros_like(new_adv))
        next_adv = torch.where(is_action, new_adv, next_adv)
        next_val = torch.where(is_action, values[:, t], next_val)

    returns = advantages + values
    return advantages, returns
