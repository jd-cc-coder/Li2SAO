"""Critic / GAE metrics for LiSAO (explained variance)."""

from __future__ import annotations

import logging
from typing import Any

import torch

logger = logging.getLogger("lisao.sao.metrics")


def explained_variance(
    returns: torch.Tensor,
    values: torch.Tensor,
    masks: torch.Tensor | None = None,
) -> torch.Tensor:
    """EV = 1 − Var(R − V) / Var(R) over (optionally masked) tokens.

    Returns a 0-dim float tensor. If ``Var(R)`` is 0, EV is 0.
    """
    residual = returns - values
    if masks is not None:
        keep = masks.to(dtype=torch.bool)
        if keep.any():
            residual = residual[keep]
            returns = returns[keep]
        else:
            return returns.new_zeros(())
    var_r = returns.var(unbiased=False)
    if var_r <= 0:
        return returns.new_zeros(())
    return 1.0 - residual.var(unbiased=False) / var_r


def masked_return_value_pairs(
    returns: torch.Tensor,
    values: torch.Tensor,
    mask: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Action-token (R, V) pairs. ``mask==0`` (observations) are dropped."""
    ret = returns.reshape(-1).float()
    val = values.reshape(-1).float().to(device=ret.device)
    n = min(ret.numel(), val.numel())
    ret = ret[:n]
    val = val[:n]
    if mask is None or mask.numel() == 0:
        return ret, val
    keep = mask.reshape(-1).to(device=ret.device) != 0
    if keep.numel() != n:
        return ret, val
    return ret[keep], val[keep]


def split_pairs_in_half(
    returns: torch.Tensor,
    values: torch.Tensor,
) -> tuple[tuple[torch.Tensor, torch.Tensor], tuple[torch.Tensor, torch.Tensor]]:
    """Split action tokens into prefix / suffix halves (mid = n//2)."""
    n = min(returns.numel(), values.numel())
    mid = n // 2
    ret = returns.reshape(-1)[:n]
    val = values.reshape(-1)[:n]
    return (ret[:mid], val[:mid]), (ret[mid:], val[mid:])


def compute_critic_explained_variance(rollout_data: dict[str, Any]) -> torch.Tensor | None:
    """EV over the critic batch in ``rollout_data`` (CPU-safe scalar tensor)."""
    values = rollout_data.get("values")
    returns = rollout_data.get("returns")
    if not values or not returns:
        return None
    value_cat = torch.cat([v.detach().flatten().float() for v in values], dim=0)
    return_cat = torch.cat([r.detach().flatten().float() for r in returns], dim=0)
    masks = rollout_data.get("loss_masks")
    mask_cat = None
    if masks:
        mask_cat = torch.cat([m.detach().flatten().float() for m in masks], dim=0)
        if mask_cat.numel() != value_cat.numel():
            mask_cat = None
    return explained_variance(return_cat, value_cat, mask_cat)


def log_critic_explained_variance(args: Any, rollout_id: int, rollout_data: dict[str, Any]) -> float | None:
    """Log ``train/critic/explained_variance``; stash on ``args`` for train metrics."""
    ev = compute_critic_explained_variance(rollout_data)
    if ev is None:
        return None
    ev_f = float(ev.item())
    metrics = {"train/critic/explained_variance": ev_f}
    args._lisao_critic_metrics = metrics
    logger.info("lisao critic rollout %d: explained_variance=%.4f", rollout_id, ev_f)
    try:
        from slime.utils import logging_utils
        from slime.utils.metric_utils import compute_rollout_step

        payload = dict(metrics)
        payload["train/step"] = compute_rollout_step(args, rollout_id)
        logging_utils.log(args, payload, step_key="train/step")
    except Exception:
        logger.debug("lisao critic EV wandb log skipped", exc_info=True)
    return ev_f
