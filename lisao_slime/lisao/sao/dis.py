"""SAO Direct double-sided Importance Sampling (DIS).

Plug in via slime's TIS hook:

    --use-dis
    --tis-clip-low 0.7
    --tis-clip 6.0
    --custom-tis-function-path lisao.sao.dis.dis_tis_function

``--use-dis`` turns on ``use_tis`` + ``use_rollout_logprobs`` together.
Tokens with ratio π_θ / π_rollout outside ``[tis_clip_low, tis_clip]`` get
``pg_loss = 0`` (hard mask). Unlike ``icepop_function``, this does **not**
multiply ``pg_loss`` by the ratio again.
"""

from __future__ import annotations

from typing import Any

import torch


def dis_tis_function(
    args,
    *,
    pg_loss: torch.Tensor,
    train_log_probs: list[torch.Tensor],
    rollout_log_probs: list[torch.Tensor],
    loss_masks: list[torch.Tensor],
    **kwargs: Any,
) -> tuple[torch.Tensor, list[torch.Tensor], dict[str, torch.Tensor]]:
    """Token-level DIS mask matching SAO Eq.(1)–(3).

    slime already computed ``pg_loss = -ratio · A``. Multiplying by the trust-
    region mask yields gradient ``r · A · ∇log π_θ`` inside the interval and
    zero outside.

    Returns ``(pg_loss, loss_masks, metrics)`` with ``tis``, ``tis_clipfrac``,
    ``tis_abs``.
    """
    train_lp = torch.cat(train_log_probs, dim=0)
    rollout_lp = torch.cat(rollout_log_probs, dim=0)
    ratio = torch.exp(train_lp - rollout_lp)
    in_trust_region = (ratio >= args.tis_clip_low) & (ratio <= args.tis_clip)
    mask = in_trust_region.to(dtype=pg_loss.dtype)
    pg_loss = pg_loss * mask
    metrics = {
        "tis": ratio.detach(),
        "tis_clipfrac": (1.0 - mask).detach(),
        "tis_abs": (ratio - 1.0).abs().detach(),
    }
    return pg_loss, loss_masks, metrics
