"""Optional Li²SAO critic targets: mixed-group residual MSE and fork ranking.

Default ``absolute`` is a no-op: critic still fits uncentered GAE returns, actor
``A`` / switch logic are untouched. ``residual`` and ``residual_rank`` rewrite
*critic* labels only:

- drop collapsed (all-equal / singleton) groups from the value loss
- fit ``R_t - μ_g(raw)`` instead of absolute ``R_t``
- zero tokens before the first local pos/neg response divergence
- ``residual_rank`` adds a last-action hinge over the batch's mixed pairs

After the n=1 switch there are no mixed groups; this module no-ops so the
critic keeps training on absolute GAE.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from typing import Any

import torch
import torch.nn.functional as F

from lisao.sao.li2sao import (
    _as_float_list,
    _as_int_list,
    group_raw_means,
    last_action_value,
)

logger = logging.getLogger("lisao.sao.critic_process")

CRITIC_TARGET_ABSOLUTE = "absolute"
CRITIC_TARGET_RESIDUAL = "residual"
CRITIC_TARGET_RESIDUAL_RANK = "residual_rank"
CRITIC_TARGET_CHOICES = (
    CRITIC_TARGET_ABSOLUTE,
    CRITIC_TARGET_RESIDUAL,
    CRITIC_TARGET_RESIDUAL_RANK,
)
DEFAULT_CRITIC_TARGET = CRITIC_TARGET_ABSOLUTE
DEFAULT_RANK_COEF = 1.0
DEFAULT_RANK_MARGIN = 0.1


def resolve_critic_target(args: Any) -> str:
    raw = getattr(args, "li2sao_critic_target", None) or DEFAULT_CRITIC_TARGET
    target = str(raw).strip().lower()
    if target not in CRITIC_TARGET_CHOICES:
        raise ValueError(
            f"--li2sao-critic-target must be one of {CRITIC_TARGET_CHOICES}, got {raw!r}"
        )
    return target


def last_action_index(values: torch.Tensor, loss_mask: torch.Tensor | None) -> int:
    """Index of the last action token in a response-aligned tensor, else -1."""
    vec = values.reshape(-1)
    if vec.numel() == 0:
        return -1
    if loss_mask is not None:
        mask = loss_mask.reshape(-1)
        if mask.numel() == vec.numel():
            action = (mask != 0).nonzero(as_tuple=True)[0]
            if action.numel():
                return int(action[-1].item())
            return -1
    return int(vec.numel() - 1)


def first_response_divergence(tokens_a: torch.Tensor, tokens_b: torch.Tensor) -> int:
    """First response index where two token sequences differ; ``min(len)`` if prefix-equal."""
    a = tokens_a.detach().reshape(-1).to(device="cpu")
    b = tokens_b.detach().reshape(-1).to(device="cpu")
    n = min(a.numel(), b.numel())
    if n == 0:
        return 0
    neq = a[:n] != b[:n]
    idx = neq.nonzero(as_tuple=True)[0]
    if idx.numel():
        return int(idx[0].item())
    return n


def response_token_slice(
    tokens: torch.Tensor | None,
    total_len: int,
    resp_len: int,
) -> torch.Tensor | None:
    """Prompt-stripped response tokens, or ``None`` if the payload is unusable."""
    if tokens is None or resp_len <= 0:
        return None
    t = tokens.reshape(-1)
    if t.numel() == 0:
        return None
    prompt_len = max(int(total_len) - int(resp_len), 0)
    end = prompt_len + int(resp_len)
    if t.numel() >= end:
        return t[prompt_len:end]
    if t.numel() >= int(resp_len):
        return t[-int(resp_len) :]
    return t


def _keep_last_action(template: torch.Tensor, loss_mask: torch.Tensor | None) -> torch.Tensor:
    keep = torch.zeros_like(template)
    idx = last_action_index(template, loss_mask)
    if idx >= 0:
        keep.reshape(-1)[idx] = 1
    return keep


def mixed_group_last_action_rank_loss(
    values_list: list[torch.Tensor],
    group_ids: list[int | None],
    raw_rewards: list[float],
    loss_masks: list[torch.Tensor] | None,
    *,
    margin: float,
) -> tuple[torch.Tensor | None, int]:
    """Mean hinge ``max(0, m - (V^+_T - V^-_T))`` over mixed pairs in this micro-batch.

    Higher raw reward is the positive member. Empty / unmatched batches return
    ``(None, 0)`` so the value loss stays unchanged.
    """
    n = len(values_list)
    group_ids = _as_int_list(group_ids) or []
    raw_rewards = _as_float_list(raw_rewards) or []
    if n < 2 or len(group_ids) != n or len(raw_rewards) != n:
        return None, 0
    buckets: dict[int, list[int]] = defaultdict(list)
    for i, gid in enumerate(group_ids):
        if gid is None or values_list[i] is None or values_list[i].numel() == 0:
            continue
        buckets[int(gid)].append(i)

    pair_losses: list[torch.Tensor] = []
    for members in buckets.values():
        if len(members) < 2:
            continue
        for a_i in range(len(members)):
            for b_i in range(a_i + 1, len(members)):
                i = members[a_i]
                j = members[b_i]
                ri = float(raw_rewards[i])
                rj = float(raw_rewards[j])
                if ri == rj:
                    continue
                pos, neg = (i, j) if ri > rj else (j, i)
                mask_pos = loss_masks[pos] if loss_masks is not None and pos < len(loss_masks) else None
                mask_neg = loss_masks[neg] if loss_masks is not None and neg < len(loss_masks) else None
                v_pos = last_action_value(values_list[pos], mask_pos)
                v_neg = last_action_value(values_list[neg], mask_neg)
                pair_losses.append(F.relu(float(margin) - (v_pos - v_neg)))
    if not pair_losses:
        return None, 0
    stacked = torch.stack([p.reshape(()) for p in pair_losses])
    return stacked.mean(), int(stacked.numel())


def apply_critic_process_targets(args: Any, rollout_data: dict[str, Any]) -> dict[str, Any] | None:
    """Rewrite critic returns / keep-mask. Never touches actor ``advantages``."""
    target = resolve_critic_target(args)
    args.li2sao_critic_target = target
    if target == CRITIC_TARGET_ABSOLUTE:
        args._lisao_critic_process_metrics = None
        rollout_data.pop("critic_keep", None)
        rollout_data.pop("critic_group_ids", None)
        rollout_data.pop("critic_raw_rewards", None)
        return None

    returns = rollout_data.get("returns")
    if not returns:
        args._lisao_critic_process_metrics = None
        return None

    group_ids = _as_int_list(rollout_data.get("group_indices"))
    raw_rewards = _as_float_list(rollout_data.get("raw_reward"))
    if group_ids is None or raw_rewards is None:
        args._lisao_critic_process_metrics = None
        return None

    counts: dict[int, int] = defaultdict(int)
    for gid in group_ids:
        if gid is not None:
            counts[int(gid)] += 1
    n_batch = max(counts.values()) if counts else 1
    if n_batch <= 1:
        metrics = {
            "target": target,
            "n_batch": int(n_batch),
            "n_mixed_kept": 0,
            "n_collapsed_drop": 0,
            "n_last_token_only": 0,
            "n_rank_pairs": 0,
            "mean_div_frac": 0.0,
            "noop": 1,
        }
        args._lisao_critic_process_metrics = metrics
        rollout_data.pop("critic_keep", None)
        rollout_data.pop("critic_group_ids", None)
        rollout_data.pop("critic_raw_rewards", None)
        logger.info("li2sao critic_target=%s n_batch=%s: no mixed groups, keep absolute GAE", target, n_batch)
        return metrics

    n_local = len(returns)
    partition = rollout_data.get("partition")
    if partition is not None:
        partition = [int(i) for i in partition]
    if partition is None:
        if n_local != len(group_ids):
            logger.warning(
                "li2sao critic_target=%s: partition missing and local returns %s != full group_ids %s; skip",
                target,
                n_local,
                len(group_ids),
            )
            args._lisao_critic_process_metrics = None
            return None
        partition = list(range(n_local))
    if len(partition) != n_local:
        raise ValueError(f"partition {len(partition)} != returns {n_local}")

    means, _stds, skip, mixed_frac = group_raw_means(group_ids, raw_rewards)
    loss_masks = rollout_data.get("loss_masks")
    tokens_list = rollout_data.get("tokens")
    total_lengths = rollout_data.get("total_lengths")
    response_lengths = rollout_data.get("response_lengths")

    local_gids: list[int | None] = []
    local_raws: list[float] = []
    local_by_gid: dict[int, list[int]] = defaultdict(list)
    for local_i, global_i in enumerate(partition):
        gid = group_ids[global_i] if 0 <= global_i < len(group_ids) else None
        raw = raw_rewards[global_i] if 0 <= global_i < len(raw_rewards) else 0.0
        local_gids.append(gid)
        local_raws.append(float(raw))
        if gid is not None:
            local_by_gid[int(gid)].append(local_i)

    critic_keep: list[torch.Tensor] = []
    n_mixed_kept = 0
    n_collapsed_drop = 0
    n_last_token_only = 0
    div_frac_sum = 0.0
    n_div = 0

    for local_i in range(n_local):
        ret = returns[local_i]
        mask = loss_masks[local_i] if loss_masks is not None and local_i < len(loss_masks) else None
        resp_len = int(response_lengths[local_i]) if response_lengths is not None else int(ret.numel())
        template = ret.new_zeros(resp_len)
        gid = local_gids[local_i]
        collapsed = gid is None or bool(skip.get(int(gid), True))
        if collapsed:
            critic_keep.append(template)
            n_collapsed_drop += 1
            continue

        mu = float(means.get(int(gid), 0.0))
        rollout_data["returns"][local_i] = ret - mu
        n_mixed_kept += 1

        total_len = int(total_lengths[local_i]) if total_lengths is not None else int(ret.numel())
        opposites = [
            j
            for j in local_by_gid.get(int(gid), [])
            if j != local_i and float(local_raws[j]) != float(local_raws[local_i])
        ]
        keep: torch.Tensor
        if not opposites or tokens_list is None:
            keep = _keep_last_action(template, mask)
            n_last_token_only += 1
        else:
            t_star = 0
            my_tok = response_token_slice(tokens_list[local_i], total_len, resp_len)
            if my_tok is None:
                keep = _keep_last_action(template, mask)
                n_last_token_only += 1
            else:
                for j in opposites:
                    j_resp = int(response_lengths[j]) if response_lengths is not None else int(returns[j].numel())
                    j_total = int(total_lengths[j]) if total_lengths is not None else int(returns[j].numel())
                    other_tok = response_token_slice(tokens_list[j], j_total, j_resp)
                    if other_tok is None:
                        continue
                    t_star = max(t_star, first_response_divergence(my_tok, other_tok))
                keep = torch.ones_like(template)
                cut = min(t_star, keep.numel())
                if cut > 0:
                    keep.reshape(-1)[:cut] = 0
                if mask is not None and mask.numel() == keep.numel():
                    keep = keep * (mask.reshape_as(keep) != 0).to(dtype=keep.dtype)
                if keep.reshape(-1).sum().item() <= 0:
                    keep = _keep_last_action(template, mask)
                    n_last_token_only += 1
                else:
                    n_div += 1
                    div_frac_sum += float(cut) / max(keep.numel(), 1)
        critic_keep.append(keep)

    rollout_data["critic_keep"] = critic_keep
    rollout_data["critic_group_ids"] = local_gids
    rollout_data["critic_raw_rewards"] = local_raws

    n_rank_pairs = 0
    for gid, members in local_by_gid.items():
        if skip.get(int(gid), True):
            continue
        highs = [i for i in members if float(local_raws[i]) > 0.0]
        lows = [i for i in members if float(local_raws[i]) <= 0.0]
        n_rank_pairs += len(highs) * len(lows)

    metrics = {
        "target": target,
        "n_batch": int(n_batch),
        "n_mixed_kept": int(n_mixed_kept),
        "n_collapsed_drop": int(n_collapsed_drop),
        "n_last_token_only": int(n_last_token_only),
        "n_rank_pairs": int(n_rank_pairs),
        "mean_div_frac": (div_frac_sum / n_div) if n_div else 0.0,
        "group_mixed_frac": float(mixed_frac),
        "noop": 0,
    }
    args._lisao_critic_process_metrics = metrics
    logger.info(
        "li2sao critic_target=%s mixed_kept=%s collapsed_drop=%s last_token_only=%s "
        "rank_pairs=%s mean_div_frac=%.3f mixed_frac=%.3f",
        target,
        n_mixed_kept,
        n_collapsed_drop,
        n_last_token_only,
        n_rank_pairs,
        float(metrics["mean_div_frac"]),
        float(mixed_frac),
    )
    return metrics
