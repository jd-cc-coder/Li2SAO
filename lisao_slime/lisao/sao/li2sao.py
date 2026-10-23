"""Li2SAO: group-relative bootstrap, then one switch to single-rollout GAE.

While the group size is greater than one, actor advantages are group-relative outcome rewards and the
critic fits group-centered returns plus a ranking loss. Value clip is 0.2 in
that stage and inactive after the switch. The readiness gate is residual
explained variance together with ranking accuracy.

After the switch, the actor uses GAE and the critic fits absolute returns.
Clip and DIS stay at the LiSAO values unless ``--li2sao-boost-grpo-loss`` is
set. ``--li2sao-gae-mix`` is an optional blend and is disabled by default.
"""

from __future__ import annotations

import logging
import math
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any

import torch

from lisao.sao.metrics import masked_return_value_pairs, split_pairs_in_half

logger = logging.getLogger("lisao.sao.li2sao")

DEFAULT_N_BOOST = 8
DEFAULT_N_STABLE = 1
DEFAULT_TRAJ_PER_STEP = 128
DEFAULT_EV_EMA_BETA = 0.9
DEFAULT_EV_THRESHOLD = 0.2
DEFAULT_EV_CONSECUTIVE = 10
DEFAULT_BOOST_MIN_STEPS = 60
DEFAULT_BOOST_MAX_STEPS = 150
SWITCH_CRITERION_SEQ = "seq"
SWITCH_CRITERION_TOKEN = "token"
SWITCH_CRITERION_RESIDUAL = "residual"
SWITCH_CRITERION_CHOICES = (
    SWITCH_CRITERION_SEQ,
    SWITCH_CRITERION_TOKEN,
    SWITCH_CRITERION_RESIDUAL,
)
DEFAULT_SWITCH_CRITERION = SWITCH_CRITERION_SEQ
DEFAULT_SWITCH_RANK_THRESHOLD = 0.0
DEFAULT_GAE_MIX = False
TOKEN_DEFAULT_EV_THRESHOLD = 0.2
TOKEN_DEFAULT_BOOST_MIN_STEPS = 80
TOKEN_DEFAULT_BOOST_MAX_STEPS = 200
GRPO_STD_EPS = 1e-6
DEFAULT_BOOST_GRPO_LOSS = False
DEFAULT_BOOST_EPS_CLIP = 0.2
DEFAULT_BOOST_EPS_CLIP_HIGH = 0.28
LOSS_MODE_BOOST_GRPO = "boost_grpo"
LOSS_MODE_STABLE_LISAO = "stable_lisao"


def resolve_switch_criterion(args: Any) -> str:
    raw = getattr(args, "li2sao_switch_criterion", None) or DEFAULT_SWITCH_CRITERION
    criterion = str(raw).strip().lower()
    if criterion not in SWITCH_CRITERION_CHOICES:
        raise ValueError(
            f"--li2sao-switch-criterion must be one of {SWITCH_CRITERION_CHOICES}, got {raw!r}"
        )
    return criterion


def gae_mix_enabled(args: Any) -> bool:
    return bool(getattr(args, "li2sao_gae_mix", DEFAULT_GAE_MIX))


def boost_grpo_loss_enabled(args: Any) -> bool:
    """While n>1, match native GRPO: no DIS/TIS, PPO clip 0.2/0.28."""
    return bool(getattr(args, "li2sao_boost_grpo_loss", DEFAULT_BOOST_GRPO_LOSS))


def li2sao_loss_fields(args: Any) -> dict[str, Any]:
    """Actor-side fields that must stay in sync with the driver Namespace."""
    return {
        "value_clip": float(getattr(args, "value_clip", 0.2)),
        "eps_clip": float(getattr(args, "eps_clip", DEFAULT_BOOST_EPS_CLIP)),
        "eps_clip_high": float(getattr(args, "eps_clip_high", DEFAULT_BOOST_EPS_CLIP_HIGH) or DEFAULT_BOOST_EPS_CLIP),
        "use_tis": bool(getattr(args, "use_tis", False)),
        "li2sao_loss_mode": getattr(args, "li2sao_loss_mode", None),
        "li2sao_stable_eps_clip": getattr(args, "li2sao_stable_eps_clip", None),
        "li2sao_stable_eps_clip_high": getattr(args, "li2sao_stable_eps_clip_high", None),
        "li2sao_stable_use_tis": getattr(args, "li2sao_stable_use_tis", None),
        "_li2sao_stable_loss_stashed": bool(getattr(args, "_li2sao_stable_loss_stashed", False)),
    }


def _stash_stable_loss(args: Any) -> None:
    if getattr(args, "_li2sao_stable_loss_stashed", False):
        return
    eps = float(getattr(args, "eps_clip", DEFAULT_BOOST_EPS_CLIP))
    high = getattr(args, "eps_clip_high", None)
    args.li2sao_stable_eps_clip = eps
    args.li2sao_stable_eps_clip_high = float(high if high is not None else eps)
    args.li2sao_stable_use_tis = bool(getattr(args, "use_tis", False))
    args._li2sao_stable_loss_stashed = True


INACTIVE_VALUE_CLIP = float("inf")


def apply_li2sao_value_clip(args: Any, n_samples: int) -> None:
    """Use value clip 0.2 while groups are larger than one. Disable it after the switch."""
    if not getattr(args, "_li2sao_value_clip_stashed", False):
        raw = getattr(args, "value_clip", 0.2)
        try:
            bootstrap_clip = float(raw)
        except (TypeError, ValueError):
            bootstrap_clip = 0.2
        if not math.isfinite(bootstrap_clip) or bootstrap_clip <= 0.0:
            bootstrap_clip = 0.2
        args._li2sao_bootstrap_value_clip = bootstrap_clip
        args._li2sao_value_clip_stashed = True
    n_stable = int(getattr(args, "li2sao_n_stable", DEFAULT_N_STABLE))
    if int(n_samples) > n_stable:
        args.value_clip = float(args._li2sao_bootstrap_value_clip)
    else:
        args.value_clip = INACTIVE_VALUE_CLIP


def apply_li2sao_loss_mode(args: Any, n_samples: int) -> dict[str, Any]:
    """Set PPO clip and TIS for the current group size.

    Value clip is 0.2 during the bootstrap and inactive after the switch to one
    sample per prompt. The GRPO clip swap runs only with ``--li2sao-boost-grpo-loss``.

    Boost (n > n_stable) turns DIS off and uses clip 0.2 and 0.28. The stable
    stage restores the LiSAO clip and DIS stashed at configure. ``use_dis`` and
    ``use_rollout_logprobs`` stay on so rollout logprobs are still collected.
    """
    apply_li2sao_value_clip(args, n_samples)
    if not boost_grpo_loss_enabled(args):
        return {}
    _stash_stable_loss(args)
    n_stable = int(getattr(args, "li2sao_n_stable", DEFAULT_N_STABLE))
    if int(n_samples) > n_stable:
        new_clip = float(getattr(args, "li2sao_boost_eps_clip", DEFAULT_BOOST_EPS_CLIP))
        new_high = float(getattr(args, "li2sao_boost_eps_clip_high", DEFAULT_BOOST_EPS_CLIP_HIGH))
        new_tis = False
        mode = LOSS_MODE_BOOST_GRPO
    else:
        new_clip = float(args.li2sao_stable_eps_clip)
        new_high = float(args.li2sao_stable_eps_clip_high)
        new_tis = bool(args.li2sao_stable_use_tis)
        mode = LOSS_MODE_STABLE_LISAO
    changed = (
        getattr(args, "li2sao_loss_mode", None) != mode
        or float(getattr(args, "eps_clip", new_clip)) != new_clip
        or float(getattr(args, "eps_clip_high", new_high) or new_high) != new_high
        or bool(getattr(args, "use_tis", False)) != new_tis
    )
    args.eps_clip = new_clip
    args.eps_clip_high = new_high
    args.use_tis = new_tis
    args.li2sao_loss_mode = mode
    if changed:
        logger.info(
            "Li2SAO loss mode=%s n=%s eps_clip=%s/%s use_tis=%s (use_dis=%s)",
            mode,
            int(n_samples),
            new_clip,
            new_high,
            new_tis,
            bool(getattr(args, "use_dis", False)),
        )
    return li2sao_loss_fields(args)


def infer_li2sao_group_n(args: Any, rollout_data: Any | None = None) -> int:
    """Prefer full-batch group size so leftover n=8 after a switch stays GRPO."""
    if rollout_data is not None:
        gids = _as_int_list(rollout_data.get("group_indices"))
        if gids:
            counts: dict[int, int] = defaultdict(int)
            for gid in gids:
                if gid is not None:
                    counts[int(gid)] += 1
            if counts:
                return max(counts.values())
    return int(getattr(args, "n_samples_per_prompt", DEFAULT_N_BOOST))


def apply_li2sao_loss_mode_from_batch(args: Any, rollout_data: Any | None = None) -> dict[str, Any]:
    """Actor-side: pick clip/TIS from the batch's real n, not the driver flag."""
    if not getattr(args, "use_li2sao", False) and not boost_grpo_loss_enabled(args):
        return {}
    return apply_li2sao_loss_mode(args, infer_li2sao_group_n(args, rollout_data))


def sync_li2sao_loss_fields(args: Any, actor_model: Any | None) -> None:
    if (not getattr(args, "use_li2sao", False) and not boost_grpo_loss_enabled(args)) or actor_model is None:
        return
    updater = getattr(actor_model, "update_args_fields", None)
    if updater is None:
        return
    updater(li2sao_loss_fields(args))


def gae_mix_alpha(ev: float | None, threshold: float) -> float:
    """α = clip(e/η, 0, 1). Missing / non-positive EV or η keeps pure GRPO."""
    if ev is None or threshold <= 0:
        return 0.0
    return float(min(1.0, max(0.0, float(ev) / float(threshold))))


def mix_ev_for_criterion(
    criterion: str,
    *,
    ev_seq: float | None,
    ev_token: float | None,
    ev_prefix: float | None,
    ev_suffix: float | None,
    ev_residual: float | None,
) -> float | None:
    """Batch EV that drives α; matches the n-boost switch metric, not critic_target."""
    if criterion == SWITCH_CRITERION_RESIDUAL:
        return ev_residual
    if criterion == SWITCH_CRITERION_TOKEN:
        if ev_token is None or ev_prefix is None or ev_suffix is None:
            return None
        return float(min(ev_token, ev_prefix, ev_suffix))
    return ev_seq


def mix_gae_grpo_advantage(
    gae: torch.Tensor,
    grpo: torch.Tensor,
    alpha: float,
    loss_mask: torch.Tensor | None,
) -> torch.Tensor:
    """Token-wise α A_GAE + (1-α) A_GRPO; observation tokens stay 0."""
    mixed = gae.to(dtype=grpo.dtype) * float(alpha) + grpo * (1.0 - float(alpha))
    if loss_mask is None or loss_mask.numel() != mixed.numel():
        return mixed
    keep = (loss_mask.reshape_as(mixed) != 0).to(dtype=mixed.dtype)
    return mixed * keep


def apply_token_switch_defaults(args: Any) -> None:
    """Later T_min / T_max when token gating is on and seq window defaults remain."""
    if float(getattr(args, "li2sao_ev_threshold", DEFAULT_EV_THRESHOLD)) == DEFAULT_EV_THRESHOLD:
        args.li2sao_ev_threshold = TOKEN_DEFAULT_EV_THRESHOLD
    if int(getattr(args, "li2sao_n_boost_min_steps", DEFAULT_BOOST_MIN_STEPS)) == DEFAULT_BOOST_MIN_STEPS:
        args.li2sao_n_boost_min_steps = TOKEN_DEFAULT_BOOST_MIN_STEPS
    if int(getattr(args, "li2sao_n_boost_max_steps", DEFAULT_BOOST_MAX_STEPS)) == DEFAULT_BOOST_MAX_STEPS:
        args.li2sao_n_boost_max_steps = TOKEN_DEFAULT_BOOST_MAX_STEPS


def configure_li2sao_args(args: Any) -> None:
    """Force boost-phase sampling and drop critic-only warmup.

    Called from ``slime_validate_args`` when ``--use-li2sao`` is set, before
    derived batch-size fields are computed. The estimator stays PPO so GAE
    returns remain the critic target; actor advantages are swapped in
    ``apply_li2sao`` while n>1.
    """
    if args.advantage_estimator != "ppo":
        raise ValueError("--use-li2sao requires --advantage-estimator ppo")
    n_boost = int(getattr(args, "li2sao_n_boost", DEFAULT_N_BOOST))
    n_stable = int(getattr(args, "li2sao_n_stable", DEFAULT_N_STABLE))
    traj = int(getattr(args, "li2sao_traj_per_step", DEFAULT_TRAJ_PER_STEP))
    if n_boost < 2:
        raise ValueError(f"--li2sao-n-boost must be >= 2, got {n_boost}")
    if n_stable < 1:
        raise ValueError(f"--li2sao-n-stable must be >= 1, got {n_stable}")
    if traj <= 0 or traj % n_boost != 0 or traj % n_stable != 0:
        raise ValueError(
            f"--li2sao-traj-per-step ({traj}) must be positive and divisible by "
            f"n_boost={n_boost} and n_stable={n_stable}"
        )
    criterion = resolve_switch_criterion(args)
    args.li2sao_switch_criterion = criterion
    from lisao.sao.critic_process import resolve_critic_target

    args.li2sao_critic_target = resolve_critic_target(args)
    if criterion == SWITCH_CRITERION_TOKEN:
        apply_token_switch_defaults(args)
    t_min = int(getattr(args, "li2sao_n_boost_min_steps", DEFAULT_BOOST_MIN_STEPS))
    t_max = int(getattr(args, "li2sao_n_boost_max_steps", DEFAULT_BOOST_MAX_STEPS))
    if t_min < 1 or t_max < t_min:
        raise ValueError(f"li2sao boost window invalid: min={t_min} max={t_max}")
    from lisao.inflight_async.switch import resolve_inflight_switch

    inflight_switch = resolve_inflight_switch(args)

    args.num_critic_only_steps = 0
    args.n_samples_per_prompt = n_boost
    args.rollout_batch_size = traj // n_boost
    if getattr(args, "global_batch_size", None) is None:
        args.global_batch_size = traj
    elif traj % int(args.global_batch_size) != 0:
        raise ValueError(
            f"--global-batch-size {args.global_batch_size} must divide "
            f"--li2sao-traj-per-step {traj}"
        )
    args.over_sampling_batch_size = args.rollout_batch_size
    args.grpo_std_normalization = False
    apply_li2sao_loss_mode(args, n_boost)
    logger.info(
        "Li2SAO enabled: start n=%d batch=%d traj=%d gbs=%s "
        "(stable n=%d inflight_switch=%s switch_criterion=%s ev_th=%s rank_th=%s "
        "gae_mix=%s boost_grpo_loss=%s min/max=%s/%s) "
        "actor=group relative rewards, critic_target=%s, num_critic_only_steps=0",
        n_boost,
        args.rollout_batch_size,
        traj,
        args.global_batch_size,
        n_stable,
        inflight_switch,
        criterion,
        getattr(args, "li2sao_ev_threshold", DEFAULT_EV_THRESHOLD),
        getattr(args, "li2sao_switch_rank_threshold", DEFAULT_SWITCH_RANK_THRESHOLD),
        int(gae_mix_enabled(args)),
        int(boost_grpo_loss_enabled(args)),
        t_min,
        t_max,
        args.li2sao_critic_target,
    )


def group_raw_means(
    group_ids: list[int | None],
    raw_rewards: list[float],
) -> tuple[dict[int, float], dict[int, float], dict[int, bool], float]:
    """Per-group mean and unbiased std of raw 0/1 rewards (slime GRPO).

    Groups whose members are all equal have std 0 (A_i=0 after norm).
    Singleton / missing ids are skip. ``mixed_frac`` is the fraction of
    multi-sample groups that are not all-equal (logged only).
    """
    if len(group_ids) != len(raw_rewards):
        raise ValueError(f"group_ids {len(group_ids)} != raw_rewards {len(raw_rewards)}")
    buckets: dict[int, list[float]] = defaultdict(list)
    for gid, reward in zip(group_ids, raw_rewards, strict=True):
        if gid is None:
            continue
        buckets[int(gid)].append(float(reward))

    means: dict[int, float] = {}
    stds: dict[int, float] = {}
    skip: dict[int, bool] = {}
    n_multi = 0
    n_mixed = 0
    for gid, vals in buckets.items():
        mean = sum(vals) / len(vals)
        means[gid] = mean
        identical = all(v == vals[0] for v in vals)
        if len(vals) <= 1:
            stds[gid] = 0.0
            skip[gid] = True
            continue
        stds[gid] = float(torch.tensor(vals, dtype=torch.float32).std(unbiased=True).item())
        n_multi += 1
        if identical:
            skip[gid] = True
        else:
            skip[gid] = False
            n_mixed += 1
    mixed_frac = (n_mixed / n_multi) if n_multi else 0.0
    return means, stds, skip, mixed_frac


def _masked_broadcast(template: torch.Tensor, scalar: float, loss_mask: torch.Tensor | None) -> torch.Tensor:
    """Fill ``template`` with ``scalar``; keep observation tokens at 0."""
    out = torch.ones_like(template) * scalar
    if loss_mask is None:
        return out
    mask = loss_mask.reshape_as(template) if loss_mask.numel() == template.numel() else None
    if mask is None:
        return out
    return out * (mask != 0).to(dtype=out.dtype)


def apply_group_relative_actor_advantages(
    advantages: list[torch.Tensor],
    group_ids: list[int | None],
    raw_rewards: list[float],
    *,
    partition: list[int] | None = None,
    loss_masks: list[torch.Tensor] | None = None,
) -> tuple[int, float, float]:
    """Replace actor advantages with GRPO A_i = (r_i − μ_g) / (σ_g + ε).

    ``raw_rewards`` / ``group_ids`` are the *full* batch. ``partition[i]`` maps
    local advantage i onto that full batch (DP split). Identical / singleton
    groups get 0. Observation tokens (loss_mask==0) stay 0. ``σ`` is unbiased
    (``torch.std``) and ``ε=1e-6``, matching slime GRPO. Returns
    ``(n_replaced, mixed_frac, mean_abs_scalar)``.
    """
    means, stds, _skip, mixed_frac = group_raw_means(group_ids, raw_rewards)
    n_local = len(advantages)
    if partition is None:
        if n_local != len(group_ids):
            raise ValueError("partition required when advantages are DP-sharded")
        partition = list(range(n_local))
    if len(partition) != n_local:
        raise ValueError(f"partition {len(partition)} != advantages {n_local}")

    n_replaced = 0
    abs_sum = 0.0
    for local_i, global_i in enumerate(partition):
        gid = group_ids[global_i]
        raw = raw_rewards[global_i]
        if gid is None:
            scalar = 0.0
        else:
            gid_i = int(gid)
            centered = float(raw) - float(means.get(gid_i, float(raw)))
            std = float(stds.get(gid_i, 0.0))
            scalar = centered / (std + GRPO_STD_EPS)
        mask = loss_masks[local_i] if loss_masks is not None and local_i < len(loss_masks) else None
        advantages[local_i] = _masked_broadcast(advantages[local_i], scalar, mask)
        n_replaced += 1
        abs_sum += abs(scalar)
    mean_abs = abs_sum / n_replaced if n_replaced else 0.0
    return n_replaced, mixed_frac, mean_abs


def center_advantages_by_group(
    advantages: list[torch.Tensor],
    group_ids: list[int | None],
    raw_rewards: list[float],
    *,
    partition: list[int] | None = None,
    loss_masks: list[torch.Tensor] | None = None,
) -> tuple[int, float]:
    """Backward-compatible alias of ``apply_group_relative_actor_advantages``."""
    n_replaced, mixed_frac, _mean_abs = apply_group_relative_actor_advantages(
        advantages,
        group_ids,
        raw_rewards,
        partition=partition,
        loss_masks=loss_masks,
    )
    return n_replaced, mixed_frac


def last_action_value(
    values: torch.Tensor,
    loss_mask: torch.Tensor | None,
) -> torch.Tensor:
    """Scalar V at the last action token (loss_mask==1), else the last token."""
    vec = values.reshape(-1)
    if vec.numel() == 0:
        return values.new_zeros(())
    if loss_mask is not None:
        mask = loss_mask.reshape(-1).to(device=vec.device)
        if mask.numel() == vec.numel():
            action = (mask != 0).nonzero(as_tuple=True)[0]
            if action.numel():
                return vec[action[-1]]
            return vec.sum() * 0
    return vec[-1]


def gather_response_pair(
    tensor: torch.Tensor,
    loss_mask: torch.Tensor | None,
    total_len: int,
    resp_len: int,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Response-length tensor (+ mask), all-gathering across CP ranks if needed."""
    v = tensor
    m = loss_mask
    if not (torch.distributed.is_available() and torch.distributed.is_initialized()):
        return v, m
    from megatron.core import mpu

    if mpu.get_context_parallel_world_size() > 1:
        from slime.backends.megatron_utils.cp_utils import all_gather_with_cp

        v = all_gather_with_cp(tensor, total_len, resp_len)[:resp_len]
        if loss_mask is not None and loss_mask.numel() != resp_len:
            m = all_gather_with_cp(loss_mask, total_len, resp_len)[:resp_len]
    return v, m


def gather_response_value(
    values: torch.Tensor,
    loss_mask: torch.Tensor | None,
    total_len: int,
    resp_len: int,
) -> torch.Tensor:
    """Last-action V, all-gathering across context-parallel ranks if needed."""
    v, m = gather_response_pair(values, loss_mask, total_len, resp_len)
    return last_action_value(v, m)


def sequence_explained_variance(rewards: torch.Tensor, values: torch.Tensor) -> torch.Tensor | None:
    """EV = 1 − Var(r − V) / Var(r) over trajectories. None if Var(r)==0."""
    if rewards.numel() == 0 or rewards.numel() != values.numel():
        return None
    r = rewards.reshape(-1).float()
    v = values.reshape(-1).float().to(device=r.device)
    residual = r - v
    var_r = r.var(unbiased=False)
    if var_r <= 0:
        return None
    return 1.0 - residual.var(unbiased=False) / var_r


def distributed_sequence_explained_variance(
    rewards: torch.Tensor,
    values: torch.Tensor,
) -> float | None:
    """Sequence EV reduced across the data-parallel group (population var)."""
    r = rewards.reshape(-1).float()
    v = values.reshape(-1).float().to(device=r.device)
    if r.numel() != v.numel():
        return None
    n = r.new_tensor(float(r.numel())) if r.numel() else r.new_zeros(())
    residual = r - v
    stats = torch.stack(
        [
            r.sum() if r.numel() else r.new_zeros(()),
            (r * r).sum() if r.numel() else r.new_zeros(()),
            residual.sum() if r.numel() else r.new_zeros(()),
            (residual * residual).sum() if r.numel() else r.new_zeros(()),
            n,
        ]
    )
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        try:
            import torch.distributed as dist
            from megatron.core import mpu

            dist.all_reduce(stats, group=mpu.get_data_parallel_group())
        except Exception:
            pass
    total_n = float(stats[4].item())
    if total_n <= 1:
        return None
    mean_r = float(stats[0].item()) / total_n
    var_r = float(stats[1].item()) / total_n - mean_r * mean_r
    if var_r <= 1e-12:
        return None
    mean_res = float(stats[2].item()) / total_n
    var_res = float(stats[3].item()) / total_n - mean_res * mean_res
    return 1.0 - var_res / var_r


def mixed_group_switch_metrics(
    last_values: list[torch.Tensor],
    group_ids: list[int | None],
    raw_rewards: list[float],
    partition: list[int] | None,
    valid: list[bool] | None = None,
) -> tuple[float | None, float | None, int]:
    """Residual EV and ranking accuracy over globally collected terminal values."""
    from lisao.sao.ranking import data_parallel_group, gather_indexed_values, outcome_pairs

    if not last_values or len(group_ids) != len(raw_rewards):
        return None, None, 0
    if partition is None:
        if len(last_values) != len(group_ids):
            return None, None, 0
        partition = list(range(len(last_values)))
    values, present = gather_indexed_values(
        torch.stack([v.detach().float().reshape(()) for v in last_values]),
        partition,
        len(group_ids),
        valid=valid,
        process_group=data_parallel_group(),
    )
    means, _stds, skip, _mixed = group_raw_means(group_ids, raw_rewards)
    mixed = torch.tensor(
        [gid is not None and not skip.get(int(gid), True) for gid in group_ids],
        dtype=torch.bool,
        device=values.device,
    ) & present
    targets = values.new_tensor([
        raw - means.get(int(gid), raw) if gid is not None else 0.0
        for gid, raw in zip(group_ids, raw_rewards, strict=True)
    ])
    ev = sequence_explained_variance(targets[mixed], values[mixed])
    pos, neg = outcome_pairs(values, group_ids, raw_rewards, present)
    rank_acc = float((values[pos] > values[neg]).float().mean().item()) if pos.numel() else None
    return None if ev is None else float(ev.item()), rank_acc, pos.numel()


def _as_float_list(values: Any) -> list[float] | None:
    if values is None:
        return None
    if torch.is_tensor(values):
        return [float(x) for x in values.detach().cpu().reshape(-1).tolist()]
    out: list[float] = []
    for item in values:
        if torch.is_tensor(item):
            out.append(float(item.detach().cpu().reshape(-1)[0].item()))
        else:
            out.append(float(item))
    return out


def _as_int_list(values: Any) -> list[int | None] | None:
    if values is None:
        return None
    if torch.is_tensor(values):
        return [int(x) for x in values.detach().cpu().reshape(-1).tolist()]
    out: list[int | None] = []
    for item in values:
        if item is None:
            out.append(None)
        elif torch.is_tensor(item):
            out.append(int(item.detach().cpu().reshape(-1)[0].item()))
        else:
            out.append(int(item))
    return out


def apply_li2sao(args: Any, rollout_data: dict[str, Any]) -> dict[str, Any] | None:
    """Post-GAE hook: GRPO actor A while n>1; log sequence EV. Returns stay GAE.

    ``--li2sao-gae-mix`` optionally blends GAE back in after the GRPO replace.
    The n=1 switch is unchanged: no groups means no GRPO term.
    """
    if not getattr(args, "use_li2sao", False):
        return None
    advantages = rollout_data.get("advantages")
    values = rollout_data.get("values")
    if not advantages or not values:
        return None

    group_ids = _as_int_list(rollout_data.get("group_indices"))
    raw_rewards = _as_float_list(rollout_data.get("raw_reward"))
    partition = rollout_data.get("partition")
    if partition is not None:
        partition = [int(i) for i in partition]
    loss_masks = rollout_data.get("loss_masks")

    mixed_frac = 0.0
    n_grpo = 0
    n_batch = 1
    mean_abs_adv = 0.0
    actor_mode = "gae"
    mix_alpha = 0.0
    gae_saved: list[torch.Tensor] | None = None
    if group_ids is not None and raw_rewards is not None:
        counts: dict[int, int] = defaultdict(int)
        for gid in group_ids:
            if gid is not None:
                counts[int(gid)] += 1
        if counts:
            n_batch = max(counts.values())
        if n_batch > 1 and (partition is not None or len(advantages) == len(group_ids)):
            if gae_mix_enabled(args):
                gae_saved = [adv.clone() for adv in advantages]
            n_grpo, mixed_frac, mean_abs_adv = apply_group_relative_actor_advantages(
                advantages,
                group_ids,
                raw_rewards,
                partition=partition,
                loss_masks=loss_masks,
            )
            rollout_data["advantages"] = advantages
            actor_mode = "grpo"
        else:
            _, _, _, mixed_frac = group_raw_means(group_ids, raw_rewards)
            if n_batch > 1:
                logger.warning(
                    "li2sao: n_batch=%s but actor A was not replaced "
                    "(partition missing and local advantages %s != full group_ids %s); "
                    "actor keeps GAE",
                    n_batch,
                    len(advantages),
                    len(group_ids),
                )

    local_v: list[torch.Tensor] = []
    local_r: list[float] = []
    full_r: list[torch.Tensor] = []
    full_v: list[torch.Tensor] = []
    prefix_r: list[torch.Tensor] = []
    prefix_v: list[torch.Tensor] = []
    suffix_r: list[torch.Tensor] = []
    suffix_v: list[torch.Tensor] = []
    total_lengths = rollout_data.get("total_lengths")
    response_lengths = rollout_data.get("response_lengths")
    returns = rollout_data.get("returns")
    for i, value in enumerate(values):
        mask = loss_masks[i] if loss_masks is not None and i < len(loss_masks) else None
        total_len = int(total_lengths[i]) if total_lengths is not None else int(value.numel())
        resp_len = int(response_lengths[i]) if response_lengths is not None else int(value.numel())
        gathered_v, gathered_m = gather_response_pair(value.detach(), mask, total_len, resp_len)
        local_v.append(last_action_value(gathered_v, gathered_m))
        if raw_rewards is not None:
            gi = partition[i] if partition is not None else i
            if 0 <= gi < len(raw_rewards):
                local_r.append(raw_rewards[gi])
        if returns is None or i >= len(returns):
            continue
        gathered_ret, _ = gather_response_pair(returns[i].detach(), mask, total_len, resp_len)
        rr, vv = masked_return_value_pairs(gathered_ret, gathered_v, gathered_m)
        if rr.numel() == 0:
            continue
        full_r.append(rr)
        full_v.append(vv)
        (pre_r, pre_v), (suf_r, suf_v) = split_pairs_in_half(rr, vv)
        if pre_r.numel():
            prefix_r.append(pre_r)
            prefix_v.append(pre_v)
        if suf_r.numel():
            suffix_r.append(suf_r)
            suffix_v.append(suf_v)
    ev_seq = None
    if local_v and len(local_r) == len(local_v):
        device = local_v[0].device
        ev_seq = distributed_sequence_explained_variance(
            torch.tensor(local_r, device=device, dtype=torch.float32),
            torch.stack([v.float().reshape(()) for v in local_v]),
        )
    ev_token = ev_prefix = ev_suffix = None
    if full_r:
        ev_token = distributed_sequence_explained_variance(torch.cat(full_r), torch.cat(full_v))
        ev_prefix = (
            distributed_sequence_explained_variance(torch.cat(prefix_r), torch.cat(prefix_v)) if prefix_r else None
        )
        ev_suffix = (
            distributed_sequence_explained_variance(torch.cat(suffix_r), torch.cat(suffix_v)) if suffix_r else None
        )
    ev_residual = rank_acc = None
    n_rank_pairs = 0
    if group_ids is not None and raw_rewards is not None and local_v:
        ev_residual, rank_acc, n_rank_pairs = mixed_group_switch_metrics(
            local_v,
            group_ids,
            raw_rewards,
            partition,
            valid=[bool(mask.any()) for mask in loss_masks] if loss_masks is not None else None,
        )

    if gae_saved is not None and actor_mode == "grpo":
        criterion = resolve_switch_criterion(args)
        mix_ev = mix_ev_for_criterion(
            criterion,
            ev_seq=ev_seq,
            ev_token=ev_token,
            ev_prefix=ev_prefix,
            ev_suffix=ev_suffix,
            ev_residual=ev_residual,
        )
        mix_alpha = gae_mix_alpha(
            mix_ev,
            float(getattr(args, "li2sao_ev_threshold", DEFAULT_EV_THRESHOLD)),
        )
        if mix_alpha > 0.0:
            abs_sum = 0.0
            for i, grpo_adv in enumerate(advantages):
                mask = loss_masks[i] if loss_masks is not None and i < len(loss_masks) else None
                mixed = mix_gae_grpo_advantage(gae_saved[i], grpo_adv, mix_alpha, mask)
                advantages[i] = mixed
                abs_sum += float(mixed.detach().abs().mean().item())
            rollout_data["advantages"] = advantages
            mean_abs_adv = abs_sum / max(len(advantages), 1)
            actor_mode = "mix"

    metrics = {
        "ev_seq": ev_seq,
        "ev_token": ev_token,
        "ev_prefix": ev_prefix,
        "ev_suffix": ev_suffix,
        "ev_residual": ev_residual,
        "rank_acc": rank_acc,
        "n_rank_pairs": int(n_rank_pairs),
        "group_mixed_frac": mixed_frac,
        "n_batch": int(n_batch),
        "n_grpo": int(n_grpo),
        "n_centered": int(n_grpo),
        "mean_abs_adv": float(mean_abs_adv),
        "actor_mode": actor_mode,
        "gae_mix_alpha": float(mix_alpha),
    }
    args._lisao_li2sao_metrics = metrics
    ev = metrics.get("ev_seq")
    logger.info(
        "li2sao: n_batch=%s mode=%s mix_a=%s ev_seq=%s ev_res=%s rank_acc=%s ev_token=%s "
        "ev_pre=%s ev_suf=%s mixed_frac=%.3f n_grpo=%s |A|=%.4f",
        metrics.get("n_batch"),
        actor_mode,
        f"{mix_alpha:.4f}",
        f"{ev:.4f}" if ev is not None else "na",
        f"{ev_residual:.4f}" if ev_residual is not None else "na",
        f"{rank_acc:.4f}" if rank_acc is not None else "na",
        f"{ev_token:.4f}" if ev_token is not None else "na",
        f"{ev_prefix:.4f}" if ev_prefix is not None else "na",
        f"{ev_suffix:.4f}" if ev_suffix is not None else "na",
        float(metrics.get("group_mixed_frac") or 0.0),
        metrics.get("n_grpo"),
        float(mean_abs_adv),
    )
    return metrics


def log_li2sao_train_metrics(args: Any, rollout_id: int) -> None:
    """Align Li2SAO scalars with the real training step (call from train_*)."""
    metrics = getattr(args, "_lisao_li2sao_metrics", None)
    if not metrics:
        return
    try:
        from slime.utils import logging_utils
        from slime.utils.metric_utils import compute_rollout_step

        ev = metrics.get("ev_seq")
        ev_token = metrics.get("ev_token")
        ev_prefix = metrics.get("ev_prefix")
        ev_suffix = metrics.get("ev_suffix")
        payload = {
            "train/li2sao/ev_seq": float(ev) if ev is not None else 0.0,
            "train/li2sao/ev_seq_valid": 0.0 if ev is None else 1.0,
            "train/li2sao/ev_token": float(ev_token) if ev_token is not None else 0.0,
            "train/li2sao/ev_token_valid": 0.0 if ev_token is None else 1.0,
            "train/li2sao/ev_prefix": float(ev_prefix) if ev_prefix is not None else 0.0,
            "train/li2sao/ev_suffix": float(ev_suffix) if ev_suffix is not None else 0.0,
            "train/li2sao/ev_residual": float(metrics["ev_residual"])
            if metrics.get("ev_residual") is not None
            else 0.0,
            "train/li2sao/ev_residual_valid": 0.0 if metrics.get("ev_residual") is None else 1.0,
            "train/li2sao/rank_acc": float(metrics["rank_acc"]) if metrics.get("rank_acc") is not None else 0.0,
            "train/li2sao/rank_acc_valid": 0.0 if metrics.get("rank_acc") is None else 1.0,
            "train/li2sao/n_rank_pairs": float(metrics.get("n_rank_pairs") or 0),
            "train/li2sao/group_mixed_frac": float(metrics.get("group_mixed_frac") or 0.0),
            "train/li2sao/n_batch": float(metrics.get("n_batch") or 1),
            "train/li2sao/n_grpo": float(metrics.get("n_grpo") or 0),
            "train/li2sao/mean_abs_adv": float(metrics.get("mean_abs_adv") or 0.0),
            "train/li2sao/gae_mix_alpha": float(metrics.get("gae_mix_alpha") or 0.0),
            "train/li2sao/actor_grpo": 1.0 if metrics.get("actor_mode") in {"grpo", "mix"} else 0.0,
            "train/step": compute_rollout_step(args, rollout_id),
        }
        critic_m = getattr(args, "_lisao_critic_process_metrics", None) or {}
        if critic_m:
            payload["train/li2sao/critic_mixed_kept"] = float(critic_m.get("n_mixed_kept") or 0)
            payload["train/li2sao/critic_collapsed_drop"] = float(critic_m.get("n_collapsed_drop") or 0)
            payload["train/li2sao/critic_last_token_only"] = float(critic_m.get("n_last_token_only") or 0)
            payload["train/li2sao/critic_mean_div_frac"] = float(critic_m.get("mean_div_frac") or 0.0)
            payload["train/li2sao/critic_rank_pairs"] = float(critic_m.get("n_rank_pairs") or 0)
            payload["train/li2sao/critic_noop"] = float(critic_m.get("noop") or 0)
        logging_utils.log(args, payload, step_key="train/step")
    except Exception:
        logger.debug("li2sao wandb log skipped", exc_info=True)


def collect_li2sao_metrics(train_results: Any) -> dict[str, Any] | None:
    """Pick ``li2sao`` metrics out of ``ray.get(async_train(...))``."""
    if train_results is None:
        return None
    if isinstance(train_results, dict):
        train_results = [train_results]
    for item in train_results:
        if isinstance(item, dict) and item.get("li2sao"):
            return item["li2sao"]
    return None


@dataclass
class Li2SAOController:
    """One-way n_boost → n_stable switch when the configured critic EV is ready."""

    n_boost: int = DEFAULT_N_BOOST
    n_stable: int = DEFAULT_N_STABLE
    traj_per_step: int = DEFAULT_TRAJ_PER_STEP
    ev_ema_beta: float = DEFAULT_EV_EMA_BETA
    ev_threshold: float = DEFAULT_EV_THRESHOLD
    ev_consecutive: int = DEFAULT_EV_CONSECUTIVE
    boost_min_steps: int = DEFAULT_BOOST_MIN_STEPS
    boost_max_steps: int = DEFAULT_BOOST_MAX_STEPS
    switch_criterion: str = DEFAULT_SWITCH_CRITERION
    rank_threshold: float = DEFAULT_SWITCH_RANK_THRESHOLD
    mixed_frac_floor: float = 0.0
    n: int = field(init=False)
    steps_in_boost: int = 0
    ev_ema: float | None = None
    ev_prefix_ema: float | None = None
    ev_suffix_ema: float | None = None
    rank_ema: float | None = None
    mixed_ema: float | None = None
    consecutive_ready: int = 0
    switched: bool = False
    last_reason: str | None = None

    def __post_init__(self) -> None:
        self.n = self.n_boost
        criterion = str(self.switch_criterion).strip().lower()
        if criterion not in SWITCH_CRITERION_CHOICES:
            raise ValueError(
                f"switch_criterion must be one of {SWITCH_CRITERION_CHOICES}, got {self.switch_criterion!r}"
            )
        self.switch_criterion = criterion

    @classmethod
    def from_args(cls, args: Any) -> Li2SAOController:
        return cls(
            n_boost=int(getattr(args, "li2sao_n_boost", DEFAULT_N_BOOST)),
            n_stable=int(getattr(args, "li2sao_n_stable", DEFAULT_N_STABLE)),
            traj_per_step=int(getattr(args, "li2sao_traj_per_step", DEFAULT_TRAJ_PER_STEP)),
            ev_ema_beta=float(getattr(args, "li2sao_ev_ema_beta", DEFAULT_EV_EMA_BETA)),
            ev_threshold=float(getattr(args, "li2sao_ev_threshold", DEFAULT_EV_THRESHOLD)),
            ev_consecutive=int(getattr(args, "li2sao_ev_consecutive", DEFAULT_EV_CONSECUTIVE)),
            boost_min_steps=int(getattr(args, "li2sao_n_boost_min_steps", DEFAULT_BOOST_MIN_STEPS)),
            boost_max_steps=int(getattr(args, "li2sao_n_boost_max_steps", DEFAULT_BOOST_MAX_STEPS)),
            switch_criterion=resolve_switch_criterion(args),
            rank_threshold=float(
                getattr(args, "li2sao_switch_rank_threshold", DEFAULT_SWITCH_RANK_THRESHOLD) or 0.0
            ),
        )

    @property
    def batch_size(self) -> int:
        return self.traj_per_step // self.n

    def _update_ema(self, current: float | None, value: Any) -> tuple[float | None, bool]:
        return self._update_ema_threshold(current, value, self.ev_threshold)

    def _update_ema_threshold(
        self, current: float | None, value: Any, threshold: float
    ) -> tuple[float | None, bool]:
        if value is None or not math.isfinite(float(value)):
            return current, False
        ev = float(value)
        updated = ev if current is None else self.ev_ema_beta * current + (1.0 - self.ev_ema_beta) * ev
        return updated, updated > threshold

    def _observe_seq(self, metrics: dict[str, Any] | None) -> None:
        ev_seq = None if metrics is None else metrics.get("ev_seq")
        if ev_seq is None:
            self.consecutive_ready = 0
            return
        self.ev_ema, ready = self._update_ema(self.ev_ema, ev_seq)
        if ready:
            self.consecutive_ready += 1
        else:
            self.consecutive_ready = 0

    def _observe_token(self, metrics: dict[str, Any] | None) -> None:
        ev_token = None if metrics is None else metrics.get("ev_token")
        ev_prefix = None if metrics is None else metrics.get("ev_prefix")
        ev_suffix = None if metrics is None else metrics.get("ev_suffix")
        self.ev_ema, ok_token = self._update_ema(self.ev_ema, ev_token)
        self.ev_prefix_ema, ok_prefix = self._update_ema(self.ev_prefix_ema, ev_prefix)
        self.ev_suffix_ema, ok_suffix = self._update_ema(self.ev_suffix_ema, ev_suffix)
        if ok_token and ok_prefix and ok_suffix:
            self.consecutive_ready += 1
        else:
            self.consecutive_ready = 0

    def _observe_residual(self, metrics: dict[str, Any] | None) -> None:
        ev_res = None if metrics is None else metrics.get("ev_residual")
        if ev_res is None:
            self.consecutive_ready = 0
            return
        self.ev_ema, ok_ev = self._update_ema(self.ev_ema, ev_res)
        ok_rank = True
        if self.rank_threshold > 0:
            rank = None if metrics is None else metrics.get("rank_acc")
            if rank is None:
                ok_rank = False
            else:
                self.rank_ema, ok_rank = self._update_ema_threshold(
                    self.rank_ema, rank, self.rank_threshold
                )
        if ok_ev and ok_rank:
            self.consecutive_ready += 1
        else:
            self.consecutive_ready = 0

    def observe(self, metrics: dict[str, Any] | None) -> tuple[int, int, str] | None:
        """Update EMA. Return ``(n, batch, reason)`` once when switching.

        ``seq`` (default): last-token V vs raw r. ``token``: token-level EV(R,V)
        and both action-token halves. ``residual``: mixed-group last-action
        EV of V vs r-μ_g(raw); optional rank accuracy if ``rank_threshold``>0.
        ``mixed_frac`` is logged only.
        """
        if self.switched:
            return None
        self.steps_in_boost += 1
        mixed_frac = None if metrics is None else metrics.get("group_mixed_frac")
        if self.switch_criterion == SWITCH_CRITERION_TOKEN:
            self._observe_token(metrics)
        elif self.switch_criterion == SWITCH_CRITERION_RESIDUAL:
            self._observe_residual(metrics)
        else:
            self._observe_seq(metrics)

        if self.steps_in_boost < self.boost_min_steps:
            self.consecutive_ready = 0

        if mixed_frac is not None:
            mf = float(mixed_frac)
            self.mixed_ema = (
                mf if self.mixed_ema is None else self.ev_ema_beta * self.mixed_ema + (1.0 - self.ev_ema_beta) * mf
            )

        reason = None
        if self.steps_in_boost >= self.boost_min_steps and self.consecutive_ready >= self.ev_consecutive:
            reason = "ev_ready"
        elif self.steps_in_boost >= self.boost_max_steps:
            reason = "timeout"

        if reason is None:
            return None
        self.switched = True
        self.n = self.n_stable
        self.last_reason = reason
        logger.info(
            "Li2SAO switch n=%d→%d reason=%s criterion=%s steps=%d ev_ema=%s "
            "ev_prefix_ema=%s ev_suffix_ema=%s rank_ema=%s mixed_ema=%s consecutive=%d",
            self.n_boost,
            self.n_stable,
            reason,
            self.switch_criterion,
            self.steps_in_boost,
            f"{self.ev_ema:.4f}" if self.ev_ema is not None else "na",
            f"{self.ev_prefix_ema:.4f}" if self.ev_prefix_ema is not None else "na",
            f"{self.ev_suffix_ema:.4f}" if self.ev_suffix_ema is not None else "na",
            f"{self.rank_ema:.4f}" if self.rank_ema is not None else "na",
            f"{self.mixed_ema:.4f}" if self.mixed_ema is not None else "na",
            self.consecutive_ready,
        )
        return self.n, self.batch_size, reason


def log_li2sao_controller(args: Any, rollout_id: int, controller: Li2SAOController, reason: str | None) -> None:
    try:
        from slime.utils import logging_utils
        from slime.utils.metric_utils import compute_rollout_step

        payload = {
            "train/li2sao/n": float(controller.n),
            "train/li2sao/ev_ema": 0.0 if controller.ev_ema is None else float(controller.ev_ema),
            "train/li2sao/ev_prefix_ema": 0.0 if controller.ev_prefix_ema is None else float(controller.ev_prefix_ema),
            "train/li2sao/ev_suffix_ema": 0.0 if controller.ev_suffix_ema is None else float(controller.ev_suffix_ema),
            "train/li2sao/rank_ema": 0.0 if controller.rank_ema is None else float(controller.rank_ema),
            "train/li2sao/mixed_ema": 0.0 if controller.mixed_ema is None else float(controller.mixed_ema),
            "train/li2sao/steps_in_boost": float(controller.steps_in_boost),
            "train/li2sao/consecutive_ready": float(controller.consecutive_ready),
            "train/li2sao/switched": 1.0 if controller.switched else 0.0,
            "train/li2sao/boost_grpo_loss": 1.0 if boost_grpo_loss_enabled(args) else 0.0,
            "train/li2sao/use_tis": 1.0 if getattr(args, "use_tis", False) else 0.0,
            "train/li2sao/eps_clip": float(getattr(args, "eps_clip", 0.0) or 0.0),
            "train/li2sao/eps_clip_high": float(getattr(args, "eps_clip_high", 0.0) or 0.0),
            "train/li2sao/switch_criterion_token": 1.0
            if controller.switch_criterion == SWITCH_CRITERION_TOKEN
            else 0.0,
            "train/li2sao/switch_criterion_residual": 1.0
            if controller.switch_criterion == SWITCH_CRITERION_RESIDUAL
            else 0.0,
            "train/step": compute_rollout_step(args, rollout_id),
        }
        try:
            from lisao.inflight_async.switch import resolve_inflight_switch

            payload["train/li2sao/inflight_switch_flatten"] = (
                1.0 if resolve_inflight_switch(args) == "flatten" else 0.0
            )
        except Exception:
            payload["train/li2sao/inflight_switch_flatten"] = 0.0
        if reason:
            payload["train/li2sao/switch_reason_" + reason] = 1.0
        logging_utils.log(args, payload, step_key="train/step")
    except Exception:
        logger.debug("li2sao controller wandb log skipped", exc_info=True)


def apply_sampling_to_args(args: Any, n_samples: int, rollout_batch_size: int) -> None:
    args.n_samples_per_prompt = int(n_samples)
    args.rollout_batch_size = int(rollout_batch_size)
    args.over_sampling_batch_size = int(rollout_batch_size)
    apply_li2sao_loss_mode(args, n_samples)
