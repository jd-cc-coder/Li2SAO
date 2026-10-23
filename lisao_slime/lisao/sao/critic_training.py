"""Step-wide normalizers and exact terminal-ranking gradients."""

from __future__ import annotations

import torch
import torch.distributed as dist

from .li2sao import gather_response_value, infer_li2sao_group_n
from .ranking import data_parallel_group, gather_indexed_values, ranking_contributions


CACHE_KEYS = (
    "critic_residual_scale",
    "critic_rank_gradient",
    "critic_rank_loss",
    "critic_rank_pairs",
    "critic_step_size",
)


def step_view(data_iterator, num_microbatches):
    source = data_iterator[0].rollout_data
    schedules = [it.micro_batch_indices[it.offset : it.offset + num_microbatches] for it in data_iterator]
    if any(schedule != schedules[0] for schedule in schedules[1:]):
        raise ValueError("Critic pipeline iterators must use the same step schedule")
    if len(schedules[0]) != num_microbatches:
        raise ValueError("Critic step exceeds its microbatch schedule")
    indices = [i for mb in schedules[0] for i in mb]
    if not indices or len(set(indices)) != len(indices):
        raise ValueError("Critic step requires distinct trajectory indices")
    fields = ("tokens", "loss_masks", "multimodal_train_inputs", "total_lengths", "response_lengths")
    view = {key: [source[key][i] for i in indices] for key in fields if source.get(key) is not None}
    remap = {index: i for i, index in enumerate(indices)}
    schedule = [[remap[i] for i in mb] for mb in schedules[0]]
    iterators = [type(it)(view, schedule) for it in data_iterator]
    return source, indices, iterators


@torch.no_grad()
def prepare_critic_step(args, model, data_iterator, num_microbatches, step_global_batch_size):
    """Refresh dL_rank/dV at current weights without retaining microbatch graphs."""
    if args.loss_type != "value_loss" or getattr(args, "li2sao_critic_target", "absolute") == "absolute":
        return
    source = data_iterator[0].rollout_data
    if infer_li2sao_group_n(args, source) <= 1:
        return
    if getattr(args, "calculate_per_token_loss", False):
        raise ValueError("Li2SAO residual training requires per-step loss normalization")

    from megatron.core import mpu

    source, indices, iterators = step_view(data_iterator, num_microbatches)
    use_rank = args.li2sao_critic_target == "residual_rank"
    current_values = None
    if use_rank:
        if any(float(getattr(args, key, 0.0) or 0.0) != 0 for key in ("hidden_dropout", "attention_dropout")):
            raise ValueError("Exact Li2SAO ranking requires zero dropout")
        if any(
            isinstance(module, torch.nn.modules.dropout._DropoutNd) and module.p
            for chunk in model
            for module in chunk.modules()
        ):
            raise ValueError("Exact Li2SAO ranking requires zero dropout in all critic modules")
        from slime.backends.megatron_utils.loss import get_values
        from slime.backends.megatron_utils.model import forward_only

        current_values = forward_only(get_values, args, model, iterators, [num_microbatches]).get("values")
    if not mpu.is_pipeline_last_stage(ignore_virtual=True):
        return
    if source.get("critic_keep") is None:
        raise ValueError("Li2SAO residual training is missing critic targets and keep masks")

    kept = torch.stack([source["critic_keep"][i].sum().float() for i in indices]).sum()
    group = data_parallel_group()
    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(kept, group=group)
    # The training loss wrapper divides accumulated contributions by the step size.
    residual_scale = step_global_batch_size / kept.clamp_min(1)
    partition = source.get("partition")
    global_indices = [partition[i] if partition is not None else i for i in indices]
    if use_rank:
        if current_values is None or len(current_values) != len(indices):
            raise ValueError("Critic ranking prepass did not return all step trajectories")
        terminal_values = torch.stack(
            [
                gather_response_value(
                    value, source["loss_masks"][i], source["total_lengths"][i], source["response_lengths"][i]
                )
                for i, value in zip(indices, current_values, strict=True)
            ]
        )
        terminal_values, present = gather_indexed_values(
            terminal_values,
            global_indices,
            len(source["group_indices"]),
            valid=[bool(source["loss_masks"][i].any()) for i in indices],
            process_group=group,
        )
        gradient, rank_loss, pairs = ranking_contributions(
            terminal_values,
            source["group_indices"],
            source["raw_reward"],
            margin=float(getattr(args, "li2sao_critic_rank_margin", 0.1)),
            valid=present,
        )
    for key in CACHE_KEYS:
        source.setdefault(key, [None] * len(source["tokens"]))
    for i, global_i in zip(indices, global_indices, strict=True):
        source["critic_residual_scale"][i] = residual_scale
        source["critic_step_size"][i] = step_global_batch_size
        source["critic_rank_gradient"][i] = gradient[global_i] if use_rank else kept.new_zeros(())
        source["critic_rank_loss"][i] = rank_loss[global_i] if use_rank else kept.new_zeros(())
        source["critic_rank_pairs"][i] = pairs[global_i] if use_rank else kept.new_zeros(())
