"""MLP-only LoRA on a **separate** critic replica for ``--lora_light_mode``.

Actor and critic are two Megatron copies in the same train process (no second
Ray actor, no train offload). That placement is shared with ``--fast_mode``;
this module only injects LoRA when ``lora_light_mode`` is on. Critic freezes
the backbone — including MLP base weights — and trains LoRA on ``fc1``/``fc2``
plus the scalar value head (``output_layer``). Adapters are not synced to SGLang.
"""

from __future__ import annotations

import logging
import math
import re
from argparse import Namespace
from collections.abc import Iterable, Iterator, Sequence
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger("lisao.sao.lora")

LORA_A_NAME = "lora_A"
LORA_B_NAME = "lora_B"
DEFAULT_LORA_TARGET_MODULES = ("fc1", "fc2")

_LAYER_INDEX_RE = re.compile(r"\.layers\.(\d+)\.")
_TARGET_ALIASES = {
    "qkv": ("linear_qkv", "query_key_value"),
    "proj": ("linear_proj", "linear_dense", "dense"),
    "fc1": ("linear_fc1", "dense_h_to_4h"),
    "fc2": ("linear_fc2", "dense_4h_to_h"),
}

_lora_enabled = False


def is_lora_enabled() -> bool:
    return _lora_enabled


def enable_lora() -> None:
    global _lora_enabled
    _lora_enabled = True


def disable_lora() -> None:
    global _lora_enabled
    _lora_enabled = False


def is_lora_param(name: str) -> bool:
    return LORA_A_NAME in name or LORA_B_NAME in name


def is_critic_trainable_param(name: str) -> bool:
    """LoRA adapters + critic value head (Megatron ``output_layer``)."""
    return is_lora_param(name) or "output_layer" in name


def is_lora_or_critic_param(name: str) -> bool:
    return is_critic_trainable_param(name)


def is_policy_sync_param(name: str) -> bool:
    """Params that should be broadcast to SGLang (policy backbone + LM head)."""
    return not is_lora_param(name)


def parse_lora_target_modules(spec: str | Sequence[str] | None) -> tuple[str, ...]:
    if spec is None:
        return DEFAULT_LORA_TARGET_MODULES
    if isinstance(spec, str):
        parts = [p.strip() for p in spec.split(",") if p.strip()]
    else:
        parts = [str(p).strip() for p in spec if str(p).strip()]
    return tuple(parts) or DEFAULT_LORA_TARGET_MODULES


def _unwrap_module(module: Any) -> Any:
    while hasattr(module, "module"):
        module = module.module
    return module


def iter_model_chunks(model: Any) -> Iterator[Any]:
    if isinstance(model, (list, tuple)):
        for chunk in model:
            yield _unwrap_module(chunk)
        return
    yield _unwrap_module(model)


def _module_layer_index(name: str) -> int | None:
    match = _LAYER_INDEX_RE.search(name)
    if match is None:
        return None
    return int(match.group(1))


def _name_matches_targets(name: str, targets: Sequence[str]) -> bool:
    leaf = name.rsplit(".", 1)[-1]
    for target in targets:
        aliases = _TARGET_ALIASES.get(target, (target,))
        if any(alias == leaf or alias in name for alias in aliases):
            return True
    return False


def _in_last_n_layers(name: str, lora_layers: int, num_layers: int | None) -> bool:
    if lora_layers < 0:
        return True
    index = _module_layer_index(name)
    if index is None:
        return False
    if num_layers is None:
        return True
    return index >= max(0, num_layers - lora_layers)


def _is_injectable_linear(module: nn.Module) -> bool:
    weight = getattr(module, "weight", None)
    return isinstance(weight, torch.Tensor) and weight.ndim == 2 and not hasattr(module, LORA_A_NAME)


def _is_row_parallel(module: nn.Module) -> bool:
    if bool(getattr(module, "input_is_parallel", False)):
        return True
    parallel_mode = getattr(module, "parallel_mode", None)
    if parallel_mode is None:
        parallel_mode = getattr(getattr(module, "weight", None), "parallel_mode", None)
    return parallel_mode == "row"


def _uses_sequence_parallel(module: nn.Module) -> bool:
    if bool(getattr(module, "sequence_parallel", False)):
        return True
    config = getattr(module, "config", None)
    return bool(getattr(config, "sequence_parallel", False)) if config is not None else False


def _tp_group(module: nn.Module) -> Any:
    return getattr(module, "tp_group", None) or getattr(module, "_tp_group", None)


def _prepare_lora_hidden(module: nn.Module, hidden: torch.Tensor) -> torch.Tensor:
    """Gather sequence-parallel activations for column-parallel linears.

    Megatron/TE ColumnParallelLinear all-gathers the SP-sharded sequence inside
    the base GEMM, so the returned activation is ``[seq, out/tp]``. LoRA must
    gather the same input or ``result + delta`` fails (e.g. 8192 vs 4096 at TP=2).
    """
    if _is_row_parallel(module) or not _uses_sequence_parallel(module):
        return hidden
    try:
        from megatron.core import tensor_parallel

        kwargs = {}
        group = _tp_group(module)
        if group is not None:
            kwargs["group"] = group
        try:
            return tensor_parallel.gather_from_sequence_parallel_region(
                hidden, tensor_parallel_output_grad=True, **kwargs
            )
        except TypeError:
            return tensor_parallel.gather_from_sequence_parallel_region(
                hidden, tensor_parallel_output_grad=True
            )
    except Exception:
        logger.debug("column-parallel LoRA sequence gather skipped", exc_info=True)
        return hidden


def _maybe_reduce_row_parallel(module: nn.Module, tensor: torch.Tensor) -> torch.Tensor:
    """Match row-parallel output comms: reduce-scatter when SP is on, else all-reduce."""
    if not _is_row_parallel(module):
        return tensor
    try:
        from megatron.core import tensor_parallel

        kwargs = {}
        group = _tp_group(module)
        if group is not None:
            kwargs["group"] = group
        if _uses_sequence_parallel(module):
            try:
                return tensor_parallel.reduce_scatter_to_sequence_parallel_region(tensor, **kwargs)
            except TypeError:
                return tensor_parallel.reduce_scatter_to_sequence_parallel_region(tensor)
        try:
            return tensor_parallel.reduce_from_tensor_model_parallel_region(tensor, **kwargs)
        except TypeError:
            return tensor_parallel.reduce_from_tensor_model_parallel_region(tensor)
    except Exception:
        logger.debug("row-parallel LoRA reduce skipped", exc_info=True)
        return tensor


def _attach_lora(module: nn.Module, rank: int, alpha: float) -> None:
    weight = module.weight
    out_features, in_features = weight.shape
    lora_a = nn.Parameter(torch.empty(rank, in_features, dtype=weight.dtype, device=weight.device))
    lora_b = nn.Parameter(torch.zeros(out_features, rank, dtype=weight.dtype, device=weight.device))
    nn.init.kaiming_uniform_(lora_a, a=math.sqrt(5))
    module.register_parameter(LORA_A_NAME, lora_a)
    module.register_parameter(LORA_B_NAME, lora_b)
    module.lora_scaling = float(alpha) / float(rank)
    # Row-parallel LoRA B is replicated across TP; SP shards the sequence, so
    # each rank sees a partial GEMM. Mark it like LayerNorm so Megatron
    # all-reduces those grads in finalize_model_grads.
    if _is_row_parallel(module) and _uses_sequence_parallel(module):
        setattr(lora_b, "sequence_parallel", True)
        setattr(lora_b, "allreduce", True)

    orig_forward = module.forward

    def forward_with_lora(*args, **kwargs):
        result = orig_forward(*args, **kwargs)
        if not _lora_enabled:
            return result
        hidden = args[0] if args else kwargs.get("hidden_states", kwargs.get("input"))
        if hidden is None:
            return result
        hidden = _prepare_lora_hidden(module, hidden)
        delta = F.linear(F.linear(hidden, module.lora_A), module.lora_B)
        delta = delta * module.lora_scaling
        delta = _maybe_reduce_row_parallel(module, delta)
        if isinstance(result, tuple):
            return (result[0] + delta, *result[1:])
        return result + delta

    module.forward = forward_with_lora


def inject_lora(model: Any, args: Namespace | None = None, **overrides) -> list[str]:
    """Attach LoRA to matching linear modules. Returns injected FQNs."""
    rank = int(overrides.get("lora_rank", getattr(args, "lora_rank", 16) if args else 16))
    alpha = float(overrides.get("lora_alpha", getattr(args, "lora_alpha", 32.0) if args else 32.0))
    targets = parse_lora_target_modules(
        overrides.get("lora_target_modules", getattr(args, "lora_target_modules", None) if args else None)
    )
    lora_layers = int(overrides.get("lora_layers", getattr(args, "lora_layers", -1) if args else -1))
    num_layers = overrides.get("num_layers", getattr(args, "num_layers", None) if args else None)

    injected: list[str] = []
    for chunk in iter_model_chunks(model):
        injected.extend(_inject_chunk(chunk, rank, alpha, targets, lora_layers, num_layers))

    if not injected and set(targets) != {"fc1", "fc2"}:
        logger.warning(
            "LoRA inject matched 0 modules for targets %s; retrying fc1,fc2",
            ",".join(targets),
        )
        return inject_lora(
            model,
            args,
            lora_rank=rank,
            lora_alpha=alpha,
            lora_target_modules="fc1,fc2",
            lora_layers=lora_layers,
            num_layers=num_layers,
        )

    logger.info("LoRA injected into %d modules (rank=%s alpha=%s): %s", len(injected), rank, alpha, injected[:24])
    return injected


def _inject_chunk(
    chunk: nn.Module,
    rank: int,
    alpha: float,
    targets: Sequence[str],
    lora_layers: int,
    num_layers: int | None,
) -> list[str]:
    injected: list[str] = []
    for name, module in chunk.named_modules():
        if name.endswith("output_layer") or ".output_layer" in name:
            continue
        if not _name_matches_targets(name, targets):
            continue
        if not _in_last_n_layers(name, lora_layers, num_layers):
            continue
        if not _is_injectable_linear(module):
            continue
        _attach_lora(module, rank, alpha)
        injected.append(name)
    return injected


def freeze_non_lora_critic_params(model: Any) -> None:
    """Train LoRA + value head only; MLP/attention/embeddings stay frozen."""
    for chunk in iter_model_chunks(model):
        for name, param in chunk.named_parameters():
            param.requires_grad = is_critic_trainable_param(name)


def maybe_attach_lora_critic(model: Any, args: Namespace, role: str, post_process: bool) -> None:
    del post_process
    if not getattr(args, "lora_light_mode", False) or role != "critic":
        return
    inject_lora(model, args)
    freeze_non_lora_critic_params(model)
    enable_lora()


def is_critic_only_warmup(args: Namespace | None, rollout_id: int) -> bool:
    """True when this rollout should train the critic only (no policy step)."""
    steps = int(getattr(args, "num_critic_only_steps", 0) or 0) if args is not None else 0
    return int(rollout_id) < steps


def zero_non_lora_critic_grads(model: Any) -> None:
    chunks: Iterable[Any] = model if isinstance(model, (list, tuple)) else [model]
    for chunk in chunks:
        for name, param in chunk.named_parameters():
            if param.grad is None:
                continue
            if not is_critic_trainable_param(name):
                param.grad.zero_()


def scale_lr_cond(name: str, param: torch.nn.Parameter | None = None) -> bool:
    del param
    return is_critic_trainable_param(name)


def filter_policy_sync_params(named_params: Iterable[tuple[str, torch.Tensor]]) -> Iterator[tuple[str, torch.Tensor]]:
    for name, param in named_params:
        if is_policy_sync_param(name):
            yield name, param
