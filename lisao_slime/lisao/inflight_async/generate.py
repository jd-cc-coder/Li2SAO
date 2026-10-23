"""AReaL-style abort → resubmit generate for In-Flight Async.

Weight updates pause SGLang and abort in-flight HTTP generates. Instead of
discarding the partial trajectory, this loop keeps already-produced tokens /
log-probs / ``weight_versions`` and continues with the remaining token budget
under the new policy weights.

Also:
  * waits on :class:`PauseState` while weights are syncing (not a fixed short sleep)
  * clamps ``max_new_tokens`` by both response budget and remaining context window

Use via::

    --custom-generate-function-path lisao.inflight_async.generate.generate

or let :func:`lisao.inflight_async.rollout.generate_rollout_inflight_async`
install this path automatically (package: ``lisao_slime/lisao/``).
"""

from __future__ import annotations

import asyncio
import inspect
import logging
from argparse import Namespace
from collections.abc import Awaitable, Callable
from typing import Any

from lisao.inflight_async.metrics import record_segment_tokens
from lisao.inflight_async.pause import get_pause_state

logger = logging.getLogger("lisao.inflight_async.generate")

# Defaults aligned with AReaL InfBridge.
_DEFAULT_MAX_RESUBMIT_RETRIES = 64
_DEFAULT_RESUBMIT_WAIT_S = 0.5


def remaining_max_new_tokens(ori_max_new_tokens: int, response_length: int) -> int:
    """Tokens still allowed before hitting the response-length budget."""
    return max(0, int(ori_max_new_tokens) - int(response_length))


def remaining_context_tokens(context_len: int | None, total_tokens: int) -> int | None:
    """Tokens still allowed before hitting the context window, or None if unset."""
    if context_len is None or int(context_len) <= 0:
        return None
    return max(0, int(context_len) - int(total_tokens))


def effective_remaining_new_tokens(
    ori_max_new_tokens: int,
    response_length: int,
    total_tokens: int,
    context_len: int | None,
) -> int:
    """Min of response-budget remainder and context-window remainder."""
    by_response = remaining_max_new_tokens(ori_max_new_tokens, response_length)
    by_context = remaining_context_tokens(context_len, total_tokens)
    if by_context is None:
        return by_response
    return min(by_response, by_context)


def _ori_max_new_tokens(args: Namespace, sampling_params: dict[str, Any]) -> int:
    """Full response budget for this sample (not the per-attempt remainder)."""
    return int(getattr(args, "rollout_max_response_len", None) or sampling_params.get("max_new_tokens") or 0)


def _context_len(args: Namespace) -> int | None:
    value = getattr(args, "rollout_max_context_len", None)
    if value is None:
        return None
    value = int(value)
    return value if value > 0 else None


def resolve_inner_generate(args: Namespace) -> tuple[Callable[..., Awaitable[Any]] | None, bool]:
    """Return (inner_generate, inner_is_stock).

    When the inflight rollout wraps a user custom generate (e.g. ReTool
    multi-turn), ``args.inflight_inner_generate_function_path`` holds that
    path. Stock slime generate subtracts ``response_length`` from
    ``max_new_tokens``; custom generates usually treat it as remaining budget.
    """
    inner_path = getattr(args, "inflight_inner_generate_function_path", None)
    if inner_path:
        from slime.utils.misc import load_function

        return load_function(inner_path), False
    return None, True


async def _call_inner_generate(
    inner: Callable[..., Awaitable[Any]],
    args: Namespace,
    sample: Any,
    sampling_params: dict[str, Any],
    evaluation: bool,
) -> Any:
    if "evaluation" in inspect.signature(inner).parameters:
        return await inner(args, sample, sampling_params, evaluation=evaluation)
    return await inner(args, sample, sampling_params)


async def generate(
    args: Namespace,
    sample: Any,
    sampling_params: dict[str, Any],
    evaluation: bool = False,
) -> Any:
    """Generate with in-flight weight-update resume.

    Signature matches slime ``--custom-generate-function-path`` callables.
    When ``inflight_inner_generate_function_path`` is set (ReTool / other
    agentic generate), that callable is wrapped instead of slime stock generate.
    """
    # Lazy import: keeps unit tests for pure helpers free of slime/sglang deps.
    from slime.utils.types import Sample

    inner, inner_is_stock = resolve_inner_generate(args)
    if inner is None:
        from slime.rollout.sglang_rollout import generate as stock_generate

        inner = stock_generate
        inner_is_stock = True

    ori_max = _ori_max_new_tokens(args, sampling_params)
    if ori_max <= 0:
        sample.status = Sample.Status.TRUNCATED
        return sample

    context_len = _context_len(args)
    max_retries = int(getattr(args, "inflight_max_resubmit_retries", _DEFAULT_MAX_RESUBMIT_RETRIES))
    wait_s = float(getattr(args, "inflight_resubmit_wait", _DEFAULT_RESUBMIT_WAIT_S))
    pause_state = get_pause_state()

    for attempt in range(max_retries):
        # Do not hit the engine while trainer is mid weight-sync.
        await pause_state.wait_until_resumed(poll_s=wait_s)

        total_tokens = len(sample.tokens) if sample.tokens else 0
        effective = effective_remaining_new_tokens(
            ori_max,
            sample.response_length,
            total_tokens,
            context_len,
        )
        if effective <= 0:
            sample.status = Sample.Status.TRUNCATED
            if context_len is not None and total_tokens >= context_len:
                logger.info(
                    "inflight context exhausted index=%s total_tokens=%d context_len=%d "
                    "response_length=%d",
                    sample.index,
                    total_tokens,
                    context_len,
                    sample.response_length,
                )
            return sample

        # Stock generate subtracts response_length from max_new_tokens; custom
        # generate (ReTool) treats it as remaining budget and clamps itself.
        sp = sampling_params.copy()
        if inner_is_stock:
            sp["max_new_tokens"] = int(sample.response_length) + int(effective)
        else:
            sp["max_new_tokens"] = int(effective)

        if sample.status not in (Sample.Status.PENDING, Sample.Status.ABORTED):
            sample.status = Sample.Status.ABORTED if sample.response_length > 0 else Sample.Status.PENDING

        prev_len = sample.response_length
        prev_versions = len(sample.weight_versions)
        sample = await _call_inner_generate(inner, args, sample, sp, evaluation)
        record_segment_tokens(sample, prev_len, prev_versions)

        if sample.status != Sample.Status.ABORTED:
            if len(sample.weight_versions) >= 2:
                logger.info(
                    "inflight multi-version sample index=%s versions=%s "
                    "response_length=%d attempts=%d",
                    sample.index,
                    sample.weight_versions,
                    sample.response_length,
                    attempt + 1,
                )
            return sample

        logger.debug(
            "inflight abort resubmit index=%s attempt=%d/%d "
            "response_length=%d(+%d) weight_versions=%s paused=%s",
            sample.index,
            attempt + 1,
            max_retries,
            sample.response_length,
            sample.response_length - prev_len,
            sample.weight_versions,
            pause_state.is_paused(),
        )
        # If still paused, the top-of-loop wait handles it; otherwise brief
        # backoff before the next attempt (engine may still be flushing).
        if pause_state.is_paused():
            await pause_state.wait_until_resumed(poll_s=wait_s)
        else:
            await asyncio.sleep(wait_s)

    logger.warning(
        "inflight: exhausted %d resubmit retries index=%s response_length=%d versions=%s",
        max_retries,
        sample.index,
        sample.response_length,
        sample.weight_versions,
    )
    return sample
