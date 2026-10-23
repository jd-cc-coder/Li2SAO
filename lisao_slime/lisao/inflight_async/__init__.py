"""In-Flight Async: long rollouts may span multiple policy weight versions."""

from lisao.inflight_async.generate import (
    effective_remaining_new_tokens,
    remaining_context_tokens,
    remaining_max_new_tokens,
)

__all__ = [
    "generate",
    "generate_rollout_inflight_async",
    "remaining_max_new_tokens",
    "remaining_context_tokens",
    "effective_remaining_new_tokens",
    "get_pause_state",
]


def __getattr__(name: str):
    # Lazy exports so `import lisao.inflight_async` does not pull slime/sglang.
    if name == "generate":
        from lisao.inflight_async.generate import generate

        return generate
    if name == "generate_rollout_inflight_async":
        from lisao.inflight_async.rollout import generate_rollout_inflight_async

        return generate_rollout_inflight_async
    if name == "get_pause_state":
        from lisao.inflight_async.pause import get_pause_state

        return get_pause_state
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
