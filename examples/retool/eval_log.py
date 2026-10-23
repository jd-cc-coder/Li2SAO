"""Eval JSON logger: reuse lisao.eval_log and record sandbox / tool-call hits."""

from __future__ import annotations

from typing import Any

from lisao.eval.eval_log import log_eval_rollout_data as _base_log


def log_eval_rollout_data(
    rollout_id: int,
    args: Any,
    data: dict[str, Any],
    extra_metrics: dict[str, Any] | None = None,
) -> bool:
    extra = dict(extra_metrics or {})
    for key, payload in data.items():
        samples = payload.get("samples") or []
        n = len(samples)
        if not n:
            continue
        n_interp = 0
        n_tool_call = 0
        n_boxed = 0
        for s in samples:
            resp = getattr(s, "response", "") or ""
            if "<interpreter>" in resp:
                n_interp += 1
            if "<tool_call>" in resp or "<code>" in resp or "```python" in resp:
                n_tool_call += 1
            if "\\boxed" in resp:
                n_boxed += 1
        extra[f"eval/{key}-sandbox_hit_ratio"] = n_interp / n
        extra[f"eval/{key}-tool_call_ratio"] = n_tool_call / n
        extra[f"eval/{key}-boxed_ratio"] = n_boxed / n
    return _base_log(rollout_id, args, data, extra_metrics=extra)
