"""Flexible math RM for Qwen3.5 thinking + AIME-style integer answers.

``deepscaler`` is format-strict and often false-negatives:
  * no ``</think>`` / ``###Response`` → 0 (even with a correct ``\\boxed{}``)
  * marker present but no ``\\boxed{}`` → 0 (e.g. ``... is 504.``)

This grader keeps the same mathd/sympy equality checks, but:
  1. prefers the post-thinking / post-``###Response`` segment when present
  2. otherwise grades the full response (thinking-off or truncated mid-think)
  3. prefers ``\\boxed{}``, else the last integer in that segment
"""

from __future__ import annotations

import re
from typing import Any

from slime.rollout.rm_hub.math_utils import extract_answer, grade_answer_mathd, grade_answer_sympy

_THINK_END = "</think>"
_RESPONSE_MARK = "###Response"
_LAST_INT_RE = re.compile(r"(-?\d+)\s*\.?\s*$")


def _solution_segment(response: str) -> str:
    if _THINK_END in response:
        return response.split(_THINK_END)[-1]
    if _RESPONSE_MARK in response:
        return response.split(_RESPONSE_MARK, 1)[1]
    return response


def _extract_candidate(response: str) -> str | None:
    text = _solution_segment(response)
    boxed = extract_answer(text)
    if boxed is not None:
        return boxed
    # Whole response may still have a box if the marker was missing.
    if text is not response:
        boxed = extract_answer(response)
        if boxed is not None:
            return boxed
    m = _LAST_INT_RE.search(text.strip())
    if m:
        return m.group(1)
    ints = re.findall(r"-?\d+", text)
    if ints:
        return ints[-1]
    return None


def grade_math_flexible(response: str, label) -> int:
    if label is None or label == "":
        return 0
    gt = str(label)
    if "\\boxed" in gt:
        gt = extract_answer(gt) or gt
    pred = _extract_candidate(response or "")
    if pred is None:
        return 0
    ok = grade_answer_mathd(pred, gt) or grade_answer_sympy(pred, gt)
    return 1 if ok else 0


def grade_math_flexible_dict(response: str, label) -> dict[str, Any]:
    """Same 0/1 decision as ``grade_math_flexible``, plus extracted pred."""
    pred = _extract_candidate(response or "") or ""
    acc = float(grade_math_flexible(response, label))
    return {"score": acc, "acc": acc, "pred": pred}


async def async_rm_math_flexible(args: Any, sample_or_samples, **kwargs):
    """Custom RM entry for ``--custom-rm-path`` (single or batch)."""
    del args, kwargs
    if isinstance(sample_or_samples, list):
        return [grade_math_flexible(s.response, s.label) for s in sample_or_samples]
    return grade_math_flexible(sample_or_samples.response, sample_or_samples.label)
