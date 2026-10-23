"""Eval / train RM aligned with ``examples/lisao``: ``lisao.eval.math_rm``.

Generation still goes through ``generate_with_retool.generate`` (code_interpreter
sandbox). Scoring is the same 0/1 flexible grader as inflight / LiSAO:
prefer ``\\boxed{}``, else the last integer; mathd or sympy equality.
``mean_reward`` equals empirical Pass@1 when n=1.

Retool-only cleanup: drop the old invalid-action hint that embeds a fake
``\\boxed{answer}``, which would otherwise become the last boxed extract.
"""

from __future__ import annotations

import re
from typing import Any

from lisao.eval.math_rm import grade_math_flexible_dict
from slime.utils.types import Sample

_INVALID_ACTION_HINT = re.compile(
    r"My previous action is invalid\..*?Let me try again\.",
    flags=re.DOTALL,
)
_PLACEHOLDER_BOXED = re.compile(r"\\boxed\{answer\}", flags=re.IGNORECASE)


def strip_retool_grade_noise(response: str) -> str:
    text = response or ""
    text = _INVALID_ACTION_HINT.sub("", text)
    text = _PLACEHOLDER_BOXED.sub("", text)
    return text


def grade_one(sample: Sample) -> dict[str, Any]:
    if not isinstance(sample, Sample):
        raise TypeError("Sample must be an instance of Sample class.")
    response = strip_retool_grade_noise(sample.response or "")
    return grade_math_flexible_dict(response, sample.label)


async def reward_func(args, sample, **kwargs):
    del args, kwargs
    if isinstance(sample, list):
        return [grade_one(s) for s in sample]
    return grade_one(sample)
