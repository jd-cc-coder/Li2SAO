"""Unit tests for ReTool generate abort → resume (in-flight weight sync).

Does not import real slime ``Sample`` (needs torch). Stubs slime modules so
``generate_with_retool`` can load in a CPU-only env.
"""

from __future__ import annotations

import asyncio
import sys
from enum import Enum
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

RETOOL_DIR = Path(__file__).resolve().parents[1] / "examples" / "retool"
sys.path.insert(0, str(RETOOL_DIR))

NUM_GPUS = 0


class _Status(Enum):
    PENDING = "pending"
    COMPLETED = "completed"
    TRUNCATED = "truncated"
    ABORTED = "aborted"


class Sample:
    Status = _Status

    def __init__(
        self,
        prompt="",
        tokens=None,
        response="",
        response_length=0,
        status=None,
        loss_mask=None,
        weight_versions=None,
        tool_call_count=0,
        metadata=None,
        rollout_log_probs=None,
        **kwargs,
    ):
        self.prompt = prompt
        self.tokens = list(tokens or [])
        self.response = response
        self.response_length = int(response_length)
        self.status = status if status is not None else _Status.PENDING
        self.loss_mask = list(loss_mask) if loss_mask is not None else None
        self.weight_versions = list(weight_versions or [])
        self.tool_call_count = int(tool_call_count)
        self.metadata = dict(metadata or {})
        self.rollout_log_probs = list(rollout_log_probs) if rollout_log_probs is not None else None
        self.rollout_top_p_token_ids = None
        self.rollout_top_p_token_offsets = None
        self.payload_text = ""
        self.payload_has_system = False
        self.payload_has_tools = False
        for key, value in kwargs.items():
            setattr(self, key, value)

    def append_response_tokens(
        self,
        args=None,
        *,
        tokens=None,
        log_probs=None,
        trainable: bool = True,
        meta_info=None,
        text=None,
        update_terminal_info: bool = True,
    ):
        tokens = list(tokens or [])
        if text is not None:
            self.response += text
        previous = self.response_length
        if tokens:
            self.tokens += tokens
            self.response_length += len(tokens)
            if self.loss_mask is None:
                self.loss_mask = [1] * previous
            self.loss_mask += [1 if trainable else 0] * len(tokens)
        if log_probs is not None:
            if self.rollout_log_probs is None:
                self.rollout_log_probs = [0.0] * previous
            self.rollout_log_probs += list(log_probs)
        elif tokens and not trainable:
            if self.rollout_log_probs is None:
                self.rollout_log_probs = [0.0] * previous
            self.rollout_log_probs += [0.0] * len(tokens)
        if meta_info and "weight_version" in meta_info:
            self.weight_versions.append(str(meta_info["weight_version"]))


_fake_types = SimpleNamespace(Sample=Sample)
sys.modules.setdefault("slime", MagicMock())
sys.modules.setdefault("slime.utils", MagicMock())
sys.modules["slime.utils.types"] = _fake_types
sys.modules.setdefault("slime.rollout", MagicMock())
sys.modules.setdefault("slime.rollout.sglang_rollout", SimpleNamespace(GenerateState=object))
sys.modules.setdefault("slime.utils.http_utils", SimpleNamespace(post=AsyncMock()))

from generate_with_retool import (  # noqa: E402
    _has_retool_prefix,
    _inflight_resume_enabled,
    _per_turn_max_new_tokens,
    _reset_sample_for_fresh_generate,
    _should_resume_retool,
    generate,
    postprocess_predictions,
)


class _FakeTokenizer:
    def __call__(self, text, add_special_tokens=False):
        return {"input_ids": [ord(c) for c in text]}

    def decode(self, ids):
        return "".join(chr(int(i)) for i in ids)


class _FakeGenerateState:
    def __init__(self, args):
        self.args = args
        self.tokenizer = _FakeTokenizer()


def _args(**kwargs):
    base = dict(
        sglang_router_ip="127.0.0.1",
        sglang_router_port=30000,
        rollout_max_context_len=4096,
        rollout_max_response_len=512,
        context_parallel_size=1,
        max_tokens_per_gpu=4096,
        partial_rollout=False,
        inflight_inner_generate_function_path=None,
        rollout_function_path=None,
        custom_generate_function_path="generate_with_retool.generate",
        inflight_resubmit_wait=0.0,
    )
    base.update(kwargs)
    return SimpleNamespace(**base)


def _http_output(text: str, finish: str, version: str = "1"):
    ids = [ord(c) for c in text]
    return {
        "text": text,
        "meta_info": {
            "finish_reason": {"type": finish},
            "output_token_logprobs": [[-0.1, tid] for tid in ids],
            "weight_version": version,
        },
    }


def test_inflight_resume_enabled_detects_wrapper_and_rollout_path():
    assert _inflight_resume_enabled(_args()) is False
    assert _inflight_resume_enabled(_args(partial_rollout=True)) is True
    assert _inflight_resume_enabled(_args(inflight_inner_generate_function_path="generate_with_retool.generate")) is True
    assert (
        _inflight_resume_enabled(
            _args(rollout_function_path="lisao.inflight_async.rollout.generate_rollout_inflight_async")
        )
        is True
    )
    assert (
        _inflight_resume_enabled(_args(custom_generate_function_path="lisao.inflight_async.generate.generate"))
        is True
    )


def test_should_resume_only_for_inflight_prefix():
    sample = Sample(prompt="q", tokens=[1, 2, 3, 4], response_length=2, status=Sample.Status.ABORTED)
    assert _has_retool_prefix(sample) is True
    assert _should_resume_retool(_args(), sample) is False
    assert _should_resume_retool(_args(partial_rollout=True), sample) is True
    assert _should_resume_retool(_args(partial_rollout=True), sample, evaluation=True) is False
    fresh = Sample(prompt="q", tokens=[1, 2, 3], response_length=0)
    assert _should_resume_retool(_args(partial_rollout=True), fresh) is False


def test_sync_reset_drops_prefix():
    sample = Sample(
        prompt="q",
        tokens=[1, 2, 3, 4],
        response="ab",
        response_length=2,
        loss_mask=[1, 1],
        weight_versions=["1"],
        tool_call_count=3,
        metadata={"retool_turn": 2, "inflight_segment_tokens": [{"weight_version": "1", "num_tokens": 2}]},
    )
    _reset_sample_for_fresh_generate(sample)
    assert sample.response == ""
    assert sample.response_length == 0
    assert sample.loss_mask == []
    assert sample.weight_versions == []
    assert sample.tool_call_count == 0
    assert sample.status == Sample.Status.PENDING
    assert "retool_turn" not in sample.metadata
    assert "inflight_segment_tokens" not in sample.metadata


def test_per_turn_max_new_tokens_clamps_response_and_context():
    args = _args(rollout_max_response_len=10, rollout_max_context_len=20)
    assert (
        _per_turn_max_new_tokens(
            args,
            {"max_new_tokens": 100},
            total_length=15,
            response_length=4,
            max_context_length=20,
        )
        == 5
    )
    assert (
        _per_turn_max_new_tokens(
            args,
            {"max_new_tokens": 100},
            total_length=5,
            response_length=8,
            max_context_length=20,
        )
        == 2
    )


def test_postprocess_predictions_still_finds_boxed_and_tool_call():
    assert postprocess_predictions("answer is \\boxed{42}")[0] == "answer"
    action, content = postprocess_predictions(
        '<tool_call>\n{"name": "code_interpreter", "arguments": {"code": "print(1)"}}\n</tool_call>'
    )
    assert action == "code"
    assert "print(1)" in content


def test_generate_abort_keeps_tokens_and_resume_completes():
    from lisao.inflight_async.pause import get_pause_state

    get_pause_state().set_paused(False)
    args = _args(
        inflight_inner_generate_function_path="generate_with_retool.generate",
        rollout_function_path="lisao.inflight_async.rollout.generate_rollout_inflight_async",
    )
    sample = Sample(prompt="What is 1+1?", status=Sample.Status.PENDING)
    posts = [
        _http_output("let me think", "abort", version="1"),
        _http_output(" \\boxed{2}", "stop", version="2"),
    ]

    async def fake_post(url, payload):
        return posts.pop(0)

    with (
        patch("generate_with_retool.GenerateState", _FakeGenerateState),
        patch("generate_with_retool.post", new=AsyncMock(side_effect=fake_post)),
        patch("generate_with_retool.tool_registry") as registry,
    ):
        registry.get_tool_specs.return_value = []
        first = asyncio.run(generate(args, sample, {"max_new_tokens": 64}))
        assert first.status == Sample.Status.ABORTED
        assert first.response_length > 0
        assert "let me think" in first.response
        assert first.weight_versions == ["1"]
        prompt_len = first.metadata["retool_prompt_len"]
        assert len(first.tokens) == prompt_len + first.response_length

        second = asyncio.run(generate(args, first, {"max_new_tokens": 64}))
        assert second.status == Sample.Status.COMPLETED
        assert "let me think" in second.response
        assert "\\boxed{2}" in second.response
        assert second.weight_versions == ["1", "2"]
        assert posts == []


def test_sync_mode_wipes_aborted_prefix_instead_of_resuming():
    args = _args()  # no inflight flags
    sample = Sample(prompt="What is 1+1?", status=Sample.Status.PENDING)
    sample.tokens = [1, 2, 3, 4, 5]
    sample.response_length = 2
    sample.response = "xx"
    sample.loss_mask = [1, 1]
    sample.weight_versions = ["9"]

    async def fake_post(url, payload):
        return _http_output(" \\boxed{2}", "stop", version="1")

    with (
        patch("generate_with_retool.GenerateState", _FakeGenerateState),
        patch("generate_with_retool.post", new=AsyncMock(side_effect=fake_post)),
        patch("generate_with_retool.tool_registry") as registry,
    ):
        registry.get_tool_specs.return_value = []
        out = asyncio.run(generate(args, sample, {"max_new_tokens": 64}))

    assert out.status == Sample.Status.COMPLETED
    assert "xx" not in (out.response or "")
    assert "9" not in out.weight_versions
    assert "\\boxed{2}" in out.response


def test_tool_call_split_across_abort_still_executes():
    """Incomplete <tool_call> on abort is completed after resume, then executed."""
    from lisao.inflight_async.pause import get_pause_state

    get_pause_state().set_paused(False)
    args = _args(inflight_inner_generate_function_path="generate_with_retool.generate")
    sample = Sample(prompt="What is 1+1?", status=Sample.Status.PENDING)
    first_chunk = '<tool_call>\n{"name": "code_interpreter", "arguments": {"code": "print(1)"'
    second_chunk = "}}\n</tool_call>"
    posts = [
        _http_output(first_chunk, "abort", version="1"),
        _http_output(second_chunk, "stop", version="2"),
        _http_output(" \\boxed{1}", "stop", version="2"),
    ]

    async def fake_post(url, payload):
        return posts.pop(0)

    with (
        patch("generate_with_retool.GenerateState", _FakeGenerateState),
        patch("generate_with_retool.post", new=AsyncMock(side_effect=fake_post)),
        patch("generate_with_retool.tool_registry") as registry,
    ):
        registry.get_tool_specs.return_value = []
        registry.execute_tool = AsyncMock(return_value="1")
        first = asyncio.run(generate(args, sample, {"max_new_tokens": 64}))
        assert first.status == Sample.Status.ABORTED
        assert "<tool_call>" in first.response
        assert "</tool_call>" not in first.response

        second = asyncio.run(generate(args, first, {"max_new_tokens": 64}))
        assert second.status == Sample.Status.COMPLETED
        assert "</tool_call>" in second.response
        assert "<interpreter>" in second.response
        assert "\\boxed{1}" in second.response
        registry.execute_tool.assert_awaited()
        assert posts == []


def test_pause_between_turns_returns_aborted():
    """Weight update between tool turns yields ABORTED so the wrapper splits versions."""
    args = _args(inflight_inner_generate_function_path="generate_with_retool.generate")
    sample = Sample(prompt="What is 1+1?", status=Sample.Status.PENDING)
    tool_text = '<tool_call>\n{"name": "code_interpreter", "arguments": {"code": "print(1)"}}\n</tool_call>'
    posts = [_http_output(tool_text, "stop", version="1")]

    class _Pause:
        def __init__(self):
            self._paused = False

        def is_paused(self):
            return self._paused

        async def wait_until_resumed(self, poll_s=0.5):
            self._paused = False

    pause = _Pause()

    async def fake_post(url, payload):
        pause._paused = True
        return posts.pop(0)

    with (
        patch("generate_with_retool.GenerateState", _FakeGenerateState),
        patch("generate_with_retool.post", new=AsyncMock(side_effect=fake_post)),
        patch("generate_with_retool.tool_registry") as registry,
        patch("lisao.inflight_async.pause.get_pause_state", return_value=pause),
    ):
        registry.get_tool_specs.return_value = []
        registry.execute_tool = AsyncMock(return_value="1")
        out = asyncio.run(generate(args, sample, {"max_new_tokens": 64}))

    assert out.status == Sample.Status.ABORTED
    assert "<interpreter>" in out.response
    assert out.metadata["retool_turn"] == 1
    assert posts == []


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
