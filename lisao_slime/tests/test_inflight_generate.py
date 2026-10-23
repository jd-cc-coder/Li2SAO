"""Unit tests for In-Flight Async abort→resume generate."""

from __future__ import annotations

import asyncio
import sys
from enum import Enum
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from lisao.inflight_async.generate import (
    effective_remaining_new_tokens,
    remaining_context_tokens,
    remaining_max_new_tokens,
)
from lisao.inflight_async.pause import PauseState, get_pause_state

NUM_GPUS = 0


def test_remaining_max_new_tokens():
    assert remaining_max_new_tokens(100, 0) == 100
    assert remaining_max_new_tokens(100, 40) == 60
    assert remaining_max_new_tokens(100, 100) == 0
    assert remaining_max_new_tokens(100, 120) == 0


def test_remaining_context_and_effective():
    assert remaining_context_tokens(None, 10) is None
    assert remaining_context_tokens(100, 40) == 60
    assert remaining_context_tokens(100, 120) == 0
    # response budget binds
    assert effective_remaining_new_tokens(50, 10, 20, 1000) == 40
    # context binds
    assert effective_remaining_new_tokens(50, 10, 95, 100) == 5
    # both zero
    assert effective_remaining_new_tokens(10, 10, 100, 100) == 0


def test_pause_state_wait_until_resumed():
    state = PauseState()
    state.set_paused(True)

    async def clear_later():
        await asyncio.sleep(0.05)
        state.set_paused(False)

    async def run():
        task = asyncio.create_task(clear_later())
        await state.wait_until_resumed(poll_s=0.01)
        await task
        assert state.is_paused() is False

    asyncio.run(run())


class _Status(Enum):
    PENDING = "pending"
    COMPLETED = "completed"
    TRUNCATED = "truncated"
    ABORTED = "aborted"


def _make_fake_sample_module():
    class Sample:
        Status = _Status

        def __init__(self):
            self.index = 7
            self.tokens = [1, 2, 3]
            self.response_length = 0
            self.weight_versions = []
            self.status = _Status.PENDING
            self.remove_sample = False
            self.metadata = {}

    return SimpleNamespace(Sample=Sample)


def test_generate_resubmits_after_abort_and_keeps_tokens():
    fake_types = _make_fake_sample_module()
    sample = fake_types.Sample()
    args = SimpleNamespace(
        rollout_max_response_len=10,
        rollout_max_context_len=None,
        inflight_max_resubmit_retries=8,
        inflight_resubmit_wait=0.0,
    )
    call_n = {"n": 0}
    get_pause_state().set_paused(False)

    async def fake_stock(args, sample, sampling_params):
        call_n["n"] += 1
        # First call: response_length=0 → max_new_tokens == effective (==10)
        # Second: response_length=4 → max_new_tokens == 4+6 == 10
        if call_n["n"] == 1:
            assert sampling_params["max_new_tokens"] == 10
            sample.response_length = 4
            sample.tokens = sample.tokens + [10, 11, 12, 13]
            sample.weight_versions.append("1")
            sample.status = fake_types.Sample.Status.ABORTED
            return sample
        assert sampling_params["max_new_tokens"] == 10  # 4 + 6
        sample.response_length = 10
        sample.tokens = sample.tokens + [20, 21, 22, 23, 24, 25]
        sample.weight_versions.append("2")
        sample.status = fake_types.Sample.Status.COMPLETED
        return sample

    fake_sglang = SimpleNamespace(generate=AsyncMock(side_effect=fake_stock))

    with patch.dict(
        sys.modules,
        {
            "slime": MagicMock(),
            "slime.utils": MagicMock(),
            "slime.utils.types": fake_types,
            "slime.rollout": MagicMock(),
            "slime.rollout.sglang_rollout": fake_sglang,
        },
    ):
        from lisao.inflight_async.generate import generate

        out = asyncio.run(generate(args, sample, {"max_new_tokens": 10}))

    assert out.status == fake_types.Sample.Status.COMPLETED
    assert out.response_length == 10
    assert out.weight_versions == ["1", "2"]
    assert call_n["n"] == 2
    assert out.metadata["inflight_segment_tokens"] == [
        {"weight_version": "1", "num_tokens": 4},
        {"weight_version": "2", "num_tokens": 6},
    ]


def test_ensure_inflight_generate_wraps_user_custom_path():
    from lisao.inflight_async.wrap import CUSTOM_GENERATE_PATH, ensure_inflight_generate

    args = SimpleNamespace(custom_generate_function_path="generate_with_retool.generate")
    ensure_inflight_generate(args)
    assert args.custom_generate_function_path == CUSTOM_GENERATE_PATH
    assert args.inflight_inner_generate_function_path == "generate_with_retool.generate"

    # Second call must not clobber the inner path.
    ensure_inflight_generate(args)
    assert args.custom_generate_function_path == CUSTOM_GENERATE_PATH
    assert args.inflight_inner_generate_function_path == "generate_with_retool.generate"


def test_ensure_inflight_generate_default_is_stock_wrapper():
    from lisao.inflight_async.wrap import CUSTOM_GENERATE_PATH, ensure_inflight_generate

    args = SimpleNamespace(custom_generate_function_path=None)
    ensure_inflight_generate(args)
    assert args.custom_generate_function_path == CUSTOM_GENERATE_PATH
    assert getattr(args, "inflight_inner_generate_function_path", None) is None


def test_generate_wraps_custom_inner_and_passes_remaining_budget():
    """Custom inner (ReTool) gets remaining tokens, not stock's inflated budget."""
    fake_types = _make_fake_sample_module()
    sample = fake_types.Sample()
    sample.response_length = 4
    sample.tokens = [1, 2, 3, 10, 11, 12, 13]
    sample.status = fake_types.Sample.Status.ABORTED
    args = SimpleNamespace(
        rollout_max_response_len=10,
        rollout_max_context_len=None,
        inflight_max_resubmit_retries=4,
        inflight_resubmit_wait=0.0,
        inflight_inner_generate_function_path="generate_with_retool.generate",
    )
    get_pause_state().set_paused(False)
    seen = {}

    async def fake_inner(args, sample, sampling_params, evaluation=False):
        seen["max_new_tokens"] = sampling_params["max_new_tokens"]
        seen["evaluation"] = evaluation
        sample.response_length = 10
        sample.tokens = sample.tokens + [20, 21, 22, 23, 24, 25]
        sample.weight_versions.append("2")
        sample.status = fake_types.Sample.Status.COMPLETED
        return sample

    with patch.dict(
        sys.modules,
        {
            "slime": MagicMock(),
            "slime.utils": MagicMock(),
            "slime.utils.types": fake_types,
            "slime.rollout": MagicMock(),
            "slime.rollout.sglang_rollout": SimpleNamespace(generate=AsyncMock()),
        },
    ):
        import lisao.inflight_async.generate as genmod

        with patch.object(genmod, "resolve_inner_generate", return_value=(fake_inner, False)):
            out = asyncio.run(genmod.generate(args, sample, {"max_new_tokens": 10}))

    assert seen["max_new_tokens"] == 6  # remaining, not 4+6
    assert seen["evaluation"] is False
    assert out.status == fake_types.Sample.Status.COMPLETED
    assert out.metadata["inflight_segment_tokens"] == [
        {"weight_version": "2", "num_tokens": 6},
    ]


def test_generate_truncates_when_budget_exhausted():
    fake_types = _make_fake_sample_module()
    sample = fake_types.Sample()
    sample.response_length = 5
    sample.tokens = [1] * 8
    sample.status = fake_types.Sample.Status.ABORTED
    args = SimpleNamespace(
        rollout_max_response_len=5,
        rollout_max_context_len=None,
        inflight_max_resubmit_retries=4,
        inflight_resubmit_wait=0.0,
    )
    stock = AsyncMock()
    fake_sglang = SimpleNamespace(generate=stock)
    get_pause_state().set_paused(False)

    with patch.dict(
        sys.modules,
        {
            "slime": MagicMock(),
            "slime.utils": MagicMock(),
            "slime.utils.types": fake_types,
            "slime.rollout": MagicMock(),
            "slime.rollout.sglang_rollout": fake_sglang,
        },
    ):
        from lisao.inflight_async.generate import generate

        out = asyncio.run(generate(args, sample, {"max_new_tokens": 5}))

    assert out.status == fake_types.Sample.Status.TRUNCATED
    stock.assert_not_called()


def test_generate_truncates_when_context_exhausted():
    fake_types = _make_fake_sample_module()
    sample = fake_types.Sample()
    sample.tokens = list(range(100))  # already at context
    sample.response_length = 3
    sample.status = fake_types.Sample.Status.ABORTED
    args = SimpleNamespace(
        rollout_max_response_len=50,
        rollout_max_context_len=100,
        inflight_max_resubmit_retries=4,
        inflight_resubmit_wait=0.0,
    )
    stock = AsyncMock()
    fake_sglang = SimpleNamespace(generate=stock)
    get_pause_state().set_paused(False)

    with patch.dict(
        sys.modules,
        {
            "slime": MagicMock(),
            "slime.utils": MagicMock(),
            "slime.utils.types": fake_types,
            "slime.rollout": MagicMock(),
            "slime.rollout.sglang_rollout": fake_sglang,
        },
    ):
        from lisao.inflight_async.generate import generate

        out = asyncio.run(generate(args, sample, {"max_new_tokens": 50}))

    assert out.status == fake_types.Sample.Status.TRUNCATED
    stock.assert_not_called()


def test_generate_waits_while_paused_before_stock_call():
    fake_types = _make_fake_sample_module()
    sample = fake_types.Sample()
    args = SimpleNamespace(
        rollout_max_response_len=8,
        rollout_max_context_len=None,
        inflight_max_resubmit_retries=4,
        inflight_resubmit_wait=0.01,
    )
    pause = get_pause_state()
    pause.set_paused(True)
    saw_unpaused = {"ok": False}

    async def fake_stock(args, sample, sampling_params):
        assert pause.is_paused() is False
        saw_unpaused["ok"] = True
        sample.response_length = 8
        sample.status = fake_types.Sample.Status.COMPLETED
        sample.weight_versions.append("1")
        return sample

    async def clear_pause():
        await asyncio.sleep(0.05)
        pause.set_paused(False)

    fake_sglang = SimpleNamespace(generate=AsyncMock(side_effect=fake_stock))

    with patch.dict(
        sys.modules,
        {
            "slime": MagicMock(),
            "slime.utils": MagicMock(),
            "slime.utils.types": fake_types,
            "slime.rollout": MagicMock(),
            "slime.rollout.sglang_rollout": fake_sglang,
        },
    ):
        from lisao.inflight_async.generate import generate

        async def run():
            clearer = asyncio.create_task(clear_pause())
            out = await generate(args, sample, {"max_new_tokens": 8})
            await clearer
            return out

        out = asyncio.run(run())

    assert saw_unpaused["ok"] is True
    assert out.status == fake_types.Sample.Status.COMPLETED


def test_should_requeue_soft_abort_logic():
    class S:
        def __init__(self, aborted=True, remove=False):
            self.status = _Status.ABORTED if aborted else _Status.COMPLETED
            self.remove_sample = remove

    def should_requeue(group):
        aborted = [s for s in group if s.status == _Status.ABORTED]
        if not aborted:
            return False
        if all(getattr(s, "remove_sample", False) for s in aborted):
            return False
        return True

    assert should_requeue([S(aborted=True, remove=False)]) is True
    assert should_requeue([S(aborted=True, remove=True)]) is False
    assert should_requeue([S(aborted=False)]) is False


def _ns(**kwargs):
    return SimpleNamespace(**kwargs)


def test_pause_state_bumps_weight_version_on_unpause():
    from lisao.inflight_async.pause import PauseState

    state = PauseState()
    assert state.current_weight_version() == 0
    state.set_paused(False)
    assert state.current_weight_version() == 0
    state.set_paused(True)
    assert state.current_weight_version() == 0
    state.set_paused(False)
    assert state.current_weight_version() == 1
    state.set_paused(True)
    state.set_paused(False)
    assert state.current_weight_version() == 2


def test_inflight_batch_metrics_ratios():
    from lisao.inflight_async.metrics import compute_inflight_batch_metrics

    on_policy = _ns(
        weight_versions=["5"],
        response_length=10,
        metadata={"inflight_segment_tokens": [{"weight_version": "5", "num_tokens": 10}]},
    )
    prev = _ns(
        weight_versions=["4"],
        response_length=10,
        metadata={"inflight_segment_tokens": [{"weight_version": "4", "num_tokens": 10}]},
    )
    cross = _ns(
        weight_versions=["4", "5"],
        response_length=10,
        metadata={
            "inflight_segment_tokens": [
                {"weight_version": "4", "num_tokens": 3},
                {"weight_version": "5", "num_tokens": 7},
            ]
        },
    )
    stale = _ns(
        weight_versions=["3"],
        response_length=10,
        metadata={"inflight_segment_tokens": [{"weight_version": "3", "num_tokens": 10}]},
    )
    # One training batch: current=5, prev=4 is 1-step lag (on-policy), 3 is stale.
    groups = [[on_policy, prev], [cross, stale]]
    metrics = compute_inflight_batch_metrics(groups, current_weight_version=5)

    assert metrics["train/inflight/num_samples"] == 4
    assert metrics["train/inflight/cross_policy_count"] == 1
    assert metrics["train/inflight/cross_policy_ratio"] == 0.25
    assert metrics["train/inflight/offpolicy_sample_count"] == 1  # only v3
    assert metrics["train/inflight/offpolicy_sample_ratio"] == 0.25
    assert metrics["train/inflight/num_response_tokens"] == 40
    assert metrics["train/inflight/offpolicy_token_count"] == 10
    assert metrics["train/inflight/offpolicy_token_ratio"] == 10 / 40
    assert metrics["train/inflight/current_weight_version"] == 5
    assert metrics["train/inflight/mean_num_weight_versions"] == 1.25
    # current=5: lags 0 (v5), 1 (v4), 1 (v4+v5 oldest=4), 2 (v3)
    assert metrics["train/inflight/policy_lag_max"] == 2
    assert metrics["train/inflight/policy_lag_mean"] == 1.0
    # tokens: 10*0 + 10*1 + 3*1 + 7*0 + 10*2 = 33 / 40
    assert metrics["train/inflight/policy_lag_mean_token"] == 33 / 40


def test_inflight_batch_metrics_all_stale_without_cross_policy():
    from lisao.inflight_async.metrics import compute_inflight_batch_metrics

    samples = [
        _ns(weight_versions=["3"], response_length=8, metadata={}),
        _ns(weight_versions=["3"], response_length=2, metadata={}),
    ]
    metrics = compute_inflight_batch_metrics(samples, current_weight_version=5)
    assert metrics["train/inflight/cross_policy_count"] == 0
    assert metrics["train/inflight/cross_policy_ratio"] == 0.0
    assert metrics["train/inflight/offpolicy_sample_count"] == 2
    assert metrics["train/inflight/offpolicy_sample_ratio"] == 1.0
    assert metrics["train/inflight/offpolicy_token_count"] == 10
    assert metrics["train/inflight/offpolicy_token_ratio"] == 1.0
    assert metrics["train/inflight/policy_lag_max"] == 2
    assert metrics["train/inflight/policy_lag_mean"] == 2.0
    assert metrics["train/inflight/policy_lag_mean_token"] == 2.0


def test_previous_version_is_not_offpolicy():
    from lisao.inflight_async.metrics import compute_inflight_batch_metrics, is_stale_weight_version

    assert is_stale_weight_version("5", 5) is False
    assert is_stale_weight_version("4", 5) is False
    assert is_stale_weight_version("3", 5) is True
    assert is_stale_weight_version(None, 5) is False

    samples = [
        _ns(weight_versions=["4"], response_length=8, metadata={}),
        _ns(weight_versions=["5"], response_length=2, metadata={}),
        _ns(weight_versions=["4", "5"], response_length=6, metadata={}),
    ]
    metrics = compute_inflight_batch_metrics(samples, current_weight_version=5)
    assert metrics["train/inflight/offpolicy_sample_count"] == 0
    assert metrics["train/inflight/offpolicy_sample_ratio"] == 0.0
    assert metrics["train/inflight/offpolicy_token_count"] == 0
    assert metrics["train/inflight/cross_policy_count"] == 1
    assert metrics["train/inflight/policy_lag_max"] == 1
    assert metrics["train/inflight/policy_lag_mean"] == 2 / 3


def test_inflight_batch_metrics_empty():
    from lisao.inflight_async.metrics import compute_inflight_batch_metrics

    metrics = compute_inflight_batch_metrics([], current_weight_version=1)
    assert metrics["train/inflight/num_samples"] == 0
    assert metrics["train/inflight/cross_policy_ratio"] == 0.0
    assert metrics["train/inflight/offpolicy_sample_ratio"] == 0.0
    assert metrics["train/inflight/offpolicy_token_ratio"] == 0.0
    assert metrics["train/inflight/policy_lag_max"] == 0
    assert metrics["train/inflight/policy_lag_mean"] == 0.0
    assert metrics["train/inflight/policy_lag_mean_token"] == 0.0


def test_policy_lag_current_minus_version():
    from lisao.inflight_async.metrics import policy_lag

    assert policy_lag("5", 5) == 0
    assert policy_lag("4", 5) == 1
    assert policy_lag("3", 5) == 2
    assert policy_lag("7", 5) == 0  # newer than current clamps to 0
    assert policy_lag(None, 5) is None
    assert policy_lag("5", None) is None


def test_samples_from_train_payload_matches_batch_metrics():
    from lisao.inflight_async.metrics import compute_inflight_batch_metrics, samples_from_train_payload

    samples = samples_from_train_payload(
        weight_versions=[["4"], ["5"]],
        segment_tokens=[
            [{"weight_version": "4", "num_tokens": 10}],
            [{"weight_version": "5", "num_tokens": 10}],
        ],
        response_lengths=[10, 10],
    )
    metrics = compute_inflight_batch_metrics(samples, current_weight_version=5)
    assert metrics["train/inflight/policy_lag_mean"] == 0.5
    assert metrics["train/inflight/policy_lag_max"] == 1


def _group(*versions: str, index: int = 0):
    samples = []
    for version in versions:
        samples.append(
            _ns(
                index=index,
                weight_versions=[version] if isinstance(version, str) else list(version),
            )
        )
    return samples


def test_resolve_max_policy_lag():
    from lisao.inflight_async.queue_control import resolve_max_policy_lag

    assert resolve_max_policy_lag(_ns()) is None
    assert resolve_max_policy_lag(_ns(max_policy_lag=None)) is None
    assert resolve_max_policy_lag(_ns(max_policy_lag=0)) == 0
    assert resolve_max_policy_lag(_ns(max_policy_lag=4)) == 4
    with pytest.raises(ValueError):
        resolve_max_policy_lag(_ns(max_policy_lag=-1))


def test_group_exceeds_max_policy_lag():
    from lisao.inflight_async.queue_control import (
        group_exceeds_max_policy_lag,
        group_max_policy_lag,
    )

    current = 5
    fresh = _group("5", "4")
    stale = _group("3", "5")  # oldest v3 → lag 2
    mixed_versions = [_ns(index=0, weight_versions=["4", "2"])]  # lag 3 from v2

    assert group_max_policy_lag(fresh, current) == 1
    assert group_exceeds_max_policy_lag(fresh, current, max_policy_lag=1) is False
    assert group_exceeds_max_policy_lag(stale, current, max_policy_lag=1) is True
    assert group_exceeds_max_policy_lag(stale, current, max_policy_lag=2) is False
    assert group_exceeds_max_policy_lag(mixed_versions, current, max_policy_lag=2) is True
    # Unknown versions: keep
    assert group_exceeds_max_policy_lag(_group(), current, max_policy_lag=0) is False


def test_cross_policy_sample_uses_oldest_first_token_version():
    from lisao.inflight_async.metrics import sample_oldest_weight_version, sample_policy_lag
    from lisao.inflight_async.queue_control import (
        group_exceeds_max_policy_lag,
        group_max_policy_lag,
        take_groups_from_completed,
    )

    current = 5
    # abort at v3, resume at v4/v5 — last segment is on-policy, first tokens are not
    cross = [
        _ns(
            index=0,
            weight_versions=["3", "4", "5"],
            metadata={
                "inflight_segment_tokens": [
                    {"weight_version": "3", "num_tokens": 8},
                    {"weight_version": "4", "num_tokens": 4},
                    {"weight_version": "5", "num_tokens": 12},
                ]
            },
        )
    ]
    assert sample_oldest_weight_version(cross[0]) == "3"
    assert sample_policy_lag(cross[0], current) == 2
    assert group_max_policy_lag(cross, current) == 2
    # Using the latest version (5) would keep this; first-token v3 must drop.
    assert group_exceeds_max_policy_lag(cross, current, max_policy_lag=1) is True
    assert group_exceeds_max_policy_lag(cross, current, max_policy_lag=2) is False

    cross_ok = [_ns(index=1, weight_versions=["4", "5"], metadata={})]
    assert sample_policy_lag(cross_ok[0], current) == 1
    assert group_exceeds_max_policy_lag(cross_ok, current, max_policy_lag=1) is False

    result = take_groups_from_completed(
        [(0, cross), (1, cross_ok)],
        need=2,
        max_policy_lag=1,
        current_weight_version=current,
    )
    assert [gid for gid, _ in result.accepted] == [1]
    assert [gid for gid, _ in result.dropped_stale] == [0]


def test_take_groups_legacy_accepts_all_no_leftover():
    from lisao.inflight_async.queue_control import take_groups_from_completed

    items = [(0, _group("1")), (1, _group("2")), (2, _group("3"))]
    result = take_groups_from_completed(items, need=2, max_policy_lag=None, current_weight_version=9)
    assert [gid for gid, _ in result.accepted] == [0, 1, 2]
    assert result.leftover == []
    assert result.dropped_stale == []


def test_take_groups_keeps_extra_and_drops_stale():
    from lisao.inflight_async.queue_control import take_groups_from_completed

    # current=5, max_policy_lag=1 → keep v5 and v4, drop v3
    items = [
        (0, _group("5")),
        (1, _group("3")),
        (2, _group("4")),
        (3, _group("5")),
        (4, _group("4")),
    ]
    result = take_groups_from_completed(items, need=2, max_policy_lag=1, current_weight_version=5)
    assert [gid for gid, _ in result.accepted] == [0, 2]
    assert [gid for gid, _ in result.leftover] == [3, 4]
    assert [gid for gid, _ in result.dropped_stale] == [1]


def test_take_groups_max_policy_lag_zero_only_current_version():
    from lisao.inflight_async.queue_control import take_groups_from_completed

    items = [(0, _group("4")), (1, _group("5")), (2, _group("5"))]
    result = take_groups_from_completed(items, need=1, max_policy_lag=0, current_weight_version=5)
    assert [gid for gid, _ in result.accepted] == [1]
    assert [gid for gid, _ in result.leftover] == [2]
    assert [gid for gid, _ in result.dropped_stale] == [0]


def _sample(group_index: int, index: int = 0):
    return _ns(group_index=group_index, index=index)


def test_flatten_queue_items_splits_n_boost_keeps_singleton_ids():
    from lisao.inflight_async.switch import flatten_queue_items

    boost = [_sample(3, i) for i in range(8)]
    singleton = [_sample(9, 99)]
    items = [(10, boost), (11, singleton)]
    flat, next_gid, next_gi, n_split = flatten_queue_items(
        items, gid_start=1000, group_index_start=50
    )
    assert n_split == 8
    assert len(flat) == 9
    assert [gid for gid, _ in flat[:8]] == list(range(1000, 1008))
    assert flat[8][0] == 11
    assert [s.group_index for _, group in flat[:8] for s in group] == list(range(50, 58))
    assert flat[8][1][0].group_index == 9
    assert next_gid == 1008
    assert next_gi == 58
    assert all(len(group) == 1 for _, group in flat)


def test_flatten_queue_items_noop_when_already_n1():
    from lisao.inflight_async.switch import flatten_queue_items

    items = [(1, [_sample(4)]), (2, [_sample(5)])]
    flat, next_gid, next_gi, n_split = flatten_queue_items(
        items, gid_start=1000, group_index_start=50
    )
    assert n_split == 0
    assert flat == items
    assert next_gid == 1000
    assert next_gi == 50


def test_flatten_buffer_groups_splits_without_touching_sample_group_index():
    from lisao.inflight_async.switch import flatten_buffer_groups

    buf = SimpleNamespace(
        buffer=[[_sample(1, 0), _sample(1, 1), _sample(1, 2)], [_sample(2, 3)]],
        sample_group_index=10,
    )
    n_split, next_gi = flatten_buffer_groups(buf, group_index_start=1_000_000_000)
    assert n_split == 3
    assert len(buf.buffer) == 4
    assert all(len(g) == 1 for g in buf.buffer)
    assert [g[0].group_index for g in buf.buffer[:3]] == [
        1_000_000_000,
        1_000_000_001,
        1_000_000_002,
    ]
    assert buf.buffer[3][0].group_index == 2
    assert next_gi == 1_000_000_003
    assert buf.sample_group_index == 10


def test_flatten_ids_do_not_overlap_data_source_counter():
    from lisao.inflight_async.switch import _FLATTEN_GID_START, flatten_queue_items

    boost = [_sample(3, i) for i in range(8)]
    live = [_sample(12, 99)]
    flat, next_gid, next_gi, n_split = flatten_queue_items(
        [(1, boost), (2, live)],
        gid_start=_FLATTEN_GID_START,
        group_index_start=_FLATTEN_GID_START,
    )
    assert n_split == 8
    split_ids = [s.group_index for _, group in flat[:8] for s in group]
    assert split_ids == list(range(_FLATTEN_GID_START, _FLATTEN_GID_START + 8))
    assert live[0].group_index == 12
    assert 12 not in split_ids
    assert next_gid == _FLATTEN_GID_START + 8
    assert next_gi == _FLATTEN_GID_START + 8


def test_resolve_inflight_switch():
    from lisao.inflight_async.switch import resolve_inflight_switch

    assert resolve_inflight_switch(_ns()) == "drain"
    assert resolve_inflight_switch(_ns(li2sao_inflight_switch="flatten")) == "flatten"
    assert resolve_inflight_switch(_ns(li2sao_inflight_switch="DRAIN")) == "drain"
    with pytest.raises(ValueError, match="li2sao-inflight-switch"):
        resolve_inflight_switch(_ns(li2sao_inflight_switch="drop"))


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
