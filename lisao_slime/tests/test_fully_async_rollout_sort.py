"""Unit tests for fully-async fan-out group sorting / abort requeue."""

from __future__ import annotations

from slime.rollout.fully_async_rollout import _group_sort_key, _should_requeue_aborted
from slime.utils.types import Sample


def _sample(index: int | None = 0, *, aborted: bool = False, remove: bool = False) -> Sample:
    s = Sample(index=index, prompt="x", tokens=[1], response_length=1, loss_mask=[1])
    if aborted:
        s.status = Sample.Status.ABORTED
    if remove:
        s.remove_sample = True
    return s


def test_group_sort_key_plain_group():
    group = [_sample(3), _sample(3)]
    assert _group_sort_key(group) == 3


def test_group_sort_key_fanout_nested_list():
    # custom-generate returns list[Sample] per member → gather yields list[list[Sample]].
    # Regression: getattr(list, "index") is list.index (a method), int() TypeError.
    group = [[_sample(7), _sample(7)], [_sample(7)]]
    assert _group_sort_key(group) == 7


def test_group_sort_key_does_not_use_list_index_method():
    group = [[_sample(None)], [_sample(None)]]
    # No Sample.index set → fall back to 0; must not call int(list.index).
    assert _group_sort_key(group) == 0


def test_should_requeue_soft_abort():
    group = [_sample(1, aborted=True), _sample(1, aborted=True)]
    assert _should_requeue_aborted(group) is True


def test_should_not_requeue_permanent_remove_sample_abort():
    group = [[_sample(1, aborted=True, remove=True)], [_sample(1, aborted=True, remove=True)]]
    assert _should_requeue_aborted(group) is False


def test_completed_group_not_requeued():
    group = [[_sample(2), _sample(2)]]
    assert _should_requeue_aborted(group) is False
