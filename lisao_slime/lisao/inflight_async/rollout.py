"""In-Flight Async rollout worker for slime.

Builds on slime's fully-async background worker: a fixed pool of in-flight
groups stays warm across rollout boundaries. Soft aborts from weight updates
are handled inside :mod:`lisao.inflight_async.generate` (abort→resubmit), so a
trajectory can be composed of segments from multiple policy versions without
being discarded and restarted from the prompt.

Use with::

    python train_async.py \\
      --rollout-function-path lisao.inflight_async.rollout.generate_rollout_inflight_async
"""

from __future__ import annotations

import asyncio
import atexit
import logging
import queue
import threading
import time
from collections import deque

from lisao.inflight_async.metrics import compute_inflight_batch_metrics
from lisao.inflight_async.pause import get_pause_state
from lisao.inflight_async.queue_control import (
    group_max_policy_lag,
    resolve_max_policy_lag,
    take_groups_from_completed,
)
from lisao.inflight_async.switch import (
    _FLATTEN_GID_START,
    as_singleton_group,
    flatten_buffer_groups,
    flatten_queue_items,
    iter_prompt_members,
)
from lisao.inflight_async.wrap import ensure_inflight_generate as _ensure_inflight_generate
from slime.rollout.base_types import RolloutFnTrainOutput
from slime.rollout.sglang_rollout import GenerateState, generate_and_rm_group
from slime.utils.async_utils import run
from slime.utils.http_utils import get_rollout_num_engines
from slime.utils.types import Sample

__all__ = [
    "InFlightAsyncRolloutWorker",
    "drain_inflight_worker",
    "enable_inflight_flatten",
    "flatten_inflight_worker",
    "generate_rollout_inflight_async",
]

logger = logging.getLogger("lisao.inflight_async.rollout")

_global_worker: InFlightAsyncRolloutWorker | None = None
_worker_lock = threading.Lock()


def _iter_samples(node: Sample | list) -> list[Sample]:
    if isinstance(node, Sample):
        return [node]
    if isinstance(node, list):
        out: list[Sample] = []
        for item in node:
            out.extend(_iter_samples(item))
        return out
    return []


def _group_sort_key(group: list) -> int:
    if group:
        head = group[0]
        if isinstance(head, list):
            if head and isinstance(head[0], Sample) and head[0].index is not None:
                return int(head[0].index)
        elif isinstance(head, Sample) and head.index is not None:
            return int(head.index)
    for sample in _iter_samples(group):
        if sample.index is not None:
            return int(sample.index)
    return 0


def _should_requeue_aborted(group: list) -> bool:
    """Requeue only when soft-abort samples remain after in-generate retries.

    Permanent ``remove_sample`` aborts ship to training (loss masked). Soft
    aborts that still carry partial tokens go back to the buffer so the next
    attempt can continue — tokens are preserved on the Sample.
    """
    samples = _iter_samples(group)
    aborted = [s for s in samples if s.status == Sample.Status.ABORTED]
    if not aborted:
        return False
    if all(getattr(s, "remove_sample", False) for s in aborted):
        return False
    return True


def _log_multi_version_group(group: list) -> None:
    multi = [s for s in _iter_samples(group) if len(getattr(s, "weight_versions", []) or []) >= 2]
    if not multi:
        return
    for sample in multi:
        logger.info(
            "inflight-async completed multi-version sample index=%s "
            "num_weight_versions=%d versions=%s response_length=%d",
            sample.index,
            len(sample.weight_versions),
            sample.weight_versions,
            sample.response_length,
        )


def _get_global_worker(args, data_buffer) -> InFlightAsyncRolloutWorker:
    global _global_worker
    with _worker_lock:
        if _global_worker is None or not _global_worker.worker_thread.is_alive():
            logger.info("starting in-flight-async rollout worker")
            _global_worker = InFlightAsyncRolloutWorker(
                args,
                data_buffer,
                concurrency=args.sglang_server_concurrency * get_rollout_num_engines(args),
            )
            _global_worker.start()
        return _global_worker


def _stop_global_worker() -> None:
    global _global_worker
    with _worker_lock:
        if _global_worker is not None:
            _global_worker.stop()
            _global_worker = None


def enable_inflight_flatten() -> None:
    """Start splitting n-boost completions; does not drop leftover."""
    global _global_worker
    with _worker_lock:
        worker = _global_worker
        if worker is not None:
            worker.enable_flatten_to_n1()


def flatten_inflight_worker(data_buffer=None) -> int:
    """Keep the worker; split leftover n-boost groups into n=1 singletons."""
    global _global_worker
    n_split = 0
    with _worker_lock:
        worker = _global_worker
        if worker is not None:
            n_split = worker.flatten_to_n1(data_buffer)
        elif data_buffer is not None:
            n_split, _ = flatten_buffer_groups(data_buffer, group_index_start=_FLATTEN_GID_START)
    logger.info("li2sao flatten inflight: split %d n-boost members into n=1", n_split)
    return n_split


def drain_inflight_worker(data_buffer=None) -> None:
    """Drop leftover in-flight groups so the next sag can change n."""
    global _global_worker
    with _worker_lock:
        worker = _global_worker
        if worker is not None:
            worker.running = False
            try:
                with worker._leftover_lock:
                    worker._leftover.clear()
            except Exception:
                logger.debug("inflight leftover clear failed", exc_info=True)
            while True:
                try:
                    worker.output_queue.get_nowait()
                except queue.Empty:
                    break
            worker.stop()
            _global_worker = None
    if data_buffer is not None and hasattr(data_buffer, "buffer"):
        try:
            data_buffer.buffer.clear()
        except Exception:
            logger.debug("inflight data_buffer clear failed", exc_info=True)


atexit.register(_stop_global_worker)


class InFlightAsyncRolloutWorker:
    """Background thread + asyncio loop for In-Flight Async generation."""

    def __init__(self, args, data_buffer, concurrency: int = 10):
        self.args = args
        self.data_buffer = data_buffer
        self.concurrency = concurrency
        self.running = True
        self.output_queue: queue.Queue[tuple[int, list]] = queue.Queue(maxsize=1000)
        # Groups taken but not used this step (max_policy_lag path). Checked
        # before output_queue so leftover stays FIFO across training steps.
        self._leftover: deque[tuple[int, list]] = deque()
        self._leftover_lock = threading.Lock()
        self._flatten_to_n1 = False
        self._flatten_gid = _FLATTEN_GID_START
        self.worker_thread: threading.Thread | None = None
        self.state = GenerateState(args)

    def start(self) -> None:
        if self.worker_thread is None or not self.worker_thread.is_alive():
            self.worker_thread = threading.Thread(
                target=self._thread_main, name="inflight-async-rollout", daemon=True
            )
            self.worker_thread.start()

    def stop(self) -> None:
        self.running = False
        if self.worker_thread and self.worker_thread.is_alive():
            self.worker_thread.join(timeout=5)

    def enable_flatten_to_n1(self) -> None:
        self._flatten_to_n1 = True

    def flatten_to_n1(self, data_buffer=None) -> int:
        """Split leftover / output_queue / abort buffer into n=1; keep running."""
        self.enable_flatten_to_n1()
        items: list[tuple[int, list]] = []
        with self._leftover_lock:
            while self._leftover:
                items.append(self._leftover.popleft())
        while True:
            try:
                items.append(self.output_queue.get_nowait())
            except queue.Empty:
                break
        n_buf = 0
        with self._leftover_lock:
            gid_start = self._flatten_gid
            flat, next_gid, next_gi, n_split = flatten_queue_items(
                items,
                gid_start=gid_start,
                group_index_start=gid_start,
            )
            self._flatten_gid = max(next_gid, next_gi)
            if data_buffer is not None:
                n_buf, next_gi = flatten_buffer_groups(
                    data_buffer, group_index_start=self._flatten_gid
                )
                self._flatten_gid = max(self._flatten_gid, next_gi)
            for item in flat:
                self._leftover.append(item)
        logger.info(
            "inflight-async flatten to n=1: queue_groups=%d -> %d split_members=%d buffer_split=%d",
            len(items),
            len(flat),
            n_split,
            n_buf,
        )
        return n_split + n_buf

    def _alloc_flatten_gids(self, n: int) -> list[int]:
        with self._leftover_lock:
            start = self._flatten_gid
            self._flatten_gid += n
            return list(range(start, start + n))

    def _put_completed(self, gid: int, group: list) -> None:
        """Non-blocking emit so flatten's 8-way split cannot stall the event loop."""
        item = (gid, group)
        try:
            self.output_queue.put_nowait(item)
        except queue.Full:
            with self._leftover_lock:
                self._leftover.append(item)

    def _stamp_flatten_ids(self, pieces: list[list]) -> list[int]:
        n = 0
        for piece in pieces:
            n += max(1, len(_iter_samples(piece)))
        ids = self._alloc_flatten_gids(n)
        k = 0
        gids: list[int] = []
        for piece in pieces:
            gid = ids[k]
            gids.append(gid)
            samples = _iter_samples(piece)
            if not samples:
                k += 1
                continue
            for sample in samples:
                if hasattr(sample, "group_index"):
                    sample.group_index = ids[k]
                k += 1
        return gids

    def _requeue_n1_groups(self, pieces: list[list]) -> None:
        """Put aborted n=1 groups back on the data buffer without mixing n."""
        buffer = getattr(self.data_buffer, "buffer", None)
        for piece in pieces:
            if buffer is not None:
                buffer.append(piece)
                continue
            try:
                self.data_buffer.add_samples([piece])
            except Exception:
                logger.exception("inflight-async: failed to requeue flattened aborted group")

    def get_completed_groups(self) -> list[tuple[int, list]]:
        completed: list[tuple[int, list]] = []
        with self._leftover_lock:
            while self._leftover:
                completed.append(self._leftover.popleft())
        while True:
            try:
                completed.append(self.output_queue.get_nowait())
            except queue.Empty:
                break
        if not self._flatten_to_n1 or not completed:
            return completed
        with self._leftover_lock:
            gid_start = self._flatten_gid
            flat, next_gid, next_gi, n_split = flatten_queue_items(
                completed,
                gid_start=gid_start,
                group_index_start=gid_start,
            )
            if n_split:
                self._flatten_gid = max(next_gid, next_gi)
                logger.info("inflight-async flatten stragglers: split %d n-boost members", n_split)
        return flat

    def put_back_groups(self, items: list[tuple[int, list]]) -> None:
        """Requeue unused groups at the front for the next training step."""
        if not items:
            return
        with self._leftover_lock:
            for item in reversed(items):
                self._leftover.appendleft(item)

    def queue_size(self) -> int:
        with self._leftover_lock:
            leftover_n = len(self._leftover)
        return leftover_n + self.output_queue.qsize()

    def _thread_main(self) -> None:
        asyncio.run(self._loop())

    async def _loop(self) -> None:
        active_tasks: set[asyncio.Task] = set()
        max_concurrent = self.concurrency
        gid_counter = 0

        while self.running:
            try:
                if active_tasks:
                    done = {t for t in active_tasks if t.done()}
                    for t in done:
                        try:
                            t.result()
                        except Exception as e:  # noqa: BLE001
                            logger.warning("inflight-async task crashed: %r", e)
                    active_tasks -= done

                while len(active_tasks) < max_concurrent and self.running:
                    groups = self.data_buffer.get_samples(1)
                    if not groups:
                        break
                    for group in groups:
                        gid = gid_counter
                        gid_counter += 1
                        task = asyncio.create_task(
                            generate_and_rm_group(
                                self.args,
                                group,
                                sampling_params=self.state.sampling_params.copy(),
                                evaluation=False,
                            )
                        )
                        task.add_done_callback(self._make_done_cb(gid))
                        active_tasks.add(task)

                await asyncio.sleep(1)
            except Exception as e:  # noqa: BLE001
                logger.exception("inflight-async loop iteration error: %s", e)
                await asyncio.sleep(1)

        if active_tasks:
            logger.info("inflight-async: waiting for %d in-flight tasks to drain", len(active_tasks))
            try:
                await asyncio.wait(active_tasks, timeout=30)
            except Exception:  # noqa: BLE001
                pass

    def _make_done_cb(self, gid: int):
        def _cb(done_task: asyncio.Task) -> None:
            try:
                result = done_task.result()
            except Exception:  # noqa: BLE001
                logger.exception("inflight-async: process task raised")
                return
            if not isinstance(result, list):
                logger.warning(
                    "inflight-async: generate_and_rm_group returned %r, expected list; dropping",
                    type(result).__name__,
                )
                return
            # After a flatten switch, n-boost completions become n=1 groups so
            # they can mix with new n-stable sampling without GRPO grouping.
            if self._flatten_to_n1 and len(iter_prompt_members(result)) > 1:
                pieces = [as_singleton_group(m) for m in iter_prompt_members(result)]
                gids = self._stamp_flatten_ids(pieces)
                aborted: list[list] = []
                for piece, new_gid in zip(pieces, gids, strict=True):
                    if _should_requeue_aborted(piece):
                        aborted.append(piece)
                        continue
                    _log_multi_version_group(piece)
                    self._put_completed(new_gid, piece)
                if aborted:
                    self._requeue_n1_groups(aborted)
                    logger.info(
                        "inflight-async: requeued %d aborted n=1 members from n-boost "
                        "gid=%d (ready=%d)",
                        len(aborted),
                        gid,
                        len(pieces) - len(aborted),
                    )
                return
            # Soft abort after generate-level retries → requeue with tokens kept.
            if _should_requeue_aborted(result):
                try:
                    self.data_buffer.add_samples([result])
                    logger.info(
                        "inflight-async: requeued aborted group with partial tokens "
                        "(gid=%d response_lengths=%s)",
                        gid,
                        [s.response_length for s in _iter_samples(result)],
                    )
                except Exception:  # noqa: BLE001
                    logger.exception("inflight-async: failed to requeue aborted group")
                return
            _log_multi_version_group(result)
            self._put_completed(gid, result)

        return _cb


async def _generate_rollout_async(args, rollout_id: int, data_buffer) -> tuple[list[list[Sample]], dict]:
    assert args.rollout_global_dataset
    _ensure_inflight_generate(args)
    worker = _get_global_worker(args, data_buffer)

    target = args.rollout_batch_size
    max_policy_lag = resolve_max_policy_lag(args)
    logger.info(
        "inflight-async rollout %d: target=%d queue_warm=%d max_policy_lag=%s",
        rollout_id,
        target,
        worker.queue_size(),
        max_policy_lag,
    )

    collected: dict[int, list] = {}
    dropped_stale = 0
    leftover_put_back = 0
    started = time.time()
    last_log = started
    log_every = 30.0

    while len(collected) < target:
        items = worker.get_completed_groups()
        if not items:
            await asyncio.sleep(0.05)
        else:
            current = get_pause_state().current_weight_version()
            result = take_groups_from_completed(
                items,
                need=target - len(collected),
                max_policy_lag=max_policy_lag,
                current_weight_version=current,
            )
            for gid, group in result.accepted:
                collected[gid] = group
            if result.dropped_stale:
                dropped_stale += len(result.dropped_stale)
                preview = [
                    (gid, group_max_policy_lag(group, current))
                    for gid, group in result.dropped_stale[:8]
                ]
                logger.info(
                    "inflight-async rollout %d: dropped %d stale groups "
                    "(max_policy_lag=%s current=%s) preview=%s",
                    rollout_id,
                    len(result.dropped_stale),
                    max_policy_lag,
                    current,
                    preview,
                )
            if result.leftover:
                leftover_put_back += len(result.leftover)
                worker.put_back_groups(result.leftover)

        now = time.time()
        if now - last_log > log_every:
            logger.info(
                "inflight-async rollout %d: collected %d/%d, queue=%d, elapsed=%.1fs",
                rollout_id,
                len(collected),
                target,
                worker.queue_size(),
                now - started,
            )
            last_log = now

    # Legacy: sort then slice, discarding overshoot already drained.
    # max_policy_lag: collected is exactly target; leftover already put back.
    out = sorted(collected.values(), key=_group_sort_key)
    if max_policy_lag is None:
        out = out[:target]
    # Policy lag for wandb is computed on the actor at train time
    # (train/inflight/*, aligned with train/step). Collection-time current
    # is PauseState and is only logged here for debugging.
    collect_metrics = compute_inflight_batch_metrics(
        out,
        current_weight_version=get_pause_state().current_weight_version(),
        prefix="rollout/inflight",
    )
    logger.info(
        "inflight-async rollout %d collect: done in %.1fs, queue_left=%d, "
        "dropped_stale=%d leftover_put_back=%d max_policy_lag=%s "
        "collect_policy_lag max=%s mean=%.3f mean_token=%.3f "
        "cross_policy=%s/%s (%.3f) offpolicy_samples=%s/%s (%.3f) "
        "current_weight_version=%s (PauseState; wandb uses train/inflight/*)",
        rollout_id,
        time.time() - started,
        worker.queue_size(),
        dropped_stale,
        leftover_put_back,
        max_policy_lag,
        collect_metrics["rollout/inflight/policy_lag_max"],
        collect_metrics["rollout/inflight/policy_lag_mean"],
        collect_metrics["rollout/inflight/policy_lag_mean_token"],
        collect_metrics["rollout/inflight/cross_policy_count"],
        collect_metrics["rollout/inflight/num_samples"],
        collect_metrics["rollout/inflight/cross_policy_ratio"],
        collect_metrics["rollout/inflight/offpolicy_sample_count"],
        collect_metrics["rollout/inflight/num_samples"],
        collect_metrics["rollout/inflight/offpolicy_sample_ratio"],
        collect_metrics["rollout/inflight/current_weight_version"],
    )
    return out, {
        **collect_metrics,
        "rollout/inflight/dropped_stale_groups": dropped_stale,
        "rollout/inflight/leftover_put_back": leftover_put_back,
    }


def generate_rollout_inflight_async(args, rollout_id, data_buffer, evaluation: bool = False):
    """Slime ``--rollout-function-path`` entrypoint for In-Flight Async."""
    if evaluation:
        raise ValueError("in-flight-async rollout doesn't support evaluation mode")
    _ensure_inflight_generate(args)
    groups, metrics = run(_generate_rollout_async(args, rollout_id, data_buffer))
    return RolloutFnTrainOutput(samples=groups, metrics=metrics)
