"""Pause flag for In-Flight Async weight-update coordination.

The training actor sets this on the RolloutManager process around
``update_weights``; the in-flight generate loop polls it before resubmit
(same idea as AReaL ``PauseState``).
"""

from __future__ import annotations

import asyncio
import logging
import threading

logger = logging.getLogger("lisao.inflight_async.pause")


class PauseState:
    """Thread-safe pause flag (set from Ray sync methods, polled from asyncio).

    ``current_weight_version`` increments each time an **actor/policy**
    weight update finishes (paused True → False). Critic training does not
    pause generation or bump this counter.
    """

    def __init__(self) -> None:
        self._paused = False
        self._lock = threading.Lock()
        self._weight_version = 0

    def set_paused(self, paused: bool) -> None:
        with self._lock:
            prev = self._paused
            self._paused = bool(paused)
            bumped = False
            if prev and not self._paused:
                self._weight_version += 1
                bumped = True
            version = self._weight_version
        if prev != self._paused:
            if bumped:
                logger.info("inflight pause_state -> %s weight_version=%s", self._paused, version)
            else:
                logger.info("inflight pause_state -> %s", self._paused)

    def is_paused(self) -> bool:
        with self._lock:
            return self._paused

    def current_weight_version(self) -> int:
        with self._lock:
            return self._weight_version

    async def wait_until_resumed(self, poll_s: float = 0.5) -> None:
        """Block (async) until weight update clears the pause flag."""
        while self.is_paused():
            await asyncio.sleep(poll_s)


_PAUSE_STATE = PauseState()


def get_pause_state() -> PauseState:
    return _PAUSE_STATE
