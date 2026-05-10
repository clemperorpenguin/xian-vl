"""Timeout-aware inference wrapper with AccuracyScore generation.

Wraps async inference calls with ``asyncio.wait_for()`` and returns a
partial result with a degraded :class:`~shared_types.models.AccuracyScore`
when the deadline fires.

Default timeouts:
* Game mode: 3 s  (speed is king during gameplay)
* Web mode:  5 s
* Document mode: 10 s  (quality over latency)
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any, Coroutine, TypeVar

from shared_types.models import AccuracyScore

logger = logging.getLogger(__name__)

T = TypeVar("T")

# ── Default Timeouts by Mode ─────────────────────────────────────────

MODE_TIMEOUTS: dict[str, float] = {
    "Game": 3.0,
    "Web": 5.0,
    "Document": 10.0,
}


def timeout_for_mode(mode: str) -> float:
    """Return the default timeout (seconds) for a translation mode."""
    return MODE_TIMEOUTS.get(mode, 5.0)


async def run_with_timeout(
    coro: Coroutine[Any, Any, T],
    timeout_seconds: float,
    *,
    partial_result: T | None = None,
) -> tuple[T, AccuracyScore]:
    """Execute *coro* under a deadline.

    Returns
    -------
    (result, accuracy)
        If the coroutine finishes in time, ``accuracy.reason`` is
        ``"full_pass"`` with a score of 1.0.  On timeout, the
        *partial_result* fallback is returned with a degraded score.

    Raises
    ------
    Exception
        Any non-timeout exception from *coro* propagates unchanged.
    """
    try:
        result = await asyncio.wait_for(coro, timeout=timeout_seconds)
        return result, AccuracyScore(score=1.0, reason="full_pass")
    except asyncio.TimeoutError:
        logger.warning(
            "Inference timed out after %.1f s — returning partial result.",
            timeout_seconds,
        )
        if partial_result is None:
            raise
        return partial_result, AccuracyScore(score=0.3, reason="timeout")
