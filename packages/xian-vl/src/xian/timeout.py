# Xian-VL — Core Vision-Language orchestration engine.
# Copyright (C) 2026  Clementine Pendragon <clem@pendragon.systems>
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.
#
# Contact: clem@pendragon.systems (Clementine Pendragon, c/o Xian Project Development)

"""Timeout-aware inference wrapper with AccuracyScore generation.

Wraps async inference calls with ``asyncio.wait_for()`` and returns a
partial result with a degraded :class:`~shared_types.models.AccuracyScore`
when the deadline fires.

Default timeouts for :func:`run_with_timeout` (text-style deadlines; not used for VLM):

* Game / Web / Document: conservative values if you wire ``run_with_timeout`` to mode.

Vision (VLM) OCR + translation needs much longer; see :data:`VISION_TIMEOUTS` and
:func:`vision_timeout_for_mode`.
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

# Whole-frame VLM calls (screenshot → OCR + translate). Lemonade / local models
# routinely need tens of seconds, especially on first load or large images.
VISION_TIMEOUTS: dict[str, float] = {
    "Game": 30.0,
    "Web": 60.0,
    "Document": 120.0,
}

# Chat / agentic flows (tool calls + follow-up) need a larger budget than live OCR.
CHAT_TIMEOUT_SECONDS = 120.0

# Sub-requests inside chat (e.g. query translation for dual search).
CHAT_AUX_TIMEOUT_SECONDS = 30.0


def timeout_for_mode(mode: str) -> float:
    """Return the default timeout (seconds) for a translation mode."""
    return MODE_TIMEOUTS.get(mode, 5.0)


def vision_timeout_for_mode(mode: str) -> float:
    """Return the asyncio deadline (seconds) for a vision-language completion."""
    return VISION_TIMEOUTS.get(mode, 120.0)


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
