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

"""Deadlines for the inference calls, by what kind of work they are.

Whole-frame vision work gets far longer than a chat sub-request: a local VLM
routinely needs tens of seconds on a large image or a cold model load, while a
query translation that takes 30 s has already failed in practice.
"""

from __future__ import annotations

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


def vision_timeout_for_mode(mode: str) -> float:
    """Return the asyncio deadline (seconds) for a vision-language completion."""
    return VISION_TIMEOUTS.get(mode, 120.0)
