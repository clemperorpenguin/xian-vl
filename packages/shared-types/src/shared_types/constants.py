# Xian-VL Shared Types — Canonical model definitions and constants.
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

"""Application-wide constants.

Migrated from the original ``xian/constants.py``.  Values here are
shared defaults; individual apps may override them via settings or CLI.
"""

import os

# ── Lemonade Server ──────────────────────────────────────────────────
DEFAULT_API_URL = os.environ.get("XIAN_API_URL", "http://localhost:13305/v1")

# ── Model Defaults ───────────────────────────────────────────────────
DEFAULT_MODEL = "LMX-Omni-5.5B-Lite"
DEFAULT_MAX_TOKENS = 2048
MODE_MAX_TOKENS: dict[str, int] = {
    "Game": 4096,
    "Web": 4096,
    "Document": 8192,
}


# ── Language Defaults ────────────────────────────────────────────────
DEFAULT_SOURCE_LANG = "Chinese"
DEFAULT_TARGET_LANG = "English"
DEFAULT_MODE = "Game"
DEFAULT_STYLES: list[str] = []

# ── Application Identity (PyQt QSettings) ────────────────────────────
ORGANIZATION_NAME = "Xian"
APPLICATION_NAME = "VideoGameTranslator"

# ── Hotkeys ──────────────────────────────────────────────────────────
DEFAULT_LEADER_KEY = "Double-Tap Shift"
DEFAULT_OVERLAY_TOGGLE_KEY = "rshift"

# ── GPU ──────────────────────────────────────────────────────────────
DEFAULT_GPU_MEMORY_UTILIZATION = "Default"

# ── NPU (AMD Ryzen AI / XDNA) ────────────────────────────────────────
# Which accelerator text and speech inference prefers. Vision always stays on
# the GPU: the Linux NPU backend (FastFlowLM) serves no vision models.
DEFAULT_BACKEND_PREFERENCE = "auto"
BACKEND_PREFERENCES = ("auto", "gpu", "npu")
# FastFlowLM power profiles, slowest/coolest to fastest/hottest.
NPU_POWER_MODES = ("powersaver", "balanced", "performance", "turbo")
DEFAULT_NPU_POWER_MODE = "balanced"

# ── Image Processing ─────────────────────────────────────────────────
QWEN_MAX_DIMENSION = 1920
IMAGE_HASH_SIZE = 16  # 16×16 perceptual hash

# ── Context ──────────────────────────────────────────────────────────
# Frames kept in the sliding window. Chat history references frames by id,
# so this bounds how far back an earlier screenshot stays describable.
CONTEXT_FRAME_WINDOW = 3

# Token budgets for the session-memory block. Chat can afford real context;
# the OCR path gets a much smaller slice because latency is what matters there
# and the block only has to keep terminology consistent.
GAME_STATE_CHAT_TOKEN_BUDGET = 1200
GAME_STATE_OCR_TOKEN_BUDGET = 300

# Days of play history kept before purging.
DEFAULT_MEMORY_RETENTION_DAYS = 30
