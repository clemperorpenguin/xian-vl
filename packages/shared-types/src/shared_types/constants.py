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
# Xian registers its own Omni collection rather than depending on a vendor
# bundle: see ``xian.collections``.  ``LMX-Omni-*`` remains a valid manual
# choice, it is just not what we install.
DEFAULT_MODEL = "Xian-Ultra"
DEFAULT_COLLECTION_TIER = "ultra"

# A single vision model, offered on first run as the small alternative to a
# full collection. It covers screen translation and nothing else — no speech,
# no TTS — which makes it the quick way to try the overlay without waiting on
# several gigabytes of download.
SINGLE_MODEL_NAME = "Qwen3.5-0.8B-GGUF"
SINGLE_MODEL_SIZE_GB = 0.8
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
# Which accelerator text and speech inference prefers. Vision may also run on
# the NPU where the server offers a vision-capable NPU model — FastFlowLM does
# now — but only under an explicit "npu" preference; "auto" leaves vision on
# the GPU and moves the small, high-frequency speech and translation calls.
DEFAULT_BACKEND_PREFERENCE = "auto"
BACKEND_PREFERENCES = ("auto", "gpu", "npu")
# FastFlowLM power profiles, slowest/coolest to fastest/hottest.
NPU_POWER_MODES = ("powersaver", "balanced", "performance", "turbo")
DEFAULT_NPU_POWER_MODE = "balanced"

# ── Image Processing ─────────────────────────────────────────────────
QWEN_MAX_DIMENSION = 1920
IMAGE_HASH_SIZE = 16  # 16×16 perceptual hash

# The live overlay re-sends a frame every time the screen changes, so it is
# sized and encoded for latency rather than for fidelity.
#
# Dimension: a vision model's prefill cost scales with image *area*, so 1024
# against 1920 is roughly a third of the vision tokens. Game text is large and
# high-contrast; it survives the downscale, and the boxes come back normalized
# so nothing depends on the pixel size.
LIVE_MAX_DIMENSION = 1024

# Format: PNG on a 1920×1200 game frame measures 176-213ms to encode and 3.8MB
# on the wire, against 5ms and ~700KB for JPEG. Lossless buys nothing here —
# the frame came from a lossy display pipeline and is about to be downscaled.
# Document mode still uses PNG, where fidelity is the point.
LIVE_IMAGE_FORMAT = "JPEG"
LIVE_IMAGE_QUALITY = 85

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

# ── Live translation ─────────────────────────────────────────────────
# How often the live overlay looks at its region. Below the vision model's
# own latency there is nothing to gain — the change gate skips the work but
# the captures still cost.
DEFAULT_LIVE_INTERVAL_MS = 700

