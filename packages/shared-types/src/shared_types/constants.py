"""Application-wide constants.

Migrated from the original ``xian/constants.py``.  Values here are
shared defaults; individual apps may override them via settings or CLI.
"""

# ── Lemonade Server ──────────────────────────────────────────────────
DEFAULT_API_URL = "http://localhost:13305/v1"

# ── Model Defaults ───────────────────────────────────────────────────
DEFAULT_MODEL = "Qwen3.5-0.8B-GGUF"
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

# ── GPU ──────────────────────────────────────────────────────────────
DEFAULT_GPU_MEMORY_UTILIZATION = "Default"

# ── Image Processing ─────────────────────────────────────────────────
QWEN_MAX_DIMENSION = 1920
IMAGE_HASH_SIZE = 16  # 16×16 perceptual hash
