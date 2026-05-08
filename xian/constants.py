"""Application-wide constants."""

# Organization and application names for QSettings
ORGANIZATION_NAME = "Xian"
APPLICATION_NAME = "VideoGameTranslator"

# Settings defaults
DEFAULT_MAX_TOKENS = 2048
DEFAULT_API_URL = "http://localhost:13305/v1"
DEFAULT_MODEL = "Qwen3.5-9B-GGUF"
DEFAULT_SOURCE_LANG = "Chinese"
DEFAULT_TARGET_LANG = "English"
DEFAULT_MODE = "Game"
DEFAULT_STYLES = []
DEFAULT_LEADER_KEY = "Shift+Space"

# Image processing
QWEN_MAX_DIMENSION = 1024
IMAGE_HASH_SIZE = 16  # 16x16 for perceptual hash
