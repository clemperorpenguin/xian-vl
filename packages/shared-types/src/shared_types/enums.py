"""Enumerations for the Xian ecosystem.

These enums define the valid values for language, mode, style, and
collection tier selections across all clients and the xian-vl engine.
"""

from enum import StrEnum


class SourceLanguage(StrEnum):
    """Languages supported as translation source."""

    CHINESE = "Chinese"
    JAPANESE = "Japanese"
    KOREAN = "Korean"
    RUSSIAN = "Russian"
    AUTO = "Auto"


class TargetLanguage(StrEnum):
    """Languages supported as translation target."""

    ENGLISH = "English"
    SPANISH = "Spanish"
    FRENCH = "French"
    GERMAN = "German"
    PORTUGUESE = "Portuguese"
    RUSSIAN = "Russian"
    HINDI = "Hindi"
    BENGALI = "Bengali"
    TURKISH = "Turkish"
    ARABIC = "Arabic"
    CHINESE = "Chinese"
    JAPANESE = "Japanese"
    KOREAN = "Korean"
    VIETNAMESE = "Vietnamese"


class TranslationMode(StrEnum):
    """Context modes that affect prompt construction.

    Game mode optimises for speed and UI text.
    Web mode is general-purpose.
    Document mode allows longer timeouts and richer output.
    """

    GAME = "Game"
    WEB = "Web"
    DOCUMENT = "Document"


class TranslationStyle(StrEnum):
    """Stylistic registers injected into the system prompt.

    Multiple styles can be active simultaneously.
    """

    ROMANCE = "Romance"
    WUXIA = "Wuxia"
    VISUAL_NOVEL = "Visual Novel"
    TECHNICAL = "Technical"
    LITERARY = "Literary"


class CollectionTier(StrEnum):
    """Xian-specific Lemonade collection tiers.

    Lite targets ≤ 8 GB VRAM (integrated / GTX 1660-class).
    Ultra targets ≥ 12 GB VRAM (RTX 3060+).
    Custom allows manual model selection.
    """

    LITE = "lite"
    ULTRA = "ultra"
    CUSTOM = "custom"
