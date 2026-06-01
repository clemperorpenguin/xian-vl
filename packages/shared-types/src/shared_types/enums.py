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
