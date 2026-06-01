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

"""Shared types for the Xian ecosystem.

Canonical Pydantic models, enums, and constants used across all apps and packages.
"""

from shared_types.enums import (
    SourceLanguage,
    TargetLanguage,
    TranslationMode,
    TranslationStyle,
    CollectionTier,
)
from shared_types.models import (
    AccuracyScore,
    TranslationRequest,
    TranslationResult,
    TextStyle,
    CinematicPayload,
    ChatRequest,
    ChatResponse,
    FeedbackPayload,
    GroundingResult,
    RobobookRequest,
    RobobookChapter,
    CollectionModel,
    LemonadeHealth,
)
from shared_types.constants import (
    DEFAULT_API_URL,
    DEFAULT_MODEL,
    DEFAULT_MAX_TOKENS,
)

__all__ = [
    # Enums
    "SourceLanguage",
    "TargetLanguage",
    "TranslationMode",
    "TranslationStyle",
    "CollectionTier",
    # Models
    "AccuracyScore",
    "TranslationRequest",
    "TranslationResult",
    "TextStyle",
    "CinematicPayload",
    "ChatRequest",
    "ChatResponse",
    "FeedbackPayload",
    "GroundingResult",
    "RobobookRequest",
    "RobobookChapter",
    "CollectionModel",
    "LemonadeHealth",
    # Constants
    "DEFAULT_API_URL",
    "DEFAULT_MODEL",
    "DEFAULT_MAX_TOKENS",
]
