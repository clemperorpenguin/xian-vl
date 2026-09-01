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

"""Pydantic v2 models for the Xian ecosystem.

These are the canonical data structures exchanged between the xian-vl
engine, Lemonade Server, and every client application.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


# ── Rendering ────────────────────────────────────────────────────────


class TextStyle(BaseModel):
    """Text styling metadata for overlay rendering."""

    font_family: str = "sans-serif"
    font_size: float = 16.0
    font_weight: str = "normal"
    text_color: tuple[int, int, int] = (255, 255, 255)
    background_color: tuple[int, int, int] | None = None
    rotation_angle: float = 0.0
    opacity: float = 1.0


# ── Accuracy ─────────────────────────────────────────────────────────


class AccuracyScore(BaseModel):
    """Quality indicator attached to every translation.

    ``score`` ranges from 0.0 (worst) to 1.0 (best).
    ``reason`` explains *why* the score is what it is.
    """

    score: float = Field(ge=0.0, le=1.0)
    reason: str = "full_pass"


# ── Translation ──────────────────────────────────────────────────────


class TranslationResult(BaseModel):
    """Single translated text region returned by the engine."""

    translated_text: str
    original_text: str = ""
    truncated: bool = False
    raw_output: str = ""
    x: float = 0.0
    y: float = 0.0
    width: float = 0.0
    height: float = 0.0
    confidence: float = 1.0
    accuracy: AccuracyScore = Field(
        default_factory=lambda: AccuracyScore(score=1.0, reason="full_pass")
    )
    style: TextStyle | None = None
    rotation_angle: float = 0.0
