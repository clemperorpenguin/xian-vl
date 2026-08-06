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

"""The OCR contract: a frame in, located lines out."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

logger = logging.getLogger(__name__)

__all__ = ["OcrLine", "OcrEngine", "ocr_available", "merge_adjacent_lines"]


@dataclass
class OcrLine:
    """One recognized run of text, in source-image pixel coordinates."""

    box: tuple[int, int, int, int]  # (left, top, right, bottom)
    text: str
    confidence: float = 0.0

    @property
    def width(self) -> int:
        return self.box[2] - self.box[0]

    @property
    def height(self) -> int:
        return self.box[3] - self.box[1]

    def is_valid(self) -> bool:
        return self.width > 0 and self.height > 0 and bool(self.text.strip())


@runtime_checkable
class OcrEngine(Protocol):
    """Anything that can find and read text in a decoded frame."""

    def run(self, frame) -> list[OcrLine]:
        """Detect and recognize text in a HxWx3 uint8 RGB array."""
        ...

    @property
    def provider(self) -> str:
        """Which execution provider is actually running the models."""
        ...


def ocr_available() -> bool:
    """Whether the optional OCR dependencies are installed."""
    try:
        import onnxruntime  # noqa: F401
        import rapidocr_onnxruntime  # noqa: F401
    except ImportError:
        return False
    return True


def merge_adjacent_lines(lines: list[OcrLine], gap_ratio: float = 0.6) -> list[OcrLine]:
    """Join detections that are really one line split by wide letter spacing.

    Detectors frequently break a spaced-out game label into fragments; merging
    them keeps the inpainted fill one rectangle instead of a row of patches.
    """
    if len(lines) < 2:
        return lines

    ordered = sorted(lines, key=lambda ln: (ln.box[1], ln.box[0]))
    merged: list[OcrLine] = [ordered[0]]

    for line in ordered[1:]:
        previous = merged[-1]
        same_row = (
            abs(line.box[1] - previous.box[1]) < previous.height * 0.5
            and abs(line.height - previous.height) < previous.height * 0.5
        )
        gap = line.box[0] - previous.box[2]
        if same_row and 0 <= gap <= previous.height * gap_ratio:
            merged[-1] = OcrLine(
                box=(
                    min(previous.box[0], line.box[0]),
                    min(previous.box[1], line.box[1]),
                    max(previous.box[2], line.box[2]),
                    max(previous.box[3], line.box[3]),
                ),
                text=f"{previous.text} {line.text}".strip(),
                confidence=min(previous.confidence, line.confidence),
            )
        else:
            merged.append(line)

    return merged
