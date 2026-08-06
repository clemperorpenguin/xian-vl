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

"""ONNX Runtime text detection and recognition.

Execution providers are tried in preference order and the one actually chosen
is reported rather than assumed.  That matters on Linux: ONNX Runtime will
happily accept ``VitisAIExecutionProvider`` and then silently run every node on
the CPU when the Ryzen AI user-space pieces are missing, so a claim of "running
on the NPU" has to be checked, not trusted.
"""

from __future__ import annotations

import logging
import threading

import numpy as np

from xian.ocr.engine import OcrLine, merge_adjacent_lines

logger = logging.getLogger(__name__)

__all__ = ["OnnxOcrEngine", "available_providers", "select_providers"]

#: Preference order. VitisAI is the NPU; ROCm the GPU; CPU always works and is
#: already fast enough for a small detector at these frame rates.
PROVIDER_PREFERENCE = (
    "VitisAIExecutionProvider",
    "ROCMExecutionProvider",
    "CPUExecutionProvider",
)

#: Human-readable name for each provider, for status text and logs.
PROVIDER_LABELS = {
    "VitisAIExecutionProvider": "NPU (Vitis AI)",
    "ROCMExecutionProvider": "GPU (ROCm)",
    "CPUExecutionProvider": "CPU",
}


def available_providers() -> list[str]:
    """Execution providers this ONNX Runtime build offers."""
    try:
        import onnxruntime
    except ImportError:
        return []
    return list(onnxruntime.get_available_providers())


def select_providers(prefer_npu: bool = True) -> list[str]:
    """Providers to request, most preferred first."""
    available = set(available_providers())
    order = [p for p in PROVIDER_PREFERENCE if p in available]
    if not prefer_npu:
        order = [p for p in order if p != "VitisAIExecutionProvider"]
    if "CPUExecutionProvider" not in order:
        order.append("CPUExecutionProvider")
    return order


class OnnxOcrEngine:
    """Detect and read text locally, with no encode or network round trip.

    Models come from ``rapidocr-onnxruntime`` (PP-OCR detection plus
    recognition), which ships them via pip so there is nothing to host.
    """

    def __init__(self, prefer_npu: bool = True, text_score: float = 0.5):
        self._lock = threading.Lock()
        self._text_score = text_score
        self._providers = select_providers(prefer_npu)
        self._reader = None
        self._provider = "unavailable"
        self._init_reader()

    def _init_reader(self) -> None:
        try:
            from rapidocr_onnxruntime import RapidOCR
        except ImportError as exc:
            raise RuntimeError(
                "Local OCR needs the optional dependencies. Install them with: "
                "uv sync --extra ocr"
            ) from exc

        self._reader = RapidOCR()
        self._provider = self._detect_active_provider()
        logger.info(
            "Local OCR ready on %s (requested: %s)",
            PROVIDER_LABELS.get(self._provider, self._provider),
            ", ".join(self._providers),
        )
        if "VitisAIExecutionProvider" in self._providers and self._provider != "VitisAIExecutionProvider":
            # Worth stating plainly: this is the common Linux outcome, and a
            # silent CPU fallback would otherwise look like the NPU working.
            logger.warning(
                "NPU execution was requested but ONNX Runtime assigned the models to %s. "
                "OCR still works; it is just not using the NPU.",
                PROVIDER_LABELS.get(self._provider, self._provider),
            )

    def _detect_active_provider(self) -> str:
        """Ask the loaded sessions which provider actually owns their nodes."""
        for attr in ("text_det", "text_rec"):
            component = getattr(self._reader, attr, None)
            session = getattr(component, "session", None)
            # RapidOCR wraps the session one level deeper in some releases.
            session = getattr(session, "session", session)
            providers = getattr(session, "get_providers", None)
            if callable(providers):
                active = providers()
                if active:
                    return active[0]
        return "CPUExecutionProvider"

    @property
    def provider(self) -> str:
        return self._provider

    @property
    def provider_label(self) -> str:
        return PROVIDER_LABELS.get(self._provider, self._provider)

    def run(self, frame) -> list[OcrLine]:
        """Detect and recognize text in an RGB frame.

        ``frame`` may be a PIL image or a HxWx3 uint8 array; an array is used
        as-is, which is the whole point of running in-process.
        """
        if self._reader is None:
            return []

        array = _as_array(frame)
        if array.size == 0:
            return []

        # RapidOCR sessions are not documented as thread-safe and the live
        # pipeline may call this from more than one worker.
        with self._lock:
            result, _elapsed = self._reader(array)

        if not result:
            return []

        height, width = array.shape[:2]
        lines: list[OcrLine] = []
        for entry in result:
            if len(entry) < 3:
                continue
            polygon, text, score = entry[0], entry[1], entry[2]
            try:
                confidence = float(score)
            except (TypeError, ValueError):
                confidence = 0.0
            if confidence < self._text_score or not str(text).strip():
                continue

            box = _polygon_to_box(polygon, width, height)
            if box is None:
                continue
            lines.append(OcrLine(box=box, text=str(text).strip(), confidence=confidence))

        return merge_adjacent_lines([ln for ln in lines if ln.is_valid()])


def _as_array(frame) -> np.ndarray:
    """Coerce a frame to a HxWx3 uint8 RGB array without a re-encode."""
    if isinstance(frame, np.ndarray):
        return frame
    if hasattr(frame, "convert"):  # PIL image
        return np.asarray(frame.convert("RGB"))
    return np.asarray(frame)


def _polygon_to_box(polygon, width: int, height: int) -> tuple[int, int, int, int] | None:
    """Reduce a detector's quadrilateral to an axis-aligned box."""
    try:
        points = np.asarray(polygon, dtype=float).reshape(-1, 2)
    except (ValueError, TypeError):
        return None
    if points.size == 0:
        return None

    left = int(max(0, min(width, points[:, 0].min())))
    right = int(max(0, min(width, points[:, 0].max())))
    top = int(max(0, min(height, points[:, 1].min())))
    bottom = int(max(0, min(height, points[:, 1].max())))

    if right <= left or bottom <= top:
        return None
    return (left, top, right, bottom)
