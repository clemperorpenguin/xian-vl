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

"""Sliding-window context manager for frames and chat history.

Manages the most recent screen captures and the conversation thread
between the user and the assistant.  Thread-safe: all public methods
acquire an internal lock so that InferenceWorker threads and the main
Qt thread can call into this safely.
"""

from __future__ import annotations

import threading
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any  # Any has no builtin equivalent

from PIL import Image

import logging

logger = logging.getLogger(__name__)


@dataclass
class FrameContext:
    """Represents a single frame and its extracted data."""

    image: Image.Image
    extracted_text: str = ""
    translations: list[Any] = field(default_factory=list)
    frame_id: int = 0
    ts: float = 0.0
    window_title: str | None = None
    region: tuple[int, int, int, int] | None = None


class ContextManager:
    """Manages the sliding window buffer for frames and chat history."""

    MAX_CHAT_MESSAGES = 50

    def __init__(self, max_frames: int = 3):
        self.max_frames = max_frames
        self.frames: deque[FrameContext] = deque(maxlen=max_frames)
        # Message format: [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]
        self.chat_history: list[dict[str, Any]] = []
        self._lock = threading.Lock()
        self._next_frame_id = 1

    def add_frame(
        self,
        image: Image.Image,
        *,
        window_title: str | None = None,
        region: tuple[int, int, int, int] | None = None,
    ) -> int:
        """Add a new frame to the buffer, sliding the window if necessary.

        Returns the id assigned to the frame so callers can reference it later
        (chat messages store the id rather than the image itself).
        """
        with self._lock:
            frame_id = self._next_frame_id
            self._next_frame_id += 1
            self.frames.append(
                FrameContext(
                    image=image.copy(),
                    frame_id=frame_id,
                    ts=time.time(),
                    window_title=window_title,
                    region=region,
                )
            )
            count = len(self.frames)
        logger.debug("ContextManager: added frame %d. Total frames: %d", frame_id, count)
        return frame_id

    def update_last_frame_data(self, extracted_text: str, translations: list[Any]):
        """Update the metadata for the most recently added frame."""
        with self._lock:
            if self.frames:
                self.frames[-1].extracted_text = extracted_text
                self.frames[-1].translations = translations

    def _trim_history(self):
        """Trim chat history to stay within the sliding window limit.

        Trims on turn boundaries so the window never opens on an assistant
        message with no preceding user turn, which some chat templates reject.
        Must be called while ``self._lock`` is held.
        """
        if len(self.chat_history) <= self.MAX_CHAT_MESSAGES:
            return
        trimmed = self.chat_history[-self.MAX_CHAT_MESSAGES:]
        while trimmed and trimmed[0].get("role") != "user":
            trimmed.pop(0)
        self.chat_history = trimmed

    def add_user_message(self, message: str, with_image: bool = True):
        """Add a user message to the chat history.
        If with_image is True, we construct a multimodal message referencing the latest frame.
        """
        content: list[dict[str, Any]] = []

        with self._lock:
            if with_image and self.frames:
                # Bind the marker to the frame that was on screen *now*, so
                # replaying history later does not resolve to a newer frame.
                content.append({"type": "image", "frame_id": self.frames[-1].frame_id})

            content.append({"type": "text", "text": message})

            self.chat_history.append({"role": "user", "content": content})
            self._trim_history()
        logger.debug("ContextManager: added user message.")

    def add_assistant_message(self, message: str):
        """Add an assistant response to the chat history."""
        with self._lock:
            self.chat_history.append({"role": "assistant", "content": [{"type": "text", "text": message}]})
            self._trim_history()
        logger.debug("ContextManager: added assistant message.")

    def get_latest_frame(self) -> Image.Image | None:
        """Return the most recent frame."""
        with self._lock:
            if self.frames:
                return self.frames[-1].image
        return None

    def get_latest_frame_id(self) -> int | None:
        """Return the id of the most recent frame, if any."""
        with self._lock:
            if self.frames:
                return self.frames[-1].frame_id
        return None

    def get_frame_by_id(self, frame_id: int) -> FrameContext | None:
        """Return a frame still inside the sliding window, or None if evicted."""
        with self._lock:
            for frame in reversed(self.frames):
                if frame.frame_id == frame_id:
                    return frame
        return None

    def get_chat_history(self) -> list[dict[str, Any]]:
        """Return a shallow copy of the chat history."""
        with self._lock:
            return list(self.chat_history)

    def get_recent_extracted_text(self) -> str | None:
        """Return the most recent non-empty extracted_text from frames."""
        with self._lock:
            for frame in reversed(self.frames):
                if frame.extracted_text:
                    return frame.extracted_text
        return None

    def clear_history(self):
        """Clear only the chat history."""
        with self._lock:
            self.chat_history.clear()

    def clear_all(self):
        """Clear both frames and chat history."""
        with self._lock:
            self.frames.clear()
            self.chat_history.clear()

