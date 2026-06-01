"""Sliding-window context manager for frames and chat history.

Manages the most recent screen captures and the conversation thread
between the user and the assistant.  Thread-safe: all public methods
acquire an internal lock so that InferenceWorker threads and the main
Qt thread can call into this safely.
"""

from __future__ import annotations

import threading
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


class ContextManager:
    """Manages the sliding window buffer for frames and chat history."""

    MAX_CHAT_MESSAGES = 50

    def __init__(self, max_frames: int = 3):
        self.max_frames = max_frames
        self.frames: deque[FrameContext] = deque(maxlen=max_frames)
        # Message format: [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]
        self.chat_history: list[dict[str, Any]] = []
        self._lock = threading.Lock()

    def add_frame(self, image: Image.Image):
        """Add a new frame to the buffer, sliding the window if necessary."""
        frame = FrameContext(image=image.copy())
        with self._lock:
            self.frames.append(frame)
        logger.debug("ContextManager: added frame. Total frames: %d", len(self.frames))

    def update_last_frame_data(self, extracted_text: str, translations: list[Any]):
        """Update the metadata for the most recently added frame."""
        with self._lock:
            if self.frames:
                self.frames[-1].extracted_text = extracted_text
                self.frames[-1].translations = translations

    def _trim_history(self):
        """Trim chat history to stay within the sliding window limit.

        Must be called while ``self._lock`` is held.
        """
        if len(self.chat_history) > self.MAX_CHAT_MESSAGES:
            self.chat_history = self.chat_history[-self.MAX_CHAT_MESSAGES:]

    def add_user_message(self, message: str, with_image: bool = True):
        """Add a user message to the chat history.
        If with_image is True, we construct a multimodal message using the latest frame.
        """
        content: list[dict[str, str]] = []

        with self._lock:
            if with_image and self.frames:
                content.append({"type": "image"})

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

