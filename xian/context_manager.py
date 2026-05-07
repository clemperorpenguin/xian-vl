from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from PIL import Image
import logging

logger = logging.getLogger(__name__)

@dataclass
class FrameContext:
    """Represents a single frame and its extracted data."""
    image: Image.Image
    extracted_text: str = ""
    translations: List[Any] = field(default_factory=list)

class ContextManager:
    """Manages the sliding window buffer for frames and chat history."""

    MAX_CHAT_MESSAGES = 50

    def __init__(self, max_frames: int = 3):
        self.max_frames = max_frames
        self.frames: List[FrameContext] = []
        # Message format: [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]
        self.chat_history: List[Dict[str, Any]] = []
        
    def add_frame(self, image: Image.Image):
        """Add a new frame to the buffer, sliding the window if necessary."""
        frame = FrameContext(image=image.copy())
        self.frames.append(frame)
        
        if len(self.frames) > self.max_frames:
            self.frames.pop(0)
            
        logger.debug(f"ContextManager: added frame. Total frames: {len(self.frames)}")
        
    def update_last_frame_data(self, extracted_text: str, translations: List[Any]):
        """Update the metadata for the most recently added frame."""
        if self.frames:
            self.frames[-1].extracted_text = extracted_text
            self.frames[-1].translations = translations
            
    def _trim_history(self):
        """Trim chat history to stay within the sliding window limit."""
        if len(self.chat_history) > self.MAX_CHAT_MESSAGES:
            self.chat_history = self.chat_history[-self.MAX_CHAT_MESSAGES:]

    def add_user_message(self, message: str, with_image: bool = True):
        """Add a user message to the chat history.
        If with_image is True, we construct a multimodal message using the latest frame.
        """
        content = []
        
        if with_image and self.frames:
            content.append({"type": "image"})
            
        content.append({"type": "text", "text": message})
        
        self.chat_history.append({"role": "user", "content": content})
        self._trim_history()
        logger.debug("ContextManager: added user message.")
        
    def add_assistant_message(self, message: str):
        """Add an assistant response to the chat history."""
        self.chat_history.append({"role": "assistant", "content": [{"type": "text", "text": message}]})
        self._trim_history()
        logger.debug("ContextManager: added assistant message.")
        
    def get_latest_frame(self) -> Optional[Image.Image]:
        """Return the most recent frame."""
        if self.frames:
            return self.frames[-1].image
        return None
        
    def get_chat_history(self) -> List[Dict[str, Any]]:
        """Return the full chat history."""
        return self.chat_history
        
    def clear_history(self):
        """Clear only the chat history."""
        self.chat_history.clear()
        
    def clear_all(self):
        """Clear both frames and chat history."""
        self.frames.clear()
        self.chat_history.clear()
