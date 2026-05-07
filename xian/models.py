from dataclasses import dataclass, field
from typing import Tuple, Optional


@dataclass
class TextStyle:
    """Text style information for rendering"""
    font_family: str = "sans-serif"
    font_size: float = 16.0
    font_weight: str = "normal"
    text_color: Tuple[int, int, int] = (255, 255, 255)
    background_color: Optional[Tuple[int, int, int]] = None
    rotation_angle: float = 0.0
    opacity: float = 1.0


@dataclass
class TranslationResult:
    """Result from translation API with style information"""
    translated_text: str
    x: float
    y: float
    width: float
    height: float
    confidence: float = 1.0
    original_text: str = ""
    style: Optional[TextStyle] = None
    rotation_angle: float = 0.0
