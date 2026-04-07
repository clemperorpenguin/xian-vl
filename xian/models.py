from dataclasses import dataclass, field
from enum import Enum
from typing import Tuple, Optional

class TranslationMode(Enum):
    FULL_SCREEN = "full_screen"
    REGION_SELECT = "region_select"

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
class TranslationRegion:
    """Represents a region to be translated"""
    x: int
    y: int
    width: int
    height: int
    name: str = ""
    enabled: bool = True

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
