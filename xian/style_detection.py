"""Text Style Detection and Background Reconstruction for context-aware translation rendering."""

import logging
import math
from dataclasses import dataclass, field
from typing import List, Tuple, Optional
from enum import Enum

import numpy as np
from PIL import Image, ImageFilter, ImageDraw
import cv2

logger = logging.getLogger(__name__)


class TextOrientation(Enum):
    """Text orientation types"""
    HORIZONTAL = 0
    VERTICAL = 1
    ROTATED_45 = 45
    ROTATED_90 = 90
    ROTATED_135 = 135
    UNKNOWN = -1


@dataclass
class TextStyle:
    """Detected text style information"""
    font_family: str = "sans-serif"
    font_size: float = 16.0
    font_weight: str = "normal"  # normal, bold
    font_style: str = "normal"  # normal, italic
    text_color: Tuple[int, int, int] = (255, 255, 255)  # RGB
    background_color: Optional[Tuple[int, int, int]] = None
    has_outline: bool = False
    outline_color: Tuple[int, int, int] = (0, 0, 0)
    outline_width: float = 0.0
    orientation: TextOrientation = TextOrientation.HORIZONTAL
    rotation_angle: float = 0.0
    letter_spacing: float = 0.0
    line_height: float = 1.2
    opacity: float = 1.0
    shadow_color: Optional[Tuple[int, int, int]] = None
    shadow_offset: Tuple[int, int] = (0, 0)
    shadow_blur: float = 0.0
    confidence: float = 1.0


@dataclass
class TextRegion:
    """Text region with style and bounding box"""
    x: float
    y: float
    width: float
    height: float
    text: str
    style: TextStyle
    confidence: float = 1.0
    rotation_angle: float = 0.0


class StyleDetector:
    """Detect text style from image regions"""
    
    def __init__(self):
        self.common_fonts = [
            "Arial", "Helvetica", "Times New Roman", "Georgia", "Verdana",
            "Tahoma", "Trebuchet MS", "Impact", "Comic Sans MS", "Courier New",
            "Palatino", "Garamond", "Bookman", "Avant Garde", "Noto Sans",
            "Noto Sans JP", "Noto Sans KR", "Noto Sans SC", "Source Han Sans",
            "Meiryo", "Yu Gothic", "Hiragino Sans", "Malgun Gothic"
        ]
    
    def detect_style(self, image: Image.Image, bbox: Tuple[float, float, float, float]) -> TextStyle:
        """
        Detect text style from a bounding box region.
        
        Args:
            image: PIL Image containing the text region
            bbox: (x, y, width, height) bounding box
            
        Returns:
            TextStyle object with detected properties
        """
        x, y, w, h = bbox
        
        # Extract region from image
        region = image.crop((int(x), int(y), int(x + w), int(y + h)))
        
        # Convert to numpy for analysis
        np_region = np.array(region)
        
        style = TextStyle()
        
        # Detect text color (most common non-background color)
        style.text_color = self._detect_text_color(np_region)
        
        # Detect background color
        style.background_color = self._detect_background_color(np_region)
        
        # Estimate font size from text height
        style.font_size = self._estimate_font_size(np_region, h)
        
        # Detect orientation
        style.orientation, style.rotation_angle = self._detect_orientation(np_region)
        
        # Detect if text has outline/shadow
        style.has_outline, style.outline_color, style.outline_width = self._detect_outline(np_region)
        
        # Estimate font weight
        style.font_weight = self._estimate_font_weight(np_region)
        
        return style
    
    def _detect_text_color(self, region: np.ndarray) -> Tuple[int, int, int]:
        """Detect the primary text color in the region"""
        if len(region.shape) == 3:
            # Reshape to list of pixels
            pixels = region.reshape(-1, 3)
            
            # Filter out very dark and very light pixels (likely background)
            # Text is usually mid-range brightness
            brightness = np.mean(pixels, axis=1)
            text_mask = (brightness > 50) & (brightness < 200)
            
            if np.sum(text_mask) > 0:
                text_pixels = pixels[text_mask]
                # Find most common color cluster
                avg_color = np.mean(text_pixels, axis=0)
                return tuple(int(c) for c in avg_color)
        
        # Default to white text
        return (255, 255, 255)
    
    def _detect_background_color(self, region: np.ndarray) -> Optional[Tuple[int, int, int]]:
        """Detect the background color in the region"""
        if len(region.shape) == 3:
            pixels = region.reshape(-1, 3)
            
            # Find most common color (likely background)
            # Use histogram approach
            unique_colors, counts = np.unique(pixels, axis=0, return_counts=True)
            if len(unique_colors) > 0:
                most_common_idx = np.argmax(counts)
                return tuple(int(c) for c in unique_colors[most_common_idx])
        
        return None
    
    def _estimate_font_size(self, region: np.ndarray, region_height: int) -> float:
        """Estimate font size from the region"""
        # Convert to grayscale
        if len(region.shape) == 3:
            gray = cv2.cvtColor(region, cv2.COLOR_RGB2GRAY)
        else:
            gray = region
        
        # Threshold to get text regions
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Find contours
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
            # Get average contour height
            heights = []
            for contour in contours:
                _, _, _, h = cv2.boundingRect(contour)
                if h > 5:  # Filter noise
                    heights.append(h)
            
            if heights:
                avg_height = np.mean(heights)
                # Font size is approximately the height of lowercase letters
                return max(8.0, min(72.0, avg_height * 0.8))
        
        # Default estimate based on region height
        return region_height * 0.6
    
    def _detect_orientation(self, region: np.ndarray) -> Tuple[TextOrientation, float]:
        """Detect text orientation"""
        if len(region.shape) == 3:
            gray = cv2.cvtColor(region, cv2.COLOR_RGB2GRAY)
        else:
            gray = region
        
        # Use Canny edge detection
        edges = cv2.Canny(gray, 50, 150)
        
        # Find lines using Hough transform
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=50, minLineLength=50, maxLineGap=10)
        
        if lines is not None and len(lines) > 0:
            # Calculate average angle
            angles = []
            for line in lines:
                x1, y1, x2, y2 = line[0]
                angle = math.degrees(math.atan2(y2 - y1, x2 - x1))
                angles.append(angle)
            
            if angles:
                avg_angle = np.median(angles)
                
                # Normalize angle
                if abs(avg_angle) < 15:
                    return TextOrientation.HORIZONTAL, 0.0
                elif abs(avg_angle - 90) < 15 or abs(avg_angle + 90) < 15:
                    return TextOrientation.VERTICAL, 90.0
                elif abs(avg_angle - 45) < 15 or abs(avg_angle + 45) < 15:
                    return TextOrientation.ROTATED_45, 45.0
                elif abs(avg_angle - 135) < 15 or abs(avg_angle + 135) < 15:
                    return TextOrientation.ROTATED_135, 135.0
        
        return TextOrientation.HORIZONTAL, 0.0
    
    def _detect_outline(self, region: np.ndarray) -> Tuple[bool, Tuple[int, int, int], float]:
        """Detect if text has an outline/stroke"""
        if len(region.shape) == 3:
            gray = cv2.cvtColor(region, cv2.COLOR_RGB2GRAY)
        else:
            gray = region
        
        # Look for edge patterns that suggest outlined text
        # Outlined text typically has sharp transitions at character boundaries
        
        # Threshold
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Dilate and compare
        kernel = np.ones((3, 3), np.uint8)
        dilated = cv2.dilate(binary, kernel, iterations=1)
        diff = cv2.absdiff(dilated, binary)
        
        # If there's significant difference at edges, might have outline
        edge_pixels = np.sum(diff > 0)
        total_pixels = diff.size
        
        if edge_pixels / total_pixels > 0.15:  # Threshold for outline detection
            # Try to detect outline color (usually darker)
            outline_mask = diff > 0
            if len(region.shape) == 3:
                outline_pixels = region.reshape(-1, 3)[outline_mask.flatten()]
                if len(outline_pixels) > 0:
                    outline_color = np.mean(outline_pixels, axis=0)
                    return True, tuple(int(c) for c in outline_color), 1.5
        
        return False, (0, 0, 0), 0.0
    
    def _estimate_font_weight(self, region: np.ndarray) -> str:
        """Estimate if text is bold or normal"""
        if len(region.shape) == 3:
            gray = cv2.cvtColor(region, cv2.COLOR_RGB2GRAY)
        else:
            gray = region
        
        # Threshold
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Calculate fill ratio (text pixels / bounding box)
        text_pixels = np.sum(binary > 0)
        total_pixels = binary.size
        fill_ratio = text_pixels / total_pixels
        
        # Bold text typically has higher fill ratio
        if fill_ratio > 0.15:
            return "bold"
        return "normal"


class BackgroundReconstructor:
    """Reconstruct background by inpainting over text regions"""
    
    def __init__(self):
        pass
    
    def reconstruct(self, image: Image.Image, mask: Image.Image) -> Image.Image:
        """
        Reconstruct background by inpainting over masked regions.
        
        Args:
            image: Original PIL Image
            mask: Binary mask where white pixels indicate regions to inpaint
            
        Returns:
            PIL Image with inpainted regions
        """
        # Convert to OpenCV format
        cv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        cv_mask = np.array(mask)
        
        # Use OpenCV inpainting
        inpainted = cv2.inpaint(cv_image, cv_mask, inpaintRadius=3, flags=cv2.INPAINT_TELEA)
        
        # Convert back to PIL
        inpainted_rgb = cv2.cvtColor(inpainted, cv2.COLOR_BGR2RGB)
        return Image.fromarray(inpainted_rgb)
    
    def reconstruct_simple(self, image: Image.Image, bbox: Tuple[float, float, float, float]) -> Image.Image:
        """
        Simple background reconstruction using blur and color matching.
        
        Args:
            image: Original PIL Image
            bbox: (x, y, width, height) bounding box
            
        Returns:
            PIL Image with reconstructed background
        """
        x, y, w, h = bbox
        x, y, w, h = int(x), int(y), int(w), int(h)
        
        # Create a copy
        result = image.copy()
        
        # Extract surrounding pixels for color matching
        margin = 10
        sample_region = image.crop((
            max(0, x - margin),
            max(0, y - margin),
            min(image.width, x + w + margin),
            min(image.height, y + h + margin)
        ))
        
        # Get average color from edges
        edge_pixels = []
        sample_np = np.array(sample_region)
        
        # Top and bottom edges
        edge_pixels.extend(sample_np[:margin, :].reshape(-1, 3))
        edge_pixels.extend(sample_np[-margin:, :].reshape(-1, 3))
        # Left and right edges
        edge_pixels.extend(sample_np[:, :margin].reshape(-1, 3))
        edge_pixels.extend(sample_np[:, -margin:].reshape(-1, 3))
        
        if edge_pixels:
            avg_color = np.mean(edge_pixels, axis=0)
            avg_color = tuple(int(c) for c in avg_color)
            
            # Fill the region with average color
            draw = ImageDraw.Draw(result)
            draw.rectangle([x, y, x + w, y + h], fill=avg_color)
            
            # Apply slight blur to blend
            region = result.crop((x, y, x + w, y + h))
            blurred = region.filter(ImageFilter.GaussianBlur(radius=2))
            result.paste(blurred, (x, y))
        
        return result


class StyledTextRenderer:
    """Render text with matched style"""
    
    def __init__(self):
        self.font_cache = {}
    
    def render(self, text: str, style: TextStyle, size: Tuple[int, int], 
               background: Optional[Image.Image] = None) -> Image.Image:
        """
        Render text with the specified style.
        
        Args:
            text: Text to render
            style: TextStyle object with style properties
            size: (width, height) of the output image
            background: Optional background image to render onto
            
        Returns:
            PIL Image with rendered text
        """
        # Create transparent image if no background
        if background is None:
            result = Image.new('RGBA', size, (0, 0, 0, 0))
        else:
            result = background.convert('RGBA') if background.mode != 'RGBA' else background
        
        draw = ImageDraw.Draw(result)
        
        # Get font
        font = self._get_font(style.font_family, style.font_size, style.font_weight, style.font_style)
        
        # Calculate text position (centered)
        bbox = draw.textbbox((0, 0), text, font=font)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]
        
        x = (size[0] - text_width) // 2
        y = (size[1] - text_height) // 2
        
        # Apply rotation if needed
        if style.rotation_angle != 0:
            # Create a temporary image for rotation
            temp = Image.new('RGBA', size, (0, 0, 0, 0))
            temp_draw = ImageDraw.Draw(temp)
            self._draw_text(temp_draw, text, (x, y), font, style)
            rotated = temp.rotate(-style.rotation_angle, expand=False, resample=Image.Resampling.BICUBIC)
            result = Image.alpha_composite(result, rotated)
        else:
            self._draw_text(draw, text, (x, y), font, style)
        
        return result
    
    def _get_font(self, family: str, size: float, weight: str, style: str):
        """Get or create font with specified properties"""
        from PIL import ImageFont
        
        # Try to load system font
        font_key = (family, size, weight, style)
        if font_key not in self.font_cache:
            try:
                # Construct font path based on properties
                font_name = self._resolve_font_name(family, weight, style)
                font = ImageFont.truetype(font_name, int(size))
            except (IOError, OSError):
                # Fall back to default font
                font = ImageFont.load_default()
            
            self.font_cache[font_key] = font
        
        return self.font_cache[font_key]
    
    def _resolve_font_name(self, family: str, weight: str, style: str) -> str:
        """Resolve font family to system font file"""
        # Common font paths (Linux)
        font_paths = [
            f"/usr/share/fonts/truetype/dejavu/DejaVuSans-{weight.capitalize()}.ttf",
            f"/usr/share/fonts/truetype/dejavu/DejaVuSans-{style.capitalize()}.ttf",
            f"/usr/share/fonts/truetype/liberation/LiberationSans-{weight.capitalize()}.ttf",
            f"/usr/share/fonts/truetype/noto/NotoSans-{weight.capitalize()}.ttf",
            f"/usr/share/fonts/TTF/{family}.ttf",
            f"/usr/share/fonts/{family}.ttf",
        ]
        
        import os
        for path in font_paths:
            if os.path.exists(path):
                return path
        
        # Try fontconfig
        try:
            import subprocess
            result = subprocess.run(
                ['fc-match', '-f', '%{file}', family],
                capture_output=True, text=True, timeout=5
            )
            if result.returncode == 0 and result.stdout.strip():
                return result.stdout.strip()
        except Exception:
            pass
        
        return font_paths[0]  # Return first path as fallback
    
    def _draw_text(self, draw: ImageDraw.ImageDraw, text: str, pos: Tuple[int, int], 
                   font, style: TextStyle):
        """Draw text with style properties"""
        x, y = pos
        
        # Draw shadow if present
        if style.shadow_color and style.shadow_offset != (0, 0):
            shadow_pos = (x + style.shadow_offset[0], y + style.shadow_offset[1])
            draw.text(shadow_pos, text, font=font, fill=style.shadow_color)
        
        # Draw outline if present
        if style.has_outline and style.outline_width > 0:
            outline_color = style.outline_color
            for dx in [-1, 0, 1]:
                for dy in [-1, 0, 1]:
                    if dx != 0 or dy != 0:
                        draw.text((x + dx, y + dy), text, font=font, fill=outline_color)
        
        # Draw main text
        text_color = style.text_color
        if len(text_color) == 3:
            text_color = (*text_color, int(255 * style.opacity))
        
        draw.text((x, y), text, font=font, fill=text_color)