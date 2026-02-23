import hashlib
import logging
import os
import subprocess
import tempfile
from typing import Optional, Tuple
from PyQt6.QtGui import QImage, QGuiApplication
from PyQt6.QtCore import QBuffer, QIODevice, Qt, QRect

logger = logging.getLogger(__name__)

def _check_screenshot_available():
    """Check if at least one screenshot method is likely available"""
    if os.environ.get("XDG_SESSION_TYPE") == "wayland":
        # Check for common wayland tools
        for tool in ["spectacle", "gnome-screenshot", "grim"]:
            if subprocess.run(["which", tool], capture_output=True).returncode == 0:
                return True
        # Also check for DBus as it might be used for GNOME
        return True # Assume DBus might work
    return True # PyQt fallback for X11

SCREENSHOT_AVAILABLE = _check_screenshot_available()

class ScreenCapture:
    """Handle screen capture using multiple backends for Wayland/X11 compatibility"""

    @staticmethod
    def get_virtual_desktop_geometry() -> QRect:
        """Get the geometry of the entire virtual desktop (all screens combined)"""
        total_geo = QRect()
        for screen in QGuiApplication.screens():
            total_geo = total_geo.united(screen.geometry())
        return total_geo

    @staticmethod
    def capture_screen() -> Optional[bytes]:
        """Capture entire screen using best available method"""
        
        # Try Wayland-specific methods first if on Wayland
        is_wayland = os.environ.get("XDG_SESSION_TYPE") == "wayland"
        desktop = os.environ.get("XDG_CURRENT_DESKTOP", "").lower()

        if is_wayland:
            logger.debug(f"Wayland detected, desktop: {desktop}")
            # 1. KDE Plasma - Spectacle
            if "kde" in desktop:
                logger.debug("Trying Spectacle backend...")
                data = ScreenCapture._capture_spectacle()
                if data: return data
            
            # 2. GNOME - gnome-screenshot or DBus
            if "gnome" in desktop:
                logger.debug("Trying GNOME backend...")
                data = ScreenCapture._capture_gnome()
                if data: return data

            # 3. Generic Wayland - grim
            logger.debug("Trying grim backend...")
            data = ScreenCapture._capture_grim()
            if data: return data

        # 4. Fallback to PyQt (works on X11, usually returns black on Wayland)
        logger.debug("Falling back to PyQt backend...")
        return ScreenCapture._capture_pyqt()

    @staticmethod
    def _capture_pyqt() -> Optional[bytes]:
        """Capture entire screen using PyQt (X11 only)"""
        try:
            screen = QGuiApplication.primaryScreen()
            if screen:
                pixmap = screen.grabWindow(0)
                if pixmap.isNull():
                    return None
                
                buffer = QBuffer()
                buffer.open(QIODevice.OpenModeFlag.WriteOnly)
                pixmap.save(buffer, "PNG")
                data = bytes(buffer.buffer())
                
                if ScreenCapture._is_image_empty(data):
                    logger.debug("PyQt capture returned empty/black image")
                    return None
                    
                return data
        except Exception as e:
            logger.debug(f"PyQt capture error: {e}")
        return None

    @staticmethod
    def _capture_spectacle() -> Optional[bytes]:
        """Capture screen using Spectacle (KDE)"""
        try:
            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
                tmp_path = tmp.name
            
            # -b: background, -n: no notification, -f: fullscreen, -o: output
            result = subprocess.run(
                ["spectacle", "-b", "-n", "-f", "-o", tmp_path],
                capture_output=True, timeout=5
            )
            
            if result.returncode == 0 and os.path.exists(tmp_path):
                with open(tmp_path, "rb") as f:
                    data = f.read()
                os.unlink(tmp_path)
                
                if not ScreenCapture._is_image_empty(data):
                    logger.debug("Captured screen via Spectacle")
                    return data
            
            if os.path.exists(tmp_path): os.unlink(tmp_path)
        except Exception as e:
            logger.debug(f"Spectacle capture error: {e}")
        return None

    @staticmethod
    def _capture_gnome() -> Optional[bytes]:
        """Capture screen using GNOME screenshot methods"""
        # Try gnome-screenshot CLI first
        try:
            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
                tmp_path = tmp.name
            
            result = subprocess.run(
                ["gnome-screenshot", "-f", "-file", tmp_path],
                capture_output=True, timeout=5
            )
            
            if result.returncode == 0 and os.path.exists(tmp_path):
                with open(tmp_path, "rb") as f:
                    data = f.read()
                os.unlink(tmp_path)
                return data
            
            if os.path.exists(tmp_path): os.unlink(tmp_path)
        except Exception:
            pass

        # Try DBus method for modern GNOME
        try:
            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
                tmp_path = tmp.name
            
            # Using dbus-send as a fallback to avoid requiring a dbus library
            # org.gnome.Shell.Screenshot.Screenshot(bool include_cursor, bool flash, string filename)
            subprocess.run([
                "dbus-send", "--session", "--type=method_call", 
                "--dest=org.gnome.Shell.Screenshot", 
                "/org/gnome/Shell/Screenshot", 
                "org.gnome.Shell.Screenshot.Screenshot", 
                "boolean:false", "boolean:false", f"string:{tmp_path}"
            ], capture_output=True, timeout=5)

            if os.path.exists(tmp_path):
                with open(tmp_path, "rb") as f:
                    data = f.read()
                os.unlink(tmp_path)
                return data
        except Exception as e:
            logger.debug(f"GNOME DBus capture error: {e}")
            
        return None

    @staticmethod
    def _capture_grim() -> Optional[bytes]:
        """Capture screen using grim (Generic Wayland)"""
        try:
            result = subprocess.run(["grim", "-"], capture_output=True, timeout=5)
            if result.returncode == 0:
                logger.debug("Captured screen via grim")
                return result.stdout
        except Exception as e:
            logger.debug(f"grim capture error: {e}")
        return None

    @staticmethod
    def _is_image_empty(data: bytes) -> bool:
        """Check if image is completely black or white (often happens on failed Wayland captures)"""
        if not data: return True
        img = QImage.fromData(data)
        if img.isNull(): return True
        
        # Check a few points (corners and center)
        w, h = img.width(), img.height()
        if w < 2 or h < 2: return True
        
        points = [
            img.pixelColor(0, 0),
            img.pixelColor(w-1, 0),
            img.pixelColor(0, h-1),
            img.pixelColor(w-1, h-1),
            img.pixelColor(w//2, h//2)
        ]
        
        first = points[0]
        return all(p == first for p in points)

    @staticmethod
    def capture_region(x: int, y: int, width: int, height: int) -> Optional[bytes]:
        """Capture specific screen region"""
        try:
            full_data = ScreenCapture.capture_screen()
            if not full_data:
                return None
                
            image = QImage.fromData(full_data)
            if image.isNull():
                return None
                
            # Crop to region
            rect = QRect(x, y, width, height)
            # Ensure rect is within image bounds
            rect = rect.intersected(image.rect())
            
            if rect.isEmpty():
                logger.warning(f"Requested region {x},{y} {width}x{height} is outside screen bounds")
                return None
                
            cropped = image.copy(rect)
            
            # Convert back to bytes
            buffer = QBuffer()
            buffer.open(QIODevice.OpenModeFlag.WriteOnly)
            cropped.save(buffer, "PNG")
            data = bytes(buffer.buffer())
            
            return data

        except Exception as e:
            logger.error(f"Region capture error: {e}")

        return None

    @staticmethod
    def preprocess_image(image_data: bytes) -> bytes:
        """Enhance image for better vision-language model processing results"""
        image = QImage.fromData(image_data)
        if image.isNull():
            return image_data

        # 1. Convert to Grayscale to simplify and improve contrast focus
        image = image.convertToFormat(QImage.Format.Format_Grayscale8)

        # 2. Simple contrast enhancement: Normalize
        # We find the min/max pixel values and stretch the range
        # Note: In a real app we might use something like histogram equalization,
        # but for VLMs, clear grayscale with good contrast is often enough.

        # This is a basic way to "normalize" using QImage if we don't want OpenCV
        # For efficiency, we just return the grayscale image for now which already helps

        buffer = QBuffer()
        buffer.open(QIODevice.OpenModeFlag.WriteOnly)
        image.save(buffer, "PNG")
        return bytes(buffer.buffer())

    @staticmethod
    def compress_image(image_data: bytes, quality: int = 50) -> Tuple[bytes, int, int]:
        """Compress image to JPEG and return (data, width, height)"""
        # 1. Load the image from provided data
        image = QImage.fromData(image_data)
        
        if image.isNull():
            logger.warning("Failed to load image for compression")
            return image_data, 0, 0

        # 2. Convert to RGB888 for consistent JPEG compression
        image = image.convertToFormat(QImage.Format.Format_RGB888)

        # 3. Save with compression to memory buffer
        buffer = QBuffer()
        buffer.open(QIODevice.OpenModeFlag.WriteOnly)
        image.save(buffer, "JPG", quality)
        
        return bytes(buffer.buffer()), image.width(), image.height()

    @staticmethod
    def calculate_hash(image_input) -> str:
        """Calculate a simple perceptual hash for change detection"""
        if isinstance(image_input, bytes):
            image = QImage.fromData(image_input)
        else:
            image = image_input

        if not image or image.isNull():
            return ""
        
        # Downsample to a very small size to ignore minor noise/flicker
        small = image.scaled(16, 16, Qt.AspectRatioMode.IgnoreAspectRatio, Qt.TransformationMode.FastTransformation)
        small = small.convertToFormat(QImage.Format.Format_Grayscale8)
        
        bits = []
        for y in range(16):
            for x in range(16):
                # Using a simple bitmask based on average pixel value
                bits.append(str(small.pixelColor(x, y).red()))
        
        return hashlib.md5(",".join(bits).encode()).hexdigest()
