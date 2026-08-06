# MAGE — Gaming HUD for real-time screen translation.
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

from __future__ import annotations

import logging
import os
import subprocess
import sys
import tempfile

from mage.utils.env import clean_subprocess_env

from PyQt6.QtGui import QImage, QGuiApplication, QPixmap, QPainter, QColor
from PyQt6.QtCore import QBuffer, QIODevice, QRect

logger = logging.getLogger(__name__)

__all__ = ["ScreenCapture"]


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
    def capture_screen() -> bytes | None:
        """Capture entire screen using best available method"""

        # Try Wayland-specific methods first if on Wayland (Linux only)
        if sys.platform == "linux":
            is_wayland = os.environ.get("XDG_SESSION_TYPE") == "wayland"
            desktop = os.environ.get("XDG_CURRENT_DESKTOP", "").lower()

            if is_wayland:
                logger.debug("Wayland detected, desktop: %s", desktop)
                # 1. KDE Plasma - Spectacle
                if "kde" in desktop or "plasma" in desktop:
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

        # Fallback to PyQt (works on X11, Windows, and macOS)
        logger.debug("Using PyQt backend...")
        data = ScreenCapture._capture_pyqt()

        if data:
            return data

        # Platform-specific guidance when capture fails
        if sys.platform == "darwin":
            logger.warning(
                "Screen capture returned empty on macOS. "
                "Please grant Screen Recording permission: "
                "System Settings → Privacy & Security → Screen Recording. "
                "You must add and enable the terminal or app running Xian-VL, "
                "then restart it."
            )
        else:
            logger.warning("Screen capture returned empty")

        return None

    @staticmethod
    def capture_region(rect: QRect) -> tuple[bytes | None, bool]:
        """Capture a single region of the desktop.

        Returns ``(png_bytes, already_cropped)``.  When ``already_cropped`` is
        False the caller still has to crop the returned full-desktop frame —
        only grim can capture a sub-rectangle directly, and it is the fast path
        that continuous modes depend on (no full-desktop encode, no temp file).
        """
        if rect is None or rect.isEmpty():
            return None, False

        if sys.platform == "linux" and os.environ.get("XDG_SESSION_TYPE") == "wayland":
            data = ScreenCapture._capture_grim_region(rect)
            if data:
                return data, True

        return ScreenCapture.capture_screen(), False

    @staticmethod
    def _capture_grim_region(rect: QRect) -> bytes | None:
        """Capture a sub-rectangle using grim's geometry flag."""
        geometry = f"{rect.x()},{rect.y()} {rect.width()}x{rect.height()}"
        try:
            result = subprocess.run(
                ["grim", "-g", geometry, "-"],
                capture_output=True, env=clean_subprocess_env(), timeout=5,
            )
            if result.returncode == 0 and result.stdout:
                # No all-black check here, unlike the full-desktop path. That
                # heuristic exists because some compositors return a black
                # frame instead of failing; grim reports failure with a
                # non-zero exit code. Applying it to regions would reject
                # legitimately dark HUD panels and silently drop every frame
                # back to a full-desktop capture — losing the fast path
                # continuous translation depends on.
                logger.debug("Captured region %s via grim", geometry)
                return result.stdout
        except FileNotFoundError:
            logger.debug("grim not installed; region capture unavailable")
        except Exception as e:
            logger.debug("grim region capture error: %s", e)
        return None

    @staticmethod
    def _capture_pyqt() -> bytes | None:
        """Capture entire virtual desktop using PyQt (X11, Windows, macOS).

        Composites all screens into a single image so multi-monitor setups
        are fully captured.
        """
        try:
            screens = QGuiApplication.screens()
            if not screens:
                return None

            total_geo = ScreenCapture.get_virtual_desktop_geometry()

            combined = QPixmap(total_geo.width(), total_geo.height())
            combined.fill(QColor(0, 0, 0))

            painter = QPainter(combined)
            for screen in screens:
                geo = screen.geometry()
                pixmap = screen.grabWindow(0)
                if not pixmap.isNull():
                    painter.drawPixmap(
                        geo.x() - total_geo.x(),
                        geo.y() - total_geo.y(),
                        pixmap,
                    )
            painter.end()

            if combined.isNull():
                return None

            buffer = QBuffer()
            buffer.open(QIODevice.OpenModeFlag.WriteOnly)
            combined.save(buffer, "PNG")
            data = bytes(buffer.buffer())

            if ScreenCapture._is_image_empty(data):
                logger.debug("PyQt capture returned empty/black image")
                return None

            return data
        except Exception as e:
            logger.debug("PyQt capture error: %s", e)
        return None

    @staticmethod
    def _capture_spectacle() -> bytes | None:
        """Capture screen using Spectacle (KDE)"""
        try:
            with tempfile.TemporaryDirectory() as tmpdir:
                tmp_path = os.path.join(tmpdir, "capture.png")

                # -b: background, -n: no notification, -f: fullscreen, -o: output
                result = subprocess.run(
                    ["spectacle", "-b", "-n", "-f", "-o", tmp_path],
                    capture_output=True, env=clean_subprocess_env(), timeout=5
                )

                if result.returncode == 0 and os.path.exists(tmp_path):
                    with open(tmp_path, "rb") as f:
                        data = f.read()

                    if not ScreenCapture._is_image_empty(data):
                        logger.debug("Captured screen via Spectacle")
                        return data
        except Exception as e:
            logger.debug("Spectacle capture error: %s", e)
        return None

    @staticmethod
    def _capture_gnome() -> bytes | None:
        """Capture screen using GNOME screenshot methods"""
        # Try gnome-screenshot CLI first
        try:
            with tempfile.TemporaryDirectory() as tmpdir:
                tmp_path = os.path.join(tmpdir, "capture.png")
                result = subprocess.run(
                    ["gnome-screenshot", "--file", tmp_path],
                    capture_output=True, env=clean_subprocess_env(), timeout=5
                )

                if result.returncode == 0 and os.path.exists(tmp_path):
                    with open(tmp_path, "rb") as f:
                        data = f.read()
                    if not ScreenCapture._is_image_empty(data):
                        return data
        except Exception:
            pass

        # Try DBus method for modern GNOME
        try:
            with tempfile.TemporaryDirectory() as tmpdir:
                tmp_path = os.path.join(tmpdir, "capture.png")
                # Using dbus-send as a fallback to avoid requiring a dbus library
                # org.gnome.Shell.Screenshot.Screenshot(bool include_cursor, bool flash, string filename)
                result = subprocess.run([
                    "dbus-send", "--session", "--type=method_call",
                    "--dest=org.gnome.Shell.Screenshot",
                    "/org/gnome/Shell/Screenshot",
                    "org.gnome.Shell.Screenshot.Screenshot",
                    "boolean:false", "boolean:false", f"string:{tmp_path}"
                ], capture_output=True, env=clean_subprocess_env(), timeout=5)

                if result.returncode == 0 and os.path.exists(tmp_path):
                    with open(tmp_path, "rb") as f:
                        data = f.read()
                    if not ScreenCapture._is_image_empty(data):
                        return data
        except Exception as e:
            logger.debug("GNOME DBus capture error: %s", e)

        return None

    @staticmethod
    def _capture_grim() -> bytes | None:
        """Capture screen using grim (Generic Wayland)"""
        try:
            result = subprocess.run(["grim", "-"], capture_output=True, env=clean_subprocess_env(), timeout=5)
            if result.returncode == 0:
                logger.debug("Captured screen via grim")
                return result.stdout
        except Exception as e:
            logger.debug("grim capture error: %s", e)
        return None

    @staticmethod
    def _is_image_empty(data: bytes) -> bool:
        """Check if image is completely black or white (often happens on failed Wayland captures)"""
        if not data: return True
        img = QImage.fromData(data)
        if img.isNull(): return True

        # Sample a grid of up to 5x5 points. Region captures can be narrower
        # than the grid, so the step count adapts rather than rejecting them.
        w, h = img.width(), img.height()
        if w < 1 or h < 1: return True

        cols, rows = min(5, w), min(5, h)
        points = []
        for i in range(cols):
            for j in range(rows):
                x = int(i * (w - 1) / (cols - 1)) if cols > 1 else 0
                y = int(j * (h - 1) / (rows - 1)) if rows > 1 else 0
                points.append(img.pixelColor(x, y))

        # Only reject if all sampled points are truly black or white
        black = QColor(0, 0, 0)
        white = QColor(255, 255, 255)
        return all(p.rgb() == black.rgb() for p in points) or all(p.rgb() == white.rgb() for p in points)
