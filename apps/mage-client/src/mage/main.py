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

"""MAGE entry point.

Boots the PyQt6 application, sets up the system tray, and hands off
to the XianApp controller.
"""

import logging
import os
import sys

from PyQt6.QtWidgets import QApplication
from PyQt6.QtGui import QIcon

from shared_types.constants import APPLICATION_NAME, ORGANIZATION_NAME
from xian.logging_config import setup_logger


import signal
from PyQt6.QtCore import QTimer

logger = logging.getLogger(__name__)


def _prefer_xwayland_on_wayland() -> None:
    """Route MAGE through XWayland when running in a native Wayland session.

    Wayland forbids clients from positioning their own top-level windows, so
    ``QWidget.move()`` is silently ignored and the compositor centres every
    overlay (the HUD bubbles, and most visibly the desktop familiar). MAGE's
    entire overlay stack — global positioning, stay-on-top, and the X11 window
    binder — is built for the ``xcb`` backend, so we default to it under
    XWayland, where all of that works as designed.

    This is a no-op unless we are on Linux, in a Wayland session, with an X
    display actually available (XWayland present), and the user has not already
    pinned ``QT_QPA_PLATFORM`` themselves. Set ``QT_QPA_PLATFORM`` to opt out.
    Must run before ``QApplication`` is constructed, since the platform plugin
    is chosen at that point.
    """
    if not sys.platform.startswith("linux"):
        return
    if os.environ.get("QT_QPA_PLATFORM"):
        return  # user (or launcher) already chose a backend — respect it
    if os.environ.get("XDG_SESSION_TYPE") != "wayland":
        return
    if not os.environ.get("DISPLAY"):
        # No X display => XWayland is not available; stay on the wayland plugin
        # rather than forcing an xcb backend that would fail to load.
        return
    os.environ["QT_QPA_PLATFORM"] = "xcb"
    logger.info(
        "Native Wayland session detected; routing through XWayland "
        "(QT_QPA_PLATFORM=xcb) so overlays can position correctly. "
        "Set QT_QPA_PLATFORM yourself to override."
    )


def main() -> None:
    """Launch the MAGE Gaming HUD."""
    setup_logger(level=logging.DEBUG)
    _prefer_xwayland_on_wayland()
    
    # Configure logging for the 'mage' namespace so logs print to console
    mage_logger = logging.getLogger("mage")
    mage_logger.setLevel(logging.DEBUG)
    if not mage_logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setLevel(logging.DEBUG)
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%H:%M:%S'
        )
        handler.setFormatter(formatter)
        mage_logger.addHandler(handler)

    from mage.resources import get_resource_path

    app = QApplication(sys.argv)
    app.setOrganizationName(ORGANIZATION_NAME)
    app.setApplicationName(APPLICATION_NAME)
    app.setQuitOnLastWindowClosed(False)
    app.setWindowIcon(QIcon(get_resource_path("xian.png")))

    # Allow Ctrl+C to terminate the application
    signal.signal(signal.SIGINT, lambda *args: QApplication.quit())

    # Periodically "poke" the event loop so Python can process signals
    timer = QTimer()
    timer.start(500)
    timer.timeout.connect(lambda: None)

    # Defer heavy imports until the event loop is ready
    from mage.app import XianApp  # noqa: E402
    from mage.lemond_manager import start_lemond_if_embedded, stop_lemond

    # Start embeddable lemonade if it exists in the bundle
    start_lemond_if_embedded()

    # Bound to a name so the app object isn't garbage-collected during exec().
    xian = XianApp()  # noqa: F841

    exit_code = app.exec()
    
    # Gracefully stop the server on exit
    stop_lemond()
    
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
