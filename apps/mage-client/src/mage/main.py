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
import sys

from PyQt6.QtWidgets import QApplication
from PyQt6.QtGui import QIcon

from shared_types.constants import APPLICATION_NAME, ORGANIZATION_NAME
from xian.logging_config import setup_logger


import signal
from PyQt6.QtCore import QTimer

def main() -> None:
    """Launch the MAGE Gaming HUD."""
    setup_logger(level=logging.DEBUG)

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

    xian = XianApp()
    
    exit_code = app.exec()
    
    # Gracefully stop the server on exit
    stop_lemond()
    
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
