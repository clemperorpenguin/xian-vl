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

    app = QApplication(sys.argv)
    app.setOrganizationName(ORGANIZATION_NAME)
    app.setApplicationName(APPLICATION_NAME)
    app.setQuitOnLastWindowClosed(False)
    app.setWindowIcon(QIcon("xian.png"))

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
