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


def main() -> None:
    """Launch the MAGE Gaming HUD."""
    setup_logger(level=logging.DEBUG)

    app = QApplication(sys.argv)
    app.setOrganizationName(ORGANIZATION_NAME)
    app.setApplicationName(APPLICATION_NAME)
    app.setQuitOnLastWindowClosed(False)
    app.setWindowIcon(QIcon("xian.png"))

    # Defer heavy imports until the event loop is ready
    from mage.app import XianApp  # noqa: E402

    xian = XianApp()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
