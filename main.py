#!/usr/bin/env python3
"""
Xian-VL — Stateful Wayland Assistant
A tray-resident Lens & Chat assistant powered by Lemonade-SDK.
"""

import sys
import logging

from PyQt6.QtWidgets import QApplication
from PyQt6.QtGui import QIcon
from xian.app import XianApp
from xian.logging_config import setup_logger


def main():
    """Main application entry point."""
    setup_logger(level=logging.DEBUG)

    app = QApplication(sys.argv)
    app.setApplicationName("Xian-VL")
    app.setQuitOnLastWindowClosed(False)  # Keep running in tray
    app.setWindowIcon(QIcon("xian.png"))

    xian = XianApp()  # noqa: F841 — prevent garbage collection

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
