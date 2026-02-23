#!/usr/bin/env python3
"""
Xian - Real-time Video Game Translation Overlay
A PyQt6-based translation overlay for Linux Wayland KDE Plasma
"""

import sys
from PyQt6.QtWidgets import QApplication
from PyQt6.QtGui import QIcon
from xian.main_window import MainWindow
from xian.screen_capture import SCREENSHOT_AVAILABLE
from xian.logging_config import setup_logger
import logging

def main():
    """Main application entry point"""
    # Initialize logging
    setup_logger(level=logging.DEBUG)
    
    app = QApplication(sys.argv)
    app.setWindowIcon(QIcon("xian.png"))

    # Check for required dependencies
    if not SCREENSHOT_AVAILABLE:
        logging.warning("Screenshot dependencies not available")

    # Create and show main window
    window = MainWindow()
    # Launch directly into the overlay control panel; keep legacy window hidden by default
    window.show_overlay_settings_panel()

    sys.exit(app.exec())

if __name__ == "__main__":
    main()
