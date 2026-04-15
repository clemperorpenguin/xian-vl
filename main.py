#!/usr/bin/env python3
"""
Xian - Real-time Video Game Translation Overlay
A PyQt6-based translation overlay for Linux Wayland KDE Plasma
"""

import os
import sys

# Set CPU threading limits BEFORE torch is imported
import multiprocessing
num_cores = max(4, multiprocessing.cpu_count() - 1) 
os.environ["OMP_NUM_THREADS"] = str(num_cores)
os.environ["MKL_NUM_THREADS"] = str(num_cores)
os.environ["OPENBLAS_NUM_THREADS"] = str(num_cores)
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["ATEN_CPU_CAPABILITY"] = "DEFAULT"

from PyQt6.QtWidgets import QApplication
from PyQt6.QtGui import QIcon
from xian.main_window import MainWindow
from xian.screen_capture import SCREENSHOT_AVAILABLE
from xian.logging_config import setup_logger
import logging

# Disable MKLDNN which crashes on aarch64
try:
    import torch
    torch.backends.mkldnn.enabled = False
    torch.set_num_threads(num_cores)
    # Set default device to CPU
    if hasattr(torch, 'set_default_device'):
        torch.set_default_device('cpu')
    logging.getLogger(__name__).info(f"MKLDNN disabled, torch threads set to {num_cores}")
except Exception as e:
    pass

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
