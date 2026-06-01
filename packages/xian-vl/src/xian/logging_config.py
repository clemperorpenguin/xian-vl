# Xian-VL — Core Vision-Language orchestration engine.
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

import logging
import sys

def setup_logger(name="xian", level=logging.INFO):
    """Set up and return a logger with a standard configuration"""
    if not name:
        name = "xian"
    elif name != "xian" and not name.startswith("xian."):
        name = f"xian.{name}"

    logger = logging.getLogger(name)
    
    # Avoid duplicate handlers if setup_logger is called multiple times
    if not logger.handlers:
        logger.setLevel(level)
        
        # Create console handler with a specific format
        handler = logging.StreamHandler(sys.stdout)
        handler.setLevel(level)
        
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%H:%M:%S'
        )
        handler.setFormatter(formatter)
        
        logger.addHandler(handler)
        
    return logger

# Create a default logger instance removed to avoid duplicate handlers on import
