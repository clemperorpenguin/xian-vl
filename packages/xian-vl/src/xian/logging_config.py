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
import os
import sys


def resolve_log_level(default=logging.INFO):
    """Resolve the desired log level from the environment.

    Honours ``XIAN_LOG_LEVEL`` (e.g. ``DEBUG``, ``INFO``, ``WARNING``, or a
    numeric level). Defaults to INFO so normal runs stay readable; set
    ``XIAN_LOG_LEVEL=DEBUG`` for the full firehose when diagnosing issues.
    """
    raw = os.environ.get("XIAN_LOG_LEVEL")
    if not raw:
        return default
    raw = raw.strip()
    if raw.isdigit():
        return int(raw)
    return logging.getLevelName(raw.upper()) if isinstance(
        logging.getLevelName(raw.upper()), int
    ) else default


def setup_logger(name="xian", level=None):
    """Set up and return a logger with a standard configuration"""
    if level is None:
        level = resolve_log_level()
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

    _quiet_noisy_dependencies(level)

    return logger


def _quiet_noisy_dependencies(level):
    """Hold chatty third-party loggers at WARNING unless we're in DEBUG.

    httpx logs an INFO line per request and httpcore/urllib3 emit verbose
    connection traces — useful when debugging, pure noise otherwise.
    """
    if level <= logging.DEBUG:
        return
    for noisy in ("httpx", "httpcore", "urllib3", "openai", "asyncio"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

# Create a default logger instance removed to avoid duplicate handlers on import
