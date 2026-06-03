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

"""Manager for the embedded Lemonade server."""

import os
import sys
import subprocess
import logging
import atexit
from pathlib import Path

from mage.utils.env import clean_subprocess_env

logger = logging.getLogger("xian.mage.lemond_manager")

_lemond_process = None
_lemond_log_file = None

def get_base_dir() -> Path:
    """Return the application base directory."""
    if getattr(sys, 'frozen', False):
        # If bundled with PyInstaller, use the directory containing the executable
        return Path(sys.executable).parent
    # For development, just use the current working directory or a specific known path
    # In practice, embedded lemonade is only packaged for frozen builds
    return Path(os.getcwd())

def get_lemond_executable() -> Path | None:
    """Find the lemond executable if it exists in the base directory."""
    base_dir = get_base_dir()
    
    # Check for Windows .exe or Unix binary
    for name in ["lemond.exe", "lemond"]:
        exe_path = base_dir / name
        if exe_path.exists() and exe_path.is_file():
            return exe_path
    
    return None

def start_lemond_if_embedded():
    """Start the embedded lemond server if it exists."""
    global _lemond_process, _lemond_log_file
    
    exe_path = get_lemond_executable()
    if not exe_path:
        logger.debug("No embedded lemond found at %s", get_base_dir())
        return
        
    if _lemond_process is not None:
        logger.debug("lemond is already running")
        return
        
    logger.info("Found embedded lemond at %s. Starting it...", exe_path)
    
    try:
        # Start lemond and redirect stdout/stderr to a log file in the cache directory
        log_dir = Path.home() / ".cache" / "lemonade"
        log_dir.mkdir(parents=True, exist_ok=True)
        log_file_path = log_dir / "mage_lemond.log"
        _lemond_log_file = open(log_file_path, "w", encoding="utf-8")

        # Start lemond without passing the read-only get_base_dir() as cache_dir.
        # Let lemond choose its default user-writable cache directory (~/.cache/lemonade).
        # We run it with cwd set to a writable home directory.
        _lemond_process = subprocess.Popen(
            [str(exe_path), "--port", "13305"],
            cwd=str(Path.home()),
            stdout=_lemond_log_file,
            stderr=subprocess.STDOUT,
            env=clean_subprocess_env(),
        )
        logger.info("Embedded lemond started with PID %d", _lemond_process.pid)
        
        # Start socket polling loop in a background thread to prevent blocking main thread
        import threading
        def wait_for_port():
            import socket
            import time
            logger.info("Waiting for embedded lemond to start on port 13305...")
            for i in range(20): # Up to 10 seconds (20 * 0.5s)
                if _lemond_process is None or _lemond_process.poll() is not None:
                    code = _lemond_process.poll() if _lemond_process else "unknown"
                    logger.error("Embedded lemond exited prematurely with code %s", code)
                    break
                try:
                    with socket.create_connection(("127.0.0.1", 13305), timeout=0.5):
                        logger.info("Embedded lemond is active and responding on port 13305.")
                        break
                except (ConnectionRefusedError, socket.timeout):
                    time.sleep(0.5)
            else:
                logger.warning("Timed out waiting for embedded lemond to respond on port 13305.")
                
        thread = threading.Thread(target=wait_for_port, name="lemond-startup-waiter", daemon=True)
        thread.start()
        
        # Register atexit handler to ensure it gets killed
        atexit.register(stop_lemond)
    except Exception as e:
        logger.error("Failed to start embedded lemond: %s", e)

def stop_lemond():
    """Stop the embedded lemond server."""
    global _lemond_process, _lemond_log_file
    if _lemond_process is not None:
        logger.info("Stopping embedded lemond (PID %d)...", _lemond_process.pid)
        try:
            _lemond_process.terminate()
            _lemond_process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            logger.warning("lemond did not terminate gracefully, killing it.")
            _lemond_process.kill()
        except Exception as e:
            logger.error("Error stopping lemond: %s", e)
        finally:
            _lemond_process = None
            
    if _lemond_log_file is not None:
        try:
            _lemond_log_file.close()
        except Exception:
            pass
        _lemond_log_file = None
