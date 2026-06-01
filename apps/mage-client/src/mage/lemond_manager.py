"""Manager for the embedded Lemonade server."""

import os
import sys
import subprocess
import logging
import atexit
from pathlib import Path

logger = logging.getLogger(__name__)

_lemond_process = None

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
    global _lemond_process
    
    exe_path = get_lemond_executable()
    if not exe_path:
        logger.debug("No embedded lemond found at %s", get_base_dir())
        return
        
    if _lemond_process is not None:
        logger.debug("lemond is already running")
        return
        
    logger.info("Found embedded lemond at %s. Starting it...", exe_path)
    
    try:
        # Start lemond with the base directory as an argument (or working directory)
        # The embeddable guide says: `lemond ./` places files in the same directory
        # Let's use the base directory so it generates config.json there.
        _lemond_process = subprocess.Popen(
            [str(exe_path), str(get_base_dir())],
            cwd=str(get_base_dir()),
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        logger.info("Embedded lemond started with PID %d", _lemond_process.pid)
        
        # Register atexit handler to ensure it gets killed
        atexit.register(stop_lemond)
    except Exception as e:
        logger.error("Failed to start embedded lemond: %s", e)

def stop_lemond():
    """Stop the embedded lemond server."""
    global _lemond_process
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
