import os
import sys

def get_resource_path(filename: str) -> str:
    """Get the absolute path to a resource, supporting both development and PyInstaller modes."""
    if getattr(sys, 'frozen', False):
        # PyInstaller creates a temporary folder and stores its path in _MEIPASS
        base_path = getattr(sys, '_MEIPASS', os.path.dirname(sys.executable))
    else:
        # In development, the resource is in the root of the workspace.
        # This file is in apps/mage-client/src/mage/resources.py, so we go up 4 levels to get the workspace root.
        this_dir = os.path.dirname(os.path.abspath(__file__))
        base_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(this_dir))))

    path = os.path.join(base_path, filename)
    if os.path.exists(path):
        return path
    
    # Fallback to current working directory
    return filename
