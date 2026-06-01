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
