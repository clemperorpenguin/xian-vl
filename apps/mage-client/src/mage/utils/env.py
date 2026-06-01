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

"""Environment utility functions."""

import os
import sys

def clean_subprocess_env() -> dict[str, str]:
    """Return a copy of os.environ with PyInstaller's library paths restored or removed.
    
    This ensures spawned system subprocesses (like lemond, audio recorder, or screenshot tools)
    don't try to load bundled PyInstaller dynamic libraries, avoiding crashes and driver conflicts.
    """
    env = os.environ.copy()
    
    # Restore Linux original library path if set by PyInstaller
    if "LD_LIBRARY_PATH_ORIG" in env:
        env["LD_LIBRARY_PATH"] = env["LD_LIBRARY_PATH_ORIG"]
        env.pop("LD_LIBRARY_PATH_ORIG", None)
    elif "LD_LIBRARY_PATH" in env:
        # If we are in a frozen bundle, but there was no _ORIG, it means LD_LIBRARY_PATH was NOT set in the parent system.
        if getattr(sys, 'frozen', False):
            env.pop("LD_LIBRARY_PATH", None)

    # Restore macOS original library path if set by PyInstaller
    if "DYLD_LIBRARY_PATH_ORIG" in env:
        env["DYLD_LIBRARY_PATH"] = env["DYLD_LIBRARY_PATH_ORIG"]
        env.pop("DYLD_LIBRARY_PATH_ORIG", None)
    elif "DYLD_LIBRARY_PATH" in env:
        if getattr(sys, 'frozen', False):
            env.pop("DYLD_LIBRARY_PATH", None)

    return env
