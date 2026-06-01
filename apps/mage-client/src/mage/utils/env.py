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
    """Return a copy of os.environ with PyInstaller's and AppImage's library paths removed.
    
    This ensures spawned system subprocesses (like lemond, whisper-server, audio recorder,
    or screenshot tools) don't try to load bundled dynamic libraries from the AppImage
    or PyInstaller environment, avoiding Vulkan/GPU driver crashes and library conflicts.
    """
    env = os.environ.copy()
    
    # Identify mount and extraction folders to strip them
    appdir = env.get("APPDIR", "")
    meipass = getattr(sys, '_MEIPASS', "")
    
    # Process LD_LIBRARY_PATH (Linux)
    ld_path = env.get("LD_LIBRARY_PATH_ORIG") or env.get("LD_LIBRARY_PATH")
    if ld_path:
        parts = ld_path.split(os.pathsep)
        cleaned_parts = []
        for part in parts:
            if not part:
                continue
            is_bundled = False
            if appdir and part.startswith(appdir):
                is_bundled = True
            if meipass and part.startswith(meipass):
                is_bundled = True
            if "/.mount_" in part or "/_MEI" in part:
                is_bundled = True
                
            if not is_bundled:
                cleaned_parts.append(part)
                
        if cleaned_parts:
            env["LD_LIBRARY_PATH"] = os.pathsep.join(cleaned_parts)
        else:
            env.pop("LD_LIBRARY_PATH", None)
    else:
        env.pop("LD_LIBRARY_PATH", None)

    env.pop("LD_LIBRARY_PATH_ORIG", None)

    # Process DYLD_LIBRARY_PATH (macOS)
    dyld_path = env.get("DYLD_LIBRARY_PATH_ORIG") or env.get("DYLD_LIBRARY_PATH")
    if dyld_path:
        parts = dyld_path.split(os.pathsep)
        cleaned_parts = []
        for part in parts:
            if not part:
                continue
            is_bundled = False
            if appdir and part.startswith(appdir):
                is_bundled = True
            if meipass and part.startswith(meipass):
                is_bundled = True
            if "/.mount_" in part or "/_MEI" in part:
                is_bundled = True
                
            if not is_bundled:
                cleaned_parts.append(part)
                
        if cleaned_parts:
            env["DYLD_LIBRARY_PATH"] = os.pathsep.join(cleaned_parts)
        else:
            env.pop("DYLD_LIBRARY_PATH", None)
    else:
        env.pop("DYLD_LIBRARY_PATH", None)

    env.pop("DYLD_LIBRARY_PATH_ORIG", None)

    return env
