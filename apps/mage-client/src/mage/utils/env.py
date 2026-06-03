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
    """Return a copy of os.environ with PyInstaller's environment variables cleaned or removed.
    
    This ensures spawned system subprocesses (like lemond, whisper-server, spectacle, audio recorder,
    or screenshot tools) don't try to load bundled dynamic libraries, plugins, or configurations
    from the PyInstaller environment, avoiding crashes and driver conflicts.
    """
    env = os.environ.copy()
    
    # Identify PyInstaller extraction folder to strip from paths
    meipass = getattr(sys, '_MEIPASS', "")
    
    def clean_path_var(var_name: str, restore_orig: bool = False):
        val = env.get(var_name, "")
        if restore_orig:
            orig_name = f"{var_name}_ORIG"
            if orig_name in env:
                val = env[orig_name]
                env.pop(orig_name, None)
                
        if not val:
            env.pop(var_name, None)
            return
            
        parts = val.split(os.pathsep)
        cleaned_parts = []
        for part in parts:
            if not part:
                continue
            is_bundled = False
            # Normalize paths to forward slashes for cross-platform robustness
            normalized_part = part.replace('\\', '/')
            normalized_meipass = meipass.replace('\\', '/') if meipass else ""
            if normalized_meipass and normalized_part.startswith(normalized_meipass):
                is_bundled = True
            if "/_MEI" in normalized_part:
                is_bundled = True
                
            if not is_bundled:
                cleaned_parts.append(part)
                
        if cleaned_parts:
            env[var_name] = os.pathsep.join(cleaned_parts)
        else:
            env.pop(var_name, None)

    # 1. Clean library paths (restoring original if set by PyInstaller)
    clean_path_var("LD_LIBRARY_PATH", restore_orig=True)
    clean_path_var("DYLD_LIBRARY_PATH", restore_orig=True)
    
    # 2. Clean executable path
    clean_path_var("PATH")
    
    # 3. Clean Qt plugin path
    clean_path_var("QT_PLUGIN_PATH")
    
    # 4. Clean XDG data directories (crucial for Vulkan ICD configuration files)
    clean_path_var("XDG_DATA_DIRS")
    
    # 5. Remove Python environment overrides completely if they point to the bundle
    for py_var in ["PYTHONHOME", "PYTHONPATH"]:
        val = env.get(py_var, "")
        if val:
            is_bundled = False
            normalized_val = val.replace('\\', '/')
            normalized_meipass = meipass.replace('\\', '/') if meipass else ""
            if normalized_meipass and normalized_meipass in normalized_val:
                is_bundled = True
            if "/_MEI" in normalized_val:
                is_bundled = True
            if is_bundled:
                env.pop(py_var, None)

    return env
