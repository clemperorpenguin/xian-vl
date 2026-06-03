#!/usr/bin/env python3
# Xian-VL Scripts — Development and automation scripts.
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

"""Build script for MAGE using PyInstaller."""

import argparse
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path


def _generate_icon(png_path: Path, out_dir: Path) -> str | None:
    """Convert xian.png to the platform-native icon format.

    Returns the path to the converted icon, or None if conversion fails
    (in which case PyInstaller will use its default icon).
    """
    try:
        from PIL import Image
    except ImportError:
        print("Warning: Pillow not available — skipping icon conversion.", file=sys.stderr)
        return None

    if sys.platform == "darwin":
        # macOS requires .icns
        icns_path = out_dir / "xian.icns"
        try:
            img = Image.open(png_path).convert("RGBA")
            # .icns needs specific sizes; save the 512×512 and let macOS pick
            img = img.resize((512, 512), Image.LANCZOS)
            img.save(str(icns_path), format="ICNS")
            print(f"Generated macOS icon: {icns_path}")
            return str(icns_path)
        except Exception as e:
            print(f"Warning: Failed to generate .icns icon: {e}", file=sys.stderr)
            return None

    elif sys.platform == "win32":
        # Windows requires .ico
        ico_path = out_dir / "xian.ico"
        try:
            img = Image.open(png_path).convert("RGBA")
            # .ico supports multiple sizes; include common ones
            sizes = [(16, 16), (32, 32), (48, 48), (64, 64), (128, 128), (256, 256)]
            resized = [img.resize(s, Image.LANCZOS) for s in sizes]
            resized[0].save(str(ico_path), format="ICO", sizes=sizes, append_images=resized[1:])
            print(f"Generated Windows icon: {ico_path}")
            return str(ico_path)
        except Exception as e:
            print(f"Warning: Failed to generate .ico icon: {e}", file=sys.stderr)
            return None

    # Linux / other — PNG is fine
    return str(png_path)


def build(include_lemonade: bool, lemonade_dir: Path | None):
    base_dir = Path(__file__).resolve().parent.parent
    app_dir = base_dir / "apps" / "mage-client"
    
    if not (app_dir / "src" / "mage" / "main.py").exists():
        print(f"Error: Could not find main.py at {app_dir}")
        sys.exit(1)

    # Clean previous build and dist directories to prevent artifact bleeding
    build_dir = app_dir / "build"
    dist_dir = app_dir / "dist"
    if build_dir.exists():
        print(f"Cleaning build directory: {build_dir}")
        shutil.rmtree(build_dir)
    if dist_dir.exists():
        print(f"Cleaning dist directory: {dist_dir}")
        shutil.rmtree(dist_dir)

    # ── Icon conversion ──────────────────────────────────────────────────
    icon_tmp = Path(tempfile.mkdtemp(prefix="mage-icon-"))
    icon_path = _generate_icon(base_dir / "xian.png", icon_tmp)

    # ── Paths to bundle ──────────────────────────────────────────────────
    locales_dir = base_dir / "packages" / "shared-types" / "locales"

    # Build with PyInstaller
    print("Running PyInstaller...")
    cmd = [
        sys.executable, "-m", "PyInstaller",
        "--name", "mage-client",
        "--windowed",
        "--onedir",
        # --- Data files ---
        "--add-data", f"{base_dir / 'xian.png'}{os.pathsep}.",
        "--add-data", f"{locales_dir}{os.pathsep}locales",
        # --- Hidden imports (dynamic / conditional imports PyInstaller can't trace) ---
        "--hidden-import", "pynput.keyboard",
        "--hidden-import", "pynput.mouse",
        "--hidden-import", "pynput._util",
        "--hidden-import", "httpx",
        "--hidden-import", "httpx._transports",
        "--hidden-import", "httpx._transports.default",
        "--hidden-import", "openai",
        "--hidden-import", "yaml",
        "--hidden-import", "imagehash",
        "--hidden-import", "numpy",
        "--hidden-import", "cv2",
        "--hidden-import", "lmdb",
        # --- Collect-all for packages with compiled extensions / complex internals ---
        "--collect-all", "pydantic",
        "--collect-all", "pydantic_core",
        "--noconfirm",
        str(app_dir / "src" / "mage" / "main.py")
    ]

    # Insert --icon only if we have a valid converted icon
    if icon_path:
        cmd.insert(cmd.index("--noconfirm"), "--icon")
        cmd.insert(cmd.index("--noconfirm"), icon_path)

    
    # Run PyInstaller from app_dir
    try:
        subprocess.run(cmd, cwd=app_dir, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Build failed with PyInstaller exit code {e.returncode}", file=sys.stderr)
        sys.exit(1)
    
    dist_dir = app_dir / "dist"
    
    # Determine the actual app directory where our executable lives
    if sys.platform == "darwin":
        bundle_dir = dist_dir / "mage-client.app" / "Contents" / "MacOS"
        root_app = dist_dir / "mage-client.app"
    else:
        bundle_dir = dist_dir / "mage-client"
        root_app = bundle_dir
        
    if include_lemonade:
        if not lemonade_dir or not lemonade_dir.exists():
            print(f"Error: Lemonade directory {lemonade_dir} does not exist.")
            sys.exit(1)
            
        print(f"Copying embeddable Lemonade from {lemonade_dir} to {bundle_dir}...")
        
        # Copy lemond binary
        lemond_name = "lemond.exe" if sys.platform == "win32" else "lemond"
        src_lemond = lemonade_dir / lemond_name
        if src_lemond.exists():
            shutil.copy2(src_lemond, bundle_dir)
            if sys.platform != "win32":
                (bundle_dir / lemond_name).chmod(0o755)
        else:
            print(f"Warning: {src_lemond} not found")
            
        # Copy lemonade cli binary (optional but good to have)
        cli_name = "lemonade.exe" if sys.platform == "win32" else "lemonade"
        src_cli = lemonade_dir / cli_name
        if src_cli.exists():
            shutil.copy2(src_cli, bundle_dir)
            if sys.platform != "win32":
                (bundle_dir / cli_name).chmod(0o755)
            
        # Copy resources
        src_res = lemonade_dir / "resources"
        if src_res.exists():
            dest_res = bundle_dir / "resources"
            if dest_res.exists():
                shutil.rmtree(dest_res)
            shutil.copytree(src_res, dest_res)
            
    print("Build complete.")
    
    if sys.platform == "win32":
        # Create a zip file
        zip_name = "mage-client-full" if include_lemonade else "mage-client-lite"
        zip_path = app_dir / f"{zip_name}.zip"
        print(f"Creating {zip_path}...")
        shutil.make_archive(str(app_dir / zip_name), 'zip', dist_dir, "mage-client")
        print(f"Generated {zip_path}")
        
    elif sys.platform == "darwin":
        archive_name = "mage-client-full" if include_lemonade else "mage-client-lite"
        zip_path = app_dir / f"{archive_name}.zip"
        print(f"Creating zip archive {zip_path}...")
        shutil.make_archive(str(app_dir / archive_name), 'zip', dist_dir, "mage-client.app")
        print(f"Generated {zip_path}")

        dmg_path = app_dir / f"{archive_name}.dmg"
        if shutil.which("create-dmg"):
            print(f"Creating {dmg_path}...")
            if dmg_path.exists():
                dmg_path.unlink()
            try:
                subprocess.run([
                    "create-dmg",
                    "--volname", "MAGE",
                    "--window-pos", "200", "120",
                    "--window-size", "800", "400",
                    "--icon-size", "100",
                    "--icon", "mage-client.app", "200", "190",
                    "--hide-extension", "mage-client.app",
                    "--app-drop-link", "600", "185",
                    str(dmg_path),
                    str(root_app)
                ], cwd=app_dir, check=True)
                print(f"Generated {dmg_path}")
            except subprocess.CalledProcessError as e:
                print(f"DMG creation failed with exit code {e.returncode}. Proceeding with ZIP archive.", file=sys.stderr)
        else:
            print("create-dmg not found. Skipping DMG creation.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build MAGE Client")
    parser.add_argument("--include-lemonade", action="store_true", help="Include embeddable Lemonade")
    parser.add_argument("--lemonade-dir", type=Path, help="Path to extracted embeddable Lemonade")
    args = parser.parse_args()
    
    if args.include_lemonade and not args.lemonade_dir:
        parser.error("--lemonade-dir is required when --include-lemonade is set")
        
    build(args.include_lemonade, args.lemonade_dir)
