#!/usr/bin/env python3
"""Build script for MAGE using PyInstaller."""

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

def build(include_lemonade: bool, lemonade_dir: Path | None):
    base_dir = Path(__file__).resolve().parent.parent
    app_dir = base_dir / "apps" / "mage-client"
    
    if not (app_dir / "src" / "mage" / "main.py").exists():
        print(f"Error: Could not find main.py at {app_dir}")
        sys.exit(1)
        
    # Build with PyInstaller
    print("Running PyInstaller...")
    cmd = [
        sys.executable, "-m", "PyInstaller",
        "--name", "mage-client",
        "--windowed",
        "--onedir",
        "--icon", str(base_dir / "xian.png"),
        "--add-data", f"{base_dir / 'xian.png'}:.",
        "--noconfirm",
        str(app_dir / "src" / "mage" / "main.py")
    ]
    
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
        else:
            print(f"Warning: {src_lemond} not found")
            
        # Copy lemonade cli binary (optional but good to have)
        cli_name = "lemonade.exe" if sys.platform == "win32" else "lemonade"
        src_cli = lemonade_dir / cli_name
        if src_cli.exists():
            shutil.copy2(src_cli, bundle_dir)
            
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
        dmg_name = "mage-client-full" if include_lemonade else "mage-client-lite"
        dmg_path = app_dir / f"{dmg_name}.dmg"
        print(f"Creating {dmg_path}...")
        if shutil.which("create-dmg"):
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
            except subprocess.CalledProcessError as e:
                print(f"DMG creation failed with exit code {e.returncode}", file=sys.stderr)
                sys.exit(1)
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
