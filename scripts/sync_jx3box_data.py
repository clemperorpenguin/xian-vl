import os
import subprocess
import shutil
from pathlib import Path

def sync_data():
    project_root = Path(__file__).parent.parent
    target_dir = project_root / "xian" / "knowledge" / "jx3box-data"
    repo_url = "https://github.com/JX3BOX/jx3box-data.git"

    print(f"Syncing JX3Box Data to {target_dir}...")
    
    if target_dir.exists():
        print("Directory exists. Removing to fetch a fresh shallow copy...")
        shutil.rmtree(target_dir)
        
    subprocess.run(
        ["git", "clone", "--depth", "1", repo_url, str(target_dir)],
        check=True
    )
    print("Sync complete.")

if __name__ == "__main__":
    sync_data()
