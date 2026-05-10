import os
import subprocess
import shutil
import tempfile
from pathlib import Path

def sync_data():
    project_root = Path(__file__).parent.parent
    target_dir = project_root / "xian" / "knowledge" / "jx3box-data"
    repo_url = "https://github.com/JX3BOX/jx3box-data.git"

    print(f"Syncing JX3Box Data to {target_dir}...")
    
    # Use a safer pattern: clone to a temp dir first, then replace the old target.
    # This prevents leaving the directory empty if the clone fails.
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_target = Path(tmp_dir) / "jx3box-data"
        print(f"Cloning to temporary directory {tmp_target}...")
        
        try:
            subprocess.run(
                ["git", "clone", "--depth", "1", repo_url, str(tmp_target)],
                check=True
            )
        except subprocess.CalledProcessError as e:
            print(f"Failed to clone repository: {e}")
            print("Aborting sync. Old data preserved.")
            return
            
        if target_dir.exists():
            print("Removing old data...")
            shutil.rmtree(target_dir)
            
        # Move the cloned repo to the final destination
        # Ensure parent directory exists
        target_dir.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(tmp_target), str(target_dir))
        
    print("Sync complete.")

if __name__ == "__main__":
    sync_data()
