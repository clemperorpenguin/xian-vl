# Xian-VL — Core Vision-Language orchestration engine.
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

import logging
import pathlib
import threading
import zipfile
import httpx

logger = logging.getLogger(__name__)

# The CC-CEDICT hash check has been replaced with an ETag-based update mechanism
# to automatically stay current with upstream changes without hardcoded hashes.

class LocalDictionary:
    def __init__(self, data_dir: str = "./data"):
        self.data_dir = pathlib.Path(data_dir)
        self.dict_path = self.data_dir / "cedict_ts.u8"
        self.etag_path = self.data_dir / "cedict_ts.u8.etag"
        self.zip_url = "https://www.mdbg.net/chinese/export/cedict/cedict_1_0_ts_utf-8_mdbg.zip"
        
        # Maps simplified -> list of (traditional, pinyin, english)
        self.entries: dict[str, list[tuple[str, str, str]]] = {}
        self._ready_event = threading.Event()
        
        # Ensure data directory exists
        self.data_dir.mkdir(parents=True, exist_ok=True)
            
        # Start load/download in background
        threading.Thread(target=self._init_dictionary, daemon=True).start()
        
    def _init_dictionary(self):
        try:
            # 1. Check for updates and download if necessary
            self._check_for_updates()
            
            # 2. Parse the dictionary
            if not self.dict_path.exists():
                raise FileNotFoundError("CC-CEDICT dictionary file not found.")
            
            self._parse_dict()
        except Exception as e:
            logger.error("Failed to initialize dictionary: %s", e)
            # If initialization failed and the file exists, it might be corrupt
            if self.dict_path.exists():
                logger.info("Parsing failed; the dictionary file might be corrupt. It will be re-downloaded on next start.")
                self.dict_path.unlink(missing_ok=True)
                self.etag_path.unlink(missing_ok=True)

    def _check_for_updates(self):
        """Checks if a new version of the dictionary is available using ETags."""
        current_etag = self.etag_path.read_text().strip() if self.etag_path.exists() else None
        
        try:
            headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) MAGE/1.0"}
            if current_etag and self.dict_path.exists():
                headers["If-None-Match"] = current_etag

            with httpx.Client(follow_redirects=True, timeout=30.0) as client:
                logger.info("Checking for CC-CEDICT updates...")
                resp = client.get(self.zip_url, headers=headers)
                
                if resp.status_code == 200:
                    logger.info("New CC-CEDICT version found. Downloading...")
                    zip_path = self.data_dir / "cedict.zip"
                    with open(zip_path, "wb") as f:
                        f.write(resp.content)
                    
                    self._extract_dictionary(zip_path)
                    
                    # Store the new ETag
                    new_etag = resp.headers.get("ETag")
                    if new_etag:
                        self.etag_path.write_text(new_etag)
                elif resp.status_code == 304:
                    logger.info("CC-CEDICT is already up to date.")
                else:
                    logger.warning("Unexpected status code %d from MDBG. Using local copy if available.", resp.status_code)
        except Exception as e:
            logger.warning("Failed to check for CC-CEDICT updates: %s. Using local copy.", e)

    def _extract_dictionary(self, zip_path: pathlib.Path):
        logger.info("Extracting CC-CEDICT...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            # Validate all members to prevent Zip Slip
            data_dir_resolved = self.data_dir.resolve()
            for member in zip_ref.namelist():
                if member.startswith("/") or member.startswith("\\") or ".." in member:
                    raise ValueError("Unsafe zip member: %s" % member)
                
                target = (data_dir_resolved / member).resolve()
                if not target.is_relative_to(data_dir_resolved):
                    raise ValueError("Zip member escapes target directory: %s" % member)
            zip_ref.extractall(self.data_dir)
        zip_path.unlink()
        logger.info("CC-CEDICT extracted successfully.")

    def _parse_dict(self):
        logger.info("Parsing CC-CEDICT...")
        count = 0
        with open(self.dict_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.startswith('#'): continue
                
                # Format: Traditional Simplified [pin1 yin1] /English equivalent 1/equivalent 2/
                parts = line.split(' /')
                if len(parts) < 2: continue
                
                words_pinyin = parts[0].split(' [')
                if len(words_pinyin) < 2: continue
                
                words = words_pinyin[0].split()
                if len(words) != 2: continue
                
                trad, simp = words[0], words[1]
                pinyin = words_pinyin[1].replace(']', '')
                english = ' / '.join(parts[1:]).strip().strip('/')
                
                if simp not in self.entries:
                    self.entries[simp] = []
                self.entries[simp].append((trad, pinyin, english))
                count += 1
                
        if count < 1000:  # Sanity check: CC-CEDICT should have >100,000 entries
            raise ValueError(f"Dictionary parsing failed: only {count} entries found. File may be truncated or invalid.")

        self._ready_event.set()
        logger.info("CC-CEDICT parsed successfully. Loaded %d entries.", count)
        
    def lookup(self, word: str) -> list[tuple[str, str, str]]:
        if not self._ready_event.is_set():
            return []
        return self.entries.get(word, [])
