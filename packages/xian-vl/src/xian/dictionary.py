import pathlib
import zipfile
import urllib.request
import logging
import threading
import hashlib
from typing import Dict, List, Tuple

logger = logging.getLogger(__name__)

# SHA-256 hash of the extracted cedict_ts.u8 file.
# Update this if the upstream file changes.
# To update: run `sha256sum data/cedict_ts.u8` and paste the hash here.
CEDICT_EXTRACTED_SHA256 = "59f786954858947cca7a108e3c630399b0473ce503205dcec55266df36ada011"

class LocalDictionary:
    def __init__(self, data_dir: str = "./data"):
        self.data_dir = pathlib.Path(data_dir)
        self.dict_path = self.data_dir / "cedict_ts.u8"
        self.zip_url = "https://www.mdbg.net/chinese/export/cedict/cedict_1_0_ts_utf-8_mdbg.zip"
        
        # Maps simplified -> list of (traditional, pinyin, english)
        self.entries: Dict[str, List[Tuple[str, str, str]]] = {}
        self.ready = False
        
        # Fix TOCTOU: use exist_ok=True
        self.data_dir.mkdir(parents=True, exist_ok=True)
            
        # Start load/download in background
        threading.Thread(target=self._init_dictionary, daemon=True).start()
        
    def _init_dictionary(self):
        try:
            if not self.dict_path.exists():
                logger.info("CC-CEDICT not found. Downloading...")
                zip_path = self.data_dir / "cedict.zip"
                urllib.request.urlretrieve(self.zip_url, zip_path)
                
                with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                    # Validate all members to prevent Zip Slip (path traversal)
                    data_dir_resolved = self.data_dir.resolve()
                    for member in zip_ref.namelist():
                        target = (data_dir_resolved / member).resolve()
                        if not target.is_relative_to(data_dir_resolved):
                            raise ValueError(f"Zip member escapes target directory: {member}")
                    zip_ref.extractall(self.data_dir)
                zip_path.unlink()
                logger.info("CC-CEDICT downloaded and extracted.")
                
                # Verify integrity of the extracted file
                self._verify_integrity()
                
            self._parse_dict()
        except Exception as e:
            logger.error(f"Failed to initialize dictionary: {e}")
            
    def _verify_integrity(self):
        logger.info("Verifying CC-CEDICT integrity...")
        hasher = hashlib.sha256()
        with open(self.dict_path, 'rb') as f:
            while chunk := f.read(8192):
                hasher.update(chunk)
        
        current_hash = hasher.hexdigest()
        if current_hash != CEDICT_EXTRACTED_SHA256:
            logger.error(f"CC-CEDICT integrity check failed! Expected {CEDICT_EXTRACTED_SHA256}, got {current_hash}. Removing file.")
            self.dict_path.unlink(missing_ok=True)
            raise ValueError("Integrity check failed")
        logger.info("CC-CEDICT integrity verified.")

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
                
        self.ready = True
        logger.info(f"CC-CEDICT parsed successfully. Loaded {count} entries.")
        
    def lookup(self, word: str) -> List[Tuple[str, str, str]]:
        if not self.ready:
            return []
        return self.entries.get(word, [])
