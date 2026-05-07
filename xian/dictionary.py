import os
import pathlib
import zipfile
import urllib.request
import logging
import threading
from typing import Dict, List, Tuple

logger = logging.getLogger(__name__)

class LocalDictionary:
    def __init__(self, data_dir: str = "./data"):
        self.data_dir = data_dir
        self.dict_path = os.path.join(data_dir, "cedict_ts.u8")
        self.zip_url = "https://www.mdbg.net/chinese/export/cedict/cedict_1_0_ts_utf-8_mdbg.zip"
        
        # Maps simplified -> list of (traditional, pinyin, english)
        self.entries: Dict[str, List[Tuple[str, str, str]]] = {}
        self.ready = False
        
        if not os.path.exists(data_dir):
            os.makedirs(data_dir)
            
        # Start load/download in background
        threading.Thread(target=self._init_dictionary, daemon=True).start()
        
    def _init_dictionary(self):
        try:
            if not os.path.exists(self.dict_path):
                logger.info("CC-CEDICT not found. Downloading...")
                zip_path = os.path.join(self.data_dir, "cedict.zip")
                urllib.request.urlretrieve(self.zip_url, zip_path)
                
                with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                    # Validate all members to prevent Zip Slip (path traversal)
                    data_dir_resolved = pathlib.Path(self.data_dir).resolve()
                    for member in zip_ref.namelist():
                        target = (data_dir_resolved / member).resolve()
                        if not target.is_relative_to(data_dir_resolved):
                            raise ValueError(f"Zip member escapes target directory: {member}")
                    zip_ref.extractall(self.data_dir)
                os.remove(zip_path)
                logger.info("CC-CEDICT downloaded and extracted.")
                
            self._parse_dict()
        except Exception as e:
            logger.error(f"Failed to initialize dictionary: {e}")
            
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
