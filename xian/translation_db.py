"""Translation Database Layer with LMDB for persistent caching."""

import json
import logging
import shutil
from typing import Optional, Dict, Any, List
import lmdb
import threading

from . import constants

logger = logging.getLogger(__name__)

class TranslationDB:
    """Wrapper class for translation persistence using LMDB."""
    
    def __init__(self, db_path: str = "./translation_cache.lmdb"):
        self.db_path = db_path
        # Set map_size from constants
        self.env = lmdb.open(
            self.db_path,
            map_size=constants.DB_MAP_SIZE_BYTES,
            writemap=True,
            metasync=False,
            sync=False,
            map_async=True
        )
        self.lock = threading.RLock()
    
    def put_translation(self, dhash: str, translation_data: Dict[str, Any]) -> bool:
        """Store a translation in the database."""
        try:
            with self.lock:
                with self.env.begin(write=True) as txn:
                    # Serialize the translation data to JSON string
                    serialized_data = json.dumps(translation_data)
                    txn.put(dhash.encode('utf-8'), serialized_data.encode('utf-8'))
                    return True
        except Exception as e:
            logger.error(f"Error storing translation in DB: {e}")
            return False
    
    def get_translation(self, dhash: str) -> Optional[Dict[str, Any]]:
        """Retrieve a translation from the database."""
        try:
            with self.lock:
                with self.env.begin() as txn:
                    data = txn.get(dhash.encode('utf-8'))
                    if data:
                        # Deserialize from JSON string
                        return json.loads(data.decode('utf-8'))
                    return None
        except Exception as e:
            logger.error(f"Error retrieving translation from DB: {e}")
            return None
    
    def close(self):
        """Close the database connection."""
        if self.env:
            self.env.close()
    
    def clear(self):
        """Clear all entries from the database."""
        try:
            with self.lock:
                self.env.close()
                shutil.rmtree(self.db_path, ignore_errors=True)
                self.env = lmdb.open(
                    self.db_path,
                    map_size=constants.DB_MAP_SIZE_BYTES,
                    writemap=True,
                    metasync=False,
                    sync=False,
                    map_async=True
                )
                logger.info("Cleared translation DB")
                return 0
        except Exception as e:
            logger.error(f"Error clearing translation DB: {e}")
            return 0
    
    def __del__(self):
        """Cleanup on deletion."""
        self.close()