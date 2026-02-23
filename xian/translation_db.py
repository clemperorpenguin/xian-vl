"""Translation Database Layer with LMDB for persistent caching."""

import json
import logging
import pickle
from typing import Optional, Dict, Any
import lmdb
import threading

logger = logging.getLogger(__name__)

class TranslationDB:
    """Wrapper class for translation persistence using LMDB."""
    
    def __init__(self, db_path: str = "./translation_cache.lmdb"):
        self.db_path = db_path
        # Set map_size to 1GB initially, can grow as needed
        self.env = lmdb.open(
            self.db_path,
            map_size=1024*1024*1024,  # 1GB
            writemap=True,  # Use memory-mapped writes for better performance
            metasync=False,  # Disable meta sync for better performance
            sync=False,      # Disable sync for better performance
            map_async=True   # Use asynchronous flushes
        )
        self.lock = threading.RLock()  # Thread-safe access
    
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
                with self.env.begin(write=True) as txn:
                    cursor = txn.cursor()
                    count = cursor.delete()
                    logger.info(f"Cleared {count} entries from translation DB")
                    return count
        except Exception as e:
            logger.error(f"Error clearing translation DB: {e}")
            return 0
    
    def __del__(self):
        """Cleanup on deletion."""
        self.close()