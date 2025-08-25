# src/llm_utils/caching.py

import sqlite3
import threading
import hashlib
import pickle
from typing import Optional, Any

class PersistentCache:
    def __init__(self, db_path: str):
        self.db_path = db_path
        self.lock = threading.Lock()
        self._init_db()

    def _init_db(self):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS cache (
                    key TEXT PRIMARY KEY,
                    value BLOB
                )
            """)

    def get(self, key: str) -> Optional[Any]:
        with self.lock, sqlite3.connect(self.db_path) as conn:
            cur = conn.execute("SELECT value FROM cache WHERE key=?", (key,))
            row = cur.fetchone()
            if row:
                return pickle.loads(row[0])
            return None

    def set(self, key: str, value: Any):
        with self.lock, sqlite3.connect(self.db_path) as conn:
            conn.execute("REPLACE INTO cache (key, value) VALUES (?, ?)", (key, pickle.dumps(value)))

def normalize_cache_key(parts: list[str]) -> str:
    """Creates a consistent, normalized cache key from a list of parts."""
    key_raw = ":".join(str(p).strip() for p in parts)
    return hashlib.sha256(key_raw.encode("utf-8")).hexdigest()
