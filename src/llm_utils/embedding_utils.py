# src/llm_utils/embedding_utils.py

import numpy as np
from typing import List, Dict, Optional

 # removed unused import
import sqlite3
import threading
import hashlib
import logging


from src.llm_utils.llm_factory import LLMFactory

# Persistent cache helper (copied from thematic_analyzer, but for embeddings)
class PersistentEmbeddingCache:
    def __init__(self, db_path: str):
        self.db_path = db_path
        self.lock = threading.Lock()
        self._init_db()

    def _init_db(self):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS embedding_cache (
                    key TEXT PRIMARY KEY,
                    value BLOB
                )
            """)

    def get(self, key: str):
        with self.lock, sqlite3.connect(self.db_path) as conn:
            cur = conn.execute("SELECT value FROM embedding_cache WHERE key=?", (key,))
            row = cur.fetchone()
            if row:
                import pickle
                return pickle.loads(row[0])
            return None

    def set(self, key: str, value: object) -> None:
        with self.lock, sqlite3.connect(self.db_path) as conn:
            import pickle
            conn.execute("REPLACE INTO embedding_cache (key, value) VALUES (?, ?)", (key, pickle.dumps(value)))

embedding_cache = PersistentEmbeddingCache(db_path="data/output/embedding_cache.sqlite")


def _normalize_cache_key(text: str, model_name: str) -> str:
    # Use a hash for long texts
    key_raw = f"embedding:{model_name}:{text.strip()}"
    return hashlib.sha256(key_raw.encode("utf-8")).hexdigest()

def get_embedding(text: str, task_name: str = "embedding") -> List[float]:
    """Computes and caches the embedding for a given text, using persistent cache."""
    client = LLMFactory.get_client(task_name)
    config = LLMFactory.get_task_config(task_name)
    model_name = config.embedding_model
    cache_key = _normalize_cache_key(text, model_name)
    cached = embedding_cache.get(cache_key)
    if cached is not None:
        logging.debug(f"[CACHE] Embedding hit for: {text[:50]} (model={model_name})")
        return cached
    embedding = client.get_embedding(text, model_name=model_name)
    embedding_cache.set(cache_key, embedding)
    logging.debug(f"[CACHE] Embedding miss, computed and cached for: {text[:50]} (model={model_name})")
    return embedding

def cosine_similarity(vec1: List[float], vec2: List[float]) -> float:
    """Calculates the cosine similarity between two vectors."""
    dot_product = np.dot(vec1, vec2)
    norm_vec1 = np.linalg.norm(vec1)
    norm_vec2 = np.linalg.norm(vec2)
    
    if norm_vec1 == 0 or norm_vec2 == 0:
        return 0.0
        
    return dot_product / (norm_vec1 * norm_vec2)