# src/data_processing/quality_scorer.py

from typing import List, Dict, Optional
from src.llm_utils.llm_factory import LLMFactory, load_prompt
import sqlite3
import threading
import hashlib
import logging

# Persistent cache for LLM completions
class PersistentLLMCache:
    def __init__(self, db_path: str):
        self.db_path = db_path
        self.lock = threading.Lock()
        self._init_db()

    def _init_db(self):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS llm_cache (
                    key TEXT PRIMARY KEY,
                    value TEXT
                )
            """)

    def get(self, key: str) -> str:
        with self.lock, sqlite3.connect(self.db_path) as conn:
            cur = conn.execute("SELECT value FROM llm_cache WHERE key=?", (key,))
            row = cur.fetchone()
            return row[0] if row else None

    def set(self, key: str, value: str):
        with self.lock, sqlite3.connect(self.db_path) as conn:
            conn.execute("REPLACE INTO llm_cache (key, value) VALUES (?, ?)", (key, value))

llm_cache = PersistentLLMCache(db_path="data/output/llm_cache.sqlite")

def _normalize_cache_key(prompt: str, model_name: str) -> str:
    key_raw = f"quality:{model_name}:{prompt.strip()}"
    return hashlib.sha256(key_raw.encode("utf-8")).hexdigest()

def assess_response_quality(response_text: str) -> str:
    """
    Uses an LLM to classify a response as high_effort, low_effort, or non_answer.
    """
    if not response_text or len(response_text.split()) < 2:
        return "non_answer"

    try:
        client = LLMFactory.get_client("quality_assessment")
        config = LLMFactory.get_task_config("quality_assessment")
        prompt_template = load_prompt("assess_response_quality")
        prompt = prompt_template.format(user_response_text=response_text)
        messages = [{"role": "user", "content": prompt}]
        cache_key = _normalize_cache_key(prompt, config.fast_model)
        cached = llm_cache.get(cache_key)
        if cached is not None:
            logging.debug(f"[CACHE] Quality assessment hit for response: {response_text[:50]}")
            quality = cached.strip().lower()
        else:
            quality = client.chat_completion(messages, model_name=config.fast_model, temperature=0.0)
            quality = quality.strip().lower()
            llm_cache.set(cache_key, quality)
            logging.debug(f"[CACHE] Quality assessment miss, computed and cached for response: {response_text[:50]}")
        if quality in ["high_effort", "low_effort", "non_answer"]:
            return quality
        return "low_effort"
    except Exception as e:
        print(f"Error assessing response quality: {e}")
        return "unknown"