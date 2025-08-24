# src/analysis/classifier.py

import json
from typing import List, Dict, Optional


from src.llm_utils.llm_factory import LLMFactory, load_prompt
import sqlite3
import threading
import hashlib
import logging

# Persistent cache for LLM completions

from typing import Optional
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

    def get(self, key: str) -> Optional[str]:
        with self.lock, sqlite3.connect(self.db_path) as conn:
            cur = conn.execute("SELECT value FROM llm_cache WHERE key=?", (key,))
            row = cur.fetchone()
            return row[0] if row else None

    def set(self, key: str, value: str) -> None:
        with self.lock, sqlite3.connect(self.db_path) as conn:
            conn.execute("REPLACE INTO llm_cache (key, value) VALUES (?, ?)", (key, value))

llm_cache = PersistentLLMCache(db_path="data/output/llm_cache.sqlite")

def _normalize_cache_key(prompt: str, model_name: str) -> str:
    import re
    norm_prompt = re.sub(r"\s+", " ", prompt).strip()
    key_raw = f"classify:{model_name}:{norm_prompt}"
    return hashlib.sha256(key_raw.encode("utf-8")).hexdigest()

def classify_responses(
    responses: Dict[str, str], 
    themes: List[Dict[str, str]]
) -> Dict[str, str]:
    """
    Classifies each participant's response into one of the final themes.

    Args:
        responses: A dictionary mapping participant_id to their flattened response text.
        themes: The list of final, stable themes.

    Returns:
        A dictionary mapping participant_id to their assigned theme_title.
    """
    if not themes or not responses:
        return {}

    classifications = {}
    client = LLMFactory.get_client("classification")
    config = LLMFactory.get_task_config("classification")
    prompt_template = load_prompt("classify_response")
    
    themes_json_str = json.dumps(themes, indent=2)

    for pid, response_text in responses.items():
        prompt = prompt_template.format(
            themes_json=themes_json_str,
            user_response=response_text
        )
        messages = [{"role": "user", "content": prompt}]
        cache_key = _normalize_cache_key(prompt, config.fast_model)
        try:
            cached = llm_cache.get(cache_key)
            if cached is not None:
                logging.debug(f"[CACHE] Classification hit for participant {pid}")
                assigned_theme = cached.strip()
            else:
                assigned_theme = client.chat_completion(
                    messages,
                    model_name=config.fast_model,
                    temperature=0.0
                ).strip()
                llm_cache.set(cache_key, assigned_theme)
                logging.debug(f"[CACHE] Classification miss, computed and cached for participant {pid}")
            valid_titles = {t['theme_title'] for t in themes}
            if assigned_theme in valid_titles:
                classifications[pid] = assigned_theme
            else:
                print(f"Warning: Classifier returned an invalid theme title '{assigned_theme}' for participant {pid}. Skipping.")
        except Exception as e:
            print(f"Warning: Failed to classify response for participant {pid}. Error: {e}")
    return classifications