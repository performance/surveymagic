# src/analysis/quote_selector.py

import json
from typing import List, Dict, Any

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
    key_raw = f"quotes:{model_name}:{norm_prompt}"
    return hashlib.sha256(key_raw.encode("utf-8")).hexdigest()
from config.project_config import project_config

def select_quotes_for_theme(
    theme: Dict[str, str],
    classified_responses: List[Dict[str, str]]
) -> List[Dict[str, str]]:
    """
    Selects the best ~3 supporting quotes for a single theme.

    Args:
        theme: The theme dictionary (containing title and description).
        classified_responses: A list of dicts [{'participant_id': pid, 'response': text}] for this theme.

    Returns:
        A list of dicts, each representing a quote with participantId and quoteText.
    """
    if not classified_responses:
        return []

    client = LLMFactory.get_client("synthesis") # Use a smart model for this nuanced task
    config = LLMFactory.get_task_config("synthesis")
    prompt_template = load_prompt("select_quotes")
    
    # Format responses for the prompt, keeping it clean
    responses_for_prompt = [
        {"participantId": r['participant_id'], "responseText": r['response']} 
        for r in classified_responses
    ]
    
    prompt = prompt_template.format(
        theme_title=theme['theme_title'],
        theme_description=theme['theme_description'],
        responses_json=json.dumps(responses_for_prompt, indent=2)
    )
    
    messages = [{"role": "user", "content": prompt}]
    
    try:
        cache_key = _normalize_cache_key(prompt, config.smart_model)
        cached = llm_cache.get(cache_key)
        if cached is not None:
            logging.debug(f"[CACHE] Quote selection hit for theme '{theme['theme_title']}'")
            raw_ids = cached
        else:
            raw_ids = client.chat_completion(
                messages,
                model_name=config.smart_model,
                temperature=0.2
            )
            llm_cache.set(cache_key, raw_ids)
            logging.debug(f"[CACHE] Quote selection miss, computed and cached for theme '{theme['theme_title']}'")
        selected_pids = {pid.strip() for pid in raw_ids.split(',') if pid.strip()}
        quotes = []
        for resp in classified_responses:
            if resp['participant_id'] in selected_pids and len(quotes) < project_config.max_quotes_per_theme:
                quotes.append({
                    "participantId": resp['participant_id'],
                    "quoteText": resp['response']
                })
                selected_pids.remove(resp['participant_id'])
        return quotes
    except Exception as e:
        print(f"Warning: Failed to select quotes for theme '{theme['theme_title']}'. Error: {e}")
        return []