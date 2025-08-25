# src/analysis/quote_selector.py

import json
import json
from typing import List, Dict, Any

from src.llm_utils.llm_factory import LLMFactory, load_prompt, prompt_factory
from src.llm_utils.caching import PersistentCache, normalize_cache_key
from config.project_config import project_config
import logging

llm_cache = PersistentCache(db_path=project_config.cache_db)

def select_quotes_for_theme(theme: Dict[str, str], classified_responses: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    import json
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
    responses_for_prompt = [
        {"participantId": r['participant_id'], "responseText": r['response']} 
        for r in classified_responses
    ]
    substitutions = {
        "theme_title": theme['theme_title'],
        "theme_description": theme['theme_description'],
        "responses_json": json.dumps(responses_for_prompt, sort_keys=True, separators=(",", ":"))
    }
    prompt = prompt_factory.render("select_quotes", substitutions)
    messages = [{"role": "user", "content": prompt}]
    
    import json
    try:
        cache_key = normalize_cache_key(["select_quotes", config.smart_model, json.dumps(substitutions, sort_keys=True)])
        cached = llm_cache.get(cache_key)
        if cached is not None:
            logging.debug(f"[CACHE HIT] Quote selection hit for theme '{theme['theme_title']}'")
            raw_ids = cached
        else:
            raw_ids = client.chat_completion(
                messages,
                model_name=config.smart_model,
                temperature=0.0,
                substitutions=substitutions
            )
            llm_cache.set(cache_key, raw_ids)
            logging.debug(f"[CACHE MISS] Quote selection miss, computed and cached for theme '{theme['theme_title']}'")
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
        logging.info(f"Warning: Failed to select quotes for theme '{theme['theme_title']}'. Error: {e}")
        return []