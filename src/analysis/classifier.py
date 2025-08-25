# src/analysis/classifier.py

import json
from typing import List, Dict
import logging

from src.llm_utils.llm_factory import LLMFactory, prompt_factory
from src.llm_utils.caching import PersistentCache, normalize_cache_key
from config.project_config import project_config

llm_cache = PersistentCache(db_path=project_config.cache_db)

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
    
    themes_json_str = json.dumps(themes, indent=2)

    for pid, response_text in responses.items():
        substitutions = {
            "themes_json": themes_json_str,
            "user_response": response_text
        }
        prompt = prompt_factory.render("classify_response", substitutions)
        messages = [{"role": "user", "content": prompt}]
        
        cache_key = normalize_cache_key(["classify_response", config.fast_model, json.dumps(substitutions, sort_keys=True)])
        try:
            cached = llm_cache.get(cache_key)
            if cached is not None:
                logging.debug(f"[CACHE HIT] Classification hit for participant {pid}")
                assigned_theme = cached.strip()
            else:
                assigned_theme = client.chat_completion(
                    messages,
                    model_name=config.fast_model,
                    temperature=0.0,
                    substitutions=substitutions
                ).strip()
                llm_cache.set(cache_key, assigned_theme)
                logging.debug(f"[CACHE MISS] Classification miss, computed and cached for participant {pid}")
            valid_titles = {t['theme_title'] for t in themes}
            if assigned_theme in valid_titles:
                classifications[pid] = assigned_theme
            else:
                logging.warning(f"Warning: Classifier returned an invalid theme title '{assigned_theme}' for participant {pid}. Skipping.")
        except Exception as e:
            logging.warning(f"Warning: Failed to classify response for participant {pid}. Error: {e}")
    return classifications