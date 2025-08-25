# src/data_processing/quality_scorer.py

from typing import List, Dict, Optional
from src.llm_utils.llm_factory import LLMFactory, load_prompt
from src.llm_utils.caching import PersistentCache, normalize_cache_key
from config.project_config import project_config
import logging

llm_cache = PersistentCache(db_path=project_config.cache_db)

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
        cache_key = normalize_cache_key(["quality_assessment", config.fast_model, prompt])
        cached = llm_cache.get(cache_key)
        if cached is not None:
            logging.debug(f"[CACHE HIT] Quality assessment hit for response: {response_text[:50]}")
            quality = cached.strip().lower()
        else:
            quality = client.chat_completion(messages, model_name=config.fast_model, temperature=0.0)
            quality = quality.strip().lower()
            llm_cache.set(cache_key, quality)
            logging.debug(f"[CACHE MISS] Quality assessment miss, computed and cached for response: {response_text[:50]}")
        if quality in ["high_effort", "low_effort", "non_answer"]:
            return quality
        return "low_effort"
    except Exception as e:
        logging.error(f"Error assessing response quality: {e}")
        return "unknown"