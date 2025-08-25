# src/llm_utils/embedding_utils.py

import numpy as np
from typing import List, Dict, Optional
import logging
from src.llm_utils.llm_factory import LLMFactory
from src.llm_utils.caching import PersistentCache, normalize_cache_key
from config.project_config import project_config


embedding_cache = PersistentCache(db_path=project_config.cache_db)

def get_embedding(text: str, task_name: str = "embedding", cache_only: bool = False) -> List[float]:
    """Computes and caches the embedding for a given text, using persistent cache. If cache_only is True, raises on cache miss."""
    client = LLMFactory.get_client(task_name)
    config = LLMFactory.get_task_config(task_name)
    model_name = config.embedding_model
    cache_key = normalize_cache_key(["embedding", model_name, text])
    cached = embedding_cache.get(cache_key)
    if cached is not None:
        logging.debug(f"[CACHE HIT] Embedding hit for: {text[:50]} (model={model_name})")
        return cached
    else:
        logging.debug(f"[CACHE MISS] Embedding miss for: {text[:50]} (model={model_name})")
        if cache_only:
            raise RuntimeError(f"[CACHE-ONLY] Embedding cache miss for: {text[:50]} (model={model_name})")
    embedding = client.get_embedding(text, model_name=model_name)
    embedding_cache.set(cache_key, embedding)
    logging.debug(f"[CACHE] Embedding computed and cached for: {text[:50]} (model={model_name})")
    return embedding

def cosine_similarity(vec1: List[float], vec2: List[float]) -> float:
    """Calculates the cosine similarity between two vectors."""
    dot_product = np.dot(vec1, vec2)
    norm_vec1 = np.linalg.norm(vec1)
    norm_vec2 = np.linalg.norm(vec2)
    
    if norm_vec1 == 0 or norm_vec2 == 0:
        return 0.0
        
    return dot_product / (norm_vec1 * norm_vec2)