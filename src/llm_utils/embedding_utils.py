# src/llm_utils/embedding_utils.py

import numpy as np
from typing import List, Dict, Optional
from functools import lru_cache

from src.llm_utils.llm_factory import LLMFactory

# A simple in-memory cache for embeddings to avoid redundant API calls
# The lru_cache is perfect for this.
@lru_cache(maxsize=2048)
def get_embedding(text: str, task_name: str = "embedding") -> List[float]:
    """Computes and caches the embedding for a given text."""
    client = LLMFactory.get_client(task_name)
    config = LLMFactory.get_task_config(task_name)
    return client.get_embedding(text, model_name=config.embedding_model)

def cosine_similarity(vec1: List[float], vec2: List[float]) -> float:
    """Calculates the cosine similarity between two vectors."""
    dot_product = np.dot(vec1, vec2)
    norm_vec1 = np.linalg.norm(vec1)
    norm_vec2 = np.linalg.norm(vec2)
    
    if norm_vec1 == 0 or norm_vec2 == 0:
        return 0.0
        
    return dot_product / (norm_vec1 * norm_vec2)