import os
from config.project_config import project_config
import logging
# --- Logging setup ---
def setup_logging():
    log_path = project_config.log_file
    log_dir = os.path.dirname(log_path)
    if log_dir and not os.path.exists(log_dir):
        os.makedirs(log_dir, exist_ok=True)
    log_level = getattr(logging, project_config.log_level.upper(), logging.INFO)
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(log_path),
            logging.StreamHandler()
        ]
    )

setup_logging()
import sqlite3
import threading

# Persistent cache helper
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
# src/analysis/thematic_analyzer.py

import random
import json
from typing import List, Dict, Any
from collections import Counter
from itertools import chain

from src.data_processing.trie_builder import Trie, TrieNode
from src.llm_utils.llm_factory import LLMFactory, load_prompt
from src.llm_utils.embedding_utils import get_embedding, cosine_similarity
from config.project_config import project_config
import logging

class Theme(Dict):
    """A dictionary-like object representing a theme for easier type hinting."""
    theme_title: str
    theme_description: str
    embedding: List[float]


def _extract_keywords_from_trie(trie: Trie) -> Dict[str, List[str]]:
    """Traverses the trie and extracts keywords, mapping them to participant IDs."""
    from collections import defaultdict
    keyword_map = defaultdict(list) # Aggregate keywords per participant

    # Annotate user nodes with keywords using an LLM
    client = LLMFactory.get_client("keyword_extraction")
    config = LLMFactory.get_task_config("keyword_extraction")
    prompt_template = load_prompt("extract_keywords")

    def persistent_keyword_extraction(user_response_text: str) -> str:
        prompt = prompt_template.format(user_response_text=user_response_text)
        cache_key = f"keyword:{user_response_text.strip()}"
        cached = llm_cache.get(cache_key)
        if cached is not None:
            logging.debug(f"[CACHE] Keyword extraction hit for: {user_response_text[:50]}")
            return cached
        messages = [{"role": "user", "content": prompt}]
        try:
            result = client.chat_completion(
                messages,
                model_name=config.fast_model,
                temperature=0.1
            )
            llm_cache.set(cache_key, result)
            return result
        except Exception as e:
            logging.warning(f"Could not extract keywords for node '{user_response_text[:50]}...': {e}")
            return ""

    def _traverse(node: TrieNode):
        if node.speaker == 'user' and node.participant_ids:
            raw_keywords = persistent_keyword_extraction(node.text)
            keywords = [k.strip() for k in raw_keywords.split(',') if k.strip()]
            for pid in node.participant_ids:
                keyword_map[pid].extend(keywords)
        for child in node.children:
            _traverse(child)

    _traverse(trie.root)
    return dict(keyword_map) # Return as a standard dict


def _generate_themes_for_sample(
    participant_ids_sample: List[str],
    keyword_map: Dict[str, List[str]],
    question_text: str
) -> List[Theme]:
    """Generates a candidate set of themes for a single sample of participants."""
    

    # Collect all keywords from the participants in the current sample
    sample_keywords = list(chain.from_iterable(
        keyword_map.get(pid, []) for pid in participant_ids_sample
    ))

    logging.debug(f"Sample participant IDs: {participant_ids_sample}")
    logging.debug(f"Sample keywords: {sample_keywords}")

    if not sample_keywords:
        logging.debug("No keywords found for this sample. Skipping LLM call.")
        return []

    # Use a Counter to get unique keywords and their frequencies for better prompting
    keyword_counts = Counter(sample_keywords)
    keyword_list_str = ", ".join([f"{k} ({v} mentions)" for k, v in keyword_counts.items()])

    logging.debug(f"Keyword list string for prompt: {keyword_list_str}")

    client = LLMFactory.get_client("theme_generation")
    config = LLMFactory.get_task_config("theme_generation")
    prompt_template = load_prompt("generate_candidate_themes")

    prompt = prompt_template.format(
        question_text=question_text,
    project_background=project_config.resolved_project_background,
        keyword_list=keyword_list_str
    )

    logging.debug(f"Prompt sent to LLM:\n{prompt}")

    messages = [{"role": "user", "content": prompt}]

    def strip_json_markdown(response: str) -> str:
        if response.startswith("```json"):
            response = response.split("```json", 1)[1].strip()
            response = response.split("```", 1)[0].strip()
        elif response.startswith("```"):
            response = response.split("```", 1)[1].strip()
            response = response.split("```", 1)[0].strip()
        return response

    def persistent_theme_generation(prompt: str) -> str:
        cache_key = f"theme:{prompt.strip()}"
        cached = llm_cache.get(cache_key)
        if cached is not None:
            logging.debug(f"[CACHE] Theme generation hit for prompt: {prompt[:80]}")
            return cached
        messages = [{"role": "user", "content": prompt}]
        try:
            result = client.chat_completion(
                messages,
                model_name=config.smart_model,
                temperature=0.3
            )
            llm_cache.set(cache_key, result)
            return result
        except Exception as e:
            logging.warning(f"Could not generate themes for prompt: {prompt[:80]}... Error: {e}")
            return ""

    try:
        response_json_str = persistent_theme_generation(prompt)
        response_json_str = strip_json_markdown(response_json_str)
        logging.debug(f"Raw LLM response: {response_json_str}")
        themes_data = json.loads(response_json_str)

        # Add embeddings to each theme for the merging step
        themes: List[Theme] = []
        for theme_data in themes_data:
            title = theme_data.get('theme_title')
            desc = theme_data.get('theme_description')
            if title and desc:
                theme_embedding = get_embedding(f"{title}: {desc}")
                themes.append({
                    "theme_title": title,
                    "theme_description": desc,
                    "embedding": theme_embedding
                })
        return themes

    except json.JSONDecodeError as e:
        logging.error(f"JSON decode error. Raw response: {response_json_str}")
        logging.warning(f"Failed to generate or parse themes for a sample. Error: {e}")
        return []
    except Exception as e:
        logging.error(f"Exception during LLM call or parsing. Raw response: {response_json_str}")
        logging.warning(f"Failed to generate or parse themes for a sample. Error: {e}")
        return []


def _merge_and_finalize_themes(all_candidate_themes: List[Theme]) -> List[Dict[str, str]]:
    """Clusters and merges themes from all K samples to find stable, final themes."""
    clusters = []
    
    for theme in all_candidate_themes:
        found_cluster = False
        for cluster in clusters:
            # Get the embedding of the representative theme for the cluster
            representative_embedding = cluster[0]['embedding']
            similarity = cosine_similarity(theme['embedding'], representative_embedding)
            
            if similarity > project_config.similarity_threshold_theme_merging:
                cluster.append(theme)
                found_cluster = True
                break
        if not found_cluster:
            clusters.append([theme])
            
    # Filter for stable themes (appeared in enough K-samples)
    min_occurrences = int(project_config.k_samples_for_validation * project_config.min_theme_occurrence_percentage)
    stable_clusters = [cluster for cluster in clusters if len(cluster) >= min_occurrences]
    
    final_themes = []
    for cluster in stable_clusters:
        # Choose the most frequent title and a well-phrased description as representative
        titles = Counter(theme['theme_title'] for theme in cluster)
        representative_title = titles.most_common(1)[0][0]
        
        # A simple heuristic for a good description: one of the longest ones
        representative_description = max((theme['theme_description'] for theme in cluster), key=len)
        
        final_themes.append({
            "theme_title": representative_title,
            "theme_description": representative_description
        })
        
    return final_themes


def find_stable_themes(trie: Trie, all_participant_ids: List[str], question_text: str) -> List[Dict[str, str]]:
    """
    The main function to perform thematic analysis using bootstrapping validation.
    """
    logging.info("Step 1: Extracting and annotating keywords from the trie...")
    keyword_map = _extract_keywords_from_trie(trie)
    
    logging.info(f"Step 2: Generating themes for {project_config.k_samples_for_validation} bootstrap samples...")
    all_candidate_themes: List[Theme] = []
    
    for i in range(project_config.k_samples_for_validation):
        # Bootstrap sampling (sampling with replacement)
        sample_pids = random.choices(all_participant_ids, k=len(all_participant_ids))
        logging.info(f"  - Running sample {i+1}/{project_config.k_samples_for_validation}...")
        sample_themes = _generate_themes_for_sample(sample_pids, keyword_map, question_text)
        all_candidate_themes.extend(sample_themes)
        
    logging.info("Step 3: Merging and finalizing stable themes...")
    final_themes = _merge_and_finalize_themes(all_candidate_themes)
    
    logging.info(f"Analysis complete. Found {len(final_themes)} stable themes.")
    return final_themes