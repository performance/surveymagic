from src.llm_utils.llm_factory import prompt_factory
import os
from config.project_config import project_config
import logging
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

    def get(self, key: str) -> str | None:
        with self.lock, sqlite3.connect(self.db_path) as conn:
            cur = conn.execute("SELECT value FROM llm_cache WHERE key=?", (key,))
            row = cur.fetchone()
            return row[0] if row else None

    def set(self, key: str, value: str):
        with self.lock, sqlite3.connect(self.db_path) as conn:
            conn.execute("REPLACE INTO llm_cache (key, value) VALUES (?, ?)", (key, value))

llm_cache = PersistentLLMCache(db_path=project_config.cache_db)
# src/analysis/thematic_analyzer.py

import random
import json
from typing import List, Dict, Any
from collections import Counter
from itertools import chain

from src.data_processing.trie_builder import Trie, TrieNode
from src.llm_utils.llm_factory import LLMFactory
from src.llm_utils.embedding_utils import get_embedding, cosine_similarity
from config.project_config import project_config
import logging


class ThematicAnalyzer:
    def __init__(self, use_bootstrap: bool = False):
        self.use_bootstrap = use_bootstrap

    def _extract_keywords_from_trie(self, trie: Trie) -> Dict[str, List[str]]:
        """Traverses the trie and extracts keywords, mapping them to participant IDs."""
        from collections import defaultdict
        keyword_map: defaultdict[str, List[str]] = defaultdict(list)

        client = LLMFactory.get_client("keyword_extraction")
        config = LLMFactory.get_task_config("keyword_extraction")

        def persistent_keyword_extraction(user_response_text: str) -> str:
            import re
            norm_text = re.sub(r"\s+", " ", user_response_text).strip()
            substitutions = {"user_response_text": norm_text}
            prompt = prompt_factory.render("extract_keywords", substitutions)
            messages = [{"role": "user", "content": prompt}]
            import json
            try:
                canonical_subs = json.dumps(substitutions, sort_keys=True, separators=( ",", ":"))
                cache_key = f"extract_keywords:{canonical_subs}"
                cached = llm_cache.get(cache_key)
                if cached:
                    logging.debug(f"[CACHE HIT] Keyword extraction for: {norm_text[:50]}")
                    return cached
                else:
                    logging.debug(f"[CACHE MISS] Keyword extraction for: {norm_text[:50]}")
                    result = client.chat_completion(
                        messages,
                        model_name=config.fast_model,
                        temperature=0.0,
                        substitutions=substitutions
                    )
                    llm_cache.set(cache_key, result)
                    return result
            except Exception as e:
                logging.warning(f"Could not extract keywords for node '{norm_text[:50]}...': {e}")
                return ""

        def _traverse(node: TrieNode):
            if node.speaker == 'user' and node.participant_ids:
                raw_keywords = persistent_keyword_extraction(node.text)
                keywords = [k.strip() for k in raw_keywords.split(',') if k.strip()]
                for pid in node.participant_ids:
                    if keywords:
                        keyword_map[pid].extend(keywords)
            for child in node.children:
                _traverse(child)

        _traverse(trie.root)
        return dict(keyword_map)

    def _generate_themes_for_sample(
        self,
        participant_ids_sample: List[str],
        keyword_map: Dict[str, List[str]],
        question_text: str
    ) -> List[Dict[str, Any]]:
        """Generates a candidate set of themes for a single sample of participants."""
        sample_keywords = list(chain.from_iterable(
            keyword_map.get(pid, []) for pid in participant_ids_sample
        ))

        logging.debug(f"Sample participant IDs: {participant_ids_sample}")
        logging.debug(f"Sample keywords: {sample_keywords}")

        if not sample_keywords:
            logging.debug("No keywords found for this sample. Skipping LLM call.")
            return []

        keyword_counts = Counter(sample_keywords)
        sorted_keyword_counds = sorted(keyword_counts.items(), key=lambda x: x[1], reverse=True)
        keyword_list_str = ",\n".join([f"{k} ({v} mentions)" for k, v in sorted_keyword_counds])
        logging.debug(f"Keyword list string for prompt: {keyword_list_str}")

        client = LLMFactory.get_client("theme_generation")
        config = LLMFactory.get_task_config("theme_generation")
        substitutions: Dict[str, str] = {
            "question_text": question_text,
            "project_background": project_config.resolved_project_background,
            "keyword_list": keyword_list_str
        }

        def strip_json_markdown(response: str) -> str:
            if response.startswith("```json"):
                response = response.split("```json", 1)[1].strip()
                response = response.split("```", 1)[0].strip()
            elif response.startswith("```"):
                response = response.split("```", 1)[1].strip()
                response = response.split("```", 1)[0].strip()
            return response

        def persistent_theme_generation(substitutions: dict[str, str], cache_only: bool = False) -> str:
            import json
            canonical_subs = json.dumps(substitutions, sort_keys=True, separators=( ",", ":"))
            cache_key = f"generate_candidate_themes:{canonical_subs}"
            try:
                cached = llm_cache.get(cache_key)
                if cached:
                    logging.debug(f"[CACHE HIT] Theme generation for prompt: {canonical_subs[:80]}")
                    return cached
                else:
                    logging.debug(f"[CACHE MISS] Theme generation for prompt: {canonical_subs[:80]}")
                    if cache_only:
                        raise RuntimeError(f"[CACHE-ONLY] Theme generation cache miss for prompt: {canonical_subs[:80]}")
                    prompt = prompt_factory.render("generate_candidate_themes", substitutions)
                    result = client.chat_completion(
                        [{"role": "user", "content": prompt}],
                        model_name=config.smart_model,
                        temperature=0.0
                    )
                    llm_cache.set(cache_key, result)
                    return result
            except Exception as e:
                logging.warning(f"Could not generate themes for prompt: {canonical_subs[:80]}... Error, result not cached: {e}")
                return ""

        response_json_str = ""
        try:
            response_json_str = persistent_theme_generation(substitutions, cache_only=os.environ.get("CACHE_ONLY_MODE", "0") == "1")
            response_json_str = strip_json_markdown(response_json_str)
            logging.debug(f"Raw LLM response: {response_json_str}")
            themes_data = json.loads(response_json_str)

            themes: List[Dict[str, Any]] = []
            for theme_data in themes_data:
                title = theme_data.get('theme_title')
                desc = theme_data.get('theme_description')
                if title and desc:
                    theme_embedding = get_embedding(f"{title}: {desc}", cache_only=os.environ.get("CACHE_ONLY_MODE", "0") == "1")
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

    def _merge_and_finalize_themes(self, all_candidate_themes: List[Dict[str, Any]]) -> List[Dict[str, str]]:
        """Clusters and merges themes from all K samples to find stable, final themes."""
        clusters: List[List[Dict[str, Any]]] = []
        
        for theme in all_candidate_themes:
            found_cluster = False
            for cluster in clusters:
                representative_embedding = cluster[0].get('embedding') if cluster and 'embedding' in cluster[0] else None
                if representative_embedding is not None and 'embedding' in theme:
                    similarity = cosine_similarity(theme['embedding'], representative_embedding)
                    if similarity > project_config.similarity_threshold_theme_merging:
                        cluster.append(theme)
                        found_cluster = True
                        break
            if not found_cluster:
                clusters.append([theme])
        
        logging.info(f"Found {len(clusters)} initial clusters.")
        for i, cluster in enumerate(clusters):
            logging.info(f"Cluster {i+1} size: {len(cluster)}")

        min_occurrences = 1
        if self.use_bootstrap:
            min_occurrences = int(project_config.k_samples_for_validation * project_config.min_theme_occurrence_percentage)
        
        stable_clusters = [cluster for cluster in clusters if len(cluster) >= min_occurrences]
        
        final_themes = []
        for cluster in stable_clusters:
            titles = Counter(theme['theme_title'] for theme in cluster)
            representative_title = titles.most_common(1)[0][0]
            
            representative_description = max((theme['theme_description'] for theme in cluster), key=len)
            
            final_themes.append({
                "theme_title": representative_title,
                "theme_description": representative_description
            })
            
        return final_themes

    def find_stable_themes(self, trie: Trie, all_participant_ids: List[str], question_text: str) -> List[Dict[str, str]]:
        """
        The main function to perform thematic analysis.
        """
        logging.info("Step 1: Extracting and annotating keywords from the trie...")
        keyword_map = self._extract_keywords_from_trie(trie)
        
        all_candidate_themes: List[Dict[str, Any]] = []
        
        if self.use_bootstrap:
            logging.info(f"Step 2: Generating themes for {project_config.k_samples_for_validation} bootstrap samples...")
            for i in range(project_config.k_samples_for_validation):
                sample_pids = random.choices(all_participant_ids, k=len(all_participant_ids))
                logging.info(f"  - Running sample {i+1}/{project_config.k_samples_for_validation}...")
                sample_themes = self._generate_themes_for_sample(sample_pids, keyword_map, question_text)
                all_candidate_themes.extend(sample_themes)
        else:
            logging.info("Step 2: Generating themes for all participants (no bootstrapping)...")
            sample_themes = self._generate_themes_for_sample(all_participant_ids, keyword_map, question_text)
            all_candidate_themes.extend(sample_themes)
            
        logging.info("Step 3: Merging and finalizing stable themes...")
        final_themes = self._merge_and_finalize_themes(all_candidate_themes)
        
        logging.info(f"Analysis complete. Found {len(final_themes)} stable themes.")
        return final_themes
