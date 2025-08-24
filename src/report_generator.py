# src/report_generator.py

import json
from typing import List, Dict, Any

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

def _normalize_cache_key(prompt: str, model_name: str, task: str) -> str:
    key_raw = f"{task}:{model_name}:{prompt.strip()}"
    return hashlib.sha256(key_raw.encode("utf-8")).hexdigest()
from config.project_config import project_config


def generate_question_narrative(
    question_text: str, analysis_data: Dict[str, Any]
) -> Dict[str, str]:
    """
    Generates the headline and summary for a single question's analysis.
    """
    client = LLMFactory.get_client("synthesis")
    config = LLMFactory.get_task_config("synthesis")
    prompt_template = load_prompt("generate_headline_summary")

    # We only need to provide a summary of the analysis data, not every single quote
    data_summary = {
        "participants_analyzed": analysis_data.get("participants_analyzed"),
        "themes": [
            {
                "theme_title": theme.get("theme_title"),
                "participant_count": theme.get("participant_count"),
                "participant_percentage": f"{theme.get('participant_percentage', 0):.0%}",
            }
            for theme in analysis_data.get("themes", [])
        ],
    }

    prompt = prompt_template.format(
        question_text=question_text,
        project_background=project_config.resolved_project_background,
        analysis_data_json=json.dumps(data_summary, indent=2),
    )

    messages = [{"role": "user", "content": prompt}]

    try:
        cache_key = _normalize_cache_key(prompt, config.smart_model, "headline_summary")
        cached = llm_cache.get(cache_key)
        if cached is not None:
            logging.debug(f"[CACHE] Headline/summary hit for question '{question_text}'")
            response_json_str = cached
        else:
            response_json_str = client.chat_completion(
                messages,
                model_name=config.smart_model,
                temperature=0.5
            )
            llm_cache.set(cache_key, response_json_str)
            logging.debug(f"[CACHE] Headline/summary miss, computed and cached for question '{question_text}'")
        narrative = json.loads(response_json_str)
        return {
            "headline": narrative.get("headline", "No headline generated."),
            "summary": narrative.get("summary", "No summary generated."),
        }
    except (json.JSONDecodeError, Exception) as e:
        print(
            f"Warning: Failed to generate narrative for question '{question_text}'. Error: {e}"
        )
        return {"headline": "Error", "summary": "Error generating summary."}


def generate_executive_summary(all_analyses: List[Dict[str, Any]]) -> str:
    """
    Generates the final executive summary based on all individual question analyses.
    """
    client = LLMFactory.get_client("synthesis")
    config = LLMFactory.get_task_config("synthesis")
    prompt_template = load_prompt("generate_executive_summary")

    # Create a concise summary string to feed into the final prompt
    summary_strings = []
    for analysis in all_analyses:
        q_text = analysis.get("questionText", "Unknown Question")
        headline = analysis.get("headline", "")
        summary = analysis.get("summary", "")
        summary_strings.append(
            f"For the question '{q_text}', the key finding was: '{headline}' - {summary}"
        )

    all_summaries_text = "\n\n".join(summary_strings)

    prompt = prompt_template.format(
        project_background=project_config.resolved_project_background,
        all_question_summaries=all_summaries_text,
    )
    messages = [{"role": "user", "content": prompt}]

    try:
        cache_key = _normalize_cache_key(prompt, config.smart_model, "executive_summary")
        cached = llm_cache.get(cache_key)
        if cached is not None:
            logging.debug(f"[CACHE] Executive summary hit.")
            return cached
        else:
            result = client.chat_completion(
                messages, model_name=config.smart_model, temperature=0.5
            )
            llm_cache.set(cache_key, result)
            logging.debug(f"[CACHE] Executive summary miss, computed and cached.")
            return result
    except Exception as e:
        print(f"Warning: Failed to generate executive summary. Error: {e}")
        return "Error generating executive summary."


def assemble_final_report(
    all_question_analyses: List[Dict[str, Any]],
    report_title: str = "Consumer Privacy Market: Thematic Analysis",
) -> Dict[str, Any]:
    """Assembles the full report JSON object, including the executive summary."""
    print("Generating final executive summary...")
    executive_summary = generate_executive_summary(all_question_analyses)

    final_report = {
        "reportTitle": report_title,
        "executiveSummary": executive_summary,
        "questionAnalyses": all_question_analyses,
    }

    return final_report
