# src/report_generator.py

import json
from typing import List, Dict, Any

from src.llm_utils.llm_factory import LLMFactory, load_prompt, prompt_factory
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
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS llm_cache (
                    key TEXT PRIMARY KEY,
                    value TEXT
                )
            """
            )

    def get(self, key: str) -> str:
        with self.lock, sqlite3.connect(self.db_path) as conn:
            cur = conn.execute("SELECT value FROM llm_cache WHERE key=?", (key,))
            row = cur.fetchone()
            return row[0] if row else None

    def set(self, key: str, value: str):
        with self.lock, sqlite3.connect(self.db_path) as conn:
            conn.execute(
                "REPLACE INTO llm_cache (key, value) VALUES (?, ?)", (key, value)
            )


llm_cache = PersistentLLMCache(db_path="data/output/llm_cache.sqlite")


def _normalize_cache_key(prompt: str, model_name: str, task: str) -> str:
    import re

    norm_prompt = re.sub(r"\s+", " ", prompt).strip()
    key_raw = f"{task}:{model_name}:{norm_prompt}"
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
    themes_list = analysis_data.get("themes", [])
    sorted_themes = sorted(themes_list, key=lambda x: x.get("theme_title", ""))
    data_summary = {
        "participants_analyzed": analysis_data.get("participants_analyzed"),
        "themes": [
            {
                "theme_title": theme.get("theme_title"),
                "participant_count": theme.get("participant_count"),
                "participant_percentage": f"{theme.get('participant_percentage', 0):.0%}",
            }
            for theme in sorted_themes
        ],
    }
    substitutions = {
        "question_text": question_text,
        "project_background": project_config.resolved_project_background,
        "analysis_data_json": json.dumps(data_summary, sort_keys=True, separators=(",", ":"))
    }
    prompt = prompt_factory.render("generate_headline_summary", substitutions)
    messages = [{"role": "user", "content": prompt}]

    import json
    try:
        canonical_subs = json.dumps(substitutions, sort_keys=True, separators=(",", ":"))
        cache_key = _normalize_cache_key("generate_headline_summary:" + canonical_subs, config.smart_model, "headline_summary")
        cached = llm_cache.get(cache_key)
        if cached is not None:
            logging.debug(f"[CACHE] Headline/summary hit for question '{question_text}'")
            response_json_str = cached
        else:
            if os.environ.get("CACHE_ONLY_MODE", "0") == "1":
                raise RuntimeError(f"[CACHE-ONLY] Headline/summary cache miss for question '{question_text}'")
            response_json_str = client.chat_completion(
                messages, model_name=config.smart_model, temperature=0.0, substitutions=substitutions
            )
            llm_cache.set(cache_key, response_json_str)
            logging.debug(f"[CACHE] Headline/summary miss, computed and cached for question '{question_text}'")
        narrative = json.loads(response_json_str)
        return {
            "headline": narrative.get("headline", "No headline generated."),
            "summary": narrative.get("summary", "No summary generated."),
        }
    except json.JSONDecodeError as e:
        print(f"Warning: Failed to generate narrative for question '{question_text}'. Error: {e}")
        return {"headline": "Error", "summary": "Error generating summary."}
    except Exception as e:
        print(f"Warning: Failed to generate narrative for question '{question_text}'. Error: {e}")
        return {"headline": "Error", "summary": "Error generating summary."}


def generate_executive_summary(all_analyses: List[Dict[str, Any]]) -> str:
    """
    Generates the final executive summary based on all individual question analyses.
    """
    client = LLMFactory.get_client("synthesis")
    config = LLMFactory.get_task_config("synthesis")
    sorted_analyses = sorted(all_analyses, key=lambda x: x.get("question_text", ""))
    summary_strings = []
    for analysis in sorted_analyses:
        q_text = analysis.get("question_text", "Unknown Question")
        headline = analysis.get("headline", "")
        summary = analysis.get("summary", "")
        summary_strings.append(
            f"For the question '{q_text}', the key finding was: '{headline}' - {summary}"
        )
    all_summaries_text = "\n\n".join(summary_strings)
    substitutions = {
        "project_background": project_config.resolved_project_background,
        "all_question_summaries": all_summaries_text
    }
    prompt = prompt_factory.render("generate_executive_summary", substitutions)
    messages = [{"role": "user", "content": prompt}]

    import json
    try:
        canonical_subs = json.dumps(substitutions, sort_keys=True, separators=(",", ":"))
        cache_key = _normalize_cache_key("generate_executive_summary:" + canonical_subs, config.smart_model, "executive_summary")
        cached = llm_cache.get(cache_key)
        if cached is not None:
            logging.debug("[CACHE] Executive summary hit.")
            return cached
        else:
            if os.environ.get("CACHE_ONLY_MODE", "0") == "1":
                raise RuntimeError("[CACHE-ONLY] Executive summary cache miss.")
            result = client.chat_completion(
                messages, model_name=config.smart_model, temperature=0.0, substitutions=substitutions
            )
            llm_cache.set(cache_key, result)
            logging.debug("[CACHE] Executive summary miss, computed and cached.")
            return result
    except Exception as e:
        print(f"Warning: Failed to generate executive summary. Error: {e}")
        return "Error generating executive summary."


# Add these new functions inside src/report_generator.py


def map_questions_to_objectives(
    objectives: List[str], all_analyses: List[Dict[str, Any]]
) -> Dict[str, List[Dict[str, Any]]]:
    """Maps learning objectives to the question analyses that inform them."""
    client = LLMFactory.get_client("synthesis")
    config = LLMFactory.get_task_config("synthesis")
    prompt_template = load_prompt("map_objectives_to_questions")

    question_texts = [qa["question_text"] for qa in all_analyses]

    prompt = prompt_template.format(
        learning_objectives_json=json.dumps(objectives, indent=2),
        question_texts_json=json.dumps(question_texts, indent=2),
    )

    messages = [{"role": "user", "content": prompt}]

    try:
        cache_key = _normalize_cache_key(
            prompt, config.smart_model, "objective_mapping"
        )
        cached = llm_cache.get(cache_key)
        if cached is not None:
            logging.debug("[CACHE] Objective mapping hit.")
            mapping_json = json.loads(cached)
        else:
            if os.environ.get("CACHE_ONLY_MODE", "0") == "1":
                raise RuntimeError("[CACHE-ONLY] Objective mapping cache miss.")
            response_str = client.chat_completion(
                messages, model_name=config.smart_model, temperature=0.0
            )
            llm_cache.set(cache_key, response_str)
            logging.debug("[CACHE] Objective mapping miss, computed and cached.")
            mapping_json = json.loads(response_str)

        # Reconstruct the map with full analysis objects
        final_map = {}
        analyses_by_text = {qa["question_text"]: qa for qa in all_analyses}
        for objective, q_texts in mapping_json.items():
            final_map[objective] = [
                analyses_by_text[qt] for qt in q_texts if qt in analyses_by_text
            ]
        return final_map

    except (json.JSONDecodeError, Exception) as e:
        print(f"Warning: Failed to map objectives to questions. Error: {e}")
        return {obj: [] for obj in objectives}


def synthesize_objective_insights(
    objective_map: Dict[str, List[Dict[str, Any]]],
) -> List[Dict[str, Any]]:
    """Generates a synthesized insight for each learning objective."""
    insights = []
    client = LLMFactory.get_client("synthesis")
    config = LLMFactory.get_task_config("synthesis")
    prompt_template = load_prompt("synthesize_objective_insight")

    for objective, analyses in objective_map.items():
        if not analyses:
            continue

        key_findings = []
        for qa in analyses:
            key_findings.append(
                f"- From Q: '{qa['headline']}', we learned: {qa['summary']}"
            )

        prompt = prompt_template.format(
            objective_text=objective,
            key_findings_text="\n".join(key_findings),
        )

        messages = [{"role": "user", "content": prompt}]

        try:
            cache_key = _normalize_cache_key(
                prompt, config.smart_model, "objective_synthesis"
            )
            cached = llm_cache.get(cache_key)
            if cached is not None:
                logging.debug(
                    f"[CACHE] Synthesis hit for objective: {objective[:30]}..."
                )
                synthesis_text = cached
            else:
                if os.environ.get("CACHE_ONLY_MODE", "0") == "1":
                    raise RuntimeError(f"[CACHE-ONLY] Synthesis cache miss for objective: {objective[:30]}...")
                synthesis_text = client.chat_completion(
                    messages, model_name=config.smart_model, temperature=0.0
                )
                llm_cache.set(cache_key, synthesis_text)
                logging.debug(
                    f"[CACHE] Synthesis miss for objective: {objective[:30]}..."
                )

            insights.append(
                {
                    "objectiveText": objective,
                    "synthesis": synthesis_text.strip(),
                    "supportingAnalyses": [
                        {
                            "question_text": qa["question_text"],
                            "headline": qa["headline"],
                        }
                        for qa in analyses
                    ],
                }
            )
        except Exception as e:
            print(
                f"Warning: Failed to synthesize insight for objective '{objective}'. Error: {e}"
            )

    return insights


def assemble_final_report(
    all_question_analyses: List[Dict[str, Any]],
    report_title: str = "Consumer Privacy Market: Thematic Analysis",
) -> Dict[str, Any]:
    """Assembles the full report, now including the objective-driven layer."""

    # --- New Objective Synthesis Steps ---
    print("Mapping questions to learning objectives...")
    objectives_list = project_config.resolved_learning_objectives_list
    objective_map = map_questions_to_objectives(objectives_list, all_question_analyses)

    print("Synthesizing insights for each learning objective...")
    objective_insights = synthesize_objective_insights(objective_map)

    # --- Existing Executive Summary Step ---
    print("Generating final executive summary...")
    executive_summary = generate_executive_summary(all_question_analyses)

    final_report = {
        "report_title": report_title,
        "executive_summary": executive_summary,
        "insights_by_objective": objective_insights,  # <-- NEW SECTION
        "question_analyses": all_question_analyses,
    }

    return final_report
