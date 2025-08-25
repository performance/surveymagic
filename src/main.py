# src/main.py

import pandas as pd
import os
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Any
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from config.project_config import project_config
from src.data_processing.parser import parse_conversation, flatten_user_responses
from src.data_processing.trie_builder import Trie
from src.analysis.thematic_analyzer import find_stable_themes
from src.analysis.classifier import classify_responses
from src.analysis.quote_selector import select_quotes_for_theme
from src.report_generator import generate_question_narrative, assemble_final_report


def analyze_single_question(question_column: str, df: pd.DataFrame) -> Dict[str, Any]:
    """
    Runs the full analysis pipeline for a single question column.
    """
    print(f"\n--- Starting analysis for question: {question_column} ---")

    # --- 1. Data Processing & Trie Building ---
    question_trie = Trie(question_name=question_column)
    participant_responses = {}  # {pid: flattened_response_text}
    all_pids = []

    # Extract canonical question text from the first valid entry
    first_valid_text = df[question_column].dropna().iloc[0]
    canonical_question_text = parse_conversation(first_valid_text)[0]["text"]

    for index, row in df.iterrows():
        pid = str(row.iloc[0])  # First column is ID
        raw_text = row[question_column]

        if pd.notna(raw_text):
            conversation_turns = parse_conversation(raw_text)
            if conversation_turns:
                question_trie.insert(conversation_turns, pid)
                flattened_response = flatten_user_responses(conversation_turns)
                participant_responses[pid] = flattened_response
                all_pids.append(pid)

    if not participant_responses:
        print(f"No valid responses found for {question_column}. Skipping.")
        return {}

    # --- 2. Thematic Analysis ---
    stable_themes = find_stable_themes(question_trie, all_pids, canonical_question_text)
    if not stable_themes:
        print(f"Could not determine stable themes for {question_column}. Skipping.")
        return {}

    # --- 3. Classification ---
    print(
        f"Classifying {len(participant_responses)} responses for {question_column}..."
    )
    classifications = classify_responses(participant_responses, stable_themes)

    # Save classification details for inspection
    classification_df = pd.DataFrame(
        classifications.items(), columns=["ParticipantID", "Theme"]
    )
    classification_df["Response"] = classification_df["ParticipantID"].map(
        participant_responses
    )
    output_path = os.path.join(
        project_config.output_dir, f"{question_column}_classifications.xlsx"
    )
    classification_df.to_excel(output_path, index=False)
    print(f"Classification results saved to {output_path}")

    # --- 4. Quote Selection & Theme Assembly ---
    final_themes_with_quotes = []
    total_participants_analyzed = len(participant_responses)

    for theme in stable_themes:
        # Get all responses classified under this theme
        pids_in_theme = [
            pid for pid, t in classifications.items() if t == theme["theme_title"]
        ]
        responses_in_theme = [
            {"participant_id": pid, "response": participant_responses[pid]}
            for pid in pids_in_theme
        ]

        quotes = select_quotes_for_theme(theme, responses_in_theme)

        theme_data = {
            "theme_title": theme["theme_title"],
            "theme_description": theme["theme_description"],
            "participant_count": len(pids_in_theme),
            "participant_percentage": (
                len(pids_in_theme) / total_participants_analyzed
                if total_participants_analyzed > 0
                else 0
            ),
            "supporting_quotes": quotes,
        }
        final_themes_with_quotes.append(theme_data)

    # --- 5. Narrative Generation ---
    preliminary_analysis_data = {
        "question_text": canonical_question_text,
        "participants_analyzed": total_participants_analyzed,
        "themes": final_themes_with_quotes,
    }
    narrative = generate_question_narrative(
        canonical_question_text, preliminary_analysis_data
    )

    # Combine everything for this question
    final_question_analysis = {**preliminary_analysis_data, **narrative}

    print(f"--- Finished analysis for question: {question_column} ---")
    return final_question_analysis


def main():
    """Main function to orchestrate the entire pipeline."""
    # Force single-threaded execution
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["NUMEXPR_NUM_THREADS"] = "1"
    os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
    os.environ["NUMBA_NUM_THREADS"] = "1"
    # Ensure output directory exists
    os.makedirs(project_config.output_dir, exist_ok=True)

    # Load data
    try:
        df = pd.read_excel(project_config.input_file)
    except FileNotFoundError:
        print(f"Error: Input file not found at {project_config.input_file}")
        return

    # Run questions sequentially for strict single-threaded cache validation
    all_question_analyses = []
    for col in project_config.question_columns:
        try:
            result = analyze_single_question(col, df)
            if result:
                all_question_analyses.append(result)
        except Exception as e:
            print(f"An error occurred while processing {col}: {e}")

    # --- Final Report Assembly ---
    if not all_question_analyses:
        print("No questions were successfully analyzed. Exiting.")
        return

    final_report = assemble_final_report(all_question_analyses)

    # Save the final report
    report_path = os.path.join(project_config.output_dir, project_config.report_file)
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(final_report, f, indent=2)

    print(f"\n✅ Full analysis complete. Report saved to {report_path}")


if __name__ == "__main__":
    main()
