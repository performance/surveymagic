# src/analysis/classifier.py

import json
from typing import List, Dict, Optional

from src.llm_utils.llm_factory import LLMFactory, load_prompt

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
    prompt_template = load_prompt("classify_response")
    
    themes_json_str = json.dumps(themes, indent=2)

    for pid, response_text in responses.items():
        prompt = prompt_template.format(
            themes_json=themes_json_str,
            user_response=response_text
        )
        
        messages = [{"role": "user", "content": prompt}]
        
        try:
            assigned_theme = client.chat_completion(
                messages,
                model_name=config.fast_model,
                temperature=0.0 # We want deterministic classification
            ).strip()
            
            # Ensure the LLM returned a valid theme title
            valid_titles = {t['theme_title'] for t in themes}
            if assigned_theme in valid_titles:
                classifications[pid] = assigned_theme
            else:
                # Handle cases where the LLM hallucinates or returns a non-matching title
                print(f"Warning: Classifier returned an invalid theme title '{assigned_theme}' for participant {pid}. Skipping.")
                
        except Exception as e:
            print(f"Warning: Failed to classify response for participant {pid}. Error: {e}")
            
    return classifications