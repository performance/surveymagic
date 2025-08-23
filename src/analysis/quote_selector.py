# src/analysis/quote_selector.py

import json
from typing import List, Dict, Any

from src.llm_utils.llm_factory import LLMFactory, load_prompt
from config.project_config import project_config

def select_quotes_for_theme(
    theme: Dict[str, str],
    classified_responses: List[Dict[str, str]]
) -> List[Dict[str, str]]:
    """
    Selects the best ~3 supporting quotes for a single theme.

    Args:
        theme: The theme dictionary (containing title and description).
        classified_responses: A list of dicts [{'participant_id': pid, 'response': text}] for this theme.

    Returns:
        A list of dicts, each representing a quote with participantId and quoteText.
    """
    if not classified_responses:
        return []

    client = LLMFactory.get_client("synthesis") # Use a smart model for this nuanced task
    config = LLMFactory.get_task_config("synthesis")
    prompt_template = load_prompt("select_quotes")
    
    # Format responses for the prompt, keeping it clean
    responses_for_prompt = [
        {"participantId": r['participant_id'], "responseText": r['response']} 
        for r in classified_responses
    ]
    
    prompt = prompt_template.format(
        theme_title=theme['theme_title'],
        theme_description=theme['theme_description'],
        responses_json=json.dumps(responses_for_prompt, indent=2)
    )
    
    messages = [{"role": "user", "content": prompt}]
    
    try:
        raw_ids = client.chat_completion(
            messages,
            model_name=config.smart_model,
            temperature=0.2
        )
        
        selected_pids = {pid.strip() for pid in raw_ids.split(',') if pid.strip()}
        
        # Retrieve the original, verbatim quotes using the selected IDs
        quotes = []
        # Ensure we don't add more than the max limit
        for resp in classified_responses:
            if resp['participant_id'] in selected_pids and len(quotes) < project_config.max_quotes_per_theme:
                quotes.append({
                    "participantId": resp['participant_id'],
                    "quoteText": resp['response']
                })
                selected_pids.remove(resp['participant_id']) # Prevent duplicate additions

        return quotes

    except Exception as e:
        print(f"Warning: Failed to select quotes for theme '{theme['theme_title']}'. Error: {e}")
        return []