# src/data_processing/quality_scorer.py

from typing import List, Dict, Optional
from src.llm_utils.llm_factory import LLMFactory, load_prompt

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
        
        # Use a faster model for this simple classification task
        quality = client.chat_completion(messages, model_name=config.fast_model, temperature=0.1)
        
        # Clean up the response
        quality = quality.strip().lower()
        if quality in ["high_effort", "low_effort", "non_answer"]:
            return quality
        return "low_effort" # Default to low_effort if LLM gives an unexpected response

    except Exception as e:
        print(f"Error assessing response quality: {e}")
        return "unknown" # Don't let this optional step fail the whole pipeline