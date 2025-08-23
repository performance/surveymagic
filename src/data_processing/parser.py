# src/data_processing/parser.py

import re
from typing import List, Dict, Optional, TypedDict

class ConversationTurn(TypedDict):
    speaker: str
    text: str

def parse_conversation(text: Optional[str]) -> List[ConversationTurn]:
    """
    Parses a raw string from a spreadsheet cell into a list of conversation turns.
    
    Args:
        text: The raw string, e.g., "assistant: Q1 user: A1 assistant: Q2 user: A2"

    Returns:
        A list of dictionaries, where each dictionary represents a turn.
        e.g., [{'speaker': 'assistant', 'text': 'Q1'}, {'speaker': 'user', 'text': 'A1'}]
    """
    if not text or not isinstance(text, str) or text.strip() == "":
        return []

    # Regex to split the text by "assistant:" or "user:" markers
    # The `(?=...)` is a positive lookahead to keep the delimiters
    parts = re.split(r'(?=assistant:|user:)', text, flags=re.IGNORECASE)
    
    turns: List[ConversationTurn] = []
    for part in parts:
        part = part.strip()
        if not part:
            continue
            
        if part.lower().startswith('assistant:'):
            speaker = 'assistant'
            content = part[len('assistant:'):].strip()
        elif part.lower().startswith('user:'):
            speaker = 'user'
            content = part[len('user:'):].strip()
        else:
            # This handles cases where the first part of the split is empty
            continue
        
        # Filter out empty or placeholder content
        if content and content.lower() not in ['n/a', 'na']:
            turns.append({'speaker': speaker, 'text': content})
            
    return turns

def flatten_user_responses(turns: List[ConversationTurn]) -> str:
    """
    Extracts and concatenates all user responses from a conversation.
    """
    return " ".join([turn['text'] for turn in turns if turn['speaker'] == 'user'])