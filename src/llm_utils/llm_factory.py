# src/llm_utils/llm_factory.py

from openai import OpenAI
from anthropic import Anthropic
from typing import Dict, List
import os

from config.llm_config import llm_config, LLMTaskConfig

class LLMClient:
    """A unified client for interacting with different LLM providers."""
    def __init__(self, provider: str, api_key: str):
        self.provider = provider
        if self.provider == 'openai':
            self._client = OpenAI(api_key=api_key)
        elif self.provider == 'anthropic':
            self._client = Anthropic(api_key=api_key)
        else:
            raise ValueError(f"Unsupported LLM provider: {self.provider}")

    def chat_completion(self, messages: List[Dict], model_name: str, temperature: float = 0.2, max_tokens: int = 2048) -> str:
        if self.provider == 'openai':
            response = self._client.chat.completions.create(
                model=model_name,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens
            )
            return response.choices[0].message.content
        elif self.provider == 'anthropic':
            # Anthropic expects the system prompt separately
            system_prompt = ""
            if messages[0]['role'] == 'system':
                system_prompt = messages[0]['content']
                messages = messages[1:]
            
            response = self._client.messages.create(
                model=model_name,
                max_tokens=max_tokens,
                messages=messages,
                system=system_prompt,
                temperature=temperature
            )
            return response.content[0].text
        return ""

    def get_embedding(self, text: str, model_name: str) -> List[float]:
        text = text.replace("\n", " ")
        if self.provider == 'openai':
            response = self._client.embeddings.create(input=[text], model=model_name)
            return response.data[0].embedding
        else:
            # Fallback to OpenAI for embeddings if Anthropic is the primary provider,
            # as they don't have a dedicated embedding API.
            print("Warning: Anthropic provider selected but does not have an embedding API. Falling back to OpenAI for embeddings.")
            openai_client = OpenAI(api_key=llm_config.openai_api_key)
            response = openai_client.embeddings.create(input=[text], model=model_name)
            return response.data[0].embedding

class LLMFactory:
    """Factory to create and manage LLMClient instances."""
    _instances: Dict[str, LLMClient] = {}

    @classmethod
    def get_client(cls, task_name: str) -> LLMClient:
        task_config = llm_config.tasks.get(task_name, llm_config.tasks["classification"]) # Default to a fast model config
        
        provider = task_config.provider
        api_key = llm_config.openai_api_key if provider == 'openai' else llm_config.anthropic_api_key

        if provider not in cls._instances:
            cls._instances[provider] = LLMClient(provider, api_key)
        return cls._instances[provider]
    
    @classmethod
    def get_task_config(cls, task_name: str) -> LLMTaskConfig:
        return llm_config.tasks.get(task_name, llm_config.tasks["classification"])

def load_prompt(prompt_name: str) -> str:
    """Loads a prompt from the prompts/ directory."""
    path = os.path.join('prompts', f'{prompt_name}.txt')
    try:
        with open(path, 'r', encoding='utf-8') as f:
            return f.read()
    except FileNotFoundError:
        raise FileNotFoundError(f"Prompt file not found at: {path}")