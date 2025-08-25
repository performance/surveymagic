import re

class PromptFactory:
    """Loads and manages prompt templates from the prompts/ directory."""
    def __init__(self, prompts_dir: str = 'prompts'):
        self.prompts_dir = prompts_dir
        self.templates = {}
        self._load_all_templates()

    def _load_all_templates(self):
        import os
        for fname in os.listdir(self.prompts_dir):
            if fname.endswith('.txt'):
                path = os.path.join(self.prompts_dir, fname)
                with open(path, 'r', encoding='utf-8') as f:
                    self.templates[fname[:-4]] = f.read()

    def get_template(self, name: str) -> str:
        if name in self.templates:
            return self.templates[name]
        raise KeyError(f"Prompt template '{name}' not found.")

    def list_identifiers(self, name: str) -> list:
        """Returns a list of identifiers (keys) required for rendering the template."""
        template = self.get_template(name)
        # Find all {key} occurrences not inside double braces
        return sorted(set(re.findall(r'(?<!\{)\{([a-zA-Z0-9_]+)\}(?!\})', template)))

    def render(self, name: str, substitutions: dict) -> str:
        """Renders the template with the given substitutions. Raises if any required key is missing."""
        required = self.list_identifiers(name)
        missing = [k for k in required if k not in substitutions]
        if missing:
            raise ValueError(f"Missing required substitutions for prompt '{name}': {missing}")
        template = self.get_template(name)
        # Substitute in sorted key order for stability
        for k in sorted(substitutions.keys()):
            v = substitutions[k]
            if isinstance(v, dict):
                import json
                v = json.dumps(v, sort_keys=True, separators=(",", ":"))
            template = template.replace(f'{{{k}}}', str(v))
        # Collapse whitespace
        template = re.sub(r'\s+', ' ', template).strip()
        return template

# Singleton instance
prompt_factory = PromptFactory()
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

    def chat_completion(self, messages: List[Dict], model_name: str, temperature: float = 0.0, max_tokens: int = 2048, substitutions: Dict[str, any] = None) -> str:
        import re, json
        def render_message(msg, subs):
            content = msg.get('content', '')
            # If substitutions provided, replace {key} with value in sorted order
            if subs:
                for k in sorted(subs.keys()):
                    v = subs[k]
                    # If value is dict, render as canonical JSON
                    if isinstance(v, dict):
                        v = json.dumps(v, sort_keys=True, separators=(",", ":"))
                    content = content.replace(f'{{{k}}}', str(v))
            # Collapse whitespace
            content = re.sub(r'\s+', ' ', content).strip()
            return {**msg, 'content': content}
        # Render all messages
        rendered_messages = [render_message(m, substitutions) for m in messages]
        if self.provider == 'openai':
            response = self._client.chat.completions.create(
                model=model_name,
                messages=rendered_messages,
                temperature=temperature,
                max_tokens=max_tokens
            )
            return response.choices[0].message.content
        elif self.provider == 'anthropic':
            # Anthropic expects the system prompt separately
            system_prompt = ""
            if rendered_messages[0]['role'] == 'system':
                system_prompt = rendered_messages[0]['content']
                rendered_messages = rendered_messages[1:]
            response = self._client.messages.create(
                model=model_name,
                max_tokens=max_tokens,
                messages=rendered_messages,
                system=system_prompt,
                temperature=temperature
            )
            return response.content[0].text
        return ""

    def get_embedding(self, text: any, model_name: str) -> List[float]:
        import re, json
        # If text is dict, render as canonical JSON
        if isinstance(text, dict):
            text = json.dumps(text, sort_keys=True, separators=(",", ":"))
        text = text.replace("\n", " ")
        text = re.sub(r'\s+', ' ', text).strip()
        if self.provider == 'openai':
            response = self._client.embeddings.create(input=[text], model=model_name)
            return response.data[0].embedding
        else:
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