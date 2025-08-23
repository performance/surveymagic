# config/llm_config.py

from pydantic_settings import BaseSettings
from pydantic import Field
from typing import Literal, Dict

class LLMTaskConfig(BaseSettings):
    """Configuration for a specific LLM task."""
    provider: Literal['openai', 'anthropic'] = 'openai'
    # Model names can be overridden per task
    fast_model: str = 'gpt-3.5-turbo'
    smart_model: str = 'gpt-4-turbo-preview'
    embedding_model: str = 'text-embedding-3-small'

class ProjectLLMConfig(BaseSettings):
    """Top-level LLM configuration, reads from .env file."""
    openai_api_key: str = Field(..., env='OPENAI_API_KEY')
    anthropic_api_key: str = Field(..., env='ANTHROPIC_API_KEY')

    # Default task configurations can be overridden by the user
    tasks: Dict[str, LLMTaskConfig] = {
        "keyword_extraction": LLMTaskConfig(provider="openai"),
        "theme_generation": LLMTaskConfig(provider="openai"),
        "classification": LLMTaskConfig(provider="openai"),
        "synthesis": LLMTaskConfig(provider="openai"),
        "embedding": LLMTaskConfig(provider="openai"),
        "quality_assessment": LLMTaskConfig(provider="openai"),
    }

    class Config:
        env_file = '.env'
        env_file_encoding = 'utf-8'

# Instantiate once to be imported by other modules
llm_config = ProjectLLMConfig()