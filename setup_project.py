import os

def create_project_structure():
    project_root = os.getcwd()

    # Define directories
    dirs = [
        "config",
        "prompts",
        "src/llm_utils",
        "src/data_processing",
        "src/analysis",
        "data/input",
        "data/output"
    ]

    for d in dirs:
        os.makedirs(os.path.join(project_root, d), exist_ok=True)
        # Create __init__.py for Python packages
        if "src" in d and not d.endswith("src"):
            with open(os.path.join(project_root, d, "__init__.py"), 'w') as f:
                pass # Empty __init__.py

    # Define placeholder files
    placeholder_files = {
        "config": [
            "llm_config.py",
            "project_config.py"
        ],
        "prompts": [
            "extract_keywords.txt",
            "generate_candidate_themes.txt",
            "merge_and_finalize_themes.txt",
            "classify_response.txt",
            "select_quotes.txt",
            "generate_headline_summary.txt",
            "generate_executive_summary.txt",
            "assess_response_quality.txt" # For optional Step 0
        ],
        "src/llm_utils": [
            "llm_factory.py",
            "embedding_utils.py"
        ],
        "src/data_processing": [
            "parser.py",
            "trie_builder.py",
            "quality_scorer.py"
        ],
        "src/analysis": [
            "thematic_analyzer.py",
            "classifier.py",
            "quote_selector.py"
        ],
        "src": [
            "report_generator.py",
            "main.py"
        ],
        "data/input": [
            "raw_survey_data.xlsx" # For the actual data
        ],
        "": ["README.md"] # Root level
    }

    # Create placeholder files with initial content (for some)
    for folder, files in placeholder_files.items():
        for file_name in files:
            path = os.path.join(project_root, folder, file_name)
            if not os.path.exists(path):
                with open(path, 'w') as f:
                    if file_name == "llm_config.py":
                        f.write("""from pydantic import BaseModel, Field
from typing import Literal

class LLMConfig(BaseModel):
    provider: Literal['openai', 'anthropic'] = 'openai'
    api_key: str = Field(..., env='OPENAI_API_KEY' if provider == 'openai' else 'ANTHROPIC_API_KEY')
    # Default models for various tasks
    fast_model: str = 'gpt-3.5-turbo' # For quick tasks like keyword extraction
    smart_model: str = 'gpt-4-turbo' # For complex tasks like theme generation, summaries
    embedding_model: str = 'text-embedding-3-small' # For semantic similarity

class ProjectLLMConfig(BaseModel):
    openai_api_key: str = Field(None, env='OPENAI_API_KEY')
    anthropic_api_key: str = Field(None, env='ANTHROPIC_API_KEY')
    
    # Specific configurations for each LLM provider/task
    llm_configs: dict[str, LLMConfig] = {
        "default": LLMConfig(), # Default config can be overridden
        "keyword_extraction": LLMConfig(provider="openai", fast_model="gpt-3.5-turbo"),
        "theme_generation": LLMConfig(provider="openai", smart_model="gpt-4-turbo"),
        "theme_merging": LLMConfig(provider="openai", smart_model="gpt-4-turbo"),
        "classification": LLMConfig(provider="openai", fast_model="gpt-3.5-turbo"),
        "summary_generation": LLMConfig(provider="openai", smart_model="gpt-4-turbo"),
        "embedding": LLMConfig(provider="openai", embedding_model="text-embedding-3-small"),
        "quality_assessment": LLMConfig(provider="openai", fast_model="gpt-3.5-turbo"),
    }
""")
                    elif file_name == "project_config.py":
                        f.write("""from pydantic import BaseModel
from typing import List, Dict

class ProjectConfig(BaseModel):
    project_background: str = \"\"\"
The primary goal of this research study is to understand the consumer privacy market,
specifically in the areas of network privacy (VPNs) and data deletion services. We aim
to size the market, identify key customer needs and use cases, and validate product-
market fit for CLIENT’s offerings in these spaces. The insights from this study will inform
CLIENT’s go-to-market strategy and product roadmap to best address the needs of the
target market.
\"\"\"
    learning_objectives: List[Dict[str, str]] = [
        {"Objective 1": "Understand the size and segmentation of the consumer privacy market, including current usage of VPNs, identity protection, and data deletion services."},
        {"Application 1": "Validate and refine market sizing assumptions to guide business planning and resource allocation."},
        {"Objective 2": "Identify the key use cases, pain points, and unmet needs driving consumer adoption of privacy solutions."},
        {"Application 2": "Prioritize product features and messaging based on the most compelling value propositions for target users."},
        {"Objective 3": "Assess willingness to pay and preferred pricing models for VPN, identity protection, and data deletion offerings."},
        {"Application 3": "Optimize packaging and pricing of solutions to maximize market penetration and revenue."}
    ]
    k_samples_for_validation: int = 5 # Number of bootstrap samples for theme validation
    min_theme_occurrence_percentage: float = 0.7 # Theme must appear in 70% of K samples to be stable
    similarity_threshold_trie_coalescing: float = 0.95 # Cosine similarity for merging nodes in trie
    similarity_threshold_theme_merging: float = 0.85 # Cosine similarity for merging themes across samples
    max_quotes_per_theme: int = 3
    output_dir: str = "data/output"
""")
                    elif file_name == "llm_factory.py":
                        f.write("""from openai import OpenAI
from anthropic import Anthropic
from pydantic import BaseModel, Field
from typing import Literal, Dict

from config.llm_config import LLMConfig

class LLMClient:
    def __init__(self, config: LLMConfig):
        self.config = config
        if self.config.provider == 'openai':
            self._client = OpenAI(api_key=self.config.api_key)
        elif self.config.provider == 'anthropic':
            self._client = Anthropic(api_key=self.config.api_key)
        else:
            raise ValueError(f"Unsupported LLM provider: {self.config.provider}")

    def chat_completion(self, messages: list[dict], model_name: str, temperature: float = 0.7, max_tokens: int = 1000):
        if self.config.provider == 'openai':
            response = self._client.chat.completions.create(
                model=model_name,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens
            )
            return response.choices[0].message.content
        elif self.config.provider == 'anthropic':
            response = self._client.messages.create(
                model=model_name,
                max_tokens=max_tokens,
                messages=messages,
                temperature=temperature
            )
            return response.content[0].text # Anthropic returns a list of content blocks
        # Add other providers here if needed

    def get_embedding(self, text: str, model_name: str):
        if self.config.provider == 'openai':
            response = self._client.embeddings.create(
                input=[text],
                model=model_name
            )
            return response.data[0].embedding
        elif self.config.provider == 'anthropic':
            # Anthropic currently does not offer direct embedding API
            # For this case, we might fall back to OpenAI's embedding model even if primary provider is Anthropic,
            # or use a local embedding model. For simplicity, we'll assume OpenAI's embedding is used here if needed.
            # In a production setup, a dedicated embedding service/model would be used.
            raise NotImplementedError("Anthropic does not have a direct embedding API. Configure a dedicated embedding model via OpenAI or local for Anthropic primary provider.")
        # Add other providers here if needed

class LLMFactory:
    _instances: Dict[str, LLMClient] = {}

    @classmethod
    def get_llm_client(cls, task_name: str) -> LLMClient:
        # Load the overall project LLM config
        from config.llm_config import ProjectLLMConfig
        project_llm_config = ProjectLLMConfig()

        # Get specific LLM config for the task, or fallback to default
        llm_config_for_task = project_llm_config.llm_configs.get(task_name, project_llm_config.llm_configs["default"])

        # Determine the API key based on the provider chosen for this task
        if llm_config_for_task.provider == 'openai':
            api_key = project_llm_config.openai_api_key
        elif llm_config_for_task.provider == 'anthropic':
            api_key = project_llm_config.anthropic_api_key
        else:
            api_key = llm_config_for_task.api_key # Fallback if provider not in top-level project config
        
        # Override the task-specific config's api_key if found from project_llm_config
        if api_key:
            llm_config_for_task.api_key = api_key

        instance_key = f"{llm_config_for_task.provider}-{llm_config_for_task.fast_model}-{llm_config_for_task.smart_model}-{llm_config_for_task.embedding_model}-{api_key}"
        
        if instance_key not in cls._instances:
            cls._instances[instance_key] = LLMClient(llm_config_for_task)
        return cls._instances[instance_key]
""")
                    elif file_name == "README.md":
                        f.write("""# Qualitative Thematic Analysis Pipeline

This project implements a robust and modular pipeline for performing thematic analysis on qualitative interview data, leveraging Large Language Models (LLMs) and advanced data structures (conversational tries).

## Project Structure

*   `config/`: Configuration files for LLM settings and project-specific parameters.
*   `prompts/`: A collection of all LLM prompts, one per file, ensuring modularity and easy iteration.
*   `src/`: Contains the core Python source code, organized into modules for LLM utilities, data processing, analysis, and report generation.
*   `data/`: Input and output directories for raw survey data and generated reports.

## Setup

1.  **Clone the repository:**
    ```bash
    git clone <repository_url>
    cd <project_directory>
    ```

2.  **Create a virtual environment (recommended):**
    ```bash
    python -m venv .venv
    source .venv/bin/activate  # On Windows: .venv\\Scripts\\activate
    ```

3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt # (You'll need to create this file, e.g., pandas, openai, anthropic, pydantic, numpy, scikit-learn)
    ```

4.  **Configure API Keys:**
    Create a `.env` file in the project root or set environment variables for your LLM API keys:
    ```
    OPENAI_API_KEY="sk-your-openai-key"
    ANTHROPIC_API_KEY="sk-your-anthropic-key"
    ```
    Ensure `config/llm_config.py` is updated to reflect your chosen LLM providers and models.

5.  **Place Data:**
    Place your `raw_survey_data.xlsx` file in the `data/input/` directory.

## Usage

To run the full analysis:

  ~~~bash
python src/main.py
   ~~~

The generated report (`report.json`) and individual question classification files (`q*_classifications.xlsx`) will be saved in `data/output/`.

## Key Features

*   **Conversational Trie:** Organizes multi-turn conversations for deeper analysis of interaction patterns.
*   **Semantic Coalescing:** Automatically groups semantically similar questions and user responses within the trie.
*   **K-Fold Thematic Validation (Bootstrapping):** Ensures robust and stable themes by running analysis on multiple data samples.
*   **Modular LLM Interaction:** Uses an LLM factory and external prompt files for easy customization and provider switching.
*   **Pydantic Configuration:** Manages LLM and project parameters with type safety and validation.
*   **Scalable Design:** Built with future expansion to larger datasets and more questions in mind.
""")
                    # Create empty files otherwise
                    else:
                        pass

if __name__ == "__main__":
    create_project_structure()
    print("Project structure generated successfully.")
    print("Please install dependencies (e.g., pip install pandas openai anthropic pydantic numpy scikit-learn) and set up your .env file with API keys.")