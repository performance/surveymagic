
### **Explanation of Components:**

*   **`config/`:** Contains all configurable parameters. `llm_config.py` uses Pydantic to enforce structure and validation for LLM settings (provider, API keys, model names for different tasks). `project_config.py` holds project-specific settings like the background, objectives, and thresholds for similarity/sampling.
*   **`prompts/`:** A dedicated directory for all LLM prompts. Each prompt is a separate `.txt` file. This makes it trivial to iterate on prompt engineering without touching Python code, improving reusability across projects.
*   **`src/llm_utils/llm_factory.py`:** This implements the LLM factory pattern. It's responsible for:
    *   Reading the `llm_config.py`.
    *   Instantiating and caching LLM client objects (e.g., `openai.OpenAI`, `anthropic.Anthropic`).
    *   Providing a consistent interface (`chat_completion`, `get_embedding`) regardless of the underlying LLM provider.
    *   It intelligently pulls API keys from environment variables, enhancing security.
*   **`src/llm_utils/embedding_utils.py`:** Will contain helper functions for computing and comparing embeddings, crucial for trie node coalescing and theme merging.
*   **`src/data_processing/`:** Modules dedicated to preparing the raw data: parsing conversation cells into trees, building the full trie, and scoring respondent quality.
*   **`src/analysis/`:** Modules for the core thematic analysis: running the K-fold sampling, generating themes, classifying responses, and selecting quotes.
*   **`src/report_generator.py`:** A simple script to take the final structured output from `analysis/` and assemble it into the agreed-upon JSON schema.
*   **`src/main.py`:** The entry point. It will read config, load data, orchestrate the parallel processing of questions, and trigger report generation.
*   **`data/`:** Standard input/output directories.
*   **`README.md`:** Essential for project documentation, setup, and usage.
