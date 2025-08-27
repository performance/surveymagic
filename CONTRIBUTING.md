
# Contributing Guidelines

Thank you for considering contributing to this project! We welcome improvements, bug fixes, new features, and prompt engineering.

## How to Contribute

1. **Fork the repository** and create a feature branch:
    ```bash
    git checkout -b feature/your-feature-name
    ```
2. **Code Style:**
    - Follow PEP8 for Python code.
    - Use clear, descriptive variable and function names.
    - Add docstrings to new functions/classes.
3. **Prompt Engineering:**
    - Add new prompt files to `prompts/`.
    - Name files clearly by task (e.g., `extract_keywords.txt`).
    - Document prompt changes in your PR.
4. **Configuration:**
    - Update `config/llm_config.py` or `config/project_config.py` for new settings.
    - Use Pydantic models for validation.
5. **Testing:**
    - Add or update tests for new modules or features.
    - Manual testing: Run `python src/main.py` and verify output in `data/output/`.
6. **Pull Requests:**
    - Submit a PR with a clear description of your changes.
    - Reference related issues if applicable.
    - Ensure your branch is up to date with `main` before submitting.

## Environment Setup

- Use a virtual environment (`python -m venv .venv`).
- Install dependencies with `pip install -r requirements.txt`.
- Add your LLM API keys to `.env`.

## Adding New LLM Providers

- Update `config/llm_config.py` and `src/llm_utils/llm_factory.py`.
- Add provider-specific logic and document usage in README.md.

## Issues & Feedback

- Please use GitHub Issues for bug reports and feature requests.

---
Thank you for helping improve this project!
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
