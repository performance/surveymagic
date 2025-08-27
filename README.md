
# Qualitative Thematic Analysis Pipeline

This project provides a robust, modular pipeline for thematic analysis of qualitative survey/interview data using Large Language Models (LLMs) and advanced data structures (conversational tries). It supports scalable, reproducible research and is ready for collaboration.

## Project Structure

- `config/` — LLM and project configuration (Pydantic-based)
- `prompts/` — Modular prompt files for LLM tasks
- `src/` — Core Python code: LLM utilities, data processing, analysis, report generation
- `data/input/` — Raw survey/interview data (Excel)
- `data/output/` — Generated reports and classification files

## Features

- Conversational trie for multi-turn analysis
- Semantic coalescing and K-fold thematic validation
- Modular LLM interaction (OpenAI, Anthropic, etc.)
- Pydantic config management
- Scalable, cache-enabled design

## Installation

1. Clone the repository:
    ```bash
    git clone <repository_url>
    cd <project_directory>
    ```
2. Create a virtual environment:
    ```bash
    python -m venv .venv
    source .venv/bin/activate  # On Windows: .venv\Scripts\activate
    ```
3. Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```
4. Configure API keys:
    - Create a `.env` file in the project root:
      ```
      OPENAI_API_KEY="sk-..."
      ANTHROPIC_API_KEY="sk-..."
      ```
    - See `config/llm_config.py` for details.
5. Place your data file (e.g., `raw_survey_data.xlsx`) in `data/input/`.

## Usage

To run the full analysis pipeline:
```bash
python src/main.py
```
Output files will be saved in `data/output/`:
- `report.json` — Main report
- `q*_classifications.xlsx` — Per-question classifications

## Visualizer

To view reports and logs in a browser, run the Flask visualizer:
```bash
python src/visualizer/main.py
```
Then open `http://localhost:5000` in your browser.

## Contributing

We welcome contributions! See `CONTRIBUTING.md` for guidelines.

- Use feature branches and submit pull requests.
- Follow PEP8 and project code style.
- Add or update prompt files in `prompts/` for new LLM tasks.
- Add tests for new modules.

## License

MIT License (see LICENSE file)

## Key Features

*   **Conversational Trie:** Organizes multi-turn conversations for deeper analysis of interaction patterns.
*   **Semantic Coalescing:** Automatically groups semantically similar questions and user responses within the trie.
*   **K-Fold Thematic Validation (Bootstrapping):** Ensures robust and stable themes by running analysis on multiple data samples.
*   **Modular LLM Interaction:** Uses an LLM factory and external prompt files for easy customization and provider switching.
*   **Pydantic Configuration:** Manages LLM and project parameters with type safety and validation.
*   **Scalable Design:** Built with future expansion to larger datasets and more questions in mind.
