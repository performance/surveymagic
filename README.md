# Qualitative Thematic Analysis Pipeline

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
    source .venv/bin/activate  # On Windows: .venv\Scriptsctivate
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
