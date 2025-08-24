# config/project_config.py

from pydantic import BaseModel
from typing import List, Dict

class ProjectConfig(BaseModel):
    project_background: str = "data/input/project_background.txt"  # Can be a string or a file path
    learning_objectives: str = "data/input/learning_objectives.txt"  # Can be a string or a file path

    def get_text_or_file(self, value: str) -> str:
        import os
        # If value is a file path and exists, load from file
        if isinstance(value, str) and os.path.isfile(value):
            with open(value, 'r', encoding='utf-8') as f:
                return f.read()
        return value

    @property
    def resolved_project_background(self) -> str:
        return self.get_text_or_file(self.project_background)

    @property
    def resolved_learning_objectives(self) -> str:
        return self.get_text_or_file(self.learning_objectives)
    # Analysis parameters
    k_samples_for_validation: int = 5
    min_theme_occurrence_percentage: float = 0.6 # Theme must appear in 60% of K samples
    similarity_threshold_trie_coalescing: float = 0.90
    similarity_threshold_theme_merging: float = 0.85
    max_quotes_per_theme: int = 3
    
    # File paths
    input_file: str = "data/input/vpn_sample_data.xlsx" # raw_survey_data.xlsx"
    output_dir: str = "data/output"
    report_file: str = "report.json"
    
    # The columns in the Excel file to be analyzed
    # Note: Column A (ID) is assumed to be the first column
    @property
    def question_columns(self) -> List[str]:
        import pandas as pd
        try:
            df = pd.read_excel(self.input_file)
            # Exclude the first column (assumed to be ID)
            return [col for col in df.columns if col.lower() != 'id']
        except Exception as e:
            print(f"Warning: Could not read columns from {self.input_file}: {e}")
            # Fallback to empty list
            return []
    # Logging configuration
    log_level: str = "INFO"  # Options: DEBUG, INFO, WARNING, ERROR, CRITICAL
    log_file: str = "data/output/app_info.log"


# Instantiate once to be imported by other modules
project_config = ProjectConfig()