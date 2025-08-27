# config/project_config.py

from pydantic import BaseModel, Field
from typing import List
from datetime import datetime
from pathlib import Path
import pandas as pd
import logging


class ProjectConfig(BaseModel):
    # --- Analysis parameters ---
    k_samples_for_validation: int = 5
    min_theme_occurrence_percentage: float = 0.6  # Theme must appear in 60% of K samples
    similarity_threshold_trie_coalescing: float = 0.90
    similarity_threshold_theme_merging: float = 0.75
    max_quotes_per_theme: int = 3

    # --- Input files ---
    project_background: str = "data/input_vpn/project_background.txt"
    learning_objectives: str = "data/input_vpn/learning_objectives.txt"
    input_file: str = "data/input_vpn/vpn_sample_data.xlsx"  # "data/input/olsdr.xlsx"

    # --- Output config ---
    output_dir: str = "data/output"
    use_timestamped_output_dir: bool = True
    report_file: str = "report.json"

    # --- Logging / cache ---
    log_level: str = "DEBUG"  # Options: DEBUG, INFO, WARNING, ERROR, CRITICAL
    log_file: str = "app_info.log"
    cache_db: str = "data/cache.sqlite"

    # --- Internal (not part of schema) ---
    init_timestamp: str = Field(default_factory=lambda: datetime.now().strftime("%Y%m%d-%H%M%S"), exclude=True)

    def get_text_or_file(self, value: str) -> str:
        """Return file contents if value is a valid path, else return the value itself."""
        p = Path(value)
        if p.is_file():
            return p.read_text(encoding="utf-8")
        return value

    @property
    def resolved_project_background(self) -> str:
        return self.get_text_or_file(self.project_background)

    @property
    def resolved_learning_objectives(self) -> str:
        return self.get_text_or_file(self.learning_objectives)

    @property
    def resolved_learning_objectives_list(self) -> List[str]:
        """Parses the learning objectives text into a clean list of objectives."""
        objectives : List[str] = []
        for line in self.resolved_learning_objectives.splitlines():
            line = line.strip()
            if line.lower().startswith("objective"):
                # Clean up "Objective 1: " prefix
                objective_text = line.split(":", 1)[-1].strip()
                objectives.append(objective_text)
        return objectives

    @property
    def run_output_dir_path(self) -> str:
        """Stable per-run output directory path (cached with timestamp)."""
        if self.use_timestamped_output_dir:
            return str(Path(self.output_dir).absolute() / self.init_timestamp)
        return str(Path(self.output_dir).absolute())

    @property
    def latest_output_dir_symlink(self) -> str:
        return str(Path(self.output_dir).absolute() / "latest")

    @property
    def question_columns(self) -> List[str]:
        """Try to read columns from the input Excel file (excluding ID)."""
        try:
            df = pd.read_excel(self.input_file)
            return [col for col in df.columns if col.lower() != "id"]
        except Exception as e:
            logging.warning(f"Could not read columns from {self.input_file}: {e}")
            return []


# Instantiate once to be imported by other modules
project_config = ProjectConfig()
