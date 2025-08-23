# config/project_config.py

from pydantic import BaseModel
from typing import List, Dict

class ProjectConfig(BaseModel):
    project_background: str = """
The primary goal of this research study is to understand the consumer privacy market,
specifically in the areas of network privacy (VPNs) and data deletion services. We aim
to size the market, identify key customer needs and use cases, and validate product-
market fit for CLIENT’s offerings in these spaces. The insights from this study will inform
CLIENT’s go-to-market strategy and product roadmap to best address the needs of the
target market.
"""
    learning_objectives: str = """
- Objective 1: Understand the size and segmentation of the consumer privacy market.
- Objective 2: Identify the key use cases, pain points, and unmet needs.
- Objective 3: Assess willingness to pay and preferred pricing models.
"""
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
    question_columns: List[str] = [
        "vpn_selection",
        "unmet_needs_private_location",
        "unmet_needs_always_avail",
        "current_vpn_feedback",
        "remove_data_steps_probe_yes",
        "remove_data_steps_probe_no"
    ]


# Instantiate once to be imported by other modules
project_config = ProjectConfig()