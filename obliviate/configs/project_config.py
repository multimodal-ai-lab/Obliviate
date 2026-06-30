from dataclasses import dataclass
from typing import Optional


@dataclass
class ProjectConfig:
    exp_path: Optional[str] = "default"

    checkpoints_folder: str = "checkpoints"
    data_folder: str = "data"
    output_folder: str = "outputs"

    eval_gemini_model: str = "gemini-2.5-flash"
