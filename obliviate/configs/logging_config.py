from dataclasses import dataclass
from typing import Optional


@dataclass
class LoggingConfig:
    use_wandb: bool = False
    wandb_entity: Optional[str] = None
    wandb_project: str = "obliviate"
    steps_between_validation: int = 100
