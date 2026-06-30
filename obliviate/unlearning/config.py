from dataclasses import dataclass
from typing import Optional
from obliviate.configs.model_config import ModelConfig
from obliviate.configs.project_config import ProjectConfig
from obliviate.configs.logging_config import LoggingConfig
from obliviate.unlearning.target_config import Target


@dataclass
class UnlearningConfig:
    """Configuration for unlearning training."""
    
    target: Target
    
    # Model configuration
    model_config: ModelConfig = None
    project_config: ProjectConfig = None
    logging_config: LoggingConfig = None
    
    # Training parameters
    seed: int = 42
    eta: float = 2.0  # Negative strength parameter for negation formula
    max_steps: int = 1000
    batch_size: int = 1
    learning_rate: float = 1e-4
    gradient_accumulation_steps: int = 1
    max_grad_norm: float = 0.5
    warmup_ratio: float = 0.01
    scheduler_name: str = "linear"
    
    # LoRA parameters (if using LoRA)
    lora_rank: int = 32
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    target_modules: Optional[str] = None

    # Checkpointing and naming
    steps_between_checkpoints: Optional[int] = 250
    run_name_template: str = "<model_type>_<target>_eta<eta>_<timestamp>"
    run_name: Optional[str] = None  # Generated from template by fill_templated_run_name()
    
    # Device
    device: str = 'cuda'
    
    def __post_init__(self):
        if self.model_config is None:
            self.model_config = ModelConfig()
        if self.project_config is None:
            self.project_config = ProjectConfig()
        if self.logging_config is None:
            self.logging_config = LoggingConfig()
        if self.device is None:
            self.device = self.model_config.device



