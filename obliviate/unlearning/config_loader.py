import os
import yaml
from obliviate.unlearning.config import UnlearningConfig
from obliviate.configs.model_config import ModelConfig, ModelType
from obliviate.configs.project_config import ProjectConfig
from obliviate.configs.logging_config import LoggingConfig
from obliviate.unlearning.target_config import Target


def load_config_from_yaml(config_path: str) -> UnlearningConfig:
    """
    Load UnlearningConfig from a YAML file.
    
    Args:
        config_path: Path to YAML config file
        
    Returns:
        UnlearningConfig instance
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with open(config_path, 'r') as f:
        config_dict = yaml.safe_load(f)
    
    # Extract nested configs
    model_dict = config_dict.get('model', {})
    project_dict = config_dict.get('project', {})
    logging_dict = config_dict.get('logging', {})
    
    # Create ModelConfig
    model_config = ModelConfig(
        model_type=model_dict.get('model_type', 'liquid'),
        checkpoint=model_dict.get('checkpoint', None),
        device=model_dict.get('device', 'cuda'),
        image_size=model_dict.get('image_size', 512)
    )
    
    # Create ProjectConfig
    project_config = ProjectConfig(
        exp_path=project_dict.get('exp_path', 'default'),
        checkpoints_folder=project_dict.get('checkpoints_folder', 'checkpoints'),
        data_folder=project_dict.get('data_folder', 'data'),
        output_folder=project_dict.get('output_folder', 'outputs'),
        eval_gemini_model=project_dict.get('eval_gemini_model', 'gemini-2.5-flash')
    )

    # Create LoggingConfig
    logging_config = LoggingConfig(
        use_wandb=logging_dict.get('use_wandb', False),
        wandb_entity=logging_dict.get('wandb_entity'),
        wandb_project=logging_dict.get('wandb_project', 'obliviate'),
        steps_between_validation=logging_dict.get('steps_between_validation', 100),
    )
    
    # Convert target string to enum
    target_str = config_dict.get('data', {}).get('target')
    if target_str is None:
        raise ValueError("'target' is required in config file under 'data' section")
    target = Target(target_str)
    
    # Create UnlearningConfig
    config = UnlearningConfig(
        model_config=model_config,
        project_config=project_config,
        logging_config=logging_config,
        seed=config_dict.get('training', {}).get('seed', 42),
        eta=config_dict.get('training', {}).get('eta', 7.0),
        max_steps=config_dict.get('training', {}).get('max_steps', 1000),
        batch_size=config_dict.get('training', {}).get('batch_size', 1),
        learning_rate=config_dict.get('training', {}).get('learning_rate', 1e-4),
        gradient_accumulation_steps=config_dict.get('training', {}).get('gradient_accumulation_steps', 1),
        max_grad_norm=config_dict.get('training', {}).get('max_grad_norm', 0.5),
        warmup_ratio=config_dict.get('training', {}).get('warmup_ratio', 0.01),
        scheduler_name=config_dict.get('training', {}).get('scheduler_name', 'linear'),
        target=target,
        lora_rank=config_dict.get('lora', {}).get('lora_rank', 32),
        lora_alpha=config_dict.get('lora', {}).get('lora_alpha', 16),
        lora_dropout=config_dict.get('lora', {}).get('lora_dropout', 0.05),
        target_modules=config_dict.get('lora', {}).get('target_modules', None),
        steps_between_checkpoints=config_dict.get('experiment', {}).get('steps_between_checkpoints', 100),
        run_name_template=config_dict.get('experiment', {}).get('run_name_template', '<target>_eta<eta>_<timestamp>'),
        device=config_dict.get('training', {}).get('device', 'cuda')
    )
    
    return config

