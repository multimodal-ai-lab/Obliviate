import argparse
from enum import Enum

from obliviate.configs.model_config import ModelType
from obliviate.unlearning.config_loader import load_config_from_yaml
from obliviate.unlearning.training import run_unlearning_training


def apply_config_override(config, key, value):
    """Apply a single KEY=VALUE override to UnlearningConfig."""
    model_config_aliases = {
        "model_type": (ModelType, "model_config", "model_type"),
        "image_size": (int, "model_config", "image_size"),
        "checkpoint": (str, "model_config", "checkpoint"),
    }
    project_config_aliases = {
        "exp_path": (str, "project_config", "exp_path"),
        "data_folder": (str, "project_config", "data_folder"),
    }
    logging_config_aliases = {
        "use_wandb": ("bool", "logging_config", "use_wandb"),
        "wandb_entity": (str, "logging_config", "wandb_entity"),
        "wandb_project": (str, "logging_config", "wandb_project"),
        "steps_between_validation": (int, "logging_config", "steps_between_validation"),
    }

    nested_aliases = {**model_config_aliases, **project_config_aliases, **logging_config_aliases}

    if key in nested_aliases:
        cast, parent_attr, child_attr = nested_aliases[key]
        if cast == "bool":
            casted = value.lower() in ("true", "1", "yes")
        elif cast is str:
            casted = None if value.lower() == "null" else value
        elif cast is int:
            casted = int(value)
        else:
            casted = cast(value)
        setattr(getattr(config, parent_attr), child_attr, casted)
        return True

    if not hasattr(config, key):
        return False

    current_value = getattr(config, key)
    if isinstance(current_value, bool):
        setattr(config, key, value.lower() in ("true", "1", "yes"))
    elif isinstance(current_value, Enum):
        setattr(config, key, type(current_value)(value))
    elif isinstance(current_value, int):
        setattr(config, key, int(value))
    elif isinstance(current_value, float):
        setattr(config, key, float(value))
    else:
        setattr(config, key, value)
    return True


def main():
    parser = argparse.ArgumentParser(description="Train unlearning from a YAML config file")
    
    parser.add_argument(
        "--config",
        type=str,
        help="Path to YAML config file"
    )
    
    parser.add_argument(
        "--override",
        nargs="*",
        metavar="KEY=VALUE",
        help="Override config values. Format: --override key1=value1 key2=value2"
    )
    
    args = parser.parse_args()
    
    # Load config from file
    print(f"Loading config from: {args.config}")
    config = load_config_from_yaml(args.config)
    
    # Apply overrides if provided
    if args.override:
        for override in args.override:
            if '=' not in override:
                parser.error(f"Invalid override format: {override}. Use KEY=VALUE")
            key, value = override.split('=', 1)
            if apply_config_override(config, key, value):
                print(f"  Override: {key} = {value}")
            else:
                print(f"  Warning: Unknown config key '{key}', skipping")
    
    # Run training
    print("\nStarting unlearning training...")
    print(f"Model: {config.model_config.model_type.value}")
    print(f"Target: {config.target.value}")
    print(f"Experiment path: {config.project_config.exp_path}")
    print(f"Eta (negation strength): {config.eta}")
    print(f"Max steps: {config.max_steps}\n")
    
    checkpoint_path = run_unlearning_training(config)
    print(f"\nTraining complete! Checkpoint saved to: {checkpoint_path}")


if __name__ == "__main__":
    main()

