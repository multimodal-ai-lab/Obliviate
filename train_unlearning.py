import argparse
from obliviate.unlearning.config import UnlearningConfig
from obliviate.unlearning.training import run_unlearning_training
from obliviate.configs.model_config import ModelType, ModelConfig
from obliviate.configs.project_config import ProjectConfig
from obliviate.configs.logging_config import LoggingConfig
from obliviate.unlearning.target_config import Target


def main():
    parser = argparse.ArgumentParser(description="Train unlearning for concept erasure")
    
    # Model arguments
    parser.add_argument(
        "--model_type",
        type=str,
        required=True,
        choices=[m.value for m in ModelType],
        help="Type of model to use"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Optional checkpoint to load (for student model)"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to use for training"
    )
    parser.add_argument(
        "--image_size",
        type=int,
        default=512,
        help="Image size for generation and training"
    )
    
    # Data arguments
    parser.add_argument(
        "--target",
        type=str,
        required=True,
        choices=[t.value for t in Target],
        help="Target concept to unlearn (see choices)"
    )
    # Training arguments
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed for reproducibility"
    )
    parser.add_argument(
        "--eta",
        type=float,
        default=1.0,
        help="Negative strength parameter for negation formula"
    )
    parser.add_argument(
        "--max_steps",
        type=int,
        default=1000,
        help="Maximum training steps"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="Batch size"
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-4,
        help="Learning rate"
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Gradient accumulation steps"
    )
    parser.add_argument(
        "--max_grad_norm",
        type=float,
        default=0.5,
        help="Maximum gradient norm for clipping"
    )
    parser.add_argument(
        "--warmup_ratio",
        type=float,
        default=0.01,
        help="Warmup ratio for learning rate scheduler"
    )
    parser.add_argument(
        "--scheduler_name",
        type=str,
        default="linear",
        help="Learning rate scheduler name"
    )
    
    # LoRA arguments
    parser.add_argument(
        "--lora_rank",
        type=int,
        default=32,
        help="LoRA rank"
    )
    parser.add_argument(
        "--lora_alpha",
        type=int,
        default=16,
        help="LoRA alpha"
    )
    parser.add_argument(
        "--lora_dropout",
        type=float,
        default=0.05,
        help="LoRA dropout"
    )
    # Checkpointing
    parser.add_argument(
        "--steps_between_checkpoints",
        type=int,
        default=250,
        help="Steps between checkpoints"
    )
    parser.add_argument(
        "--steps_between_validation",
        type=int,
        default=250,
        help="Steps between validation (currently unused)"
    )
    parser.add_argument(
        "--run_name",
        type=str,
        default=None,
        help="Name for this training run"
    )
    parser.add_argument(
        "--exp_path",
        type=str,
        default=None,
        help="Experiment path for grouping runs (defaults to ProjectConfig.exp_path, typically 'default')"
    )
    parser.add_argument(
        "--run_name_template",
        type=str,
        default="<target>_eta<eta>_<timestamp>",
        help="Template for run name. Supported placeholders: <target>, <eta>, <model_type>, <max_steps>, <learning_rate>, <timestamp>. Default: '<target>_eta<eta>_<timestamp>'"
    )
    
    args = parser.parse_args()
    
    # Convert target string to enum
    target_enum = Target(args.target)
    
    # Create configs
    model_config = ModelConfig(
        model_type=args.model_type,
        checkpoint=args.checkpoint,
        device=args.device,
        image_size=args.image_size
    )
    
    project_config = ProjectConfig()
    logging_config = LoggingConfig()
    
    # Override exp_path if provided via CLI
    if args.exp_path is not None:
        project_config.exp_path = args.exp_path
    
    logging_config.steps_between_validation = args.steps_between_validation
    
    # Create unlearning config
    config = UnlearningConfig(
        model_config=model_config,
        project_config=project_config,
        logging_config=logging_config,
        seed=args.seed,
        eta=args.eta,
        max_steps=args.max_steps,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        max_grad_norm=args.max_grad_norm,
        warmup_ratio=args.warmup_ratio,
        scheduler_name=args.scheduler_name,
        target=target_enum,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        steps_between_checkpoints=args.steps_between_checkpoints,
        run_name_template=args.run_name_template,
        device=args.device,
    )
    
    # Run training
    print("Starting unlearning training...")
    print(f"Model: {config.model_config.model_type.value}")
    print(f"Target: {config.target.value}")
    print(f"Experiment path: {config.project_config.exp_path}")
    print(f"Eta (negation strength): {config.eta}")
    print(f"Max steps: {config.max_steps}")
    
    checkpoint_path = run_unlearning_training(config)
    print(f"Training complete! Checkpoint saved to: {checkpoint_path}")


if __name__ == "__main__":
    main()

