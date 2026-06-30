import os
import math
import time
import yaml
from dataclasses import asdict
from typing import Optional, List

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from transformers import get_scheduler
import tqdm
import wandb

from obliviate.unlearning.config import UnlearningConfig
from obliviate.unlearning.objectives import KLObjective
from obliviate.data.dataset import UnlearningDataset, generate_target_images_cache
from obliviate.unlearning.target_config import get_validation_prompts_for_concept
from obliviate.models.model_wrapper import ModelWrapper
from obliviate.models import create_wrapper_from_config
from obliviate.configs.model_config import ModelConfig, ModelType
from obliviate.utils.misc import create_checkpoint_saver, handle_checkpoint_saving, fill_templated_run_name, set_seed, setup_lora_for_wrapper
from obliviate.utils.logging import log_generated_images_to_wandb, format_elapsed_seconds
from obliviate.data.dataset_features import get_collate_function_for_tokenizer, extend_collate_fn_for_target_logits
from obliviate.data.data_utils import get_cached_image_count
from obliviate.unlearning.target_config import TARGET_PROMPTS


def create_validation_function(
    student_wrapper: ModelWrapper,
    validation_prompts: List[str],
    log_id: str = "validation/samples"
):
    """
    Create a validation function that generates images from validation prompts and logs them to wandb.
    
    Args:
        student_wrapper: The student model wrapper
        validation_prompts: List of validation prompts to generate images from
        log_id: Wandb log identifier for the images
        
    Returns:
        A validation function that takes a step number and runs validation
    """
    def do_validation(step: int):
        """
        Run validation: generate images from validation prompts and log to wandb.
        
        Args:
            step: Current training step
        """
        print(f"Running validation at step {step}")
        
        student_wrapper.eval()
        
        # Set image size to 720 for emu3_gen validation
        validation_image_size = 720 if student_wrapper.config.model_type == ModelType.EMU3_GEN else None
        
        with torch.no_grad():
            # Generate images from validation prompts
            generated_images = list(student_wrapper.sample_text_to_image(
                prompts=validation_prompts,
                image_size=validation_image_size
            ))
        
        # Log images to wandb as a grid
        log_generated_images_to_wandb(
            generated_images=generated_images,
            log_id=log_id,
            n_cols=len(validation_prompts),
            step=step
        )
        
        student_wrapper.train()
    
    return do_validation


def run_unlearning_training(
    config: UnlearningConfig,
    student_wrapper: Optional[ModelWrapper] = None,
):
    """
    Run unlearning training loop.
    Uses a single wrapper with LoRA; extracts teacher logits by temporarily disabling adapters.
    
    Args:
        config: UnlearningConfig
        student_wrapper: Optional student model wrapper (if None, creates from config)
    """
    # Create model wrapper
    if student_wrapper is None:
        student_wrapper = create_wrapper_from_config(config.model_config)

    # student_wrapper.model.gradient_checkpointing_enable()
    
    # Generate cached images BEFORE adding LoRA (using base model)
    # Images are generated with base model; logits are extracted with disable_adapter() later
    image_cache_dir = os.path.join(
        config.project_config.data_folder,
        "unlearning_cache",
        config.model_config.model_type.value,
        config.project_config.exp_path,
        config.target.value
    )
    
    # Check how many images are already cached (supports resume when max_steps is increased)
    cached_count = get_cached_image_count(image_cache_dir)
    num_images = config.max_steps
    if cached_count < num_images:
        print("Generating target images with base model (before LoRA)...")
        print("Using concept-level training: single target prompt with different seeds")
        if cached_count > 0:
            print(f"Resuming: {cached_count} images already cached, generating {num_images - cached_count} more (indices {cached_count}..{num_images - 1})")

        # Get single target prompt
        target_prompt = TARGET_PROMPTS.get(config.target, TARGET_PROMPTS[config.target])

        # Create full prompts list (all same prompt, will use different seeds during generation)
        prompts = [target_prompt] * num_images
        print(f"Generating up to {num_images} images from target prompt: '{target_prompt}'")
        print("Using different random seeds for diversity (ESD-style)")

        generate_target_images_cache(
            prompts=prompts,
            cache_dir=image_cache_dir,
            wrapper=student_wrapper,  # Use base model (before LoRA)
            start_index=cached_count,
        )
    else:
        print(f"✅ Image cache complete: {cached_count} images already present for {num_images} steps.")
    
    # Set up LoRA for student model BEFORE dataset building
    student_wrapper = setup_lora_for_wrapper(student_wrapper, config)
    
    # Create dataset (after LoRA setup)
    # Teacher logits are extracted by temporarily disabling adapters
    dataset = UnlearningDataset(
        config=config,
        wrapper=student_wrapper,
    )
    
    # Create dataloader with extended collate function
    collate_fn = extend_collate_fn_for_target_logits(
        get_collate_function_for_tokenizer(student_wrapper.right_padded_tokenizer)
    )
    dataloader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=True,
        drop_last=True,
        collate_fn=collate_fn
    )
    
    # Fill templated run_name
    fill_templated_run_name(config)
    run_name = config.run_name
    
    # Checkpoint path: {checkpoints_folder}/{model_type}/{exp_path}/{target}/{run_name}/
    run_base_path = os.path.join(
        config.project_config.checkpoints_folder,
        config.model_config.model_type.value,
        config.project_config.exp_path,
        config.target.value,
        run_name
    )
    os.makedirs(run_base_path, exist_ok=True)
    
    # Save config
    with open(os.path.join(run_base_path, "config.yaml"), "w") as f:
        yaml.dump(asdict(config), f, default_flow_style=False, sort_keys=False)
    
    # Set random seed for reproducibility
    set_seed(config.seed)

    use_wandb = config.logging_config.use_wandb

    # Initialize wandb
    if use_wandb:
        wandb_init_kwargs = {
            "project": config.logging_config.wandb_project,
            "name": run_name,
            "config": asdict(config),
        }
        wandb_entity = config.logging_config.wandb_entity
        if wandb_entity:
            wandb_init_kwargs["entity"] = wandb_entity
        wandb.init(**wandb_init_kwargs)

        # Function to count parameters
        def count_parameters(model):
            total_params = sum(p.numel() for p in model.parameters())
            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            return {
                "total_params": total_params,
                "trainable_params": trainable_params,
                "trainable_percentage": 100 * trainable_params / total_params
            }

        param_stats = count_parameters(student_wrapper.model)
        wandb.config.update({
            "total_params": param_stats["total_params"],
            "trainable_params": param_stats["trainable_params"],
            "trainable_percentage": param_stats["trainable_percentage"]
        })
    else:
        print("W&B logging disabled (logging.use_wandb=false)")

    # Create checkpoint saver function
    save_checkpoint = create_checkpoint_saver(student_wrapper.model, run_base_path, use_wandb=use_wandb)
    last_checkpoint_path = None
    
    # Prepare model for training
    student_wrapper.train()
    student_wrapper.model.to(config.device)
    
    # Create objective
    objective_fn = KLObjective()
    
    # Create optimizer
    optimizer = optim.AdamW(
        student_wrapper.model.parameters(),
        lr=config.learning_rate
    )
    
    # Create scheduler
    total_training_steps = math.ceil(config.max_steps / config.gradient_accumulation_steps)
    scheduler = get_scheduler(
        config.scheduler_name,
        optimizer=optimizer,
        num_warmup_steps=int(config.warmup_ratio * total_training_steps),
        num_training_steps=total_training_steps
    )
    
    # Create validation function if validation is enabled and target is provided
    validation_fn = None
    if use_wandb and config.logging_config.steps_between_validation > 0 and config.target is not None:
        validation_prompts = get_validation_prompts_for_concept(config.target)
        validation_fn = create_validation_function(
            student_wrapper,
            validation_prompts,
            log_id="validation/samples"
        )
    elif use_wandb and config.logging_config.steps_between_validation > 0:
        print("Warning: Validation is enabled but no target provided. Skipping validation.")
    
    # Initial validation
    if validation_fn is not None:
        validation_fn(step=0)
    
    # Training loop
    data_iter = iter(dataloader)
    training_start = time.perf_counter()

    for global_step in tqdm.tqdm(range(config.max_steps), desc="Unlearning Training"):
        try:
            batch = next(data_iter)
            if batch is None:
                continue
        except StopIteration:
            data_iter = iter(dataloader)
            batch = next(data_iter)

        # Forward pass
        with torch.amp.autocast('cuda', dtype=torch.bfloat16):
            log_dict = {"step": global_step + 1}

            # Compute the loss on the batch
            batch_loss, metrics = objective_fn(
                student_wrapper.model,
                batch,
                config,
                student_wrapper
            )

            # Log metrics
            log_dict.update({f"train/{k}": v for k, v in metrics.items()})

        # Backward pass
        (batch_loss / config.gradient_accumulation_steps).backward()

        # Log to wandb
        log_dict.update({
            f"train/total_loss": batch_loss.item(),
            f"train/learning_rate": scheduler.get_last_lr()[0]
        })

        # Optimizer step
        if (global_step + 1) % config.gradient_accumulation_steps == 0:
            torch.nn.utils.clip_grad_norm_(
                student_wrapper.model.parameters(),
                config.max_grad_norm
            )
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

        if use_wandb:
            wandb.log(log_dict, step=global_step + 1)

        # Validation
        if validation_fn is not None and (global_step + 1) % config.logging_config.steps_between_validation == 0:
            validation_fn(step=global_step + 1)

        # Checkpoint saving
        last_checkpoint_path = handle_checkpoint_saving(save_checkpoint, config.steps_between_checkpoints,
                                                        config.max_steps, global_step=global_step)

    elapsed = time.perf_counter() - training_start
    print(f"Unlearning training complete! Training took {format_elapsed_seconds(elapsed)} ({elapsed:.1f}s)")
    if use_wandb:
        wandb.finish()
    return last_checkpoint_path

