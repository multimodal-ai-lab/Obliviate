import os
import random
import numpy as np
from datetime import datetime
import re
from enum import Enum

import torch
from tqdm import tqdm
from transformers import StoppingCriteria
from peft import LoraConfig, get_peft_model
import wandb


def build_random_generator(seed, device):
    """Build a random generator for reproducibility."""
    if seed is not None:
        g = torch.Generator(device=device)
        g.manual_seed(seed)
    else:
        g = None
    return g


def batch_iterator(iterable, batch_size):
    """Yield successive batches from an iterable."""
    total = len(iterable)
    if batch_size is None or batch_size <= 0 or batch_size >= total:
        yield iterable
    else:
        for i in range(0, total, batch_size):
            yield iterable[i:i + batch_size]


class TQDMProgressBar(StoppingCriteria):
    """Progress bar for generation."""
    def __init__(self, total_steps):
        self.pbar = tqdm(total=total_steps, desc="Generating", dynamic_ncols=True)

    def __call__(self, input_ids, scores, **kwargs):
        self.pbar.update(1)
        return False  # never stop early

    def close(self):
        self.pbar.close()


def create_checkpoint_saver(model, checkpoint_base_path, use_wandb=True):
    """Create a checkpoint saving function with closure over model and path."""

    def save_checkpoint(step, is_final=False):
        """Save model checkpoint at specified step."""
        if is_final:
            checkpoint_name = "checkpoint_final"
        else:
            checkpoint_name = f"checkpoint_step_{step}"

        checkpoint_path = os.path.join(checkpoint_base_path, checkpoint_name)
        model.save_pretrained(checkpoint_path)
        print(f"Model checkpoint saved to: {checkpoint_path}")

        if use_wandb:
            wandb.log({
                "checkpoint/step": step,
                "checkpoint/path": checkpoint_path
            }, step=step)

        return checkpoint_path

    return save_checkpoint


def handle_checkpoint_saving(save_checkpoint_fn, steps_between_checkpoints, max_steps, global_step):
    """Handle checkpoint saving logic based on save_every parameter."""
    if steps_between_checkpoints is not None and (global_step + 1) % steps_between_checkpoints == 0:
        return save_checkpoint_fn(step=global_step + 1)
    elif (global_step + 1) == max_steps:
        return save_checkpoint_fn(step=global_step + 1, is_final=True)


def fill_templated_run_name(config):
    """
    Fill templated run_name with actual values.
    
    Supports dotted attribute paths:
    - <target>: Target value (e.g., "nudity", "gore")
    - <eta>: Negation strength parameter
    - <model_config.model_type>: Model type (e.g., "liquid", "emu3")
    - <max_steps>: Maximum training steps
    - <learning_rate>: Learning rate
    - <timestamp>: Current timestamp (YYYYMMDD_HHMMSS)
    - Any other config attribute via dotted path (e.g., <model_config.model_type.value>)
    """
    
    if config.run_name_template is None:
        return
    
    def safe_get(path: str):
        """Helper to safely resolve dotted attributes like 'model_config.model_type'."""
        parts = path.split(".")
        obj = config
        for part in parts:
            obj = getattr(obj, part, None)
            if obj is None:
                return None
        
        # Convert Enums to their value
        if isinstance(obj, Enum):
            return obj.value
        
        return obj
    
    def replace_placeholder(match):
        expr = match.group(1)
        # Try to get from config using dotted path
        obj = safe_get(expr)
        if obj is None:
            # If not found, return placeholder as-is
            return f"<{expr}>"
        insertion = str(obj)
        return insertion
    
    config.run_name = re.sub(r"<([^>]+)>", replace_placeholder, config.run_name_template)
    
    # Handle timestamp separately (always available)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    config.run_name = config.run_name.replace("<timestamp>", timestamp)


def set_seed(seed: int = 42):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def setup_lora_for_wrapper(wrapper, config):
    """
    Set up LoRA for the model wrapper: inject LoRA into all target modules
    """
    target_modules = wrapper.get_modules_for_adaptation(ignore_keywords=None)
    lora_config = LoraConfig(
        r=config.lora_rank,
        lora_alpha=config.lora_alpha,
        target_modules=target_modules,
        lora_dropout=config.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
    )
    wrapper.model = get_peft_model(wrapper.model, lora_config)
    print(f"✅ Injected LoRA adapters (all layers trainable)!")
    wrapper.model.print_trainable_parameters()
    return wrapper


def extract_checkpoint_step(checkpoint_path):
    """
    Extract checkpoint step from checkpoint path.
    """
    if checkpoint_path is None:
        return None
    
    checkpoint_name = os.path.basename(checkpoint_path.rstrip(os.sep))
    
    if checkpoint_name.startswith("checkpoint_step_"):
        step = checkpoint_name.replace("checkpoint_step_", "")
        return f"step_{step}"
    return None


def parse_image_folder_path(image_folder, project_config):
    """
    Parse image folder path to extract exp_path, checkpoint_step, cfg_mode, target (optional), prompts_id.

    With target (negative_prompt): {exp_path}/{checkpoint_step}/{cfg_mode}/{target}/{prompts_id} (5+ parts).
    Without target: {exp_path}/{checkpoint_step}/{cfg_mode}/{prompts_id} (4 parts).
    """
    if not image_folder or not project_config or project_config.output_folder not in image_folder:
        return None, None, None, None, None

    rel_path = os.path.relpath(image_folder, os.path.join(project_config.output_folder, "images"))
    parts = [p for p in rel_path.split(os.sep) if p]

    if not parts:
        return None, None, None, None, None

    # Original model: original_{model_type}/{cfg_mode}/{prompts_id} (3 parts)
    if parts[0].startswith("original_"):
        if len(parts) >= 3:
            return parts[0], None, parts[1], None, parts[2]
        return None, None, None, None, None

    # Checkpoint path with target (negative_prompt): at least 5 parts
    if len(parts) >= 5:
        return os.path.join(*parts[:-4]), parts[-4], parts[-3], parts[-2], parts[-1]
    # Checkpoint path without target: 4 parts
    if len(parts) >= 4:
        return os.path.join(*parts[:-3]), parts[-3], parts[-2], None, parts[-1]

    return None, None, None, None, None


def derive_exp_path(checkpoint=None, image_folder=None, model_type=None, project_config=None):
    """
    Derive experiment path from checkpoint or image_folder.
    Returns only the experiment path, without checkpoint_step/cfg_mode/prompts_id.
    """
    if checkpoint:
        checkpoint_parts = checkpoint.split(os.sep)
        if 'checkpoints' in checkpoint_parts:
            idx = checkpoint_parts.index('checkpoints')
            exp_path_parts = checkpoint_parts[idx + 1:-1]
            return os.path.join(*exp_path_parts) if exp_path_parts else ''
        return os.path.basename(os.path.dirname(checkpoint))
    
    if image_folder and project_config:
        exp_path, _, _, _, _ = parse_image_folder_path(image_folder, project_config)
        if exp_path:
            return exp_path
    
    if not model_type:
        return 'original'
    model_str = model_type.value if hasattr(model_type, 'value') else model_type
    return f'original_{model_str}'


def build_results_log_path(exp_path, checkpoint_step=None, cfg_mode=None, prompts_id=None, filename=None, target=None):
    """
    Build relative results log file path for individual results.

    Structure: {exp_path}/{checkpoint_step}/{cfg_mode}/{target?}/{prompts_id}/{filename}
    For original models: {exp_path}/{cfg_mode}/{prompts_id}/{filename}
    """
    path_parts = [exp_path]

    if checkpoint_step:
        path_parts.append(checkpoint_step)

    if cfg_mode:
        path_parts.append(cfg_mode)

    if target:
        path_parts.append(target)

    if prompts_id:
        path_parts.append(prompts_id)

    if filename:
        path_parts.append(filename)

    return os.path.join(*path_parts) if path_parts else exp_path


def extract_path_components(checkpoint, image_folder, prompts_file, cfg_mode, project_config):
    """
    Extract path components (checkpoint_step, cfg_mode, prompts_id, target) from checkpoint or image_folder.
    """
    if image_folder:
        # Parse from image_folder if provided
        _, checkpoint_step, cfg_mode_str, target_str, prompts_id = parse_image_folder_path(image_folder, project_config)
        return checkpoint_step, cfg_mode_str, prompts_id, target_str
    else:
        # Extract from checkpoint or use provided values
        checkpoint_step = extract_checkpoint_step(checkpoint) if checkpoint else None
        cfg_mode_str = cfg_mode.value if hasattr(cfg_mode, 'value') else str(cfg_mode)
        prompts_id = os.path.splitext(os.path.basename(prompts_file))[0] if prompts_file else None
        return checkpoint_step, cfg_mode_str, prompts_id, None


def build_evaluation_metadata(checkpoint=None, image_folder=None, prompts_file=None, cfg_mode=None,
                                guidance_scale=None, model_type=None, project_config=None, **kwargs):
    """
    Build evaluation metadata for CSV logging.
    """
    if guidance_scale is None and model_type is not None:
        from obliviate.models import resolve_guidance_scale
        guidance_scale = resolve_guidance_scale(model_type, None)

    metadata = {}
    
    # Extract checkpoint_step
    if checkpoint:
        checkpoint_step = extract_checkpoint_step(checkpoint)
        if checkpoint_step:
            metadata["checkpoint_step"] = checkpoint_step
    elif image_folder and project_config:
        _, checkpoint_step, _, _, _ = parse_image_folder_path(image_folder, project_config)
        if checkpoint_step:
            metadata["checkpoint_step"] = checkpoint_step

    # Extract cfg_mode
    if cfg_mode:
        metadata["cfg_mode"] = cfg_mode.value if hasattr(cfg_mode, 'value') else str(cfg_mode)
    elif image_folder and project_config:
        _, _, cfg_mode_parsed, _, _ = parse_image_folder_path(image_folder, project_config)
        if cfg_mode_parsed:
            metadata["cfg_mode"] = cfg_mode_parsed

    # Extract prompts_id
    if prompts_file:
        prompts_id = os.path.splitext(os.path.basename(prompts_file))[0]
        metadata["prompts_id"] = prompts_id
    elif image_folder and project_config:
        _, _, _, _, prompts_id = parse_image_folder_path(image_folder, project_config)
        if prompts_id:
            metadata["prompts_id"] = prompts_id

    # Extract target (for negative_prompt runs) from path if not in kwargs
    if "target" not in metadata and image_folder and project_config:
        _, _, _, target_parsed, _ = parse_image_folder_path(image_folder, project_config)
        if target_parsed:
            metadata["target"] = target_parsed
    
    # Add guidance_scale if provided
    if guidance_scale is not None:
        metadata["guidance_scale"] = guidance_scale
    
    # Add any additional kwargs
    metadata.update(kwargs)
    
    return metadata

