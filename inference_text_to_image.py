import argparse
import os
import torch

from obliviate.configs.project_config import ProjectConfig
from obliviate.models import create_wrapper_from_config, resolve_guidance_scale
from obliviate.configs.model_config import ModelType, ModelConfig
from obliviate.configs.cfg_config import CFGMode
from obliviate.unlearning.target_config import Target
from obliviate.utils.load import load_prompts_from_csv
from obliviate.utils.misc import derive_exp_path, extract_checkpoint_step


def inference_text_to_image(
    model_type,
    checkpoint=None,
    prompts=None,
    prompts_file=None,
    batch_size=25,
    output_relpath=None,
    guidance_scale=None,
    temperature=0.99,
    top_p=0.96,
    top_k=4096,
    seed=None,
    cfg_mode=CFGMode.NONE,
    start_index=0,
    target=None,
    image_size=512,
):
    """Run inference with given model type and prompts, skipping existing images."""

    print("Starting Inference Setup...")
    project_config = ProjectConfig()

    # Build model configuration and wrapper
    model_config = ModelConfig(model_type=model_type, checkpoint=checkpoint)

    # Load prompts from file if provided
    if prompts_file:
        prompts = load_prompts_from_csv(prompts_file)

    # Apply start_index to skip prompts at the beginning
    if start_index > 0:
        if start_index >= len(prompts):
            raise ValueError(f"start_index ({start_index}) is >= number of prompts ({len(prompts)})")
        prompts = prompts[start_index:]
        print(f"Skipping first {start_index} prompts. Processing {len(prompts)} prompts starting from index {start_index}.")

    # Print loaded prompts
    n_printed_prompts = 10
    for idx, prompt in enumerate(prompts[:n_printed_prompts]):
        print(f"[{idx + start_index}] {prompt.encode('unicode_escape').decode('ascii')}")
    if len(prompts) > n_printed_prompts:
        print(f"[...] and {len(prompts) - n_printed_prompts} more prompts")

    # Derive output directory
    if output_relpath:
        save_dir = f"{project_config.output_folder}/images/{output_relpath}"
    else:
        exp_path = derive_exp_path(checkpoint=model_config.checkpoint, model_type=model_config.model_type, project_config=project_config)
        checkpoint_step = extract_checkpoint_step(model_config.checkpoint) if model_config.checkpoint else None
        prompts_id = 'adhoc' if not prompts_file else os.path.splitext(os.path.basename(prompts_file))[0]
        cfg_mode_str = cfg_mode.value
        
        path_parts = [exp_path]
        if checkpoint_step:
            path_parts.append(checkpoint_step)
        path_parts.append(cfg_mode_str)
        if cfg_mode == CFGMode.NEGATIVE_PROMPT and target is not None:
            path_parts.append(target.value)
        path_parts.append(prompts_id)

        save_dir = os.path.join(project_config.output_folder, "images", *path_parts)

    os.makedirs(save_dir, exist_ok=True)

    # Determine which indices already exist
    existing_indices = {
        int(os.path.splitext(f)[0])
        for f in os.listdir(save_dir)
        if f.endswith(".png") and os.path.splitext(f)[0].isdigit()
    }
    # Filter existing indices to only those in our range (start_index onwards)
    existing_indices = {idx for idx in existing_indices if idx >= start_index}
    print(f"Found {len(existing_indices)} existing images; skipping those.")

    # Collect prompts to process
    indices_to_generate = [i for i in range(len(prompts)) if (i + start_index) not in existing_indices]
    if not indices_to_generate:
        print("All prompts already processed. Nothing to generate.")
        return save_dir

    guidance_scale = resolve_guidance_scale(model_type, guidance_scale)
    print(f"Using guidance_scale={guidance_scale}")

    # Create the wrapper
    print(f"Loading model: {model_type}")
    wrapper = create_wrapper_from_config(model_config)

    # Run inference, skipping existing images
    print(f"Generating {len(indices_to_generate)} images with {cfg_mode.value} CFG...")
    with torch.no_grad():
        for batch_start in range(0, len(indices_to_generate), batch_size):
            batch_indices = indices_to_generate[batch_start:batch_start + batch_size]
            batch_prompts = [prompts[i] for i in batch_indices]

            print(f"Generating batch {batch_start // batch_size + 1}: indices {batch_indices}")
            wrapper.eval()
            images = wrapper.sample_text_to_image(
                prompts=batch_prompts,
                batch_size=len(batch_prompts),
                guidance_scale=guidance_scale,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                seed=seed,
                cfg_mode=cfg_mode,
                target=target,
                image_size=image_size,
            )

            for i, img in zip(batch_indices, images):
                # Adjust output index to account for start_index offset
                # batch_indices are relative to the sliced prompts list, so add start_index
                output_index = i + start_index
                out_path = os.path.join(save_dir, f"{output_index}.png")
                img.save(out_path)
                print(f"Saved: {out_path}")

    print("Sampling completed.")
    return save_dir


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run model inference with custom CFG")

    # Model arguments
    parser.add_argument(
        "--model_type",
        type=str,
        required=True,
        choices=[m.value for m in ModelType],
        help="Type of model to use for inference"
    )

    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Optional path to a specific model checkpoint"
    )

    # Prompt arguments
    parser.add_argument(
        "--prompts",
        type=str,
        nargs="+",
        default=["a cute beaver on a bike."],
        help="Text prompts for the model (space-separated or quoted)"
    )

    parser.add_argument(
        "--prompts_file",
        type=str,
        default=None,
        help="Alternative to --prompts: path to a csv file containing prompts"
    )

    # Generation parameters
    parser.add_argument(
        "--batch_size",
        type=int,
        default=25,
        help="Batch size for inference"
    )

    parser.add_argument(
        "--guidance_scale",
        type=float,
        default=None,
        help="CFG guidance scale (default: model-specific value from wrapper inference_config_defaults)"
    )

    parser.add_argument(
        "--temperature",
        type=float,
        default=0.99,
        help="Sampling temperature"
    )

    parser.add_argument(
        "--top_p",
        type=float,
        default=0.96,
        help="Top-p sampling parameter"
    )

    parser.add_argument(
        "--top_k",
        type=int,
        default=4096,
        help="Top-k sampling parameter"
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducibility"
    )

    # CFG mode arguments
    parser.add_argument(
        "--cfg_mode",
        type=str,
        default="none",
        choices=[m.value for m in CFGMode],
        help="CFG mode: 'none' (standard CFG) or 'negative_prompt' (CFG with target prompt)"
    )

    parser.add_argument(
        "--target",
        type=str,
        default=None,
        choices=[t.value for t in Target],
        help="Target concept for target prompts (only used with negative_prompt CFG mode)."
    )

    parser.add_argument(
        "--image_size",
        type=int,
        default=512,
        help="Image size for generation (default: 512)"
    )

    # Output arguments
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Custom output directory name"
    )

    # Parse and run
    args = parser.parse_args()
    
    # Convert cfg_mode string to enum
    cfg_mode = CFGMode(args.cfg_mode)
    
    # Convert target string to enum if provided
    target = Target(args.target) if args.target else None

    inference_text_to_image(
        model_type=args.model_type,
        checkpoint=args.checkpoint,
        prompts=args.prompts,
        prompts_file=args.prompts_file,
        batch_size=args.batch_size,
        output_relpath=args.output,
        guidance_scale=args.guidance_scale,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        seed=args.seed,
        cfg_mode=cfg_mode,
        target=target,
        image_size=args.image_size,
    )

