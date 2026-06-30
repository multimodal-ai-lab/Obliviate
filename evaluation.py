import argparse
import os

from inference_text_to_image import inference_text_to_image
from obliviate.configs.project_config import ProjectConfig

from obliviate.metrics.q16_evaluator import Q16Evaluator
from obliviate.metrics.clip_evaluator import CLIPEvaluator
from obliviate.metrics.fid_evaluator import FIDEvaluator
from obliviate.metrics.nudity_evaluator import NudityEvaluator
from obliviate.metrics.gore_evaluator import GoreEvaluator
from obliviate.metrics.style_evaluator import StyleEvaluator
from obliviate.metrics.object_evaluator import ObjectEvaluator
from obliviate.metrics.brand_evaluator import BrandEvaluator
from obliviate.configs.model_config import ModelType
from obliviate.configs.cfg_config import CFGMode
from obliviate.utils.load import load_prompts_from_csv, resolve_prompts_file
from obliviate.utils.misc import derive_exp_path, build_results_log_path, build_evaluation_metadata, extract_path_components
from obliviate.unlearning.target_config import Target, TARGET_TO_IMAGENET_INDEX, TARGET_TO_STYLE_ARTIST_NAME, TARGET_TO_BRAND_NAME

from dotenv import load_dotenv


load_dotenv()


def evaluate_fid(model_type, ref_dataset, checkpoint=None, batch_size=25, image_folder=None,
                prompts_file=None, precompute_stats=False, use_precomputed=True, force_recompute=False, start_index=0, cfg_mode=CFGMode.NONE, guidance_scale=None, image_size=512, target=None):
    """
    Evaluate FID score between generated images and reference dataset.

    When cfg_mode is NEGATIVE_PROMPT, target (e.g. Target.NUDITY) is used as the negative prompt for generation.
    
    Args:
        model_type: Type of model to use for inference
        checkpoint: Optional path to model checkpoint
        ref_dataset: Path to reference dataset directory (required)
        precompute_stats: Whether to precompute FID statistics
        use_precomputed: Whether to use precomputed statistics
        target: Target concept for negative_prompt CFG (required when cfg_mode is negative_prompt).
    """

    project_config = ProjectConfig()

    # Initialize FID evaluator
    fid_evaluator = FIDEvaluator(
        dataset_path=ref_dataset
    )
    
    # Precompute statistics if requested
    if precompute_stats:
        print(f"Precomputing FID statistics...")
        fid_evaluator.precompute_stats(force_recompute=force_recompute)
    
    if image_folder is None:
        prompts_file = resolve_prompts_file(prompts_file)
        print(f"Generating images using prompts from {prompts_file}...")
        image_folder = inference_text_to_image(
            model_type, checkpoint, batch_size=batch_size, prompts_file=prompts_file,
            cfg_mode=cfg_mode, guidance_scale=guidance_scale, start_index=start_index,
            image_size=image_size, target=target
        )
    else:
        print(f"Using images from {image_folder}...")

    exp_path = derive_exp_path(checkpoint=checkpoint, image_folder=image_folder, model_type=model_type, project_config=project_config)
    checkpoint_step, cfg_mode_str, prompts_id, target_str = extract_path_components(checkpoint, image_folder, prompts_file, cfg_mode, project_config)

    log_file_path = os.path.join(exp_path, "fid.csv")
    results_log_path = build_results_log_path(exp_path, checkpoint_step=checkpoint_step, cfg_mode=cfg_mode_str, prompts_id=prompts_id, filename="fid.csv", target=target_str)

    # Build evaluation metadata (include target when using negative_prompt so report distinguishes nudity vs gore etc.)
    eval_metadata = build_evaluation_metadata(
        checkpoint=checkpoint,
        image_folder=image_folder,
        prompts_file=prompts_file,
        cfg_mode=cfg_mode,
        guidance_scale=guidance_scale,
        model_type=model_type,
        project_config=project_config,
        **({"target": target.value} if target is not None else {})
    )

    print("Computing FID score...")
    fid_evaluator.evaluate(
        image_folder=image_folder,
        log_path=log_file_path,
        results_log_path=results_log_path,
        use_precomputed=use_precomputed,
        row_metadata=eval_metadata
    )

def evaluate_clip_score(model_type, checkpoint=None, batch_size=25, image_folder=None, prompts_file=None, cfg_mode=CFGMode.NONE, guidance_scale=None, start_index=0, image_size=512, target=None):
    """
    Evaluate CLIP score between generated images and their prompts.

    When cfg_mode is NEGATIVE_PROMPT, target (e.g. Target.NUDITY) is used as the negative prompt for generation.
    
    Args:
        model_type: Type of model to use for inference
        checkpoint: Optional path to model checkpoint
        target: Target concept for negative_prompt CFG (required when cfg_mode is negative_prompt).
    """
    project_config = ProjectConfig()

    if image_folder is None:
        prompts_file = resolve_prompts_file(prompts_file)
        print(f"Generating images using prompts from {prompts_file}...")
        image_folder = inference_text_to_image(
            model_type, checkpoint, batch_size=batch_size, prompts_file=prompts_file,
            cfg_mode=cfg_mode, guidance_scale=guidance_scale, start_index=start_index,
            image_size=image_size, target=target
        )
    else:
        print(f"Using images from {image_folder}...")
        # Resolve prompts_file for loading prompts list
        prompts_file = resolve_prompts_file(prompts_file)
    
    # Load prompts from CSV file
    prompts_list = load_prompts_from_csv(prompts_file)

    # Initialize CLIP evaluator
    clip_evaluator = CLIPEvaluator()
        
    exp_path = derive_exp_path(checkpoint=checkpoint, image_folder=image_folder, model_type=model_type, project_config=project_config)
    checkpoint_step, cfg_mode_str, prompts_id, target_str = extract_path_components(checkpoint, image_folder, prompts_file, cfg_mode, project_config)

    log_file_path = os.path.join(exp_path, "clip.csv")
    results_log_path = build_results_log_path(exp_path, checkpoint_step=checkpoint_step, cfg_mode=cfg_mode_str, prompts_id=prompts_id, filename="clip.csv", target=target_str)
    
    print("Computing CLIP score...")
    
    eval_metadata = build_evaluation_metadata(
        checkpoint=checkpoint,
        image_folder=image_folder,
        prompts_file=prompts_file,
        cfg_mode=cfg_mode,
        guidance_scale=guidance_scale,
        model_type=model_type,
        project_config=project_config,
        **({"target": target.value} if target is not None else {})
    )

    clip_evaluator.evaluate(
        image_folder=image_folder,
        log_path=log_file_path,
        results_log_path=results_log_path,
        prompts=prompts_list,
        row_metadata=eval_metadata
    )

def evaluate_q16(model_type, checkpoint=None, batch_size=25, image_folder=None,
                  prompts_file=None, cfg_mode=CFGMode.NONE, guidance_scale=None,
                  start_index=0, image_size=512):
    """
    Evaluate inappropriate content using Q16 classifier (CLIP + Soft-Prompts).
    
    Args:
        model_type: Type of model to use for inference
        checkpoint: Optional path to model checkpoint
        image_folder: Path to image folder (if None, will generate images)
        prompts_file: Path to CSV file containing prompts (for generation)
    """
    project_config = ProjectConfig()
    
    if image_folder is None:
        prompts_file = resolve_prompts_file(prompts_file, target=Target.GORE)
        print(f"Generating images from {prompts_file}...")
        image_folder = inference_text_to_image(model_type, checkpoint, batch_size=batch_size, prompts_file=prompts_file, 
                                               cfg_mode=cfg_mode, guidance_scale=guidance_scale, 
                                               start_index=start_index, target=Target.GORE, image_size=image_size)
    else:
        print(f"Using images from {image_folder}...")
    
    exp_path = derive_exp_path(checkpoint=checkpoint, image_folder=image_folder, model_type=model_type, project_config=project_config)
    checkpoint_step, cfg_mode_str, prompts_id, target_str = extract_path_components(checkpoint, image_folder, prompts_file, cfg_mode, project_config)

    log_file_path = os.path.join(exp_path, "q16.csv")
    results_log_path = build_results_log_path(exp_path, checkpoint_step=checkpoint_step, cfg_mode=cfg_mode_str, prompts_id=prompts_id, filename="q16.csv", target=target_str)
    
    q16_evaluator = Q16Evaluator()
    
    print("Computing Q16 safety scores...")
    
    eval_metadata = build_evaluation_metadata(
        checkpoint=checkpoint,
        image_folder=image_folder,
        prompts_file=prompts_file,
        cfg_mode=cfg_mode,
        guidance_scale=guidance_scale,
        model_type=model_type,
        project_config=project_config
    )
    
    q16_evaluator.evaluate(
        image_folder=image_folder,
        log_path=log_file_path,
        results_log_path=results_log_path,
        row_metadata=eval_metadata
    )

def evaluate_nudity(model_type, checkpoint=None, batch_size=25, image_folder=None,
                    prompts_file=None, cfg_mode=CFGMode.NONE, guidance_scale=None, start_index=0, image_size=512):
    """
    Evaluate nudity by detecting if the target is present in generated images.
    
    Args:
        model_type: Type of model to use for inference
        checkpoint: Optional path to model checkpoint
        image_folder: Path to image folder (if None, will generate images)
        prompts_file: Path to CSV file containing prompts
    """
    project_config = ProjectConfig()
    
    if image_folder is None:
        prompts_file = resolve_prompts_file(prompts_file, target=Target.NUDITY)
        print(f"Generating images from {prompts_file}...")
        image_folder = inference_text_to_image(model_type, checkpoint, batch_size=batch_size, prompts_file=prompts_file, cfg_mode=cfg_mode, guidance_scale=guidance_scale, start_index=start_index, target=Target.NUDITY, image_size=image_size)
    else:
        print(f"Using images from {image_folder}...")
    
    exp_path = derive_exp_path(checkpoint=checkpoint, image_folder=image_folder, model_type=model_type, project_config=project_config)
    checkpoint_step, cfg_mode_str, prompts_id, target_str = extract_path_components(checkpoint, image_folder, prompts_file, cfg_mode, project_config)

    log_file_path = os.path.join(exp_path, "nudity.csv")
    results_log_path = build_results_log_path(exp_path, checkpoint_step=checkpoint_step, cfg_mode=cfg_mode_str, prompts_id=prompts_id, filename="nudity.csv", target=target_str)
    
    nudity_evaluator = NudityEvaluator()
    
    print("Computing nudity...")
    
    eval_metadata = build_evaluation_metadata(
        checkpoint=checkpoint,
        image_folder=image_folder,
        prompts_file=prompts_file,
        cfg_mode=cfg_mode,
        guidance_scale=guidance_scale,
        model_type=model_type,
        project_config=project_config
    )
    
    nudity_evaluator.evaluate(
        image_folder=image_folder,
        log_path=log_file_path,
        results_log_path=results_log_path,
        row_metadata=eval_metadata
    )

def evaluate_gore(model_type, checkpoint=None, batch_size=25, image_folder=None,
                  prompts_file=None, cfg_mode=CFGMode.NONE, guidance_scale=None, start_index=0, image_size=512):
    """
    Evaluate gore by detecting if the target is present in generated images.
    
    Args:
        model_type: Type of model to use for inference
        checkpoint: Optional path to model checkpoint
        image_folder: Path to image folder (if None, will generate images)
        prompts_file: Path to CSV file containing prompts
    """
    project_config = ProjectConfig()
    
    gemini_api_key = os.getenv("GEMINI_API_KEY")
    if not gemini_api_key:
        raise ValueError("GEMINI_API_KEY is required for gore evaluation")
    
    if image_folder is None:
        prompts_file = resolve_prompts_file(prompts_file, target=Target.GORE)
        print(f"Generating images from {prompts_file}...")
        image_folder = inference_text_to_image(model_type, checkpoint, batch_size=batch_size, prompts_file=prompts_file, cfg_mode=cfg_mode, guidance_scale=guidance_scale, start_index=start_index, target=Target.GORE, image_size=image_size)
    else:
        print(f"Using images from {image_folder}...")
    
    exp_path = derive_exp_path(checkpoint=checkpoint, image_folder=image_folder, model_type=model_type, project_config=project_config)
    checkpoint_step, cfg_mode_str, prompts_id, target_str = extract_path_components(checkpoint, image_folder, prompts_file, cfg_mode, project_config)

    log_file_path = os.path.join(exp_path, "gore.csv")
    results_log_path = build_results_log_path(exp_path, checkpoint_step=checkpoint_step, cfg_mode=cfg_mode_str, prompts_id=prompts_id, filename="gore.csv", target=target_str)
    
    gore_evaluator = GoreEvaluator(gemini_model_name=project_config.eval_gemini_model, gemini_api_key=gemini_api_key)
    
    print("Computing gore...")
    
    eval_metadata = build_evaluation_metadata(
        checkpoint=checkpoint,
        image_folder=image_folder,
        prompts_file=prompts_file,
        cfg_mode=cfg_mode,
        guidance_scale=guidance_scale,
        model_type=model_type,
        project_config=project_config
    )
    
    gore_evaluator.evaluate(
        image_folder=image_folder,
        log_path=log_file_path,
        results_log_path=results_log_path,
        row_metadata=eval_metadata
    )


def evaluate_brand(model_type, target, checkpoint=None, batch_size=25, image_folder=None,
                   prompts_file=None, cfg_mode=CFGMode.NONE, guidance_scale=None,
                   start_index=0, image_size=512):
    """
    Evaluate brand logo visibility using Gemini Vision API.

    Args:
        model_type: Type of model for inference (used when generating images).
        target: Target brand concept (e.g. coca_cola).
        checkpoint: Optional path to checkpoint.
        image_folder: Path to image folder (if None, images are generated from prompts).
        prompts_file: Optional CSV of prompts; resolved from target if not provided.
    """
    project_config = ProjectConfig()
    if target not in TARGET_TO_BRAND_NAME:
        supported = ", ".join(t.value for t in TARGET_TO_BRAND_NAME)
        raise ValueError(f"Brand evaluation requires --target to be one of: {supported}")
    brand = TARGET_TO_BRAND_NAME[target]

    gemini_api_key = os.getenv("GEMINI_API_KEY")
    if not gemini_api_key:
        raise ValueError("GEMINI_API_KEY is required (set in environment/.env) for brand evaluation")

    if image_folder is None:
        prompts_file = resolve_prompts_file(prompts_file, target=target)
        print(f"Generating images from {prompts_file}...")
        image_folder = inference_text_to_image(
            model_type, checkpoint, batch_size=batch_size, prompts_file=prompts_file,
            cfg_mode=cfg_mode, guidance_scale=guidance_scale, start_index=start_index,
            target=target, image_size=image_size
        )
    else:
        print(f"Using images from {image_folder}...")

    exp_path = derive_exp_path(checkpoint=checkpoint, image_folder=image_folder, model_type=model_type, project_config=project_config)
    checkpoint_step, cfg_mode_str, prompts_id, target_str = extract_path_components(checkpoint, image_folder, prompts_file, cfg_mode, project_config)

    log_file_path = os.path.join(exp_path, "brand.csv")
    results_log_path = build_results_log_path(exp_path, checkpoint_step=checkpoint_step, cfg_mode=cfg_mode_str, prompts_id=prompts_id, filename="brand.csv", target=target_str)

    brand_evaluator = BrandEvaluator(brand=brand, gemini_model_name=project_config.eval_gemini_model, gemini_api_key=gemini_api_key)

    print(f"Computing brand detection for '{brand}'...")

    eval_metadata = build_evaluation_metadata(
        checkpoint=checkpoint,
        image_folder=image_folder,
        prompts_file=prompts_file,
        cfg_mode=cfg_mode,
        guidance_scale=guidance_scale,
        model_type=model_type,
        project_config=project_config,
        brand=brand
    )

    brand_evaluator.evaluate(
        image_folder=image_folder,
        log_path=log_file_path,
        results_log_path=results_log_path,
        row_metadata=eval_metadata
    )


def evaluate_style(model_type, checkpoint=None, batch_size=25, image_folder=None,
                  prompts_file=None, cfg_mode=CFGMode.NONE, guidance_scale=None, start_index=0,
                  target=None, image_size=512):
    """
    Evaluate style similarity by measuring CLIP similarity between generated images and style prompts.

    Args:
        model_type: Type of model to use for inference
        checkpoint: Optional path to model checkpoint
        image_folder: Path to image folder (if None, will generate images)
        prompts_file: Path to CSV file containing prompts
        target: Target concept for prompts and generation CFG
    """
    project_config = ProjectConfig()

    style_artist_name = TARGET_TO_STYLE_ARTIST_NAME[target]

    if image_folder is None:
        prompts_file = resolve_prompts_file(prompts_file, target=target)
        print(f"Generating images from {prompts_file}...")
        image_folder = inference_text_to_image(
            model_type, checkpoint, batch_size=batch_size, prompts_file=prompts_file,
            cfg_mode=cfg_mode, guidance_scale=guidance_scale, start_index=start_index,
            target=target, image_size=image_size,
        )
    else:
        print(f"Using images from {image_folder}...")
        prompts_file = resolve_prompts_file(prompts_file, target=target)

    exp_path = derive_exp_path(checkpoint=checkpoint, image_folder=image_folder, model_type=model_type, project_config=project_config)
    checkpoint_step, cfg_mode_str, prompts_id, target_str = extract_path_components(checkpoint, image_folder, prompts_file, cfg_mode, project_config)

    log_file_path = os.path.join(exp_path, "style.csv")
    results_log_path = build_results_log_path(exp_path, checkpoint_step=checkpoint_step, cfg_mode=cfg_mode_str, prompts_id=prompts_id, filename="style.csv", target=target_str)

    style_evaluator = StyleEvaluator(style_artist_name=style_artist_name)

    print(f"Computing style similarity for target={target.value!r} (CLIP: {style_artist_name!r})...")

    eval_metadata = build_evaluation_metadata(
        checkpoint=checkpoint,
        image_folder=image_folder,
        prompts_file=prompts_file,
        cfg_mode=cfg_mode,
        guidance_scale=guidance_scale,
        model_type=model_type,
        project_config=project_config,
        target=target.value,
        style_artist_name=style_artist_name,
    )
    
    style_evaluator.evaluate(
        image_folder=image_folder,
        log_path=log_file_path,
        results_log_path=results_log_path,
        row_metadata=eval_metadata
    )

def evaluate_object(model_type, checkpoint=None, batch_size=25, image_folder=None,
                  prompts_file=None, cfg_mode=CFGMode.NONE, guidance_scale=None,
                  start_index=0, target=None, target_class_index=None, image_size=512, **kwargs):
    
    project_config = ProjectConfig()

    print(f"Evaluating object erasure for target: {target}'")
    
    if image_folder is None:
        prompts_file = resolve_prompts_file(prompts_file, target=target)
        print(f"Generating images from {prompts_file}...")
        image_folder = inference_text_to_image(model_type, checkpoint, batch_size=batch_size, prompts_file=prompts_file, cfg_mode=cfg_mode, guidance_scale=guidance_scale, start_index=start_index, target=target, image_size=image_size, **kwargs)
    
    exp_path = derive_exp_path(checkpoint=checkpoint, image_folder=image_folder, model_type=model_type, project_config=project_config)
    checkpoint_step, cfg_mode_str, prompts_id, target_str = extract_path_components(checkpoint, image_folder, prompts_file, cfg_mode, project_config)

    log_file_path = os.path.join(exp_path, "object_erasure.csv")
    results_log_path = build_results_log_path(exp_path, checkpoint_step=checkpoint_step, cfg_mode=cfg_mode_str, prompts_id=prompts_id, filename="object_erasure.csv", target=target_str)

    if target_class_index is None:
        target_class_index = TARGET_TO_IMAGENET_INDEX[target]
    print(f"Using target class index: {target_class_index} for target: {target}")
    evaluator = ObjectEvaluator(target_class_index=target_class_index)
    
    print(f"Computing Object Erasure metrics for class index {target_class_index}...")
    
    eval_metadata = build_evaluation_metadata(
        checkpoint=checkpoint,
        image_folder=image_folder,
        prompts_file=prompts_file,
        cfg_mode=cfg_mode,
        guidance_scale=guidance_scale,
        model_type=model_type,
        project_config=project_config,
        target_class=target.value if target else "custom"
    )
    
    evaluator.evaluate(
        image_folder=image_folder,
        log_path=log_file_path,
        results_log_path=results_log_path,
        row_metadata=eval_metadata
    )



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run model evaluation")

    # --- Define arguments ---
    parser.add_argument(
        "--model_type",
        type=str,
        required=False,
        choices=[m.value for m in ModelType],
        help="Type of model to use for inference"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Optional path to a specific model checkpoint or run directory (if run directory, auto-selects best checkpoint)."
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=25,
        help="Batch size for inference (default: 25)"
    )
    parser.add_argument(
        "--ref_dataset",
        type=str,
        default=None,
        help="Path to reference dataset directory (required for --eval_type fid)"
    )
    parser.add_argument(
        "--prompts_file",
        type=str,
        default=None,
        help="Path to CSV file containing prompts"
    )
    parser.add_argument(
        "--image_folder",
        type=str,
        default=None,
        help="Path to image folder"
    )
    parser.add_argument(
        "--precompute_stats",
        action="store_true",
        help="Precompute FID statistics for the dataset"
    )
    parser.add_argument(
        "--use_precomputed",
        action="store_true",
        default=True,
        help="Use precomputed FID statistics"
    )
    parser.add_argument(
        "--force_recompute",
        action="store_true",
        default=False,
        help="Force recompute FID statistics"
    )
    parser.add_argument(
        "--start_index",
        type=int,
        default=0,
        help="Starting index for prompts (useful for resuming evaluation from a specific point)"
    )
    parser.add_argument(
        "--eval_type",
        type=str,
        choices=["fid", "clip", "nudity", "gore", "style", "q16", "object", "brand"],
        required=True,
        help="Type of evaluation to run"
    )
    parser.add_argument(
        "--cfg_mode",
        type=str,
        default="none",
        choices=[m.value for m in CFGMode],
        help="CFG mode: 'none' (standard CFG) or 'negative_prompt' (CFG with target prompt)"
    )
    parser.add_argument(
        "--guidance_scale",
        type=float,
        default=None,
        help="CFG guidance scale (default: model-specific value from wrapper inference_config_defaults)"
    )
    parser.add_argument(
        "--target",
        type=str,
        default=None,
        choices=[t.value for t in Target],
        help="Target concept (required for object and style; required for fid/clip when --cfg_mode negative_prompt; optional for others)"
    )
    parser.add_argument(
        "--target_class_index",
        type=int,
        default=None,
        help="ImageNet class index for object evaluation (alternative to --target)"
    )
    parser.add_argument(
        "--image_size",
        type=int,
        default=512,
        help="Image size for generation (default: 512)"
    )


    # --- Parse arguments ---
    args = parser.parse_args()
    
    if args.eval_type == "fid" and args.ref_dataset is None:
        parser.error("--ref_dataset is required for --eval_type fid")
    if args.eval_type == "object" and args.target is None:
        parser.error("--target is required for --eval_type object")
    if args.eval_type == "style" and args.target is None:
        parser.error("--target is required for --eval_type style")
    if args.eval_type == "brand" and args.target is None:
        parser.error("--target is required for --eval_type brand")
    if args.cfg_mode == "negative_prompt" and args.eval_type in ("fid", "clip") and not args.target:
        parser.error("--target is required when using --cfg_mode negative_prompt with --eval_type fid or clip")
    
    # Convert cfg_mode string to enum
    cfg_mode = CFGMode(args.cfg_mode)
    
    # Convert target string to Target enum if provided
    target = None
    if args.target:
        target = Target(args.target)

    if args.eval_type == "fid":
        evaluate_fid(
            model_type=args.model_type,
            checkpoint=args.checkpoint,
            ref_dataset=args.ref_dataset,
            batch_size=args.batch_size,
            prompts_file=args.prompts_file,
            image_folder=args.image_folder,
            precompute_stats=args.precompute_stats,
            use_precomputed=args.use_precomputed,
            force_recompute=args.force_recompute,
            start_index=args.start_index,
            cfg_mode=cfg_mode,
            guidance_scale=args.guidance_scale,
            image_size=args.image_size,
            target=target
        )
    elif args.eval_type == "clip":
        evaluate_clip_score(
            model_type=args.model_type,
            checkpoint=args.checkpoint,
            batch_size=args.batch_size,
            prompts_file=args.prompts_file,
            image_folder=args.image_folder,
            cfg_mode=cfg_mode,
            guidance_scale=args.guidance_scale,
            start_index=args.start_index,
            image_size=args.image_size,
            target=target
        )
    elif args.eval_type == "nudity":
        evaluate_nudity(
            model_type=args.model_type,
            checkpoint=args.checkpoint,
            image_folder=args.image_folder,
            batch_size=args.batch_size,
            prompts_file=args.prompts_file,
            cfg_mode=cfg_mode,
            guidance_scale=args.guidance_scale,
            start_index=args.start_index,
            image_size=args.image_size
        )
    elif args.eval_type == "gore":
        evaluate_gore(
            model_type=args.model_type,
            checkpoint=args.checkpoint,
            image_folder=args.image_folder,
            batch_size=args.batch_size,
            prompts_file=args.prompts_file,
            cfg_mode=cfg_mode,
            guidance_scale=args.guidance_scale,
            start_index=args.start_index,
            image_size=args.image_size
        )
    elif args.eval_type == "style":
        evaluate_style(
            model_type=args.model_type,
            checkpoint=args.checkpoint,
            image_folder=args.image_folder,
            batch_size=args.batch_size,
            prompts_file=args.prompts_file,
            cfg_mode=cfg_mode,
            guidance_scale=args.guidance_scale,
            start_index=args.start_index,
            target=target,
            image_size=args.image_size
        )

    elif args.eval_type == "q16": 
        evaluate_q16(
            model_type=args.model_type,
            checkpoint=args.checkpoint,
            batch_size=args.batch_size,
            image_folder=args.image_folder,
            prompts_file=args.prompts_file,
            cfg_mode=cfg_mode,
            guidance_scale=args.guidance_scale,
            start_index=args.start_index,
            image_size=args.image_size
        )
    elif args.eval_type == "object":
        evaluate_object(
            model_type=args.model_type,
            checkpoint=args.checkpoint,
            batch_size=args.batch_size,
            image_folder=args.image_folder,
            prompts_file=args.prompts_file,
            cfg_mode=cfg_mode,
            guidance_scale=args.guidance_scale,
            start_index=args.start_index,
            target=target,
            target_class_index=args.target_class_index,
            image_size=args.image_size
        )
    elif args.eval_type == "brand":
        evaluate_brand(
            model_type=args.model_type,
            target=target,
            checkpoint=args.checkpoint,
            batch_size=args.batch_size,
            image_folder=args.image_folder,
            prompts_file=args.prompts_file,
            cfg_mode=cfg_mode,
            guidance_scale=args.guidance_scale,
            start_index=args.start_index,
            image_size=args.image_size
        )