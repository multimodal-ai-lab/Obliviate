import os
from collections import Counter
from typing import Tuple, List

import torch
from PIL import Image
from tqdm import tqdm
from transformers import CLIPProcessor, CLIPModel

from obliviate.metrics.base_evaluator import BaseEvaluator, is_image_file


class CLIPScorer:
    """
    Computes the CLIP cosine similarity between an image and a corresponding text prompt.
    """

    def __init__(self, device: str = "cuda" if torch.cuda.is_available() else "cpu"):
        self.device = device

        # Load the Hugging Face CLIP model + processor
        # If transformers>=4.56.1, we need to use use_safetensors=True again
        self.model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(self.device).eval()
        self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

    def __call__(self, image: str, text_prompt: str):
        """
        Returns a dict with:
          - score: cosine similarity between image and text prompt (float)
        """
        try:
            # Load and preprocess the image + text
            inputs = self.processor(text=[text_prompt], images=[image], return_tensors="pt", padding=True)
            # Move inputs to the correct device
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            # Run the model (no gradient needed)
            with torch.no_grad():
                outputs = self.model(**inputs)
                image_embeds = outputs.image_embeds  # shape: (1, D)
                text_embeds = outputs.text_embeds  # shape: (1, D)

            # Normalize the embeddings
            image_embeds = image_embeds / image_embeds.norm(dim=-1, keepdim=True)
            text_embeds = text_embeds / text_embeds.norm(dim=-1, keepdim=True)

            # Cosine similarity: (1, D) @ (D, 1) → scalar
            similarity = (image_embeds @ text_embeds.T).item()

            return {"score": similarity}

        except Exception as e:
            return {"score": 0.0, "error": str(e)}
    
    def text_to_text(self, text1: str, text2: str):
        """
        Compute CLIP cosine similarity between two text strings.
        
        Args:
            text1: First text string
            text2: Second text string
        
        Returns:
            Dict with:
              - score: cosine similarity between text1 and text2 (float)
        """
        try:
            # Process both texts
            inputs = self.processor(text=[text1, text2], return_tensors="pt", padding=True, truncation=True)
            # Move inputs to the correct device
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Run the model (no gradient needed)
            with torch.no_grad():
                outputs = self.model.get_text_features(**inputs)
                # outputs shape: (2, D) - one embedding for each text
                text1_embeds = outputs[0:1]  # shape: (1, D)
                text2_embeds = outputs[1:2]  # shape: (1, D)
            
            # Normalize the embeddings
            text1_embeds = text1_embeds / text1_embeds.norm(dim=-1, keepdim=True)
            text2_embeds = text2_embeds / text2_embeds.norm(dim=-1, keepdim=True)
            
            # Cosine similarity: (1, D) @ (1, D).T → scalar
            similarity = (text1_embeds @ text2_embeds.T).item()
            
            return {"score": similarity}
        
        except Exception as e:
            return {"score": 0.0, "error": str(e)}


class CLIPEvaluator(BaseEvaluator):
    """
    Evaluates the CLIP score between a set of images and a set of prompts.

    Args:
        image_folder_or_images: Path to folder containing images or list of images
        prompts: List of prompts to evaluate the images against
        **kwargs: Additional arguments

    Returns:
        Dict containing 'total' (number of images), 'errors' (number of errors), and 'clip_score' (average CLIP score)
        List containing the individual ('total', 'errors', 'clip_score') results before summarization
    """
    scorer = CLIPScorer()

    def process_images(self, image_folder_or_images, prompts=None, start_index=0, **kwargs) -> Tuple[dict, list]:
        assert prompts, "CLIP evaluation requires reference prompts."

        if isinstance(image_folder_or_images, str):
            image_file_paths = [os.path.join(image_folder_or_images, f) for f in os.listdir(image_folder_or_images) if is_image_file(f)]
            image_file_paths.sort()
            images = [Image.open(image_path).convert("RGB") for image_path in image_file_paths]
        else:
            images = image_folder_or_images

        # Process only images from start_index onwards
        images_to_process = images[start_index:]
        prompts_to_process = prompts[start_index:]

        if start_index > 0:
            print(f"Continuing evaluation from index {start_index}...")
        print(f"Scanning {len(images_to_process)} images (out of {len(images)} total) ...\n")

        # --- Part 1: Accumulate results ---
        results = [
            self.scorer(image, prompt) for image, prompt in tqdm(zip(images_to_process, prompts_to_process), desc="Processing images")
        ]

        # --- Part 2: Summarize results ---
        summary = self.summarize(results)

        return dict(summary), results

    def summarize(self, results):
        summary = Counter()
        total = len(results)
        summary["total"] = total
        summary["errors"] = sum(1 for r in results if "error" in r)

        scores: List[float] = [float(r.get("score", 0.0)) for r in results]
        summary["clip_score"] = sum(scores) / total if total > 0 else 0.0

        return summary





