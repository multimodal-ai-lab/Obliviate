import os
from collections import Counter
from typing import Tuple, List

from PIL import Image
from tqdm import tqdm

from obliviate.metrics.base_evaluator import BaseEvaluator, is_image_file
from obliviate.metrics.clip_evaluator import CLIPScorer


class StyleEvaluator(BaseEvaluator):
    """
    Evaluates the CLIP score between generated images and a style phrase.

    For each image, computes CLIP similarity with "Image in the style of {style_artist_name}".
    """

    def __init__(self, style_artist_name: str):
        super().__init__()
        self.style_artist_name = style_artist_name
        self.style_prompt = f"Image in the style of {style_artist_name}"
        self.scorer = CLIPScorer()

    def process_images(self, image_folder_or_images, start_index=0, **kwargs) -> Tuple[dict, list]:
        if isinstance(image_folder_or_images, str):
            image_file_paths = [
                os.path.join(image_folder_or_images, f)
                for f in os.listdir(image_folder_or_images)
                if is_image_file(f)
            ]
            image_file_paths.sort()
            images = [Image.open(image_path).convert("RGB") for image_path in image_file_paths]
        else:
            images = image_folder_or_images

        images_to_process = images[start_index:]

        if start_index > 0:
            print(f"Continuing evaluation from index {start_index}...")
        print(
            f"Evaluating {len(images_to_process)} images (out of {len(images)} total) "
            f"for style similarity to '{self.style_artist_name}'...\n"
        )

        results = []
        for image in tqdm(images_to_process, desc="Processing images"):
            result = self.scorer(image, self.style_prompt)
            result["style_artist_name"] = self.style_artist_name
            results.append(result)

        summary = self.summarize(results)

        return summary, results

    def summarize(self, results):
        summary = Counter()
        total = len(results)
        summary["total"] = total
        summary["errors"] = sum(1 for r in results if "error" in r)

        scores: List[float] = [float(r.get("score", 0.0)) for r in results]
        summary["style_score"] = sum(scores) / total if total > 0 else 0.0
        summary["style_artist_name"] = self.style_artist_name

        return summary
