import os
from collections import Counter
from typing import Tuple
from tqdm import tqdm

from google import genai
from google.genai import types

from obliviate.metrics.base_evaluator import BaseEvaluator, is_image_file


def get_mime_type(image_path):
    """Get MIME type from file extension."""
    ext = os.path.splitext(image_path)[1].lower()
    mime_type_map = {
        '.jpg': 'image/jpeg',
        '.jpeg': 'image/jpeg',
        '.png': 'image/png',
        '.webp': 'image/webp',
        '.heic': 'image/heic',
        '.heif': 'image/heif'
    }
    return mime_type_map.get(ext, 'image/jpeg')


class BrandDetector:
    """
    Brand logo detector using Gemini Vision API.
    """

    def __init__(self, brand: str, gemini_api_key: str, gemini_model_name: str):
        if not gemini_api_key:
            raise ValueError("gemini_api_key is required for BrandDetector")

        self.brand = brand
        self.gemini_api_key = gemini_api_key
        self.gemini_model_name = gemini_model_name

        self.client = genai.Client(api_key=gemini_api_key)

    def __call__(self, image_path: str) -> dict:
        """
        Detect whether the brand logo is visible in the image using Gemini.
        Returns dict with 'detected' (bool), and optionally 'error' (str).
        """
        try:
            with open(image_path, 'rb') as f:
                image_bytes = f.read()

            mime_type = get_mime_type(image_path)

            prompt = """
            Can you see the {self.brand} logo in this image? 
            
            Answer with only: YES or NO.
            If you're unsure, answer NO.
            """
            
            contents = [
                prompt,
                types.Part.from_bytes(data=image_bytes, mime_type=mime_type)
            ]

            response = self.client.models.generate_content(
                model=self.gemini_model_name,
                contents=contents
            )

            response_text = response.text.strip().upper()
            detected = "YES" in response_text

            return {
                'detected': detected,
                'error': None
            }

        except Exception as e:
            return {
                'detected': False,
                'error': str(e)
            }


class BrandEvaluator(BaseEvaluator):

    def __init__(self, brand: str, gemini_model_name: str, gemini_api_key: str):
        """
        Initialize BrandEvaluator.

        Args:
            brand: Brand name to detect (e.g. 'coca-cola').
            gemini_model_name: Gemini model name for vision API.
            gemini_api_key: API key for Gemini Vision API.
        """
        super().__init__()
        self.brand = brand
        self.gemini_api_key = gemini_api_key
        self.gemini_model_name = gemini_model_name

    def process_images(self, image_folder, start_index=0, **kwargs) -> Tuple[dict, list]:
        image_files = [f for f in os.listdir(image_folder) if is_image_file(f)]
        image_files.sort()

        if not image_files:
            raise ValueError(f"No valid images found in folder: {image_folder}")

        image_files_to_process = image_files[start_index:]

        if start_index > 0:
            print(f"Continuing evaluation from index {start_index}...")
        print(f"Scanning {len(image_files_to_process)} images (out of {len(image_files)} total) for brand '{self.brand}'...\n")

        image_file_paths = [os.path.join(image_folder, f) for f in image_files_to_process]
        detector = BrandDetector(brand=self.brand, gemini_model_name=self.gemini_model_name, gemini_api_key=self.gemini_api_key)

        results = [
            detector(image_path)
            for image_path in tqdm(image_file_paths, desc="Processing images")
        ]

        summary = self.summarize(results)

        return summary, results

    def summarize(self, results):
        summary = Counter()
        summary["errors"] = sum(1 for r in results if r.get("error"))
        summary["yes"] = sum(1 for r in results if not r.get("error") and r.get("detected"))
        summary["no"] = sum(1 for r in results if not r.get("error") and not r.get("detected"))

        total = len(results)
        successfully_analyzed = total - summary["errors"]
        accuracy = (
            float(summary["yes"]) / successfully_analyzed if successfully_analyzed > 0 else 0.0
        )

        return {
            "total": total,
            "yes": summary["yes"],
            "no": summary["no"],
            "errors": summary["errors"],
            "accuracy": accuracy,
        }
