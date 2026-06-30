import os
from typing import Tuple
import torch
from torchvision import models
from PIL import Image
from tqdm import tqdm

from obliviate.metrics.base_evaluator import BaseEvaluator, is_image_file

class ObjectEvaluator(BaseEvaluator):
    """
    Evaluates object presence using a pre-trained ResNet-50 (ImageNet).
    Computes accuracy on the specific target class index.
    
    For Erasure: Low accuracy on target class = High Erasure Success.
    For Preservation: High accuracy on target class = High Preservation Success.
    """

    def __init__(self, target_class_index: int, device: str = "cuda" if torch.cuda.is_available() else "cpu"):
        super().__init__()
        self.device = device
        self.target_class_index = target_class_index
        
        # Load ResNet-50
        print(f"Loading ResNet-50 for object detection (Target Class Index: {target_class_index})...")
        weights = models.ResNet50_Weights.IMAGENET1K_V1
        self.model = models.resnet50(weights=weights).to(self.device)
        self.model.eval()
        
        # Standard ImageNet transforms
        self.preprocess = weights.transforms()

    def process_images(self, image_folder, start_index=0, **kwargs) -> Tuple[dict, list]:
        image_files = [f for f in os.listdir(image_folder) if is_image_file(f)]
        image_files.sort()
        
        if not image_files:
            raise ValueError(f"No valid images found in folder: {image_folder}")

        image_files_to_process = image_files[start_index:]
        
        if start_index > 0:
            print(f"Continuing evaluation from index {start_index}...")
        print(f"Scanning {len(image_files_to_process)} images for object class {self.target_class_index}...\n")
        
        results = []
        
        with torch.no_grad():
            for filename in tqdm(image_files_to_process, desc="Classifying images"):
                image_path = os.path.join(image_folder, filename)
                try:
                    img = Image.open(image_path).convert("RGB")
                    input_tensor = self.preprocess(img).unsqueeze(0).to(self.device)
                    
                    logits = self.model(input_tensor)
                    probs = torch.nn.functional.softmax(logits, dim=1)
                    
                    # Get the confidence for the target class
                    confidence = probs[0, self.target_class_index].item()
                    # if confidence > 0.2 detected = True else False
                    detected = confidence > 0.2
                    
                    results.append({
                        "filename": filename,
                        "detected": detected,
                        "confidence": confidence,
                        "error": None
                    })
                    
                except Exception as e:
                    results.append({
                        "filename": filename,
                        "error": str(e),
                        "detected": None,
                        "confidence": 0.0
                    })

        summary = self.summarize(results)
        return summary, results

    def summarize(self, results):
        total = len(results)
        errors = sum(1 for r in results if r.get("error"))
        valid_total = total - errors
        
        # Count how many times the target was detected
        detected_count = sum(1 for r in results if str(r.get("detected")) == "True")
        
        # Accuracy: How often did we generate the target?
        # For Erasure: We want this to be 0.
        detection_rate = detected_count / valid_total if valid_total > 0 else 0.0
        
        # Erasure Success Rate (ESR): Percentage of images where target was NOT detected
        erasure_success_rate = 1.0 - detection_rate

        return {
            "total": total,
            "errors": errors,
            "detected_count": detected_count,
            "detection_rate": detection_rate, # This is Top-1 Accuracy on target
            "erasure_success_rate": erasure_success_rate 
        }