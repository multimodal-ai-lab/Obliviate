import os
from pathlib import Path
from typing import Dict, Tuple, Any
import logging

from cleanfid import fid

from obliviate.metrics.base_evaluator import BaseEvaluator, is_image_file

logger = logging.getLogger(__name__)


class FIDEvaluator(BaseEvaluator):
    """
    FID evaluator for a dataset.

    Args:
        dataset_path: Path to the dataset directory
        device: Device to use for computation
    """
    def __init__(self, dataset_path: str, device: str = "cuda"):
        super().__init__()
        self.dataset_path = Path(dataset_path)
        self.images_path = self.dataset_path / "images"
        self.stats_name = self.dataset_path.name.lower().replace("-", "_")
        self.device = device
    
    def precompute_stats(self, force_recompute: bool = False):
        """
        Precompute FID statistics for the dataset.
        """
        if not force_recompute and self._stats_exist():
            logger.info(f"Statistics for {self.stats_name} already exist.")
            return True
        
        # Validate reference images directory
        if not self.images_path.exists():
            raise FileNotFoundError(f"Reference images directory not found: {self.images_path}")
        
        # Count reference images
        ref_images = [f for f in os.listdir(self.images_path) if is_image_file(f)]
        if not ref_images:
            raise ValueError(f"No valid images found in reference directory: {self.images_path}")
        
        logger.info(f"Precomputing FID statistics for {self.stats_name}...")
        logger.info(f"Images path: {self.images_path}")
        logger.info(f"Found {len(ref_images)} reference images")
        
        try:
            fid.make_custom_stats(self.stats_name, str(self.images_path), mode="clean")
            logger.info(f"Statistics precomputed and saved as '{self.stats_name}'")
            return True
        except Exception as e:
            logger.error(f"Failed to precompute FID statistics: {e}")
            raise
    
    def _stats_exist(self) -> bool:
        return fid.test_stats_exists(self.stats_name, mode="clean")
    
    def process_images(self, image_folder: str, use_precomputed: bool = True, **kwargs) -> Tuple[Dict[str, float], Any]:
        """
        Evaluate FID score between generated images and reference dataset.
        
        Args:
            image_folder: Path to folder containing generated images
            use_precomputed: Whether to use precomputed statistics
            
        Returns:
            Dict containing 'total' (number of images) and 'fid_score'
        """
        # Validate input image folder
        if not os.path.exists(image_folder):
            raise FileNotFoundError(f"Image folder not found: {image_folder}")
        
        if not os.path.isdir(image_folder):
            raise ValueError(f"Path is not a directory: {image_folder}")
        
        # Count and validate generated images
        generated_images = [f for f in os.listdir(image_folder) if is_image_file(f)]
        if not generated_images:
            raise ValueError(f"No valid images found in generated images folder: {image_folder}")
        
        logger.info(f"Found {len(generated_images)} generated images in {image_folder}")
        
        # Count and validate reference dataset if not using precomputed stats
        if not use_precomputed or not self._stats_exist():
            if not self.images_path.exists():
                raise FileNotFoundError(f"Reference images directory not found: {self.images_path}")
            
            ref_images = [f for f in os.listdir(self.images_path) if is_image_file(f)]
            if not ref_images:
                raise ValueError(f"No valid images found in reference directory: {self.images_path}")
            
            logger.info(f"Found {len(ref_images)} reference images")
        
        # Compute FID score
        try:
            if use_precomputed and self._stats_exist():
                logger.info(f"Using precomputed statistics for {self.stats_name}")
                fid_score = fid.compute_fid(image_folder, self.stats_name, mode="clean")
            else:
                logger.info("Computing FID without precomputed statistics...")
                fid_score = fid.compute_fid(image_folder, str(self.images_path), mode="clean")
        except Exception as e:
            logger.error(f"Failed to compute FID score: {e}")
            raise
        
        logger.info(f"FID Score: {fid_score:.4f}")

        summary = {
            "total": len(generated_images),
            "fid_score": fid_score
        }

        results = [summary]
        return summary, results

    def summarize(self, results):
        if isinstance(results, list) and len(results) > 0:
            return results[-1]
        return {}