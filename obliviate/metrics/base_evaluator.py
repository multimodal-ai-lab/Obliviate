import csv
import datetime
import os
from abc import ABC, abstractmethod
from typing import List, Any, Dict, Tuple

from obliviate.configs.project_config import ProjectConfig


def is_image_file(filename):
    return filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif', '.tiff'))



def save_dicts_to_csv(data: List[Dict[str, Any]], filepath: str) -> None:
    """
    Save a list of dictionaries to a CSV file, handling missing keys gracefully.

    Args:
        data (List[Dict]): List of dictionaries to write.
        filepath (str): Path to the output CSV file.

    Raises:
        ValueError: If the input list is empty.
    """
    if not data:
        raise ValueError("The data list is empty. Nothing to write.")

    # Collect all unique keys across all dicts
    fieldnames = sorted({key for row in data for key in row.keys()})

    with open(filepath, mode="w", newline="", encoding="utf-8") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for row in data:
            # Fill in missing keys with empty string
            safe_row = {key: row.get(key, "") for key in fieldnames}
            writer.writerow(safe_row)

    print(f"✅ Saved {len(data)} rows with {len(fieldnames)} columns to {filepath}")


def load_dicts_from_csv(filepath: str) -> List[Dict[str, Any]]:
    """
    Load a CSV file into a list of dictionaries.

    Args:
        filepath (str): Path to the CSV file.

    Returns:
        List[Dict[str, Any]]: List of dictionaries representing rows.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If the file is empty or invalid.
    """
    with open(filepath, mode="r", newline="", encoding="utf-8") as csvfile:
        reader = csv.DictReader(csvfile)
        data = [dict(row) for row in reader]

    if not data:
        raise ValueError(f"The CSV file '{filepath}' is empty or contains no data.")

    print(f"✅ Loaded {len(data)} rows from {filepath}")
    return data


class BaseEvaluator(ABC):

    @abstractmethod
    def process_images(self, image_file_paths, **kwargs) -> Tuple[dict, list]:
        raise NotImplementedError

    @abstractmethod
    def summarize(self, results) -> dict:
        pass

    def evaluate(self, image_folder, log_path, results_log_path, **kwargs):
        config = ProjectConfig()
        output_dir = config.output_folder

        # Summary log path
        log_path = os.path.join(output_dir, "metrics", log_path)
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        
        # Individual results log path
        results_log_path = os.path.join(output_dir, "results", results_log_path)
        os.makedirs(os.path.dirname(results_log_path), exist_ok=True)
        
        # Load existing results if available
        existing_results = []
        if os.path.isfile(results_log_path):
            print(f"Results file already exists, loading it from: {results_log_path}")
            existing_results = load_dicts_from_csv(results_log_path)
            print(f"> Successfully loaded {len(existing_results)} existing results. Here are the first 5:")
            for i, result in enumerate(existing_results[:5]):
                print(f"[{i}] {result}")
        
        # Check if we need to process more images
        if isinstance(image_folder, str):
            image_count = len([f for f in os.listdir(image_folder) if is_image_file(f)])
        else:
            image_count = len(image_folder)
        
        if len(existing_results) < image_count:
            # Continue evaluation from where it left off
            start_index = len(existing_results)
            print(f"📊 Found {len(existing_results)} existing results and {image_count} images.")
            print(f"🔄 Continuing evaluation from index {start_index} ({image_count - start_index} images remaining)...")
            
            # Process only the remaining images
            new_summary, new_results = self.process_images(image_folder, start_index=start_index, **kwargs)
            
            # Merge existing and new results
            results = existing_results + new_results
            summary = self.summarize(results)
            
            # Save the combined results
            save_dicts_to_csv(results, results_log_path)
            print(f"✅ Saved {len(results)} total results (merged {len(existing_results)} existing + {len(new_results)} new)")
        elif len(existing_results) == image_count:
            # All images already evaluated
            summary = self.summarize(existing_results)
            results = existing_results
            print(f"✅ All {image_count} images already evaluated. Using existing results.")
        else:
            # More results than images - something is wrong, re-evaluate
            print(f"⚠️  Warning: {len(existing_results)} results but only {image_count} images. Re-evaluating all images...")
            summary, results = self.process_images(image_folder, start_index=0, **kwargs)
            save_dicts_to_csv(results, results_log_path)
            
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        file_exists = os.path.exists(log_path)

        # Optional metadata to attach to the CSV row (e.g., variant="triggered"/"org")
        row_metadata = kwargs.get('row_metadata', {})

        print("\nSummary Report")
        print("=" * 40)
        print(f"Total Images Processed : {summary['total']}")
        for label, value in summary.items():
            if label != 'total':
                print(f"{label.upper()} : {value}")

        row = {
            "timestamp": datetime.datetime.now().isoformat(),
            #"image_folder": image_folder.replace(output_dir, ''),
            **row_metadata,
            **summary
        }
        fieldnames = list(row.keys())

        with open(log_path, mode="a", newline="") as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

            if not file_exists:
                writer.writeheader()

            writer.writerow(row)

        print(f"\nResults appended to '{log_path}' ✅")

        # --- Show full table as a pandas DataFrame ---
        try:
            import pandas as pd

            print("\n📊 Evaluation Log Table:")
            df = pd.read_csv(log_path)
            print(df.to_string(index=False))
        except ImportError:
            print("\nPandas is not installed. Skipping table view.")





