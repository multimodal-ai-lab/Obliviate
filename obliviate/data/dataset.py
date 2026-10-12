import os
from torch.utils.data import Dataset
import torch

from obliviate.unlearning.config import UnlearningConfig
from obliviate.models.model_wrapper import ModelWrapper
from obliviate.data.dataset_features import extract_target_text_to_image_logits
from obliviate.data.data_utils import IGNORE_INDEX, get_cached_image_count


def generate_target_images_cache(prompts, cache_dir, wrapper: ModelWrapper, start_index: int = 0):
    """
    Pre-generate and cache target images for training.
    """
    images_dir = os.path.join(cache_dir, "images")
    os.makedirs(images_dir, exist_ok=True)

    to_generate = prompts[start_index:]
    if not to_generate:
        print(f"✅ Image cache already complete: {start_index} images present, nothing to generate.")
        return cache_dir

    image_size = getattr(wrapper.config, 'image_size', 512)
    print(f"Generating {len(to_generate)} target images (indices {start_index}..{start_index + len(to_generate) - 1})...")
    with torch.no_grad():
        for i, target_img in enumerate(wrapper.sample_text_to_image(prompts=to_generate, image_size=image_size)):
            idx = start_index + i
            target_img.save(os.path.join(images_dir, f"{idx}.png"))
            if (i + 1) % 10 == 0:
                print(f"Generated {idx + 1}/{len(prompts)} images")
    print(f"✅ Generated and cached {len(to_generate)} images to {images_dir} (total: {start_index + len(to_generate)})")
    return cache_dir


class UnlearningDataset(Dataset):
    """
    Dataset for unlearning training.
    Uses a single wrapper with LoRA; extracts teacher logits on-the-fly by disabling adapters.
    """

    def __init__(self, config: UnlearningConfig, wrapper: ModelWrapper):
        self.config = config
        self.wrapper = wrapper
        self.device = config.device

        from obliviate.unlearning.target_config import TARGET_PROMPTS
        target_prompt = TARGET_PROMPTS[config.target]

        num_samples = config.max_steps
        self.prompts = [target_prompt] * num_samples

        print(f"Using concept-level training with target prompt: '{target_prompt}'")
        print(f"Number of training samples: {num_samples}")

        self.image_cache_dir = os.path.join(
            config.project_config.data_folder,
            "unlearning_cache",
            wrapper.config.model_type.value,
            config.project_config.exp_path,
            config.target.value
        )

        cached_count = get_cached_image_count(self.image_cache_dir)
        if cached_count < num_samples:
            raise ValueError(
                f"Image cache at {self.image_cache_dir} has {cached_count} images, but {num_samples} are required. "
                "Generate/resume image cache before LoRA setup (training does this automatically)."
            )
        print(f"✅ Using existing image cache: {self.image_cache_dir} ({cached_count} images)")

        self.images_dir = os.path.join(self.image_cache_dir, "images")
        self.image_paths = [
            os.path.join(self.images_dir, f"{idx}.png")
            for idx in range(len(self.prompts))
        ]

        print(f"Initialized UnlearningDataset with {len(self.prompts)} prompts")
        print("Teacher logits: computed on-the-fly (base model, LoRA disabled)")

    def __len__(self):
        return len(self.prompts)

    def _create_multimodal_text_to_image_input_sequence(self, prompt, image_path, include_uncond=False):
        from PIL import Image

        image = Image.open(image_path).convert("RGB")

        if include_uncond:
            prompts = [prompt, self.wrapper.get_unconditional_prompt()]
            images = [image, image]
        else:
            prompts = [prompt]
            images = [image]

        multimodal_ids = self.wrapper.process_multimodal_for_training(
            prompt=prompts,
            image=images,
            task="text_to_image",
            padding=True,
            truncation=True,
            return_tensors="pt",
            max_length=self.wrapper.right_padded_tokenizer.model_max_length
        )

        if not include_uncond:
            return multimodal_ids.input_ids.squeeze(0).to(self.device)
        cond, uncond = multimodal_ids.input_ids.to(self.device).split(split_size=1, dim=0)
        return cond.squeeze(0), uncond.squeeze(0)

    def _create_labels(self, input_ids, instruction_len):
        labels = input_ids.clone()
        labels[:instruction_len] = IGNORE_INDEX
        labels = labels.roll(-1, dims=0)
        labels[-1] = IGNORE_INDEX
        return labels

    @torch.no_grad()
    def __getitem__(self, idx):
        prompt = self.prompts[idx]
        image_path = self.image_paths[idx]

        cond_multimodal_input_ids, uncond_multimodal_input_ids = \
            self._create_multimodal_text_to_image_input_sequence(
                prompt, image_path, include_uncond=True
            )

        instruction_len = self.wrapper.find_beginning_of_image_token_position(cond_multimodal_input_ids) + 1
        labels = self._create_labels(cond_multimodal_input_ids, instruction_len)

        with self.wrapper.model.disable_adapter():
            target_logits_data = extract_target_text_to_image_logits(
                target_prompt=prompt,
                target_image_path=image_path,
                wrapper=self.wrapper
            )

        target_cond_logits = target_logits_data['target_cond_logits'].squeeze(0)
        target_uncond_logits = target_logits_data['target_uncond_logits'].squeeze(0)

        full_target_cond_logits = torch.zeros(len(cond_multimodal_input_ids), target_cond_logits.shape[-1])
        full_target_uncond_logits = torch.zeros(len(cond_multimodal_input_ids), target_uncond_logits.shape[-1])

        boi_token_position = instruction_len - 1
        full_target_cond_logits[boi_token_position:boi_token_position + len(target_cond_logits)] = target_cond_logits
        full_target_uncond_logits[boi_token_position:boi_token_position + len(target_uncond_logits)] = target_uncond_logits

        multimodal_input_ids = torch.stack([cond_multimodal_input_ids, uncond_multimodal_input_ids], dim=0)

        return {
            "input_ids": multimodal_input_ids,
            "inputs_embeds": target_logits_data['inputs_embeds'],
            "labels": labels,
            "target_cond_logits": full_target_cond_logits,
            "target_uncond_logits": full_target_uncond_logits,
        }
