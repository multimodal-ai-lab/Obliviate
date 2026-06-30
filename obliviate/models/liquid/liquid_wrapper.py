import torch
from PIL import Image
from transformers import AutoTokenizer, AutoModelForCausalLM
import os

from obliviate.models.chameleon.inference.image_tokenizer import ImageTokenizer

from typing import Optional

from obliviate.models.model_wrapper import ModelWrapper
from obliviate.models.liquid.liquid_formatter import LiquidPromptFormatter
from obliviate.configs.cfg_config import CFGMode
from obliviate.unlearning.target_config import Target
from obliviate.utils.misc import build_random_generator, batch_iterator


class LiquidModelWrapper(ModelWrapper):
    """
    Wrapper class for the Liquid model (Wu et al., 2025)
    """

    inference_config_defaults = {
        'guidance_scale': 7.0,
        'temperature': 0.99,
        'top_p': 0.96,
        'top_k': 4096,
        'do_sample': True,
        'batch_size': 25
    }

    N_GENERATED_TEXT_TOKENS = 256
    N_GENERATED_IMAGE_TOKENS = 1024
    IMAGE_VOCAB_SIZE = 8192

    def __init__(self, config, model_name_or_path='Junfeng5/Liquid_V1_7B'):
        self.model_name_or_path = model_name_or_path
        super().__init__(config)

    def _load_base_model(self):
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name_or_path,
            attn_implementation='flash_attention_2',
            torch_dtype=torch.bfloat16,
            device_map="auto"
        )
        print(f"Loaded Liquid Model!")

        # Load the Image Tokenizer
        vqgan_base = os.path.join(os.path.dirname(__file__), '../chameleon/vqgan_weights')
        self.image_tokenizer = ImageTokenizer(
            cfg_path=os.path.join(vqgan_base, "vqgan.yaml"),
            ckpt_path=os.path.join(vqgan_base, "vqgan.ckpt"),
            device=self.config.device
        )

        # Tokenizers for left (inference) and right (training) padding
        self.left_padded_tokenizer = AutoTokenizer.from_pretrained(self.model_name_or_path, padding_side='left')
        self.right_padded_tokenizer = AutoTokenizer.from_pretrained(self.model_name_or_path, padding_side='right')

        self.text_to_image_prompt_formatter = LiquidPromptFormatter(task="text_to_image")
        self.image_to_text_prompt_formatter = LiquidPromptFormatter(task="image_to_text")
        self.text_to_text_prompt_formatter = LiquidPromptFormatter(task="text_to_text")

    def get_unconditional_prompt(self):
        return "<unconditional>"

    def get_formatted_input(self, prompt, answer=None, mode="inference", task='text_to_image'):
        if task == 'text_to_image':
            return self.text_to_image_prompt_formatter(prompt=prompt, mode=mode)
        elif task == 'image_to_text':
            return self.image_to_text_prompt_formatter(
                prompt=prompt, answer=answer, mode=mode, apply_conversation_format=True
            )
        elif task == 'text_to_text':
            return self.text_to_text_prompt_formatter(
                prompt=prompt, answer=answer, mode=mode, apply_conversation_format=True
            )
        else:
            raise NotImplementedError

    def process_multimodal(self, mode, prompt, answer=None, image=None, task=None, **kwargs):
        formatted = self.get_formatted_input(prompt=prompt, answer=answer, mode=mode, task=task)
        tokenizer = self.left_padded_tokenizer if mode == "inference" else self.right_padded_tokenizer
        inputs = tokenizer(formatted, **kwargs).to(self.config.device)
        
        if image:
            if "<boi>" not in formatted and not all("<boi>" in p for p in formatted):
                raise ValueError(f"Formatted prompt must contain '<boi><eoi>' for image insertion, but got: {formatted}")

            images = image if isinstance(image, list) else [image]
            image_tokens_batch = [self.tokenize_pil_image(img) for img in images]

            multimodal_input_ids = []
            for text_ids, img_tokens in zip(inputs.input_ids, image_tokens_batch):
                updated_ids = self._insert_image_tokens_in_text_tokens(text_ids, img_tokens)
                multimodal_input_ids.append(updated_ids)

            pad_token_id = tokenizer.pad_token_id
            inputs["input_ids"] = torch.stack(multimodal_input_ids, dim=0)
            inputs["attention_mask"] = inputs["input_ids"].ne(pad_token_id)

        if task != "text_to_image" and answer is not None:
            answer_tokens = tokenizer(answer, add_special_tokens=False).input_ids
            answer_end = inputs.input_ids.shape[1]
            answer_start = answer_end - (len(answer_tokens) + 3)
            return inputs, (answer_start, answer_end)

        return inputs

    def process_multimodal_for_inference(self, prompt, task, image=None, **kwargs):
        return self.process_multimodal(mode="inference", prompt=prompt, image=image, task=task, **kwargs)

    def process_multimodal_for_training(self, prompt, task, answer=None, image=None, **kwargs):
        return self.process_multimodal(mode="training", prompt=prompt, answer=answer, image=image, task=task, **kwargs)

    @torch.no_grad()
    def tokenize_pil_image(self, image: Image.Image):
        return torch.tensor(self.image_tokenizer.img_tokens_from_pil(image), dtype=torch.long)

    def get_lower_and_upper_visual_vocab_bounds(self):
        original_vocab_size = len(self.right_padded_tokenizer)
        return original_vocab_size, original_vocab_size + self.IMAGE_VOCAB_SIZE

    @torch.no_grad()
    def sample_text_to_image(self, prompts, negative_prompts=None, batch_size=None, cfg_mode=CFGMode.NONE,
                             target: Optional[Target] = None, **kwargs):

        default_kwargs = self.inference_config_defaults.copy()

        if isinstance(prompts, str):
            prompts = [prompts]

        if negative_prompts and isinstance(negative_prompts, str):
            negative_prompts = [negative_prompts]

        for key in default_kwargs:
            default_kwargs[key] = kwargs.get(key, default_kwargs[key])

        batch_size = batch_size or default_kwargs['batch_size']
        generator = build_random_generator(seed=kwargs.pop('seed', None), device=self.config.device)

        positive_prompts = prompts
        # For negative_prompt mode, use target prompt as negative prompts
        if cfg_mode == CFGMode.NEGATIVE_PROMPT:
            negative_prompts = [
                self.get_target_prompt(target=target) for _ in prompts
            ] if negative_prompts is None else negative_prompts
        else:
            negative_prompts = [
                self.get_unconditional_prompt() for _ in prompts
            ] if negative_prompts is None else negative_prompts
        assert len(positive_prompts) == len(negative_prompts), "Positive/negative prompts count mismatch!"

        print("Sampling images for the following prompts:")
        for idx, prompt in enumerate(positive_prompts):
            print(f"[{idx}] {prompt.encode('unicode_escape').decode('ascii')}")

        # Build Liquid logits constraint function
        def constrained_fn(batch_id, current_tokens):
            start = 256000
            vocab_size = self.model.config.vocab_size
            return list(range(start, vocab_size))

        for i, (batch_positive_prompts, batch_negative_prompts) in enumerate(zip(
                batch_iterator(positive_prompts, batch_size),
                batch_iterator(negative_prompts, batch_size)
        )):
            tokenizer_kwargs = dict(task="text_to_image", return_tensors="pt", padding=True)

            positive_inputs = self.process_multimodal_for_inference(
                prompt=batch_positive_prompts, image=None, **tokenizer_kwargs
            )
            negative_inputs = self.process_multimodal_for_inference(
                prompt=batch_negative_prompts, image=None, **tokenizer_kwargs
            )

            outputs = self.perform_generation(
                positive_inputs=positive_inputs,
                negative_inputs=negative_inputs,
                constrained_fn=constrained_fn,
                n_max_new_tokens=self.N_GENERATED_IMAGE_TOKENS,
                guidance_scale=default_kwargs['guidance_scale'],
                generator=generator,
                return_logits=False,
                cfg_mode=cfg_mode,
                target=target,
                do_sample=default_kwargs['do_sample'],
                top_p=default_kwargs['top_p'],
                top_k=default_kwargs['top_k'],
                temperature=default_kwargs['temperature']
            )

            generated_sequences = outputs

            prompt_len = positive_inputs.input_ids.shape[1]
            generated_ids = generated_sequences[:, prompt_len:]

            del positive_inputs
            torch.cuda.empty_cache()

            image_token_ids = self.postprocess_image_tokens_for_decoding(generated_ids)

            for idx_in_batch in range(image_token_ids.shape[0]):
                image_tokens = image_token_ids[idx_in_batch]
                rec_img = self.image_tokenizer.pil_from_img_toks(image_tokens)

                yield rec_img

            torch.cuda.empty_cache()

    @torch.no_grad()
    def sample_image_to_text(self, images, prompts=None, batch_size=None, **kwargs):
        # Simplified implementation - can be expanded later
        raise NotImplementedError("Image-to-text sampling not yet implemented in Obliviate")

    def find_beginning_of_image_token_position(self, multimodal_input_ids):
        boi_id = self.right_padded_tokenizer.convert_tokens_to_ids("<boi>")
        boi_id_position = (multimodal_input_ids == boi_id).nonzero(as_tuple=True)[0][0]
        return boi_id_position

    def find_end_of_image_token_position(self, multimodal_input_ids):
        eoi_id = self.right_padded_tokenizer.convert_tokens_to_ids("<eoi>")
        eoi_id_position = (multimodal_input_ids == eoi_id).nonzero(as_tuple=True)[0][0]
        return eoi_id_position

    def _insert_image_tokens_in_text_tokens(self, text_token_ids, image_token_ids):
        instruction_len = self.find_beginning_of_image_token_position(text_token_ids) + 1
        original_vocab_size = len(self.right_padded_tokenizer)
        image_token_ids = image_token_ids.squeeze(0) + original_vocab_size
        multimodal_input_ids = torch.cat(
            [text_token_ids[:instruction_len], image_token_ids, text_token_ids[instruction_len:]]
        )
        return multimodal_input_ids

    def get_modules_for_adaptation(self, ignore_keywords=None):
        """
        Get module names for LoRA adaptation.
        """
        if ignore_keywords is None:
            ignore_keywords = ['mm_projector', 'vision_tower', 'vision_resampler', 'vlm_uni']
        return super().get_modules_for_adaptation(ignore_keywords=ignore_keywords)

    def get_embeddings(self) -> torch.nn.Embedding:
        return self.model.model.model.embed_tokens

    def postprocess_image_tokens_for_decoding(self, generated_ids):
        original_vocab_size = len(self.right_padded_tokenizer)
        image_token_ids = generated_ids - original_vocab_size
        image_token_ids = torch.clamp(image_token_ids, min=0, max=self.IMAGE_VOCAB_SIZE)
        assert torch.all(image_token_ids >= 0), "Predicted tokens overlap with text vocab!"
        return image_token_ids

