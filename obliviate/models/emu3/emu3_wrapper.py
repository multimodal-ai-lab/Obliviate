import torch
from PIL import Image
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoImageProcessor, AutoModel

from typing import Optional

from obliviate.models.emu3.emu3_processing import Emu3Processor
from obliviate.models.model_wrapper import ModelWrapper
from obliviate.configs.cfg_config import CFGMode
from obliviate.unlearning.target_config import Target
from obliviate.utils.misc import build_random_generator, batch_iterator

EMU3_IMAGE_CAPTIONING_INSTRUCTION_PROMPT = "Please describe the image"

class Emu3ModelWrapper(ModelWrapper):
    """
    Wrapper class for the Emu3 model (Emu3 Team, 2024)
    """

    inference_config_defaults = {
        'guidance_scale': 3.0,
        'temperature': 0.99,  # not sure
        'top_k': 16_384,  # 2048, not sure
        'do_sample': True,
        'batch_size': 25  # not sure
    }

    N_GENERATED_TEXT_TOKENS = 128
    N_GENERATED_IMAGE_TOKENS = 4164
    MAX_GENERATED_TOKENS = 40960
    IMAGE_VOCAB_SIZE = 32768

    BOS_TOKEN = "<|extra_203|>"  # 151849?
    BOI_IMAGE_TOKEN = "<|image start|>"  # 151852
    EOI_IMAGE_TOKEN = "<|image token|>"  # 151851

    def __init__(self, config, emu3_model_name_or_path='BAAI/Emu3-Stage1',
                 emu3_vq_model_name_or_path='BAAI/Emu3-VisionTokenizer'):

        self.emu3_model_name_or_path = emu3_model_name_or_path
        self.emu3_vq_model_name_or_path = emu3_vq_model_name_or_path
        super().__init__(config)

        if emu3_model_name_or_path == 'BAAI/Emu3-Gen':
            self.inference_config_defaults['batch_size'] = 10

    def _load_base_model(self):
        # prepare model and processor
        self.model = AutoModelForCausalLM.from_pretrained(
            self.emu3_model_name_or_path,
            attn_implementation="flash_attention_2",
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
            image_area=self.config.image_size * self.config.image_size,
            # force_download=True
        )

        self.model.gradient_checkpointing_enable()
        self.model.enable_input_require_grads()

        print("Loaded Emu3 Model")

        # Tokenizers for left (inference) and right (training) padding
        self.left_padded_tokenizer = AutoTokenizer.from_pretrained(
            self.emu3_model_name_or_path, trust_remote_code=True, padding_side="left"
        )
        self.right_padded_tokenizer = AutoTokenizer.from_pretrained(
            self.emu3_model_name_or_path, trust_remote_code=True, padding_side="right", use_fast=False
        )

        # Image processor and tokenizer (VQ model)
        self.image_processor = AutoImageProcessor.from_pretrained(
            self.emu3_vq_model_name_or_path, trust_remote_code=True
        )
        self.image_tokenizer = AutoModel.from_pretrained(
            self.emu3_vq_model_name_or_path, device_map="cuda:0", trust_remote_code=True
        ).eval()

        # Emu3 processors for inference and training
        self.inference_processor = Emu3Processor(self.image_processor, self.image_tokenizer, self.left_padded_tokenizer)
        self.training_processor = Emu3Processor(self.image_processor, self.image_tokenizer, self.right_padded_tokenizer)

    def get_unconditional_prompt(self):
        return ""

    def find_beginning_of_image_token_position(self, multimodal_input_ids):
        boi_id = self.right_padded_tokenizer.convert_tokens_to_ids(self.right_padded_tokenizer.boi_token)
        boi_id_position = (multimodal_input_ids == boi_id).nonzero(as_tuple=True)[0][0]
        return boi_id_position

    def find_end_of_image_token_position(self, multimodal_input_ids):
        eoi_id = self.right_padded_tokenizer.convert_tokens_to_ids(self.right_padded_tokenizer.eoi_token)
        eoi_id_position = (multimodal_input_ids == eoi_id).nonzero(as_tuple=True)[0][0]
        return eoi_id_position

    def get_modules_for_adaptation(self, ignore_keywords=None):
        if ignore_keywords is None:
            ignore_keywords = ['lm_head', 'embed_tokens']
        return super().get_modules_for_adaptation(ignore_keywords=ignore_keywords)

    def process_multimodal_for_inference(self, prompt=None, image=None, task=None, **kwargs):
        mode = "G" if task == "text_to_image" else "U"
        return self.inference_processor(text=prompt, image=image, mode=mode, return_token_type_ids=False, **kwargs).to(self.model.device)

    def format_image_prompt(self, image_tokens):
        h, w = image_tokens.shape
        img_str = self.training_processor.to_imgstr(image_tokens)

        image_prompt = (
                self.right_padded_tokenizer.boi_token +
                f"{h}*{w}" +
                self.right_padded_tokenizer.img_token +
                img_str +
                self.right_padded_tokenizer.eol_token +
                self.right_padded_tokenizer.eof_token +
                self.right_padded_tokenizer.eoi_token
        )

        return image_prompt

    @torch.no_grad()
    def process_multimodal_for_training(self, prompt, task, answer=None, image=None, **kwargs):
        mode = "G" if task == "text_to_image" else "U"

        if not isinstance(prompt, list):
            prompts = [prompt]
        else:
            prompts = prompt

        if not isinstance(image, list):
            images = [image]
        else:
            images = image

        if mode == "G":
            text_prompts = prompts
            image_prompts = [
                self.format_image_prompt(image_token_ids)
                for image_token_ids in self.training_processor.tokenize_image(images)
            ]
            multimodal_prompts = [
                "<|extra_203|>" + text_str + image_str for (text_str, image_str) in zip(text_prompts, image_prompts)
            ]

        elif mode == "U":
            # For understanding mode, format with image tokens and text prompts
            for i in range(len(prompts)):
                if prompts[i] is None or prompts[i] == "<caption>":
                    prompts[i] = EMU3_IMAGE_CAPTIONING_INSTRUCTION_PROMPT
            
            # Resize images to 384x384 for understanding training tasks
            resized_images = [img.resize((384, 384)) if img is not None else img for img in images]
            image_tokens_list = self.training_processor.tokenize_image(resized_images)
            multimodal_prompts = []
            
            for text_prompt, image_tokens in zip(prompts, image_tokens_list):
                image_prompt = self.format_image_prompt(image_tokens)
                
                # RAW FORMAT: BOS + text_prompt + image_prompt (no conversation template)
                multimodal_prompt = self.right_padded_tokenizer.bos_token + text_prompt + image_prompt
                
                # If answer is provided, add it for training
                if answer is not None:
                    if isinstance(answer, list):
                        multimodal_prompt += answer[len(multimodal_prompts)]
                    else:
                        multimodal_prompt += answer
                
                multimodal_prompts.append(multimodal_prompt)

        inputs = self.right_padded_tokenizer(multimodal_prompts, return_token_type_ids=False, **kwargs).to(self.config.device)
        
        # Track prompt and answer positions for understanding mode when answer is provided
        if mode == "U" and answer is not None:
            
            # Get tokenized components to calculate positions
            tokenizer = self.right_padded_tokenizer
            answer_tokens = tokenizer(answer if isinstance(answer, str) else answer[0], add_special_tokens=False).input_ids
            answer_start = inputs.input_ids.shape[-1] - len(answer_tokens)
            answer_end = answer_start + len(answer_tokens)

            return inputs, (answer_start, answer_end)

        return inputs

    def get_embeddings(self) -> torch.nn.Embedding:
        raise NotImplementedError

    @torch.no_grad()
    def sample_text_to_image(self, prompts, negative_prompts=None, batch_size=None, return_image_tokens=False, return_logits=False,
                             cfg_mode=CFGMode.NONE,
                             target: Optional[Target] = None, image_size: Optional[int] = None, **kwargs):
        # Load default inference config
        default_kwargs = self.inference_config_defaults.copy()

        if isinstance(prompts, str):
            prompts = [prompts]

        if negative_prompts and isinstance(negative_prompts, str):
            negative_prompts = [negative_prompts]

        # Override with user-provided kwargs
        for key in default_kwargs:
            default_kwargs[key] = kwargs.get(key, default_kwargs[key])

        batch_size = batch_size or default_kwargs['batch_size']

        # Reproducibility seed
        generator = build_random_generator(seed=kwargs.pop('seed', None), device=self.config.device)

        # Prepare positive and negative prompts (for CFG)
        positive_prompts = prompts
        # For negative_prompt mode, use target prompt as negative prompts
        if cfg_mode == CFGMode.NEGATIVE_PROMPT:
            negative_prompts = [self.get_target_prompt(target=target)] * len(prompts) if negative_prompts is None else negative_prompts
        else:
            negative_prompts = [self.get_unconditional_prompt()] * len(prompts) if negative_prompts is None else negative_prompts
        assert len(positive_prompts) == len(negative_prompts), "Positive/negative prompts count mismatch!"

        print("Sampling images for the following prompts:")
        for idx, prompt in enumerate(positive_prompts):
            print(f"[{idx}] {prompt.encode('unicode_escape').decode('ascii')}")

        # Process prompts in batches
        for i, (batch_positive_prompts, batch_negative_prompts) in enumerate(zip(
                batch_iterator(positive_prompts, batch_size),
                batch_iterator(negative_prompts, batch_size)
        )):

            # Format and tokenize the prompts
            image_area = (image_size * image_size) if image_size is not None else self.model.config.image_area
            processor_kwargs = dict(
                task='text_to_image',
                ratio=["1:1"] * len(batch_positive_prompts),
                image_area=image_area,
                return_tensors="pt",
                padding="longest"
            )
            positive_inputs = self.process_multimodal_for_inference(batch_positive_prompts, **processor_kwargs)
            negative_inputs = self.process_multimodal_for_inference(batch_negative_prompts, **processor_kwargs)

            # Build Emu3 logits constraint function
            h = positive_inputs.image_size[:, 0]
            w = positive_inputs.image_size[:, 1]
            constrained_fn = self.inference_processor.build_prefix_constrained_fn(h, w)

            # Generate images
            outputs = self.perform_generation(
                positive_inputs=positive_inputs,
                negative_inputs=negative_inputs,
                constrained_fn=constrained_fn,
                n_max_new_tokens=self.N_GENERATED_IMAGE_TOKENS*2,
                guidance_scale=default_kwargs['guidance_scale'],
                generator=generator,
                cfg_mode=cfg_mode,
                target=target,
                # Below are generation parameters
                do_sample=default_kwargs['do_sample'],
                top_k=default_kwargs['top_k']
            )

            generated_sequences = outputs
            logit_scores = None

            # Cleanup per-batch intermediate tensors
            del positive_inputs
            torch.cuda.empty_cache()

            # Convert each generated id sequence to an image and append in-order
            for idx_in_batch, seq in enumerate(generated_sequences):
                multimodal_list = self.inference_processor.decode(seq)
                images = list(filter(lambda x: isinstance(x, Image.Image), multimodal_list))
                assert len(images) == 1, "Expected a single image output per prompt!"

                # Yield the image (PIL format)
                for img in images:
                    yield img

            # Free memory for next batch
            del outputs, multimodal_list
            torch.cuda.empty_cache()

    @torch.no_grad()
    def sample_image_to_text(self, images, prompts=None, batch_size=None, **kwargs):

        if not isinstance(images, list):
            images = [images]

        if isinstance(images[0], str):
            images = [Image.open(f) for f in images]

        # Reproducibility seed
        generator = build_random_generator(seed=kwargs.pop('seed', None), device=self.config.device)

        # Default to image captioning
        if prompts is None:
            prompts = ["This image depicts:"] * len(images)

        # Copy prompt for each image in case of only a single provided prompt
        if len(prompts) == 1:
            prompts *= len(images)

        # Copy image for each prompt in case of only a single provided image
        if len(images) == 1:
            images *= len(prompts)

        assert len(prompts) == len(images), f"Prompts and images count mismatch! ({len(prompts)} != {len(images)})"

        # Iterate in batches
        for i, (batch_images, batch_prompts) in enumerate(zip(batch_iterator(images, batch_size), batch_iterator(prompts, batch_size))):

            # Downscale images to 384x384 for understanding tasks
            resized_batch_images = [img.resize((384, 384)) for img in batch_images]

            # Format and tokenize the prompts
            processor_kwargs = dict(
                task='image_to_text',
                return_tensors="pt",
                padding="longest",
                padding_image=True  # Important for batch processing with different image sizes
            )
            batch_inputs = self.process_multimodal_for_inference(prompt=batch_prompts, image=resized_batch_images, **processor_kwargs)

            # Extract generation parameters from kwargs
            max_new_tokens = kwargs.pop('max_new_tokens', self.N_GENERATED_TEXT_TOKENS) # generate less tokens compared to image generation.
            temperature = kwargs.pop('temperature', None)
            do_sample = kwargs.pop('do_sample', temperature is not None and temperature > 0.0)
            
            # Generate captions
            outputs = self.perform_generation(
                positive_inputs=batch_inputs,
                n_max_new_tokens=max_new_tokens,
                generator=generator,
                return_logits=False,
                # Below are generation parameters
                do_sample=do_sample,
                temperature=temperature if temperature and temperature > 0.0 else None,
            )

            generated_sequences = outputs

            prompt_len = batch_inputs.input_ids.shape[1]
            generated_ids = generated_sequences[:, prompt_len:]
            decoded = self.left_padded_tokenizer.batch_decode(generated_ids, skip_special_tokens=True)

            # Decode the text
            for idx_in_batch, response in enumerate(decoded):
                yield response.strip()

            # Free memory for next batch
            del outputs, batch_inputs
            torch.cuda.empty_cache()


    @torch.no_grad()
    def sample_text_to_text(self, prompts, batch_size=None, return_logits=False, **kwargs):
        raise NotImplementedError
