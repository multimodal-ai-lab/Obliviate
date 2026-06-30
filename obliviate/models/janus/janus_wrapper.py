from typing import Optional

import numpy as np
import torch
import PIL
from PIL import Image
import tqdm
from transformers import AutoModelForCausalLM

from obliviate.configs.cfg_config import CFGMode
from obliviate.models.janus.modeling_vlm import MultiModalityCausalLM
from obliviate.models.janus.processing_vlm import VLChatProcessor

from obliviate.models.model_wrapper import ModelWrapper
from obliviate.unlearning.target_config import Target
from obliviate.utils.misc import batch_iterator, build_random_generator


class JanusModelWrapper(ModelWrapper):
    """
    Wrapper class for the Janus model (Chen et al., 2025)
    """

    inference_config_defaults = {
        'guidance_scale': 5.0,
        'temperature': 1.0,
        'batch_size': 1
    }

    PATCH_SIZE = 16
    N_GENERATED_TEXT_TOKENS = 256
    N_GENERATED_IMAGE_TOKENS = 576

    def __init__(self, config, model_name_or_path="deepseek-ai/Janus-Pro-7B"):
        self.model_name_or_path = model_name_or_path
        super().__init__(config)
        self.image_size = config.image_size

    def eval(self):
        self.vlm.eval()

    def train(self):
        self.vlm.train()

    def _load_base_model(self):
        self.inference_processor: VLChatProcessor = VLChatProcessor.from_pretrained(self.model_name_or_path)
        self.training_processor: VLChatProcessor = VLChatProcessor.from_pretrained(self.model_name_or_path)
        self.training_processor.tokenizer.padding_side = "right"

        vlm: MultiModalityCausalLM = AutoModelForCausalLM.from_pretrained(
            self.model_name_or_path, trust_remote_code=True
        ).to(torch.bfloat16).cuda()

        self.model = vlm.language_model  # TBD which one
        self.vlm = vlm
        self.language_model = vlm.language_model
        self.image_tokenizer = vlm.gen_vision_model

        self.left_padded_tokenizer = self.inference_processor.tokenizer
        self.right_padded_tokenizer = self.training_processor.tokenizer

    def get_modules_for_adaptation(self, ignore_keywords=None):
        if ignore_keywords is None:
            ignore_keywords = ['vision_model', 'aligner', 'gen_vision_model', 'gen_embed', 'lm_head']
        return super().get_modules_for_adaptation(ignore_keywords=ignore_keywords)

    def get_unconditional_prompt(self):
        return ""

    def get_formatted_text_to_image_inference_prompt(self, prompt):
            conversation = [
                {"role": "<|User|>", "content": prompt},
                {"role": "<|Assistant|>", "content": ""},
            ]
            sft_format = self.inference_processor.apply_sft_template_for_multi_turn_prompts(
                conversations=conversation,
                sft_format=self.inference_processor.sft_format,
                system_prompt="",
            )
            # Assuming text-to-image inference (that's why we add the image_start_tag)
            sft_format = sft_format + self.inference_processor.image_start_tag

            # Assuming text-to-image training
            return sft_format

    def process_multimodal(self, mode, prompt, answer=None, image=None, task=None, **kwargs):

        # Inserting Prompt into the SFT template
        print("Inserting prompts (and images) into the SFT template")
        if isinstance(prompt, str):
            prompts = [prompt]
        else:
            prompts = prompt

        if mode == "inference":
            prompts = [self.get_formatted_text_to_image_inference_prompt(prompt=p) for p in prompts]
            inputs = self.left_padded_tokenizer(prompts, padding=True)

        elif mode == "training":
            if image is None:
                images = [None] * len(prompts)
            elif not isinstance(image, (list, tuple)):
                images = [image]
            else:
                images = image

            for p, i, in zip(prompts, images):
                print(f"Prompt: {p} and Image: {i}")

            assert images[0] is not None

            print("Identified an image in the input! Inserting the appropriate image tokens in the SFT template...")

            outputs = []
            for p, i, in zip(prompts, images):
                conversation = [
                    {"role": "<|User|>", "content": p},
                    {"role": "<|Assistant|>", "content": "<image_placeholder>", "image": ["placeholder"], },
                ]
                outputs.append(self.training_processor(
                    conversations=conversation,
                    images=[i],
                    return_tensors="pt",
                    force_batchify=False,
                ))

            inputs = self.training_processor.batchify(outputs)
            print("self.training_processor.pad_id", self.training_processor.pad_id)
            print("self.right_padded_tokenizer.pad_token_id", self.right_padded_tokenizer.pad_token_id)
            print("self.training_processor.image_start_id", self.training_processor.image_start_id)
            print("self.training_processor.image_id", self.training_processor.image_id)
            print("self.training_processor.image_end_id", self.training_processor.image_end_id)

        else:
            raise NotImplementedError("Either 'inference' or 'training' mode should be specified!")

        return inputs

    def process_multimodal_for_inference(self, prompt, task, image=None, **kwargs):
        return self.process_multimodal(mode="inference", prompt=prompt, task=task, image=None)

    def process_multimodal_for_training(self, prompt, task, answer=None, image=None, **kwargs):
        return self.process_multimodal(mode="training", prompt=prompt, task=task, image=image)

    @torch.no_grad()
    def tokenize_pil_image(self, image: Image.Image):
        return torch.tensor(self.image_tokenizer.img_tokens_from_pil(image), dtype=torch.long)

    @torch.no_grad()
    def sample_text_to_image(self, prompts, negative_prompts=None, batch_size=None, cfg_mode=CFGMode.NONE,
                             target: Optional[Target] = None, **kwargs):

        #assert not negative_prompts, "Negative prompts are not yet supported in Janus text-to-image generation."
        #assert cfg_mode == CFGMode.NONE, "CFG modes other than NONE are not yet supported in Janus text-to-image generation."

        # Load default inference config
        default_kwargs = self.inference_config_defaults.copy()

        # Override with user-provided kwargs
        for key in default_kwargs:
            default_kwargs[key] = kwargs.get(key, default_kwargs[key])

        # TODO: Temporary fix to avoid padding issues with Janus:
        default_kwargs['batch_size'] = 1

        # Reproducibility seed
        generator = build_random_generator(seed=kwargs.pop('seed', None), device=self.config.device)

        if isinstance(prompts, str):
            prompts = [prompts]

        print("Sampling images for the following prompts:")
        for idx, prompt in enumerate(prompts):
            print(f"[{idx}] {prompt.encode('unicode_escape').decode('ascii')}")

        # ====================================================================================
        #for idx, p in enumerate(prompts):
        #    images = self._legacy_generate(prompt=p, generator=generator)
        #    yield images[0]

        #return
        # =============== Everything below is our logic for batched generation ===============

        # TODO: Temporary fix to avoid padding issues in Janus.
        print("Batch size for inference:", default_kwargs['batch_size'])

        # Prepare positive and negative prompts (for CFG)
        positive_prompts = prompts

        if cfg_mode == CFGMode.NEGATIVE_PROMPT:
            negative_prompts = [
                self.get_target_prompt(target=target) for _ in prompts
            ] if negative_prompts is None else negative_prompts
        else:
            negative_prompts = [
                None for _ in prompts
            ] if negative_prompts is None else negative_prompts
        assert len(positive_prompts) == len(negative_prompts), "Positive/negative prompts count mismatch!"

        # Iterate in batches
        for i, (batch_positive_prompts, batch_negative_prompts) in enumerate(zip(
                batch_iterator(positive_prompts, default_kwargs['batch_size']),
                batch_iterator(negative_prompts, default_kwargs['batch_size'])
        )):

            batch_size = len(batch_positive_prompts)

            # CFGMode.NONE: Only positive prompts, "negative" ones are just unconditional paddings
            if batch_negative_prompts[0] is None:
                inputs = self.process_multimodal_for_inference(
                    prompt=batch_positive_prompts, task="text_to_image"
                )
                input_ids = torch.LongTensor(inputs.input_ids).cuda()

                # Prepare for (cond, uncond) parallel decoding
                tokens = input_ids.repeat_interleave(2, dim=0)  # [P1, P1, P2, P2...]
                # Mask the unconditional rows (1, 3, 5...)
                tokens[1::2, 1:-1] = self.inference_processor.pad_id

                print("Input IDS (Inference)", tokens)

            # CFGMode.NEGATIVE_PROMPT: Interleave positive and negative prompts for parallel decoding
            else:
                print("Using Negative Prompts:", negative_prompts)
                # 1. Combine into a single list to ensure uniform padding/sequence length
                # Layout: [P1, P2... PN, N1, N2... NN]
                combined_prompts = batch_positive_prompts + batch_negative_prompts

                # 2. Process everything together
                inputs = self.process_multimodal_for_inference(
                    prompt=combined_prompts,
                    task="text_to_image"
                )

                # 3. Move to GPU
                all_input_ids = torch.LongTensor(inputs.input_ids).cuda()

                # 4. Interleave the IDs: [P1, N1, P2, N2...]
                # Split the tensor back into the positive and negative halves
                pos_ids, neg_ids = all_input_ids.chunk(2, dim=0)

                # Stack along a new dimension [Batch, 2, SeqLen] and flatten the first two
                tokens = torch.stack([pos_ids, neg_ids], dim=1).view(batch_size * 2, -1)

            # Get the text input embeddings
            inputs_embeds = self.language_model.get_input_embeddings()(tokens)

            # Start the iterative (CFG) decoding
            generated_tokens = torch.zeros((batch_size, self.N_GENERATED_IMAGE_TOKENS), dtype=torch.int).cuda()
            outputs = None
            for i in tqdm.tqdm(range(self.N_GENERATED_IMAGE_TOKENS), desc="Generating Image Tokens w/ Janus"):
                outputs = self.language_model.model(inputs_embeds=inputs_embeds,
                                                    use_cache=True,
                                                    past_key_values=outputs.past_key_values if i != 0 else None)
                hidden_states = outputs.last_hidden_state

                # Derive CFG logits
                logits = self.vlm.gen_head(hidden_states[:, -1, :])
                logit_cond = logits[0::2, :]
                logit_uncond = logits[1::2, :]

                logits = logit_uncond + default_kwargs['guidance_scale'] * (logit_cond - logit_uncond)
                probs = torch.softmax(logits / default_kwargs['temperature'], dim=-1)

                # Sample the next token
                next_token = torch.multinomial(probs, num_samples=1, generator=generator)
                generated_tokens[:, i] = next_token.squeeze(dim=-1)

                next_token = torch.cat([next_token.unsqueeze(dim=1), next_token.unsqueeze(dim=1)], dim=1).view(-1)
                img_embeds = self.vlm.prepare_gen_img_embeds(next_token)
                inputs_embeds = img_embeds.unsqueeze(dim=1)

            # Postprocess generated tokens
            image_token_ids = self.postprocess_image_tokens_for_decoding(generated_tokens)

            # Decode each image token sequence
            for idx_in_batch in range(image_token_ids.shape[0]):
                image_tokens = image_token_ids[idx_in_batch]
                rec_img = PIL.Image.fromarray(image_tokens)

                yield rec_img

            torch.cuda.empty_cache()

    @torch.no_grad()
    def sample_image_to_text(self, images, prompts=None, batch_size=None, **kwargs):
        raise NotImplementedError("Look at other ModelWrapper classes before implementing this method!")

    @torch.no_grad()
    def sample_text_to_text(self, prompts, batch_size=None, return_logits=False, **kwargs):
        raise NotImplementedError("Look at other ModelWrapper classes before implementing this method!")

    def find_beginning_of_image_token_position(self, multimodal_input_ids):
        boi_id = self.right_padded_tokenizer.convert_tokens_to_ids("<begin_of_image>")
        boi_id_position = (multimodal_input_ids == boi_id).nonzero(as_tuple=True)[0][0]
        return boi_id_position

    def find_end_of_image_token_position(self, multimodal_input_ids):
        eoi_id = self.right_padded_tokenizer.convert_tokens_to_ids("<end_of_image>")
        eoi_id_position = (multimodal_input_ids == eoi_id).nonzero(as_tuple=True)[0][0]
        return eoi_id_position

    def get_embeddings(self) -> torch.nn.Embedding:
        return self.model.model.model.embed_tokens

    def postprocess_image_tokens_for_decoding(self, generated_ids):
        batch_size = len(generated_ids)

        decoded = self.image_tokenizer.decode_code(
            generated_ids.to(dtype=torch.int), shape=[
                batch_size, 8, self.image_size // self.PATCH_SIZE, self.image_size // self.PATCH_SIZE
            ]
        )

        decoded = decoded.to(torch.float32).cpu().numpy().transpose(0, 2, 3, 1)
        decoded = np.clip((decoded + 1) / 2 * 255, 0, 255)

        visual_img = np.zeros((batch_size, self.image_size, self.image_size, 3), dtype=np.uint8)
        visual_img[:, :, :] = decoded

        return visual_img
