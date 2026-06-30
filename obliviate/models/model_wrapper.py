from abc import ABC, abstractmethod
from typing import Optional

import torch
from peft import PeftModel
from transformers import (
    LogitsProcessorList,
    GenerationConfig,
    StoppingCriteriaList,
    PrefixConstrainedLogitsProcessor,
    UnbatchedClassifierFreeGuidanceLogitsProcessor,
)

from obliviate.configs.model_config import ModelConfig
from obliviate.configs.cfg_config import CFGMode
from obliviate.unlearning.target_config import Target, TARGET_PROMPTS
from obliviate.utils.misc import TQDMProgressBar


class ModelWrapper(ABC):
    """Abstract base class for model wrappers."""
    
    inference_config_defaults = dict()

    N_GENERATED_IMAGE_TOKENS = None  # Override in subclasses

    def __init__(self, config: ModelConfig):
        self.config = config

        self.left_padded_tokenizer = None  # for inference
        self.right_padded_tokenizer = None  # for training
        self.image_tokenizer = None
        self.model = None

        self._load_base_model()
        self._load_checkpoint(config.checkpoint)

    def eval(self):
        self.model.eval()

    def train(self):
        self.model.train()

    @abstractmethod
    def _load_base_model(self):
        """Load the base model. Must be implemented by subclasses."""
        pass

    def _load_checkpoint(self, checkpoint):
        """Load a PEFT adapter checkpoint if provided."""
        if not checkpoint:
            print("No checkpoint provided, skipping adapter load.")
            return

        print(f"Loading adapter from: {checkpoint}")
        try:
            self.model = PeftModel.from_pretrained(self.model, checkpoint)
            print(f"✅ Successfully loaded PEFT adapter: {checkpoint}")
        except Exception as e:
            print(f"❌ Failed to load adapter: {e}")

        return self.model

    @abstractmethod
    def get_unconditional_prompt(self):
        """Return the unconditional prompt for CFG."""
        pass
    
    def get_target_prompt(self, target: Optional[Target] = None):
        """
        Return the target prompt for NEGATIVE_PROMPT mode.
        """
        if target not in TARGET_PROMPTS:
            raise ValueError(f"Target '{target}' not found in TARGET_PROMPTS.")
        
        return TARGET_PROMPTS[target]
        
    @abstractmethod
    def process_multimodal_for_inference(self, prompt, task, image=None, **kwargs):
        """Process multimodal inputs for inference."""
        pass

    @abstractmethod
    def process_multimodal_for_training(self, prompt, task, answer=None, image=None, **kwargs):
        """Process multimodal inputs for training."""
        pass

    @torch.no_grad()
    @abstractmethod
    def sample_text_to_image(self, prompts, negative_prompts=None, batch_size=None, **kwargs):
        """Text-to-image sampling method."""
        pass

    @torch.no_grad()
    @abstractmethod
    def sample_image_to_text(self, images, prompts=None, batch_size=None, **kwargs):
        """Image-to-text sampling method."""
        pass

    @abstractmethod
    def get_embeddings(self) -> torch.nn.Embedding:
        """Returns the Embedding layer of this model."""
        pass

    def perform_generation(self, positive_inputs, n_max_new_tokens, negative_inputs=None, constrained_fn=None,
                           guidance_scale=1, generator=None, return_logits=False, 
                           cfg_mode=CFGMode.NONE,
                           target: Optional[Target] = None,
                           **generation_kwargs):
        model_was_in_training_mode = self.model.training
        self.model.eval()

        # prepare hyper parameters
        generation_config = GenerationConfig(
            use_cache=True,
            bos_token_id=self.model.config.bos_token_id,
            eos_token_id=self.model.config.eos_token_id,
            pad_token_id=self.model.config.pad_token_id,
            max_new_tokens=n_max_new_tokens,
            **generation_kwargs
        )

        # Define how the logits are modified during generation (both CFG AND prefix constraints)
        logits_processor = LogitsProcessorList()
        if negative_inputs:
            logits_processor.append(UnbatchedClassifierFreeGuidanceLogitsProcessor(
                guidance_scale,
                model=self.model,
                unconditional_ids=negative_inputs.input_ids,
                unconditional_attention_mask=negative_inputs.attention_mask,
            ))

        if constrained_fn:
            logits_processor.append(PrefixConstrainedLogitsProcessor(
                constrained_fn,
                num_beams=1,
            ))

        progress_bar = TQDMProgressBar(total_steps=n_max_new_tokens)
        stopping_criteria = StoppingCriteriaList([progress_bar])

        print(f"Starting generation for batch ...")

        outputs = self.model.generate(
            input_ids=positive_inputs.input_ids,
            attention_mask=positive_inputs.attention_mask,
            generation_config=generation_config,
            logits_processor=logits_processor,
            stopping_criteria=stopping_criteria,
            generator=generator,
            output_scores=return_logits,
            return_dict_in_generate=return_logits
        )
        
        if model_was_in_training_mode:
            self.model.train()
        else:
            self.model.eval()

        return outputs

    @abstractmethod
    def find_beginning_of_image_token_position(self, multimodal_input_ids):
        """Find the position of the beginning of image token."""
        raise NotImplementedError

    @abstractmethod
    def find_end_of_image_token_position(self, multimodal_input_ids):
        """Find the position of the end of image token."""
        raise NotImplementedError
    
    def get_modules_for_adaptation(self, ignore_keywords=None):
        """
        Get module names for LoRA adaptation.
        """
        print("Provided 'ignore_keywords':", ignore_keywords)
        
        cls = torch.nn.Linear
        lora_module_names = set()
        
        if ignore_keywords is None:
            ignore_keywords = []
        
        # Treat an empty string as not ignoring any layer
        if len(ignore_keywords) == 1 and ignore_keywords[0] == '':
            ignore_keywords = []
        
        print("Model:\n", self.model)
        print("Ignore keywords:", ignore_keywords)
        for name, module in self.model.named_modules():
            
            if ignore_keywords and any(keyword in name for keyword in ignore_keywords):
                continue
            
            if isinstance(module, cls):
                names = name.split('.')
                lora_module_names.add(names[0] if len(names) == 1 else names[-1])
        
        if 'lm_head' in lora_module_names:  # needed for 16-bit
            lora_module_names.remove('lm_head')
        return list(lora_module_names)

