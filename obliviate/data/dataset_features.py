import torch
from torch import no_grad
from PIL import Image
import random
from obliviate.models import JanusModelWrapper
from obliviate.models.model_wrapper import ModelWrapper
from obliviate.data.data_utils import IGNORE_INDEX


@no_grad()
def extract_target_text_to_image_logits(target_prompt, target_image_path, wrapper: ModelWrapper):
    """
    Extract target logits for KL divergence loss.
    
    For a given target prompt and corresponding target image, it computes the textually
    conditioned and textually unconditioned logits of the model, when the image tokens
    are forced to be those of the target image.
    
    Args:
        target_prompt: The harmful prompt
        target_image_path: path to image
        wrapper: ModelWrapper for the teacher model
        
    Returns:
        Dictionary with 'target_cond_logits' and 'target_uncond_logits'
    """
    # Load image if path provided
    target_image = Image.open(target_image_path).convert("RGB")

    # Process multimodal inputs for conditional and unconditional
    unconditional_prompt = wrapper.get_unconditional_prompt()
    
    inputs = wrapper.process_multimodal_for_training(
        prompt=[target_prompt, unconditional_prompt],
        image=[target_image, target_image],
        task="text_to_image",
        return_tensors="pt",
        padding=True
    )
    
    print("Extracting target logits...")
    torch.set_printoptions(threshold=10_000)

    # Forward pass through model
    # Use prepare_inputs_for_generation if available

    inputs_embeds = inputs.input_ids  # just a placeholder

    if isinstance(wrapper, JanusModelWrapper):

        inputs_embeds = wrapper.vlm.prepare_inputs_embeds(
            input_ids=inputs.input_ids.cuda().to(wrapper.config.device),
            pixel_values=inputs.pixel_values.cuda().to(wrapper.config.device),
            images_seq_mask=inputs.images_seq_mask.cuda().to(wrapper.config.device),
            images_emb_mask=inputs.images_emb_mask.cuda().to(wrapper.config.device)
        ).cuda().to(wrapper.config.device)

        print("INPUTS IDS SHAPE", inputs.input_ids.shape)
        print("inputs_embeds SHAPE", inputs_embeds.shape)

        hidden_states = wrapper.language_model.model(
            inputs_embeds=inputs_embeds.cuda(),
            attention_mask=inputs.attention_mask.to(wrapper.config.device)
        ).last_hidden_state
        logits = wrapper.vlm.gen_head(hidden_states)

    elif hasattr(wrapper.model, 'prepare_inputs_for_generation'):
        model_inputs = wrapper.model.prepare_inputs_for_generation(**inputs)
        logits = wrapper.model(**model_inputs, return_dict=True, output_hidden_states=False).logits
    else:
        # Fallback: direct call
        logits = wrapper.model(
            input_ids=inputs.input_ids.to(wrapper.config.device),
            attention_mask=inputs.attention_mask.to(wrapper.config.device)
        ).logits
    
    # Split conditional and unconditional logits
    cond_logits, uncond_logits = torch.split(logits, 1, dim=0)
    
    # Extract image token logits only (between <boi> and <eoi>)
    cond_boi_position = wrapper.find_beginning_of_image_token_position(inputs.input_ids[0])
    cond_eoi_position = wrapper.find_end_of_image_token_position(inputs.input_ids[0])
    
    uncond_boi_position = wrapper.find_beginning_of_image_token_position(inputs.input_ids[1])
    uncond_eoi_position = wrapper.find_end_of_image_token_position(inputs.input_ids[1])
    
    # Extract image token logits (from <boi> to <eoi>)
    # Logits at boi_position are for predicting first image token
    saved_data = {
        'target_cond_logits': cond_logits[:, cond_boi_position: cond_eoi_position],
        'target_uncond_logits': uncond_logits[:, uncond_boi_position: uncond_eoi_position],
        'inputs_embeds': inputs_embeds
    }
    
    return saved_data


def get_collate_function_for_tokenizer(tokenizer):
    """
    Create collate function for tokenizer.
    """
    def collate_fn(batch):
        # Filter out None samples (invalid samples)
        valid_batch = [x for x in batch if x is not None]
        
        # If no valid samples, return None (will be handled in training loop)
        if len(valid_batch) == 0:
            return None
        
        input_ids = [x["input_ids"] for x in valid_batch]
        labels = [x["labels"] for x in valid_batch]
        
        # Right-pad sequences
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=tokenizer.pad_token_id
        )
        labels = torch.nn.utils.rnn.pad_sequence(
            labels, batch_first=True, padding_value=IGNORE_INDEX
        )
        attention_mask = input_ids.ne(tokenizer.pad_token_id)
        
        return {
            "input_ids": input_ids,
            "labels": labels,
            "attention_mask": attention_mask
        }
    
    return collate_fn


def extend_collate_fn_for_target_logits(collate_fn):
    """
    Extend collate function to handle target logits.
    """
    def extended_collate_fn(batch):
        # Use existing collate function for standard fields
        base_collated = collate_fn(batch)
        
        # If base collate function returned None (due to all invalid samples), pass it through
        if base_collated is None:
            return None
        
        # Filter out None samples for additional processing
        valid_batch = [x for x in batch if x is not None]
        
        if len(valid_batch) == 0:
            return None
        
        if "target_cond_logits" in valid_batch[0]:
            # Handle target conditional and unconditional logits separately
            target_cond_logits = [x["target_cond_logits"] for x in valid_batch]
            base_collated["target_cond_logits"] = torch.stack(target_cond_logits)
        
        if "target_uncond_logits" in valid_batch[0]:
            target_uncond_logits = [x["target_uncond_logits"] for x in valid_batch]
            base_collated["target_uncond_logits"] = torch.stack(target_uncond_logits)

        if "inputs_embeds" in valid_batch[0]:
            target_uncond_logits = [x["inputs_embeds"] for x in valid_batch]
            base_collated["inputs_embeds"] = torch.stack(target_uncond_logits)

        return base_collated
    
    return extended_collate_fn
