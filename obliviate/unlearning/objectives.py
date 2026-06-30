import torch
import torch.nn as nn

from obliviate.models import JanusModelWrapper
from obliviate.unlearning.config import UnlearningConfig
from obliviate.models.model_wrapper import ModelWrapper
from obliviate.data.data_utils import IGNORE_INDEX


def move_padding_to_start(wrapper: ModelWrapper, uncond_ids: torch.Tensor) -> torch.Tensor:
    """
    Move padding from end to before <boi> token to align with conditional sequence.
    
    Args:
        wrapper: ModelWrapper instance
        uncond_ids: Unconditional input IDs tensor
        
    Returns:
        Aligned unconditional input IDs with padding moved to start
    """
    pad_token_id = wrapper.right_padded_tokenizer.pad_token_id

    if isinstance(wrapper, JanusModelWrapper):
        pad_token_id = wrapper.training_processor.pad_id
    
    # Count total padding
    num_pads = (uncond_ids == pad_token_id).sum().item()
    
    # Remove existing padding
    non_pad = uncond_ids[uncond_ids != pad_token_id]
    
    # Prepend same number of pads
    pads = torch.full((num_pads,), pad_token_id, dtype=uncond_ids.dtype, device=uncond_ids.device)
    new_uncond = torch.cat([pads, non_pad])
    
    # Trim (or pad) to original length, just in case
    new_uncond = new_uncond[-len(uncond_ids):]
    
    return new_uncond


class KLObjective:
    """
    KL divergence objective with negation formula for concept erasure.
    
    The objective trains the student model to match negated harmful logits:
    - neg_teacher_cond_logits = uncond - eta * (cond - uncond)
    - neg_teacher_uncond_logits = uncond
    
    This pushes the model away from generating harmful content.
    """
    
    def __init__(self):
        self.kl_loss = nn.KLDivLoss(reduction="batchmean")
    
    def __call__(self, model, batch, config: UnlearningConfig, wrapper: ModelWrapper):
        """
        Compute KL divergence loss with negation.
        
        Args:
            model: Student model being trained
            batch: Batch containing input_ids, labels, target_cond_logits, target_uncond_logits
            config: UnlearningConfig with training parameters
            wrapper: ModelWrapper for the student model
            
        Returns:
            loss: Scalar loss tensor
            metrics: Dictionary of metrics
            outputs: Model outputs
        """
        input_ids = batch["input_ids"].to(config.device)
        attention_mask = batch["attention_mask"].to(config.device)
        labels = batch["labels"].to(config.device)

        inputs_embeds = batch['inputs_embeds'].to(config.device)
        print("INPUT_IDs (SHAPE, objective):", input_ids.shape)
        print("INPUT_EMBEDS (SHAPE, objective):", inputs_embeds.shape)
        print("ATTENTION_MASK (SHAPE, objective):", attention_mask.shape)
        
        # Handle conditional/unconditional split
        if len(input_ids.shape) == 3:
            # input_ids shape: (batch_size, 2, seq_len) - includes both cond and uncond
            includes_uncond = True
            # Squeeze batch dim: (1, 2, SEQ_LEN) -> (2, SEQ_LEN)
            input_ids = input_ids.squeeze(dim=0)
            
            # Important! Align unconditional padding to match conditional sequence
            # The shorter unconditional sequence got padded at the end, but we need
            # padding before <boi> to align image token parts
            aligned_uncond_ids = move_padding_to_start(wrapper, input_ids[1])
            input_ids = torch.stack([input_ids[0], aligned_uncond_ids])
        else:
            includes_uncond = False
            print("No unconditional sequences found in batch. Falling back to conditional-only loss computation.")

        # Forward pass through student model
        if not isinstance(wrapper, JanusModelWrapper):
            # Re-create attention mask after alignment
            attention_mask = input_ids.ne(wrapper.right_padded_tokenizer.pad_token_id)
            logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
        else:
            #input_ids[1, 0] = 100000 # make sure the unconditional one also starts with 100000
            print("Input IDS (Objective)", input_ids)

            # Re-create attention mask after alignment
            #attention_mask = input_ids.ne(wrapper.training_processor.pad_id)
            print("Attention Mask (Objective)", attention_mask)

            hidden_states = wrapper.language_model.model(
                inputs_embeds=inputs_embeds.squeeze(dim=0).to(wrapper.config.device),
                attention_mask=attention_mask.squeeze(dim=0).to(wrapper.config.device)
            ).last_hidden_state

            logits = wrapper.vlm.gen_head(hidden_states)
        
        # Split student logits into conditional and unconditional
        if includes_uncond:
            student_cond_logits, student_uncond_logits = torch.split(logits, 1, dim=0)
        else:
            student_cond_logits, student_uncond_logits = logits, None
        
        # Get teacher logits (harmful model predictions)
        harmful_cond_logits = batch["target_cond_logits"].to(config.device)
        
        if includes_uncond:
            harmful_uncond_logits = batch["target_uncond_logits"].to(config.device)
        else:
            harmful_uncond_logits = None
        
        # Get valid mask (ignore padding and instruction tokens)
        # Use labels from conditional sequence for valid mask
        valid_mask = (labels != IGNORE_INDEX)
        if not valid_mask.any():
            raise ValueError("The valid_mask was empty!")
        
        # Apply negation formula to teacher logits
        eta = config.eta  # Negative strength parameter
        
        # Negation formula: uncond - eta * (cond - uncond) = (1 + eta) * uncond - eta * cond
        # This pushes away from the harmful conditional prediction
        if includes_uncond:
            neg_teacher_cond_logits = harmful_uncond_logits[valid_mask] - eta * (
                harmful_cond_logits[valid_mask] - harmful_uncond_logits[valid_mask]
            )
            neg_teacher_uncond_logits = harmful_uncond_logits[valid_mask]  # Neutral baseline
        else:
            # Fallback if no unconditional
            neg_teacher_cond_logits = harmful_cond_logits[valid_mask]
            neg_teacher_uncond_logits = harmful_cond_logits[valid_mask]
        
        # Extract valid student logits
        valid_student_cond_logits = student_cond_logits[valid_mask]
        
        # Compute (conditional) KL loss
        student_cond_log_probs = torch.log_softmax(valid_student_cond_logits, dim=-1)
        teacher_cond_probs = torch.softmax(neg_teacher_cond_logits, dim=-1)
        loss_cond = self.kl_loss(student_cond_log_probs, teacher_cond_probs)

        metrics = {
            "kl_loss_cond": loss_cond.item(),
        }

        return loss_cond, metrics