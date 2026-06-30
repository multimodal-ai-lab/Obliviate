import torch
import torchvision
import wandb


def format_elapsed_seconds(seconds: float) -> str:
    """Format elapsed seconds as human-readable string."""
    total = int(seconds)
    hours, remainder = divmod(total, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours > 0:
        return f"{hours}h {minutes}m {secs}s"
    if minutes > 0:
        return f"{minutes}m {secs}s"
    return f"{seconds:.1f}s"


def log_generated_images_to_wandb(generated_images, log_id, n_cols, step):
    """
    Log generated images to wandb as a grid.
    
    Args:
        generated_images: List of PIL Images or torch Tensors
        log_id: Wandb log identifier
        n_cols: Number of columns in the grid
        step: Training step number
    """
    if generated_images is not None:
        # Convert to tensors and create grid
        if isinstance(generated_images[0], torch.Tensor):
            imgs_tensor = torch.stack([
                img if img.dim() == 3 else img.permute(2, 0, 1)
                for img in generated_images
            ])
        else:
            pil2tensor = torchvision.transforms.ToTensor()
            imgs_tensor = torch.stack([pil2tensor(img) for img in generated_images])

        grid = torchvision.utils.make_grid(
            imgs_tensor, nrow=n_cols if n_cols is not None else len(generated_images), padding=2, normalize=True
        )
        wandb.log({log_id: wandb.Image(grid, caption=f"step_{step}")}, step=step)

