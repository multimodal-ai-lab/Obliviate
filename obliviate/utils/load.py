import os
import pandas as pd
from typing import Optional

import torch

from obliviate.configs.project_config import ProjectConfig
from obliviate.unlearning.target_config import Target, get_prompt_file_for_concept


def load_prompts_from_csv(file_path: str, n_prompts: int = None):
    """
    Load up to n_prompts rows from the simplified CSV.
    Returns a list of prompt strings.
    """
    df = pd.read_csv(file_path)

    if n_prompts is None:
        n_prompts = len(df)

    n_prompts = min(n_prompts, len(df))
    sampled = df.iloc[:n_prompts]

    return sampled["prompt"].tolist()


def resolve_prompts_file(prompts_file: Optional[str] = None, target: Optional[Target] = None):
    """
    Resolve prompts_file based on optional target.
    """
    if prompts_file:
        return prompts_file

    if target is None:
        base = ProjectConfig().data_folder
        return os.path.join(base, "mjhq_10k", "mjhq_10k.csv")

    return get_prompt_file_for_concept(target)


def checkpoint_loader(model, checkpoint_path: str, inner_name: str = "model"):
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    inner = getattr(model, inner_name)
    applied = 0
    for name, state in ckpt.items():
        try:
            sub = inner.get_submodule(name)
            sub.load_state_dict(state, strict=True)
            applied += 1
        except Exception as e:
            print(f"  Skip {name}: {e}")
    print(f"✅ Loaded checkpoint: {applied} modules from {checkpoint_path}")
    return model
