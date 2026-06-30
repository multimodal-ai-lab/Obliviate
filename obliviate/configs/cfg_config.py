from enum import Enum
from typing import Optional


class CFGMode(Enum):
    NONE = "none" # normal CFG
    NEGATIVE_PROMPT = "negative_prompt"  # Standard CFG with target prompt as unconditional
