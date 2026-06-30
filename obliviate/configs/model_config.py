from dataclasses import dataclass
from enum import Enum
from typing import Optional, Any


@dataclass
class ModelConfig:
    model_type: Any = 'liquid'
    checkpoint: Optional[str] = None
    device: str = 'cuda'
    image_size: int = 512

    def __post_init__(self):
        if isinstance(self.model_type, str):
            try:
                self.model_type = ModelType(self.model_type)
            except ValueError:
                raise ValueError(f"Invalid model type: {self.model_type}. "
                                 f"Valid model types: {[m.value for m in ModelType]}")

            if self.model_type == ModelType.JANUS_PRO:
                print("Reducing image resolution to 384 for Janus-Pro 7B model!")
                self.image_size = 384


class ModelType(Enum):
    LIQUID = "liquid"
    EMU3 = "emu3"
    EMU3_GEN = "emu3_gen"
    EMU3_CHAT = "emu3_chat"
    JANUS_PRO = "janus"

