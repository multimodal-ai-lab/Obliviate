from obliviate.configs.model_config import ModelType, ModelConfig
from obliviate.models.emu3.emu3_wrapper import Emu3ModelWrapper
from obliviate.models.janus.janus_wrapper import JanusModelWrapper
from obliviate.models.liquid.liquid_wrapper import LiquidModelWrapper

MODEL_WRAPPERS = {
    ModelType.LIQUID: LiquidModelWrapper,
    ModelType.EMU3: Emu3ModelWrapper,
    ModelType.EMU3_GEN: Emu3ModelWrapper,
    ModelType.EMU3_CHAT: Emu3ModelWrapper,
    ModelType.JANUS_PRO: JanusModelWrapper,
}


def create_wrapper_from_config(config: ModelConfig):
    wrapper_cls = MODEL_WRAPPERS.get(config.model_type)

    if config.model_type == ModelType.EMU3_GEN:
        return wrapper_cls(config, emu3_model_name_or_path='BAAI/Emu3-Gen')

    elif config.model_type == ModelType.EMU3_CHAT:
        return wrapper_cls(config, emu3_model_name_or_path='BAAI/Emu3-Chat')

    if wrapper_cls is None:
        raise NotImplementedError(f"Unknown model type: {config.model_type}")

    return wrapper_cls(config)


def resolve_guidance_scale(model_type, guidance_scale=None):
    """Return explicit guidance_scale or the wrapper's inference default for model_type."""
    if guidance_scale is not None:
        return guidance_scale
    if isinstance(model_type, str):
        model_type = ModelType(model_type)
    wrapper_cls = MODEL_WRAPPERS.get(model_type)
    if wrapper_cls is None:
        raise NotImplementedError(f"Unknown model type: {model_type}")
    return wrapper_cls.inference_config_defaults["guidance_scale"]


__all__ = [
    "ModelType",
    "ModelConfig",
    "MODEL_WRAPPERS",
    "create_wrapper_from_config",
    "resolve_guidance_scale",
    "Emu3ModelWrapper",
    "JanusModelWrapper",
    "LiquidModelWrapper",
]
