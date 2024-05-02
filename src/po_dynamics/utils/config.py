import warnings

import torch.cuda
from omegaconf import DictConfig, OmegaConf

from po_dynamics import utils


def register_resolvers():
    if not OmegaConf.has_resolver("eval"):
        OmegaConf.register_new_resolver("eval", eval, use_cache=True)
    if not OmegaConf.has_resolver("generate_random_seed"):
        OmegaConf.register_new_resolver("generate_random_seed", utils.seeding.generate_random_seed, use_cache=True)
    if not OmegaConf.has_resolver("resolve_device"):
        OmegaConf.register_new_resolver("resolve_device", utils.config.resolve_device, use_cache=True)


def resolve_device(target_name: str, device: str):
    if device.startswith("cuda") and not torch.cuda.is_available():
        warnings.warn(
            f"(Ignore if overridden) {target_name} set to CUDA, but CUDA is not available. Defaulting to CPU."
        )
        return "cpu"
    elif device == "mps" and not torch.backends.mps.is_available():
        warnings.warn(f"(Ignore if overridden) {target_name} set to MPS, but MPS is not available. Defaulting to CPU.")
        return "cpu"
    else:
        return device


def anonymize_config(config: DictConfig):
    if hasattr(config, "items"):
        anonymized_config = {}
        for key, value in config.items():
            anonymized_config[key] = anonymize_config(value)
        return anonymized_config
    elif isinstance(config, list):
        return [anonymize_config(value) for value in config]
    elif isinstance(config, str):
        return (
            config.replace("moalla", "anonymous")
            .replace("skander", "anonymous")
            .replace("claire", "anonymous")
            .replace("epfl", "anonymous")
        )
    else:
        return config
