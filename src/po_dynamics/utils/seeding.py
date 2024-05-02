import os
import random

import numpy as np
import torch
from omegaconf import DictConfig


def generate_random_seed():
    """Generate a random seed."""
    return random.randint(0, 2**32 - 1)


# Update this function whenever you have a library that needs to be seeded.
def seed_everything(config: DictConfig):
    """Seed all random generators.
    This is not strict reproducibility, but it should be enough for most cases.
    https://pytorch.org/docs/stable/notes/randomness.html
    """
    random.seed(config.seed)

    # This is for legacy numpy.
    # https://numpy.org/doc/stable/reference/random/generated/numpy.random.seed.html
    np.random.seed(config.seed)
    # New code should make a Generator out of the config.seed directly.

    torch.manual_seed(config.seed)
    torch.backends.cudnn.benchmark = False
    # torch.use_deterministic_algorithms(True)
    # lighter version of the above as not all algorithms have a deterministic implementation
    torch.backends.cudnn.deterministic = True
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
