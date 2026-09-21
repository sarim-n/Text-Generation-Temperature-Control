"""
Utility functions for seed management, formatting, and helper utilities.
"""

import random
import numpy as np
import torch


def set_seed(seed: int = 42) -> None:
    """
    Sets random seeds for reproducibility across random, numpy, and torch.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
