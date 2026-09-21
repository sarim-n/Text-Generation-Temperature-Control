"""
Temperature scaling module for controlling the entropy of logit distributions.
"""

import torch


def apply_temperature(logits: torch.Tensor, temperature: float) -> torch.Tensor:
    """
    Applies temperature scaling to raw prediction logits.

    =============================================================================
    MATHEMATICAL FORMULA & CONCEPTS:
    =============================================================================
    scaled_logits = logits / temperature

    - Temperature (T) controls the "sharpness" of the resulting probability distribution.
    - Temperature does NOT alter the model's underlying knowledge or ranking of logits.
    - T < 1.0 (Low Temperature): Exaggerates differences between logits, making the top
      tokens significantly more probable (sharper distribution -> more deterministic/greedy).
    - T = 1.0: Leaves logits unmodified (standard model output).
    - T > 1.0 (High Temperature): Compresses logit differences, bringing token probabilities
      closer together (flatter distribution -> higher entropy / more random/creative).
    =============================================================================

    Args:
        logits: PyTorch tensor of raw logit values.
        temperature: Temperature scaling factor (must be > 0).

    Returns:
        torch.Tensor: Temperature-scaled logit tensor of the same shape.

    Raises:
        ValueError: If temperature <= 0.
    """
    if temperature <= 0:
        raise ValueError(f"Temperature must be greater than 0. Got temperature={temperature}")

    scaled_logits = logits / temperature
    return scaled_logits
