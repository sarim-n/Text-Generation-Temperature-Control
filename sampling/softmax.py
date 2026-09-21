"""
Softmax conversion module for transforming raw model logits into a valid probability distribution.
"""

import torch


def logits_to_probs(logits: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """
    Converts raw, unnormalized model logits into a valid probability distribution using PyTorch softmax.

    =============================================================================
    WHY LOGITS ARE NOT PROBABILITIES:
    =============================================================================
    Logits are the unnormalized raw outputs from the final linear layer (LM Head) of 
    a transformer model. They range from negative infinity (-inf) to positive infinity (+inf).
    Because logits are unnormalized and can be negative, they cannot be interpreted 
    directly as probabilities.
    
    Softmax exponentiates each logit (e^{z_i}), ensuring all values are strictly non-negative,
    and divides by the sum of exponentiated logits (sum(e^{z_j})), ensuring that the resulting
    values sum to exactly 1.0.
    =============================================================================

    Args:
        logits: PyTorch tensor of raw logit values.
        dim: Dimension along which softmax is computed (default: -1).

    Returns:
        torch.Tensor: Valid probability distribution tensor matching the input shape,
                      where all elements are non-negative and sum to 1.0 along `dim`.
    """
    # PyTorch torch.softmax subtracts the max logit internally for numerical stability:
    # softmax(z_i) = exp(z_i - max(z)) / sum(exp(z_j - max(z)))
    probs = torch.softmax(logits, dim=dim)
    return probs
