"""
Top-K filtering module for logit truncation.
"""

import torch


def apply_top_k(logits: torch.Tensor, top_k: int) -> torch.Tensor:
    """
    Applies Top-K filtering to logits by masking out all tokens below the top K.

    Args:
        logits: Tensor of raw unnormalized logits [vocab_size].
        top_k: Number of highest probability tokens to keep (K >= 1).

    Returns:
        Filtered logits with non-top-K entries set to negative infinity.
    """
    # Placeholder for sampling implementation stage
    raise NotImplementedError("Top-K sampling will be implemented in the sampling phase.")
