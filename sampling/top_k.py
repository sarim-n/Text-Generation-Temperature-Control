"""
Top-K logit filtering module for truncating the candidate token pool to the top K highest logits.
"""

import torch


def apply_top_k(logits: torch.Tensor, top_k: int) -> torch.Tensor:
    """
    Applies Top-K filtering to logits by keeping only the top K highest values
    and setting all other values to negative infinity (-inf).

    =============================================================================
    MATHEMATICAL & ALGORITHMIC CONCEPT:
    =============================================================================
    Top-K truncation creates a hard cutoff based on token rank:
    1. Identify the logit threshold corresponding to the K-th largest logit.
    2. Any token with a logit strictly below this threshold is assigned -inf.
    3. When softmax is applied later, exp(-inf) = 0, giving suppressed tokens exactly
       zero probability.

    Properties:
    - K is a fixed integer threshold.
    - Preserves the original shape and index ordering of the input tensor.
    - Safe against K >= vocabulary_size (returns logits unchanged).
    =============================================================================

    Args:
        logits: PyTorch tensor of logit values [..., vocab_size].
        top_k: Number of highest-probability tokens to retain (must be > 0).

    Returns:
        torch.Tensor: Filtered logits matching the original shape and token order.

    Raises:
        ValueError: If top_k <= 0.
    """
    if top_k <= 0:
        raise ValueError(f"top_k must be greater than 0. Got top_k={top_k}")

    vocab_size = logits.size(-1)
    if top_k >= vocab_size:
        return logits.clone()

    # Make a copy to avoid mutating the original input tensor
    filtered_logits = logits.clone()

    # Find the top K logit values along the last dimension
    # topk_values has shape [..., k]
    topk_values, _ = torch.topk(filtered_logits, top_k, dim=-1)

    # The K-th largest value is at index -1 along the topk dimension
    min_topk_value = topk_values[..., -1:]

    # Mask all logits smaller than the K-th largest logit with -inf
    mask = filtered_logits < min_topk_value
    filtered_logits[mask] = -float('inf')

    return filtered_logits
