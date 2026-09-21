"""
Top-P (Nucleus) filtering module for dynamically truncating candidate tokens based on cumulative probability.
"""

import torch


def apply_top_p(logits: torch.Tensor, top_p: float) -> torch.Tensor:
    """
    Applies Top-P (nucleus) filtering to logits by retaining the smallest set of top-ranked
    tokens whose cumulative probability reaches or exceeds the threshold `top_p`. All other
    logits are set to negative infinity (-inf).

    =============================================================================
    MATHEMATICAL & ALGORITHMIC CONCEPT:
    =============================================================================
    Unlike Top-K (which keeps a fixed number of tokens), Top-P dynamically adjusts the size
    of the candidate set based on model confidence:
    
    1. Convert logits to probabilities and sort in descending order.
    2. Compute the cumulative sum of probabilities: CumSum(P).
    3. Identify the cutoff index where CumSum(P) >= top_p.
    4. Retain all tokens up to AND INCLUDING the token that causes CumSum(P) to reach/exceed top_p.
    5. Set all remaining tokens outside the nucleus to -inf.
    6. Scatter the mask back to restore the ORIGINAL vocabulary index order.
    =============================================================================

    Args:
        logits: PyTorch tensor of logit values [..., vocab_size].
        top_p: Cumulative probability threshold (0.0 < top_p <= 1.0).

    Returns:
        torch.Tensor: Filtered logits matching original shape and token order.

    Raises:
        ValueError: If top_p <= 0 or top_p > 1.0.
    """
    if top_p <= 0 or top_p > 1.0:
        raise ValueError(f"top_p must be in range (0.0, 1.0]. Got top_p={top_p}")

    if top_p == 1.0:
        return logits.clone()

    filtered_logits = logits.clone()

    # Sort logits in descending order along the vocabulary dimension
    sorted_logits, sorted_indices = torch.sort(filtered_logits, descending=True, dim=-1)

    # Compute softmax probabilities on sorted logits
    sorted_probs = torch.softmax(sorted_logits, dim=-1)

    # Compute cumulative probability distribution
    cumulative_probs = torch.cumsum(sorted_probs, dim=-1)

    # Identify tokens to remove: cumulative_probs > top_p
    sorted_indices_to_remove = cumulative_probs > top_p

    # CRITICAL BOUNDARY CONDITION:
    # Shift mask right by 1 to include the first token that pushes cumulative_probs >= top_p
    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
    sorted_indices_to_remove[..., 0] = False

    # Scatter mask back to the original unsorted vocabulary indices
    indices_to_remove = sorted_indices_to_remove.scatter(
        dim=-1, index=sorted_indices, src=sorted_indices_to_remove
    )

    # Apply -inf mask to suppressed tokens
    filtered_logits[indices_to_remove] = -float('inf')

    return filtered_logits
