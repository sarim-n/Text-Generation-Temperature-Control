"""
Categorical sampling primitives and full sampling pipeline orchestration.
"""

from typing import Optional, Tuple
import torch
from sampling.softmax import logits_to_probs
from sampling.temperature import apply_temperature
from sampling.top_k import apply_top_k
from sampling.top_p import apply_top_p


def sample_categorical(
    probs: torch.Tensor,
    generator: Optional[torch.Generator] = None
) -> int:
    """
    Samples a single token index from a 1D probability distribution using PyTorch multinomial sampling.

    =============================================================================
    ARGMAX VS. CATEGORICAL SAMPLING:
    =============================================================================
    - Argmax (Greedy Decoding):
      Always selects token_id = argmax(P). It is 100% deterministic.
      Disadvantage: Often leads to repetitive, boring, or looping text generation.

    - Categorical Sampling (torch.multinomial):
      Randomly samples a token according to the probability vector P.
      A token with probability 0.60 has a 60% chance of being selected, while a token
      with probability 0.05 has a 5% chance.
      Advantage: Introduces diversity, natural variation, and human-like creativity.
    =============================================================================

    Args:
        probs: 1D PyTorch tensor of non-negative probabilities summing to ~1.0.
        generator: Optional PyTorch random number generator for seeded reproducibility.

    Returns:
        int: Selected token ID index.
    """
    # Ensure input is a 1D tensor
    if probs.dim() > 1:
        probs = probs.squeeze(0)

    # torch.multinomial performs random categorical sampling
    sampled_index = torch.multinomial(probs, num_samples=1, generator=generator)
    return int(sampled_index.item())


def sample_next_token(
    logits: torch.Tensor,
    temperature: float = 1.0,
    top_k: int = 0,
    top_p: float = 1.0,
    seed: Optional[int] = None,
    generator: Optional[torch.Generator] = None
) -> Tuple[int, torch.Tensor, torch.Tensor]:
    """
    Complete sampling pipeline converting raw GPT-2 final-position logits into a selected next token ID.

    Pipeline Steps:
    GPT-2 final-position logits
        ↓
    Temperature scaling (apply_temperature)
        ↓
    Top-K filtering (apply_top_k)
        ↓
    Top-P filtering (apply_top_p)
        ↓
    Softmax probability conversion (logits_to_probs)
        ↓
    Categorical sampling (sample_categorical)
        ↓
    Selected token ID

    Args:
        logits: Raw logit tensor [vocab_size] or [1, vocab_size].
        temperature: Temperature scaling factor (default: 1.0).
        top_k: Top-K candidate pool size (default: 0 = disabled).
        top_p: Top-P nucleus threshold (default: 1.0 = disabled).
        seed: Optional integer random seed for single-step reproducible sampling.
        generator: Optional PyTorch torch.Generator instance for multi-step autoregressive generation.

    Returns:
        Tuple containing:
        - selected_token_id (int): Sampled token index.
        - final_probs (torch.Tensor): Calculated probability distribution tensor [vocab_size].
        - processed_logits (torch.Tensor): Processed logit tensor [vocab_size].
    """
    # Ensure 1D tensor shape [vocab_size]
    if logits.dim() > 1:
        logits = logits.squeeze(0)

    processed_logits = logits.clone()

    # Step 1: Temperature Scaling
    if temperature != 1.0:
        processed_logits = apply_temperature(processed_logits, temperature)

    # Step 2: Top-K Filtering
    if top_k > 0:
        processed_logits = apply_top_k(processed_logits, top_k)

    # Step 3: Top-P Nucleus Filtering
    if top_p < 1.0:
        processed_logits = apply_top_p(processed_logits, top_p)

    # Step 4: Softmax Probability Conversion
    final_probs = logits_to_probs(processed_logits)

    # Step 5: Optional Seed/Generator Setup
    # Priority: if an active generator is passed, use it; otherwise if seed is passed, create a new generator.
    if generator is None and seed is not None:
        generator = torch.Generator(device=processed_logits.device)
        generator.manual_seed(seed)

    # Step 6: Categorical Sampling
    selected_token_id = sample_categorical(final_probs, generator=generator)

    return selected_token_id, final_probs, processed_logits
