"""
Sampling module providing explicit PyTorch implementations of:
- Softmax probability conversion
- Categorical random sampling
- Temperature scaling
- Top-K logit filtering
- Top-P (nucleus) logit filtering
- Combined sampling pipeline (sample_next_token)
"""

from sampling.softmax import logits_to_probs
from sampling.temperature import apply_temperature
from sampling.top_k import apply_top_k
from sampling.top_p import apply_top_p
from sampling.sampler import sample_categorical, sample_next_token

__all__ = [
    "logits_to_probs",
    "apply_temperature",
    "apply_top_k",
    "apply_top_p",
    "sample_categorical",
    "sample_next_token",
]
