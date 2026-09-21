"""
Generator module for model loading and explicit autoregressive text generation.
"""

from generator.model import load_model_and_tokenizer, get_device
from generator.generation import generate_text, generate_text_detailed, GenerationResult

__all__ = [
    "load_model_and_tokenizer",
    "get_device",
    "generate_text",
    "generate_text_detailed",
    "GenerationResult",
]
