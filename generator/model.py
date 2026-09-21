"""
Model and tokenizer loading logic for Hugging Face GPT-2.
"""

from typing import Tuple
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer


def load_model_and_tokenizer(model_name: str = "gpt2") -> Tuple[GPT2LMHeadModel, GPT2Tokenizer]:
    """
    Loads pretrained GPT-2 model and tokenizer from Hugging Face Hub.

    Args:
        model_name: Pretrained Hugging Face model identifier (default: "gpt2").

    Returns:
        Tuple containing (model, tokenizer).
    """
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    model = GPT2LMHeadModel.from_pretrained(model_name)
    model.eval()
    return model, tokenizer
