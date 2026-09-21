"""
Model and tokenizer loading logic for Hugging Face GPT-2.
Includes device management (CUDA/CPU) and model evaluation mode setup.
"""

from typing import Tuple
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel, PreTrainedTokenizer


def get_device() -> torch.device:
    """
    Automatically selects an appropriate device (CUDA GPU if available, otherwise CPU).

    Returns:
        torch.device: CUDA or CPU device.
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def load_model_and_tokenizer(
    model_name: str = "gpt2"
) -> Tuple[PreTrainedModel, PreTrainedTokenizer, torch.device]:
    """
    Loads pretrained GPT-2 model and tokenizer from Hugging Face Hub,
    configures automatic device selection, and sets model to evaluation mode.

    =============================================================================
    EXPLANATION OF KEY CONCEPTS:
    =============================================================================
    1. Tokenization:
       - Process of breaking down raw text strings into discrete subword units ("tokens").
       - GPT-2 uses Byte-Pair Encoding (BPE) tokenization with a vocabulary of 50,257 tokens.

    2. input_ids:
       - A PyTorch tensor of shape [batch_size, sequence_length] containing numerical
         integer IDs corresponding to each token in the tokenizer's vocabulary.

    3. attention_mask:
       - A binary PyTorch tensor [batch_size, sequence_length] of 1s and 0s indicating
         which tokens the model's self-attention mechanism should process (1 for real tokens,
         0 for padded tokens).

    4. logits:
       - Raw, unnormalized prediction scores output by the model's final linear head layer
         for every position in the sequence across the entire vocabulary.
       - Shape: [batch_size, sequence_length, vocab_size].

    5. Final Sequence Position (logits[:, -1, :]):
       - Autoregressive models like GPT-2 generate text one token at a time.
       - The prediction for the *next* token depends on the accumulated context up to
         the last token position in the input prompt. Therefore, we extract the final position
         logits [batch_size, vocab_size] to sample the next token.
    =============================================================================

    Args:
        model_name: Pretrained Hugging Face model identifier (default: "gpt2").

    Returns:
        Tuple containing (model, tokenizer, device).
    """
    device = get_device()

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Ensure pad token is set for GPT-2 (default GPT2Tokenizer has no pad token)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load Causal LM model
    model = AutoModelForCausalLM.from_pretrained(model_name)
    model.to(device)

    # Set model to evaluation mode (disables dropout, batchnorm updates, etc.)
    model.eval()

    return model, tokenizer, device
