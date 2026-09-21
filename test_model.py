"""
Stage 2 Test Script: GPT-2 Model Loading & Forward Pass Verification.

Verifies:
1. Tokenization and token ID mapping.
2. Device placement (CUDA or CPU).
3. GPT-2 model forward pass under torch.no_grad().
4. Output logits tensor shapes (full sequence logits and final-position logits).
"""

import sys
import torch
from generator.model import load_model_and_tokenizer

# Ensure UTF-8 encoding for stdout printing on Windows
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')


def run_stage2_demo(prompt: str = "The future of artificial intelligence") -> None:
    print("=" * 65)
    print("STAGE 2: GPT-2 MODEL & TOKENIZER INFERENCE VERIFICATION")
    print("=" * 65)

    # 1. Load model, tokenizer, and device
    print("\n1. Loading Model and Tokenizer...")
    model, tokenizer, device = load_model_and_tokenizer("gpt2")
    print(f"   [SUCCESS] Model loaded on device: '{device}'")
    print(f"   [SUCCESS] Model mode: eval (training={model.training})")

    # 2. Tokenize prompt
    print("\n2. Tokenization Step...")
    print(f"   Original Prompt: \"{prompt}\"")

    # Encode prompt to input_ids and attention_mask
    inputs = tokenizer(prompt, return_tensors="pt")
    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs["attention_mask"].to(device)

    token_id_list = input_ids[0].tolist()
    num_tokens = len(token_id_list)

    print(f"   Token IDs: {token_id_list}")
    print(f"   Number of Input Tokens: {num_tokens}")

    # Decoded tokens (individual subword tokens)
    decoded_tokens = [tokenizer.decode([tid]) for tid in token_id_list]
    print(f"   Decoded Tokens: {decoded_tokens}")
    print(f"   Reconstructed Text: \"{tokenizer.decode(input_ids[0])}\"")

    # 3. Model Forward Pass (Inference / no_grad)
    print("\n3. Forward Pass Through GPT-2...")
    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)

    logits = outputs.logits
    vocab_size = tokenizer.vocab_size

    print(f"   Full Model Logits Shape [batch, seq_len, vocab_size]: {list(logits.shape)}")
    print(f"   Vocabulary Size: {vocab_size}")

    # 4. Extract Final Sequence Position Logits
    final_position_logits = logits[:, -1, :]
    print(f"   Final-Position Logits Shape [batch, vocab_size]: {list(final_position_logits.shape)}")

    print("\n" + "=" * 65)
    print("STAGE 2 VERIFICATION COMPLETE: GPT-2 inference operational.")
    print("=" * 65)


if __name__ == "__main__":
    run_stage2_demo()
