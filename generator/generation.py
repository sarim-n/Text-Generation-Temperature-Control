"""
Autoregressive text generation module using explicit PyTorch sampling.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any, Tuple
import torch
from transformers import PreTrainedModel, PreTrainedTokenizer

from generator.model import load_model_and_tokenizer
from sampling.sampler import sample_next_token


@dataclass
class GenerationResult:
    """
    Structured container holding the outputs and metadata of an autoregressive generation run.
    """
    prompt: str
    generated_text: str
    full_text: str
    prompt_token_ids: List[int]
    generated_token_ids: List[int]
    stop_reason: str  # "max_new_tokens", "eos_token", or "context_limit"
    step_traces: List[Dict[str, Any]] = field(default_factory=list)


def generate_text_detailed(
    prompt: str,
    model: Optional[PreTrainedModel] = None,
    tokenizer: Optional[PreTrainedTokenizer] = None,
    max_new_tokens: int = 50,
    temperature: float = 1.0,
    top_k: int = 0,
    top_p: float = 1.0,
    seed: Optional[int] = None,
    verbose: bool = False,
) -> GenerationResult:
    """
    Executes explicit, step-by-step autoregressive text generation using GPT-2 and custom PyTorch sampling.

    =============================================================================
    AUTOREGRESSIVE GENERATION PIPELINE:
    =============================================================================
    1. Prompt Tokenization: Prompt string -> input_ids tensor [1, sequence_length].
    2. Single Generator Initialization: If seed is provided, a torch.Generator is initialized ONCE
       before the loop to maintain continuous random state across all token generation steps.
    3. Autoregressive Loop (for step in range(max_new_tokens)):
       a. Forward Pass: Pass input_ids to model under torch.inference_mode().
       b. Slice Logits: Extract final-position logits: logits[:, -1, :] of shape [1, 50257].
       c. Sampling Engine: Apply temperature scaling, top-k filtering, top-p filtering,
          softmax, and categorical sampling to select next_token_id.
       d. Early Stopping Check: If next_token_id == eos_token_id, terminate loop.
       e. Append Token: Concatenate next_token_id to input_ids tensor along sequence dim.
       f. Repeat: Feed updated input_ids back into GPT-2 for the next step.
    4. Decode: Decode generated_token_ids back to text string using tokenizer.
    =============================================================================

    Args:
        prompt: Initial prompt string.
        model: Pre-loaded GPT-2 LM model (if None, will be loaded dynamically).
        tokenizer: Pre-loaded GPT-2 tokenizer (if None, will be loaded dynamically).
        max_new_tokens: Maximum number of tokens to generate (default: 50).
        temperature: Temperature scaling factor (default: 1.0).
        top_k: Top-K rank cutoff parameter (default: 0 = disabled).
        top_p: Top-P nucleus threshold parameter (default: 1.0 = disabled).
        seed: Optional random seed for reproducible multi-step generation.
        verbose: If True, prints step-by-step generation traces to stdout.

    Returns:
        GenerationResult dataclass instance with generated text, token IDs, and traces.
    """
    if model is None or tokenizer is None:
        model, tokenizer, _ = load_model_and_tokenizer("gpt2")

    device = next(model.parameters()).device
    eos_token_id = tokenizer.eos_token_id

    # Maximum context length supported by GPT-2 architecture
    max_context_length = getattr(model.config, "n_positions", 1024)

    # Step 1: Tokenize prompt
    prompt_inputs = tokenizer(prompt, return_tensors="pt")
    input_ids = prompt_inputs["input_ids"].to(device)
    prompt_token_ids = input_ids[0].tolist()

    generated_token_ids: List[int] = []
    step_traces: List[Dict[str, Any]] = []
    stop_reason = "max_new_tokens"

    # Step 2: Initialize RNG ONCE for the entire generation run to avoid seed-resetting per token
    generator = None
    if seed is not None:
        generator = torch.Generator(device=device)
        generator.manual_seed(seed)

    if verbose:
        print(f"\n--- Starting Autoregressive Generation ---")
        print(f"Prompt: \"{prompt}\" ({len(prompt_token_ids)} tokens)")
        print(f"Hyperparameters: Temp={temperature}, Top-K={top_k}, Top-P={top_p}, Seed={seed}")

    # Step 3: Autoregressive Loop
    model.eval()
    with torch.inference_mode():
        for step in range(max_new_tokens):
            # Check maximum context length safety
            if input_ids.size(1) >= max_context_length:
                stop_reason = "context_limit"
                if verbose:
                    print(f"⚠️ Reached max context length limit ({max_context_length} tokens). Stopping.")
                break

            # Forward pass through GPT-2
            outputs = model(input_ids=input_ids)
            
            # Extract final sequence position logits [1, vocab_size]
            next_token_logits = outputs.logits[:, -1, :]

            # Sample next token using custom PyTorch sampling pipeline
            next_token_id, probs, _ = sample_next_token(
                logits=next_token_logits,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                generator=generator
            )

            # Trace metadata if verbose
            token_str = tokenizer.decode([next_token_id])
            candidate_count = int(torch.sum(probs > 0).item())
            max_p = float(torch.max(probs).item())

            trace_entry = {
                "step": step + 1,
                "token_id": next_token_id,
                "token_str": token_str,
                "candidate_count": candidate_count,
                "max_prob": max_p
            }
            step_traces.append(trace_entry)

            if verbose:
                print(f"Step {step + 1:2d} | Context len: {input_ids.size(1):4d} | "
                      f"Selected: '{token_str}' (ID {next_token_id:5d}) | "
                      f"Candidates: {candidate_count:4d} | Max P: {max_p*100:5.2f}%")

            # Check EOS stopping condition
            if next_token_id == eos_token_id:
                stop_reason = "eos_token"
                if verbose:
                    print(f"🛑 Encountered EOS token (ID {eos_token_id}). Stopping generation.")
                break

            # Append sampled token to sequence and update generated_token_ids
            generated_token_ids.append(next_token_id)
            next_token_tensor = torch.tensor([[next_token_id]], device=device)
            input_ids = torch.cat([input_ids, next_token_tensor], dim=-1)

    # Step 4: Decode generated tokens
    generated_text = tokenizer.decode(generated_token_ids)
    full_text = prompt + generated_text

    return GenerationResult(
        prompt=prompt,
        generated_text=generated_text,
        full_text=full_text,
        prompt_token_ids=prompt_token_ids,
        generated_token_ids=generated_token_ids,
        stop_reason=stop_reason,
        step_traces=step_traces
    )


def generate_text(
    prompt: str,
    model: Optional[PreTrainedModel] = None,
    tokenizer: Optional[PreTrainedTokenizer] = None,
    max_new_tokens: int = 50,
    temperature: float = 1.0,
    top_k: int = 0,
    top_p: float = 1.0,
    seed: Optional[int] = None,
    verbose: bool = False,
) -> str:
    """
    Primary user-facing text generation function returning generated text string.

    Args:
        prompt: Initial prompt text string.
        model: Pre-loaded GPT-2 LM model.
        tokenizer: Pre-loaded GPT-2 tokenizer.
        max_new_tokens: Maximum number of tokens to generate.
        temperature: Temperature scaling factor.
        top_k: Top-K rank cutoff.
        top_p: Top-P nucleus threshold.
        seed: Random seed for multi-step reproducible generation.
        verbose: Print step-by-step trace output if True.

    Returns:
        str: Generated output text string (excluding initial prompt).
    """
    result = generate_text_detailed(
        prompt=prompt,
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        seed=seed,
        verbose=verbose
    )
    return result.generated_text
