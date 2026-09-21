"""
Sampling Visualization & Explainability Analysis Module.

Provides detailed breakdown of probability transformations across:
Raw Logits -> Temperature Scaling -> Top-K Filtering -> Top-P Nucleus -> Final Probabilities & Categorical Selection.

Calculates full-vocabulary metrics (Entropy, Active Candidate Count, Max Probability)
while extracting top-N token breakdowns for visual rendering.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple
import torch
from transformers import PreTrainedTokenizer

from sampling.softmax import logits_to_probs
from sampling.temperature import apply_temperature
from sampling.top_k import apply_top_k
from sampling.top_p import apply_top_p
from sampling.sampler import sample_categorical


@dataclass
class TokenCandidateInfo:
    """Struct holding detailed stage-by-stage metric values for a single token candidate."""
    token_id: int
    token_str: str
    display_label: str
    raw_logit: float
    raw_prob: float
    temp_scaled_logit: float
    temp_prob: float
    final_prob: float
    is_active: bool


@dataclass
class SamplingAnalysisResult:
    """Structured container holding comprehensive sampling analysis data for visualization."""
    vocab_size: int
    temperature: float
    top_k: int
    top_p: float
    active_candidate_count: int
    max_probability: float
    entropy_bits: float
    selected_token_id: int
    selected_token_str: str
    selected_display_label: str
    selected_token_prob: float
    top_candidates: List[TokenCandidateInfo]
    cumulative_curve_data: List[Dict[str, Any]]


def format_token_for_display(token_str: str) -> str:
    """
    Formats raw subword token strings for clean visual display in charts and tables.
    Replaces leading subword space indicators (e.g. 'Ġ' or leading space) with '▸ '.
    """
    if not token_str:
        return "<EMPTY>"
    
    # Handle Hugging Face GPT-2 BPE space markers
    display = token_str.replace("Ġ", "▸ ").replace(" ", "▸ ") if token_str.startswith(" ") or token_str.startswith("Ġ") else token_str
    
    # Handle newline and tab visual replacements
    display = display.replace("\n", "↵ (newline)").replace("\t", "⇥ (tab)")
    return display


def calculate_full_vocab_entropy(probs: torch.Tensor) -> float:
    """
    Calculates Shannon Entropy H(P) = -sum(p * log2(p)) in bits over the FULL vocabulary probability tensor.
    """
    nonzero_probs = probs[probs > 0]
    if len(nonzero_probs) == 0:
        return 0.0
    entropy = -torch.sum(nonzero_probs * torch.log2(nonzero_probs)).item()
    return float(entropy)


def analyze_next_token(
    logits: torch.Tensor,
    tokenizer: PreTrainedTokenizer,
    temperature: float = 1.0,
    top_k: int = 0,
    top_p: float = 1.0,
    top_n: int = 15,
    seed: Optional[int] = None,
    generator: Optional[torch.Generator] = None,
) -> SamplingAnalysisResult:
    """
    Analyzes the exact probability transformations performed by the sampling engine on GPT-2 next-token logits.

    Args:
        logits: PyTorch logit tensor [vocab_size] or [1, vocab_size].
        tokenizer: PreTrainedTokenizer for token decoding.
        temperature: Temperature scaling factor.
        top_k: Top-K rank cutoff parameter.
        top_p: Top-P nucleus threshold parameter.
        top_n: Number of top candidate tokens to extract for chart visualization.
        seed: Optional random seed for single-step reproducibility.
        generator: Optional PyTorch torch.Generator instance.

    Returns:
        SamplingAnalysisResult dataclass containing full-vocabulary metrics and top-N candidate details.
    """
    if logits.dim() > 1:
        logits = logits.squeeze(0)

    vocab_size = logits.size(-1)
    raw_logits = logits.clone()

    # Step 1: Raw Probabilities
    raw_probs = logits_to_probs(raw_logits)

    # Step 2: Temperature Scaling
    temp_logits = raw_logits.clone()
    if temperature != 1.0:
        temp_logits = apply_temperature(temp_logits, temperature)
    temp_probs = logits_to_probs(temp_logits)

    # Step 3: Top-K Filtering
    topk_logits = temp_logits.clone()
    if top_k > 0:
        topk_logits = apply_top_k(topk_logits, top_k)

    # Step 4: Top-P Nucleus Filtering
    final_logits = topk_logits.clone()
    if top_p < 1.0:
        final_logits = apply_top_p(final_logits, top_p)

    # Step 5: Final Softmax Probabilities over FULL Vocabulary
    final_probs = logits_to_probs(final_logits)

    # Step 6: Full Vocabulary Statistics
    active_candidate_count = int(torch.sum(final_probs > 0).item())
    max_probability = float(torch.max(final_probs).item())
    entropy_bits = calculate_full_vocab_entropy(final_probs)

    # Step 7: Categorical Sampling
    if generator is None and seed is not None:
        generator = torch.Generator(device=final_probs.device)
        generator.manual_seed(seed)

    selected_token_id = sample_categorical(final_probs, generator=generator)
    selected_token_str = tokenizer.decode([selected_token_id])
    selected_display_label = format_token_for_display(selected_token_str)
    selected_token_prob = float(final_probs[selected_token_id].item())

    # Step 8: Extract Top-N Token Candidate Information
    # Rank by final_probs descending, fallback to raw_logits descending
    sort_keys = final_probs + (raw_logits - raw_logits.min()) * 1e-8
    top_indices = torch.topk(sort_keys, k=min(top_n, vocab_size)).indices.tolist()

    top_candidates: List[TokenCandidateInfo] = []
    for tid in top_indices:
        tok_str = tokenizer.decode([tid])
        disp_label = format_token_for_display(tok_str)
        top_candidates.append(TokenCandidateInfo(
            token_id=tid,
            token_str=tok_str,
            display_label=disp_label,
            raw_logit=float(raw_logits[tid].item()),
            raw_prob=float(raw_probs[tid].item()),
            temp_scaled_logit=float(temp_logits[tid].item()),
            temp_prob=float(temp_probs[tid].item()),
            final_prob=float(final_probs[tid].item()),
            is_active=bool(final_probs[tid] > 0)
        ))

    # Step 9: Compute Sorted Cumulative Probability Curve for Top-P Nucleus Chart
    sorted_probs, sorted_indices = torch.sort(temp_probs, descending=True)
    cum_probs = torch.cumsum(sorted_probs, dim=-1)

    cumulative_curve_data: List[Dict[str, Any]] = []
    num_curve_tokens = min(top_n, vocab_size)
    for rank in range(num_curve_tokens):
        tid = int(sorted_indices[rank].item())
        tok_str = tokenizer.decode([tid])
        disp_label = format_token_for_display(tok_str)
        cumulative_curve_data.append({
            "rank": rank + 1,
            "token_id": tid,
            "token_str": tok_str,
            "display_label": disp_label,
            "individual_prob": float(sorted_probs[rank].item()),
            "cumulative_prob": float(cum_probs[rank].item()),
            "in_nucleus": bool(final_probs[tid] > 0)
        })

    return SamplingAnalysisResult(
        vocab_size=vocab_size,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        active_candidate_count=active_candidate_count,
        max_probability=max_probability,
        entropy_bits=entropy_bits,
        selected_token_id=selected_token_id,
        selected_token_str=selected_token_str,
        selected_display_label=selected_display_label,
        selected_token_prob=selected_token_prob,
        top_candidates=top_candidates,
        cumulative_curve_data=cumulative_curve_data
    )


def compare_temperatures_analysis(
    logits: torch.Tensor,
    tokenizer: PreTrainedTokenizer,
    temperatures: List[float] = [0.2, 0.7, 1.2],
    top_n: int = 10
) -> Dict[str, Any]:
    """
    Compares probability distribution shapes across different temperatures using the exact same raw logits tensor.

    Returns structured comparison dictionary containing top-N candidate probabilities for each temperature.
    """
    if logits.dim() > 1:
        logits = logits.squeeze(0)

    raw_probs = logits_to_probs(logits)
    top_indices = torch.topk(raw_probs, k=min(top_n, logits.size(-1))).indices.tolist()

    comparison_rows = []
    for tid in top_indices:
        tok_str = tokenizer.decode([tid])
        disp_label = format_token_for_display(tok_str)
        
        row = {"token_id": tid, "display_label": disp_label}
        for temp in temperatures:
            scaled_logits = apply_temperature(logits, temp)
            probs = logits_to_probs(scaled_logits)
            row[f"T={temp}"] = float(probs[tid].item())
        comparison_rows.append(row)

    return {
        "temperatures": temperatures,
        "rows": comparison_rows
    }
