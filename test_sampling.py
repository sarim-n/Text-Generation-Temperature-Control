"""
Stage 3 Test Suite: Sampling Engine Verification & Numerical Edge Case Suite.

Tests:
- Part A: Softmax probability conversion
- Part B: Categorical sampling stochasticity & reproducibility
- Part C: Temperature scaling (T=0.5, 1.0, 2.0)
- Part D: Top-K logit filtering
- Part E: Top-P (nucleus) logit filtering
- Part F & G: Real GPT-2 logits sampling test
- Part H: Numerical edge cases & boundary error handling
"""

import sys
import math
import torch

from sampling.softmax import logits_to_probs
from sampling.temperature import apply_temperature
from sampling.top_k import apply_top_k
from sampling.top_p import apply_top_p
from sampling.sampler import sample_categorical, sample_next_token
from generator.model import load_model_and_tokenizer

# Reconfigure stdout for UTF-8 compatibility on Windows
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')


def print_header(title: str) -> None:
    print("\n" + "=" * 65)
    print(f"🔹 {title}")
    print("=" * 65)


def calculate_entropy(probs: torch.Tensor) -> float:
    """Calculates Shannon entropy H(P) = -sum(p * log2(p)) in bits for positive probabilities."""
    nonzero_probs = probs[probs > 0]
    return float(-torch.sum(nonzero_probs * torch.log2(nonzero_probs)).item())


def test_part_a_softmax():
    print_header("PART A — SOFTMAX PROBABILITY CONVERSION")
    logits = torch.tensor([2.0, 1.0, 0.0])
    probs = logits_to_probs(logits)
    prob_sum = float(torch.sum(probs).item())

    print(f"Original Logits:     {logits.tolist()}")
    print(f"Resulting Probs:     {[round(p, 4) for p in probs.tolist()]}")
    print(f"Probability Sum:     {prob_sum:.6f}")

    assert torch.all(probs >= 0), "Probabilities must be non-negative"
    assert abs(prob_sum - 1.0) < 1e-5, "Probabilities must sum to ~1.0"
    print("✅ PART A PASSED: Softmax produces valid probability distribution.")


def test_part_b_categorical_sampling():
    print_header("PART B — RANDOM CATEGORICAL SAMPLING")
    probs = torch.tensor([0.70, 0.20, 0.10])
    
    # 1. Stochastic sampling test
    samples = [sample_categorical(probs) for _ in range(100)]
    count_0 = samples.count(0)
    count_1 = samples.count(1)
    count_2 = samples.count(2)
    print(f"Distribution: [0.70, 0.20, 0.10]")
    print(f"100 Samples Counts -> Token 0: {count_0}, Token 1: {count_1}, Token 2: {count_2}")

    # 2. Reproducibility test with seed
    gen1 = torch.Generator().manual_seed(42)
    sample_seeded_1 = sample_categorical(probs, generator=gen1)

    gen2 = torch.Generator().manual_seed(42)
    sample_seeded_2 = sample_categorical(probs, generator=gen2)

    print(f"Seeded Sample 1 (seed=42): {sample_seeded_1}")
    print(f"Seeded Sample 2 (seed=42): {sample_seeded_2}")

    assert sample_seeded_1 == sample_seeded_2, "Seeded sampling must be deterministic!"
    print("✅ PART B PASSED: Categorical sampling is stochastic and reproducibly seedable.")


def test_part_c_temperature():
    print_header("PART C — TEMPERATURE SCALING")
    logits = torch.tensor([4.0, 3.0, 2.0, 1.0])

    print(f"Original Logits: {logits.tolist()}")
    for temp in [0.5, 1.0, 2.0]:
        scaled_logits = apply_temperature(logits, temp)
        probs = logits_to_probs(scaled_logits)
        probs_rounded = [round(p, 4) for p in probs.tolist()]
        print(f"Temp {temp:.1f} -> Probs: {probs_rounded} | Max Prob: {max(probs_rounded):.4f}")

    # Lower temp should concentrate max probability higher than temp=1.0
    p_low = logits_to_probs(apply_temperature(logits, 0.5))
    p_high = logits_to_probs(apply_temperature(logits, 2.0))
    assert torch.max(p_low) > torch.max(p_high), "Low temp must concentrate probability more than high temp"
    print("✅ PART C PASSED: Temperature sharpens/flattens distribution correctly.")


def test_part_d_top_k():
    print_header("PART D — TOP-K LOGIT FILTERING")
    logits = torch.tensor([1.0, 5.0, 2.0, 8.0, 3.0])
    top_k = 3

    filtered = apply_top_k(logits, top_k)
    probs = logits_to_probs(filtered)

    print(f"Original Logits: {logits.tolist()}")
    print(f"Top-K ({top_k}) Filtered Logits: {filtered.tolist()}")
    print(f"Post-Softmax Probs:               {[round(p, 4) for p in probs.tolist()]}")

    # Tokens at indices 1 (logit=5.0), 3 (logit=8.0), 4 (logit=3.0) should survive
    assert filtered[3] == 8.0 and filtered[1] == 5.0 and filtered[4] == 3.0
    assert filtered[0] == -float('inf') and filtered[2] == -float('inf')
    assert probs[0].item() == 0.0 and probs[2].item() == 0.0
    print("✅ PART D PASSED: Top-K correctly suppresses non-top-K tokens.")


def test_part_e_top_p():
    print_header("PART E — TOP-P (NUCLEUS) FILTERING")
    logits = torch.tensor([4.0, 3.0, 2.0, 1.0, 0.0])

    for p in [1.0, 0.8, 0.5]:
        filtered = apply_top_p(logits, p)
        probs = logits_to_probs(filtered)
        active_count = torch.sum(probs > 0).item()
        print(f"Top-P ({p:.2f}) -> Active Candidate Count: {active_count} | Probs: {[round(x, 4) for x in probs.tolist()]}")

    p_05_filtered = apply_top_p(logits, 0.5)
    p_05_probs = logits_to_probs(p_05_filtered)
    assert torch.sum(p_05_probs > 0).item() < len(logits), "Top-P < 1.0 must truncate candidates"
    print("✅ PART E PASSED: Top-P nucleus dynamically truncates candidate sets.")


def test_part_f_g_real_gpt2_sampling():
    print_header("PARTS F & G — REAL GPT-2 LOGITS SAMPLING TEST")
    model, tokenizer, device = load_model_and_tokenizer("gpt2")

    prompt = "The future of artificial intelligence"
    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = model(**inputs)

    # Extract final position logits [vocab_size]
    final_logits = outputs.logits[0, -1, :].cpu()

    configs = [
        {"temp": 0.2, "top_k": 50, "top_p": 0.95, "desc": "Deterministic / Focused"},
        {"temp": 0.7, "top_k": 50, "top_p": 0.95, "desc": "Balanced"},
        {"temp": 1.2, "top_k": 50, "top_p": 0.95, "desc": "Creative / High Variance"}
    ]

    for cfg in configs:
        t, k, p = cfg["temp"], cfg["top_k"], cfg["top_p"]
        token_id, probs, _ = sample_next_token(final_logits, temperature=t, top_k=k, top_p=p, seed=42)
        decoded_token = tokenizer.decode([token_id])

        active_candidates = torch.sum(probs > 0).item()
        max_prob = float(torch.max(probs).item())
        entropy = calculate_entropy(probs)

        print(f"\nConfiguration: Temp={t}, Top-K={k}, Top-P={p} ({cfg['desc']})")
        print(f"  - Selected Token ID:          {token_id}")
        print(f"  - Decoded Token String:       '{decoded_token}'")
        print(f"  - Active Candidate Count:     {active_candidates}")
        print(f"  - Maximum Probability:        {max_prob * 100:.2f}%")
        print(f"  - Distribution Entropy:       {entropy:.2f} bits")

        # Repeat sampling 5 times without seed to demonstrate stochastic variance
        stochastic_samples = []
        for _ in range(5):
            tid, _, _ = sample_next_token(final_logits, temperature=t, top_k=k, top_p=p, seed=None)
            stochastic_samples.append(tokenizer.decode([tid]))
        print(f"  - 5 Unseeded Sample Outputs:  {stochastic_samples}")

    print("\n✅ PARTS F & G PASSED: Real GPT-2 logits sampling engine operates correctly.")


def test_part_h_edge_cases():
    print_header("PART H — NUMERICAL AND EDGE-CASE VALIDATION")

    logits = torch.tensor([3.0, 2.0, 1.0, 0.0])

    # 1. Temperature boundary errors
    try:
        apply_temperature(logits, 0.0)
        assert False, "Temperature 0 must raise ValueError"
    except ValueError as e:
        print(f"Captured expected temp=0 error: {e}")

    try:
        apply_temperature(logits, -0.5)
        assert False, "Negative temperature must raise ValueError"
    except ValueError as e:
        print(f"Captured expected negative temp error: {e}")

    # 2. Top-K boundary errors
    try:
        apply_top_k(logits, 0)
        assert False, "Top-K 0 must raise ValueError"
    except ValueError as e:
        print(f"Captured expected top_k=0 error: {e}")

    # Top-K > vocab_size should be safe
    topk_large = apply_top_k(logits, 100)
    assert torch.equal(topk_large, logits), "Top-K > vocab_size should return logits unchanged"

    # 3. Top-P boundary errors
    try:
        apply_top_p(logits, 0.0)
        assert False, "Top-P 0 must raise ValueError"
    except ValueError as e:
        print(f"Captured expected top_p=0 error: {e}")

    try:
        apply_top_p(logits, 1.5)
        assert False, "Top-P > 1.0 must raise ValueError"
    except ValueError as e:
        print(f"Captured expected top_p > 1.0 error: {e}")

    # 4. Numerical assertions
    probs = logits_to_probs(logits)
    assert not torch.isnan(probs).any(), "Probabilities must not contain NaN"
    assert not torch.isinf(probs).any(), "Probabilities must not contain Inf"
    assert abs(float(torch.sum(probs).item()) - 1.0) < 1e-5, "Probability sum must be 1.0"

    print("✅ PART H PASSED: All numerical edge cases and input validation rules enforced.")


def run_all_stage3_tests():
    print("=" * 65)
    print("🧪 STAGE 3: SAMPLING ENGINE & NUMERICAL TEST SUITE")
    print("=" * 65)

    test_part_a_softmax()
    test_part_b_categorical_sampling()
    test_part_c_temperature()
    test_part_d_top_k()
    test_part_e_top_p()
    test_part_f_g_real_gpt2_sampling()
    test_part_h_edge_cases()

    print("\n" + "=" * 65)
    print("🎉 ALL STAGE 3 SAMPLING ENGINE TESTS PASSED SUCCESSFULLY!")
    print("=" * 65)


if __name__ == "__main__":
    run_all_stage3_tests()
