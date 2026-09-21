"""
Stage 6 Test Suite: Sampling Visualization, Explainability & Regression Test Suite.

Tests:
- Test 1: Full vocabulary coverage (50,257 tokens)
- Test 2: Temperature sharpening vs. flattening validation
- Test 3: Top-K candidate count verification (top_k=5 -> 5 active)
- Test 4: Top-P nucleus boundary token retention
- Test 5: Probability normalization (~1.0)
- Test 6: Full vocabulary entropy calculation
- Test 7: Seeded reproducibility
- Test 8: Full regression suite execution (Stage 2, Stage 3, Stage 4)
"""

import sys
import torch
from generator.model import load_model_and_tokenizer
from sampling.analysis import analyze_next_token, compare_temperatures_analysis
from sampling.softmax import logits_to_probs
from sampling.temperature import apply_temperature
from sampling.top_k import apply_top_k
from sampling.top_p import apply_top_p

# Reconfigure stdout for UTF-8 compatibility on Windows
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')


def print_section(title: str) -> None:
    print("\n" + "=" * 65)
    print(f"🔹 {title}")
    print("=" * 65)


def test_stage6_analysis_suite():
    print("=" * 65)
    print("🔬 STAGE 6: SAMPLING VISUALIZATION & EXPLAINABILITY SUITE")
    print("=" * 65)

    print("\nLoading GPT-2 Model & Tokenizer...")
    model, tokenizer, device = load_model_and_tokenizer("gpt2")
    print(f"Model successfully loaded on device: '{device}'")

    prompt = "The future of artificial intelligence"
    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = model(**inputs)

    raw_logits = outputs.logits[0, -1, :].cpu()

    # -------------------------------------------------------------------------
    # TEST 1: Full Vocabulary Coverage
    # -------------------------------------------------------------------------
    print_section("TEST 1: FULL VOCABULARY COVERAGE")
    analysis = analyze_next_token(raw_logits, tokenizer, temperature=0.7, top_k=50, top_p=0.95, top_n=15, seed=42)
    print(f"Vocabulary Size:               {analysis.vocab_size}")
    print(f"Top Candidates Extracted:      {len(analysis.top_candidates)}")
    assert analysis.vocab_size == 50257, f"Expected 50257 vocab size, got {analysis.vocab_size}"
    print("✅ TEST 1 PASSED: Full 50,257 vocabulary evaluated.")

    # -------------------------------------------------------------------------
    # TEST 2: Temperature Sharpening vs Flattening
    # -------------------------------------------------------------------------
    print_section("TEST 2: TEMPERATURE SHARPENING VS. FLATTENING")
    ana_low = analyze_next_token(raw_logits, tokenizer, temperature=0.2, top_k=0, top_p=1.0)
    ana_high = analyze_next_token(raw_logits, tokenizer, temperature=1.5, top_k=0, top_p=1.0)

    print(f"Temp 0.2 -> Max Prob: {ana_low.max_probability*100:.2f}%, Entropy: {ana_low.entropy_bits:.2f} bits")
    print(f"Temp 1.5 -> Max Prob: {ana_high.max_probability*100:.2f}%, Entropy: {ana_high.entropy_bits:.2f} bits")

    assert ana_low.max_probability > ana_high.max_probability, "Low temp must have higher max prob than high temp"
    assert ana_low.entropy_bits < ana_high.entropy_bits, "Low temp must have lower entropy than high temp"
    print("✅ TEST 2 PASSED: Temperature sharpens and flattens probability correctly.")

    # -------------------------------------------------------------------------
    # TEST 3: Top-K Candidate Count
    # -------------------------------------------------------------------------
    print_section("TEST 3: TOP-K CANDIDATE COUNT")
    ana_topk5 = analyze_next_token(raw_logits, tokenizer, temperature=1.0, top_k=5, top_p=1.0)
    print(f"Top-K=5 -> Active Candidates Count: {ana_topk5.active_candidate_count}")
    assert ana_topk5.active_candidate_count == 5, f"Expected 5 active candidates for top_k=5, got {ana_topk5.active_candidate_count}"
    print("✅ TEST 3 PASSED: Top-K=5 leaves exactly 5 active candidates.")

    # -------------------------------------------------------------------------
    # TEST 4: Top-P Nucleus & Boundary Retention
    # -------------------------------------------------------------------------
    print_section("TEST 4: TOP-P NUCLEUS & BOUNDARY RETENTION")
    target_top_p = 0.80
    ana_topp = analyze_next_token(raw_logits, tokenizer, temperature=1.0, top_k=0, top_p=target_top_p, top_n=50)
    
    # Calculate cumulative probability sum of all retained nucleus tokens
    raw_probs = logits_to_probs(raw_logits)
    retained_mask = apply_top_p(raw_logits, target_top_p) > -float('inf')
    retained_sum = float(torch.sum(raw_probs[retained_mask]).item())
    
    print(f"Top-P={target_top_p:.2f} -> Retained Count: {ana_topp.active_candidate_count}, Cumulative Prob Sum: {retained_sum:.4f}")
    assert retained_sum >= target_top_p, f"Retained cumulative sum {retained_sum} must be >= {target_top_p} threshold"
    print("✅ TEST 4 PASSED: Top-P nucleus boundary token retained.")

    # -------------------------------------------------------------------------
    # TEST 5: Probability Normalization
    # -------------------------------------------------------------------------
    print_section("TEST 5: PROBABILITY NORMALIZATION (~1.0)")
    filtered_logits = apply_top_k(raw_logits, 5)
    probs = logits_to_probs(filtered_logits)
    sum_probs = float(torch.sum(probs).item())
    print(f"Final Probabilities Sum: {sum_probs:.6f}")
    assert abs(sum_probs - 1.0) < 1e-5, "Probabilities must sum to ~1.0"
    print("✅ TEST 5 PASSED: Probability distribution is validly normalized.")

    # -------------------------------------------------------------------------
    # TEST 6: Full Vocabulary Entropy Calculation
    # -------------------------------------------------------------------------
    print_section("TEST 6: FULL VOCABULARY ENTROPY CALCULATION")
    entropy = ana_low.entropy_bits
    print(f"Entropy over Full Vocab: {entropy:.4f} bits")
    assert entropy >= 0, "Entropy must be non-negative"
    print("✅ TEST 6 PASSED: Full-vocabulary entropy calculated.")

    # -------------------------------------------------------------------------
    # TEST 7: Seeded Reproducibility
    # -------------------------------------------------------------------------
    print_section("TEST 7: SEEDED REPRODUCIBILITY")
    run_1 = analyze_next_token(raw_logits, tokenizer, temperature=0.7, top_k=50, top_p=0.95, seed=42)
    run_2 = analyze_next_token(raw_logits, tokenizer, temperature=0.7, top_k=50, top_p=0.95, seed=42)
    print(f"Run 1 Selected: '{run_1.selected_token_str}' (ID {run_1.selected_token_id})")
    print(f"Run 2 Selected: '{run_2.selected_token_str}' (ID {run_2.selected_token_id})")
    assert run_1.selected_token_id == run_2.selected_token_id, "Seeded runs must select identical token IDs"
    print("✅ TEST 7 PASSED: Seeded analysis is 100% reproducible.")

    # -------------------------------------------------------------------------
    # TEST 8: Full Regression Suite Execution
    # -------------------------------------------------------------------------
    print_section("TEST 8: FULL CODEBASE REGRESSION CHECK")
    from test_model import run_stage2_demo
    from test_sampling import run_all_stage3_tests
    from test_generation import test_generation_suite

    run_stage2_demo("The future of artificial intelligence")
    run_all_stage3_tests()
    test_generation_suite()
    print("✅ TEST 8 PASSED: All Stage 2, 3, 4 regression tests executed cleanly.")

    print("\n" + "=" * 65)
    print("🎉 ALL STAGE 6 SAMPLING EXPLAINABILITY TESTS PASSED!")
    print("=" * 65)


if __name__ == "__main__":
    test_stage6_analysis_suite()
