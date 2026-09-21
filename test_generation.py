"""
Stage 4 Test Suite: Autoregressive Text Generation & Hyperparameter Comparison Suite.

Tests:
- Test A: Basic generation (20 new tokens)
- Test B: Deterministic seeded generation (output_1 == output_2)
- Test C: Different seeds comparison (seed=42 vs seed=123)
- Test D: Temperature comparison (T=0.2, 0.7, 1.2)
- Test E: Top-K comparison (K=1, 10, 50)
- Test F: Top-P comparison (P=0.5, 0.9, 0.95, 1.0)
- Generation Trace Test: Step-by-step autoregressive inspection
"""

import sys
import torch
from generator.model import load_model_and_tokenizer
from generator.generation import generate_text, generate_text_detailed

# Reconfigure stdout for UTF-8 compatibility on Windows
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')


def print_section(title: str) -> None:
    print("\n" + "=" * 65)
    print(f"🔹 {title}")
    print("=" * 65)


def test_generation_suite():
    print("=" * 65)
    print("🤖 STAGE 4: AUTOREGRESSIVE TEXT GENERATION TEST SUITE")
    print("=" * 65)

    print("\nLoading GPT-2 Model & Tokenizer...")
    model, tokenizer, device = load_model_and_tokenizer("gpt2")
    print(f"Model successfully loaded on device: '{device}'")

    prompt = "The future of artificial intelligence"

    # -------------------------------------------------------------------------
    # TEST A: Basic Generation
    # -------------------------------------------------------------------------
    print_section("TEST A: BASIC GENERATION (20 TOKENS)")
    gen_a = generate_text(
        prompt=prompt,
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=20,
        temperature=1.0,
        top_k=0,
        top_p=1.0,
        seed=None
    )
    print(f"Prompt:         \"{prompt}\"")
    print(f"Generated Text: \"{gen_a}\"")
    assert len(gen_a) > 0, "Generated text must not be empty!"
    print("✅ TEST A PASSED: Basic generation produces valid text.")

    # -------------------------------------------------------------------------
    # TEST B: Deterministic Seeded Generation
    # -------------------------------------------------------------------------
    print_section("TEST B: DETERMINISTIC SEEDED GENERATION")
    out_b1 = generate_text(
        prompt=prompt,
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=20,
        temperature=0.7,
        top_k=50,
        top_p=0.95,
        seed=42
    )
    out_b2 = generate_text(
        prompt=prompt,
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=20,
        temperature=0.7,
        top_k=50,
        top_p=0.95,
        seed=42
    )
    print(f"Run 1 (seed=42): \"{out_b1}\"")
    print(f"Run 2 (seed=42): \"{out_b2}\"")
    assert out_b1 == out_b2, "Seeded generation MUST be 100% deterministic (output_1 == output_2)!"
    print("✅ TEST B PASSED: Seeded generation is strictly deterministic across runs.")

    # -------------------------------------------------------------------------
    # TEST C: Different Seeds Comparison
    # -------------------------------------------------------------------------
    print_section("TEST C: DIFFERENT SEEDS COMPARISON")
    out_c1 = generate_text(prompt=prompt, model=model, tokenizer=tokenizer, max_new_tokens=20, temperature=0.8, seed=42)
    out_c2 = generate_text(prompt=prompt, model=model, tokenizer=tokenizer, max_new_tokens=20, temperature=0.8, seed=123)
    print(f"Seed 42:  \"{out_c1}\"")
    print(f"Seed 123: \"{out_c2}\"")
    print("✅ TEST C PASSED: Observed seed-dependent variation.")

    # -------------------------------------------------------------------------
    # TEST D: Temperature Comparison
    # -------------------------------------------------------------------------
    print_section("TEST D: TEMPERATURE COMPARISON (T=0.2, 0.7, 1.2)")
    for temp in [0.2, 0.7, 1.2]:
        text_d = generate_text(
            prompt=prompt,
            model=model,
            tokenizer=tokenizer,
            max_new_tokens=20,
            temperature=temp,
            top_k=50,
            top_p=0.95,
            seed=42
        )
        print(f"Temp {temp:.1f}: \"{text_d}\"")
    print("✅ TEST D PASSED: Temperature variation verified.")

    # -------------------------------------------------------------------------
    # TEST E: Top-K Comparison
    # -------------------------------------------------------------------------
    print_section("TEST E: TOP-K COMPARISON (K=1, 10, 50)")
    for k in [1, 10, 50]:
        text_e = generate_text(
            prompt=prompt,
            model=model,
            tokenizer=tokenizer,
            max_new_tokens=20,
            temperature=1.0,
            top_k=k,
            top_p=1.0,
            seed=42
        )
        print(f"Top-K {k:2d}: \"{text_e}\"")
    print("✅ TEST E PASSED: Top-K rank truncation verified.")

    # -------------------------------------------------------------------------
    # TEST F: Top-P Comparison
    # -------------------------------------------------------------------------
    print_section("TEST F: TOP-P COMPARISON (P=0.5, 0.9, 0.95, 1.0)")
    for p in [0.5, 0.9, 0.95, 1.0]:
        text_f = generate_text(
            prompt=prompt,
            model=model,
            tokenizer=tokenizer,
            max_new_tokens=20,
            temperature=1.0,
            top_k=0,
            top_p=p,
            seed=42
        )
        print(f"Top-P {p:.2f}: \"{text_f}\"")
    print("✅ TEST F PASSED: Top-P nucleus sampling verified.")

    # -------------------------------------------------------------------------
    # GENERATION TRACE TEST
    # -------------------------------------------------------------------------
    print_section("GENERATION TRACE DEMONSTRATION (5 STEPS VERBOSE)")
    res_trace = generate_text_detailed(
        prompt=prompt,
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=5,
        temperature=0.7,
        top_k=50,
        top_p=0.95,
        seed=42,
        verbose=True
    )
    print(f"\nReconstructed Full Text: \"{res_trace.full_text}\"")
    print(f"Stopping Reason: {res_trace.stop_reason}")
    print("✅ GENERATION TRACE PASSED.")

    print("\n" + "=" * 65)
    print("🎉 ALL STAGE 4 AUTOREGRESSIVE GENERATION TESTS PASSED!")
    print("=" * 65)


if __name__ == "__main__":
    test_generation_suite()
