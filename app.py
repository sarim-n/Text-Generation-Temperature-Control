"""
Streamlit User Interface for Text Generation Temperature Control Playground.

Presentation layer that connects user inputs to:
- Backend generation engine (generator/generation.py)
- Sampling visualization & explainability analysis engine (sampling/analysis.py)
"""

from typing import Optional, List, Dict, Any
import pandas as pd
import torch
import streamlit as st

from generator.model import load_model_and_tokenizer
from generator.generation import generate_text_detailed, GenerationResult
from sampling.analysis import (
    analyze_next_token,
    compare_temperatures_analysis,
    format_token_for_display,
    SamplingAnalysisResult
)

# -----------------------------------------------------------------------------
# 1. Page Configuration & Layout
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Text Generation Temperature Control",
    page_icon="🎛️",
    layout="wide",
    initial_sidebar_state="expanded"
)


# -----------------------------------------------------------------------------
# 2. Cached Resource Model Loader
# -----------------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def get_cached_model_and_tokenizer():
    """
    Caches the GPT-2 model and tokenizer in memory using Streamlit's @st.cache_resource.
    Prevents reloading model weights on every button click or parameter change.
    """
    return load_model_and_tokenizer("gpt2")


# -----------------------------------------------------------------------------
# 3. Sidebar Hyperparameter Controls
# -----------------------------------------------------------------------------
st.sidebar.title("🎛️ Sampling Controls")
st.sidebar.markdown("Adjust parameters to control randomness, truncation, and reproducibility.")

temperature = st.sidebar.slider(
    label="Temperature (T)",
    min_value=0.1,
    max_value=2.0,
    value=0.7,
    step=0.05,
    help="Lower values make generation more focused and deterministic. Higher values make generation more diverse and random."
)

top_k = st.sidebar.slider(
    label="Top-K Truncation",
    min_value=0,
    max_value=100,
    value=50,
    step=1,
    help="Top-K limits sampling to the K highest-scoring tokens (0 = disabled)."
)

top_p = st.sidebar.slider(
    label="Top-P Nucleus",
    min_value=0.1,
    max_value=1.0,
    value=0.95,
    step=0.01,
    help="Top-P keeps the smallest group of tokens whose cumulative probability reaches the selected threshold (1.0 = disabled)."
)

max_new_tokens = st.sidebar.slider(
    label="Maximum New Tokens",
    min_value=1,
    max_value=200,
    value=50,
    step=5,
    help="Maximum number of tokens to generate autoregressively."
)

st.sidebar.markdown("---")
use_fixed_seed = st.sidebar.checkbox(
    label="Use fixed seed",
    value=False,
    help="A fixed seed makes repeated runs reproducible when the same prompt and settings are used."
)

seed: Optional[int] = None
if use_fixed_seed:
    seed = st.sidebar.number_input(
        label="Random Seed",
        min_value=0,
        max_value=999999,
        value=42,
        step=1
    )


# -----------------------------------------------------------------------------
# 4. Main Application Workspace
# -----------------------------------------------------------------------------
st.title("🎛️ Text Generation Temperature Control")
st.markdown(
    "Explore how **Temperature**, **Top-K**, and **Top-P (Nucleus)** sampling parameters affect "
    "autoregressive GPT-2 text generation and next-token probability distributions in real time."
)

# User Prompt Input Text Area
prompt_text = st.text_area(
    label="Prompt",
    value="The future of artificial intelligence",
    height=120,
    placeholder="Type your input prompt here...",
    help="Enter the initial context string for GPT-2 to complete."
)

# Generate Button
generate_clicked = st.button("🚀 Generate Text & Analyze Distribution", type="primary", use_container_width=True)

# -----------------------------------------------------------------------------
# 5. Execution & Result Presentation Layer
# -----------------------------------------------------------------------------
if generate_clicked:
    cleaned_prompt = prompt_text.strip()
    
    if not cleaned_prompt:
        st.warning("⚠️ Please enter a non-empty text prompt before generating.")
    else:
        try:
            with st.spinner("Loading GPT-2 model resources..."):
                model, tokenizer, device = get_cached_model_and_tokenizer()

            with st.spinner("Generating text autoregressively & analyzing logits..."):
                # Execute generation
                result: GenerationResult = generate_text_detailed(
                    prompt=cleaned_prompt,
                    model=model,
                    tokenizer=tokenizer,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    top_k=top_k,
                    top_p=top_p,
                    seed=seed,
                    verbose=False
                )

                # Extract final-position logits for explainability visualization
                inputs = tokenizer(cleaned_prompt, return_tensors="pt").to(device)
                with torch.inference_mode():
                    outputs = model(**inputs)
                final_prompt_logits = outputs.logits[0, -1, :].cpu()

                # Run sampling analysis
                analysis_res: SamplingAnalysisResult = analyze_next_token(
                    logits=final_prompt_logits,
                    tokenizer=tokenizer,
                    temperature=temperature,
                    top_k=top_k,
                    top_p=top_p,
                    top_n=15,
                    seed=seed
                )

                # Run temperature comparison analysis
                temp_comp_res = compare_temperatures_analysis(
                    logits=final_prompt_logits,
                    tokenizer=tokenizer,
                    temperatures=[0.2, 0.7, 1.2],
                    top_n=10
                )

            # Store results in session state
            st.session_state["last_result"] = result
            st.session_state["last_analysis"] = analysis_res
            st.session_state["last_temp_comp"] = temp_comp_res
            st.session_state["last_device"] = str(device)
            st.session_state["last_params"] = {
                "temperature": temperature,
                "top_k": top_k,
                "top_p": top_p,
                "max_new_tokens": max_new_tokens,
                "seed": seed
            }

        except Exception as e:
            st.error(f"❌ An error occurred during text generation: {str(e)}")

# Display Generated Results & Explainability Charts
if "last_result" in st.session_state:
    result: GenerationResult = st.session_state["last_result"]
    analysis: SamplingAnalysisResult = st.session_state["last_analysis"]
    temp_comp = st.session_state["last_temp_comp"]
    params = st.session_state["last_params"]
    device_str = st.session_state["last_device"]

    st.markdown("---")
    st.subheader("📝 Generated Output")

    # Text Output Containers
    col1, col2 = st.columns([1, 1])
    with col1:
        st.markdown("**Original Prompt:**")
        st.info(f"\"{result.prompt}\"")

    with col2:
        st.markdown("**Generated Continuation:**")
        st.success(f"\"{result.generated_text}\"" if result.generated_text else "*(No tokens generated / EOS reached)*")

    st.markdown("**Full Reconstructed Sequence:**")
    st.code(result.full_text, language="text")

    # Generation Metrics Summary
    st.markdown("### 📊 Generation & Distribution Summary")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Selected First Token", f"'{analysis.selected_display_label}'")
    m2.metric("Active Candidate Pool", f"{analysis.active_candidate_count} tokens")
    m3.metric("Max Probability", f"{analysis.max_probability * 100:.2f}%")
    m4.metric("Distribution Entropy", f"{analysis.entropy_bits:.2f} bits")

    st.markdown("---")
    st.subheader("🔬 Sampling Explainability & Visualization")
    st.caption("Inspect how Temperature, Top-K, and Top-P transformed GPT-2's next-token logits for the prompt.")

    # Streamlit Tabs for Visualization
    tab1, tab2, tab3, tab4 = st.tabs([
        "📊 Candidate Probabilities (Top-15)",
        "🔥 Temperature Effect (0.2 vs 0.7 vs 1.2)",
        "✂️ Top-K & Top-P Truncation",
        "📋 Full Candidate Table"
    ])

    # -------------------------------------------------------------------------
    # TAB 1: Candidate Probabilities Chart
    # -------------------------------------------------------------------------
    with tab1:
        st.markdown("#### Final Next-Token Probability Distribution (Top-15 Candidates)")
        chart_data = pd.DataFrame({
            "Token": [c.display_label for c in analysis.top_candidates],
            "Probability (%)": [c.final_prob * 100 for c in analysis.top_candidates]
        }).set_index("Token")
        
        st.bar_chart(chart_data)

    # -------------------------------------------------------------------------
    # TAB 2: Temperature Effect Comparison
    # -------------------------------------------------------------------------
    with tab2:
        st.markdown("#### Effect of Temperature on Probability Distribution (Same Logits)")
        st.info(
            "💡 **Key Concept**: Temperature scales raw logits (`z / T`) before softmax. "
            "It does NOT alter the model's knowledge, but changes distribution sharpness."
        )
        
        comp_df = pd.DataFrame(temp_comp["rows"]).set_index("display_label")
        temp_cols = [f"T={t}" for t in temp_comp["temperatures"]]
        st.bar_chart(comp_df[temp_cols] * 100)

    # -------------------------------------------------------------------------
    # TAB 3: Top-K & Top-P Truncation Chart
    # -------------------------------------------------------------------------
    with tab3:
        st.markdown("#### Top-P (Nucleus) Cumulative Probability Curve")
        cum_df = pd.DataFrame(analysis.cumulative_curve_data)
        
        c1, c2 = st.columns([2, 1])
        with c1:
            chart_cum = pd.DataFrame({
                "Rank": cum_df["rank"],
                "Cumulative Probability (%)": cum_df["cumulative_prob"] * 100
            }).set_index("Rank")
            st.line_chart(chart_cum)
        
        with c2:
            st.markdown(f"**Sampling Configuration:**")
            st.write(f"- **Top-K Setting:** `{params['top_k']}`")
            st.write(f"- **Top-P Setting:** `{params['top_p']}`")
            st.write(f"- **Surviving Candidates:** `{analysis.active_candidate_count}`")
            st.write(f"- **Selected Token:** `{analysis.selected_display_label}` ({analysis.selected_token_prob*100:.1f}%)")

    # -------------------------------------------------------------------------
    # TAB 4: Full Detailed Candidate Table
    # -------------------------------------------------------------------------
    with tab4:
        st.markdown("#### Detailed Stage-by-Stage Token Transformation Table")
        table_rows = []
        for c in analysis.top_candidates:
            table_rows.append({
                "Rank Token": c.display_label,
                "Token ID": c.token_id,
                "Raw Logit (z)": round(c.raw_logit, 2),
                "Raw Prob (%)": round(c.raw_prob * 100, 2),
                "Temp Prob (%)": round(c.temp_prob * 100, 2),
                "Final Prob (%)": round(c.final_prob * 100, 2),
                "Status": "✅ Active" if c.is_active else "❌ Truncated (-inf)"
            })
        st.dataframe(pd.DataFrame(table_rows), use_container_width=True)
