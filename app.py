"""
Streamlit User Interface for Text Generation Temperature Control Playground.

Presentation layer that connects user inputs to the backend generation engine (generator/generation.py).
Contains NO ML model inference or sampling logic internally.
"""

from typing import Optional
import streamlit as st
from generator.model import load_model_and_tokenizer
from generator.generation import generate_text_detailed, GenerationResult

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
    Prevents reloading the model weights on every button click or parameter change.
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
    "autoregressive GPT-2 text generation in real time."
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
generate_clicked = st.button("🚀 Generate Text", type="primary", use_container_width=True)

# -----------------------------------------------------------------------------
# 5. Execution & Result Presentation Layer
# -----------------------------------------------------------------------------
if generate_clicked:
    cleaned_prompt = prompt_text.strip()
    
    # Empty prompt validation
    if not cleaned_prompt:
        st.warning("⚠️ Please enter a non-empty text prompt before generating.")
    else:
        try:
            # Load or retrieve cached GPT-2 model resources
            with st.spinner("Loading GPT-2 model resources..."):
                model, tokenizer, device = get_cached_model_and_tokenizer()

            # Execute autoregressive text generation
            with st.spinner("Generating text autoregressively step-by-step..."):
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

            # Store result in session state for persistence
            st.session_state["last_result"] = result
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

# Display Generated Results if available in session state
if "last_result" in st.session_state:
    result: GenerationResult = st.session_state["last_result"]
    params = st.session_state["last_params"]
    device_str = st.session_state["last_device"]

    st.markdown("---")
    st.subheader("📝 Generated Output")

    # Result Columns
    col1, col2 = st.columns([1, 1])

    with col1:
        st.markdown("**Original Prompt:**")
        st.info(f"\"{result.prompt}\"")

    with col2:
        st.markdown("**Generated Continuation:**")
        st.success(f"\"{result.generated_text}\"" if result.generated_text else "*(No tokens generated / EOS reached immediately)*")

    # Full Reconstructed Text Block
    st.markdown("**Full Reconstructed Sequence:**")
    st.code(result.full_text, language="text")

    # Generation Statistics & Configuration Badges
    st.markdown("### 📊 Generation Summary")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Generated Tokens", len(result.generated_token_ids))
    m2.metric("Prompt Tokens", len(result.prompt_token_ids))
    m3.metric("Stopping Reason", result.stop_reason)
    m4.metric("Inference Device", device_str.upper())

    # Sidebar parameters recap expander
    with st.expander("🔍 Applied Generation Parameters Recap"):
        st.json({
            "Prompt": result.prompt,
            "Temperature": params["temperature"],
            "Top-K": params["top_k"] if params["top_k"] > 0 else "Disabled (0)",
            "Top-P": params["top_p"] if params["top_p"] < 1.0 else "Disabled (1.0)",
            "Max New Tokens": params["max_new_tokens"],
            "Seed": params["seed"] if params["seed"] is not None else "None (Stochastic)",
            "Inference Mode": "torch.inference_mode()",
            "Model": "gpt2"
        })
