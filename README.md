# Text Generation Temperature Control Playground

## 📌 Project Overview
This project is an interactive playground designed to demonstrate how different sampling strategies (**Temperature**, **Top-K**, and **Top-P / Nucleus Sampling**) influence text generation behavior using GPT-2 (`gpt2`). The application provides an intuitive interface (built with Streamlit) to inspect and experiment with token probability distributions and generation quality in real time.

---

## 🎯 What Temperature, Top-K, and Top-P Demonstrate

1. **Temperature ($T$)**:
   - Scales raw model output logits ($z_i / T$) before applying softmax.
   - **Low Temperature ($T < 1.0$)**: Makes probability distributions sharper/more greedy, resulting in predictable, deterministic text.
   - **High Temperature ($T > 1.0$)**: Flattens distributions, increasing output randomness, diversity, and creativity (or potential incoherence).

2. **Top-K Sampling**:
   - Truncates the token pool to only the $K$ highest-probability candidate tokens.
   - Prevents very low probability / nonsensical tokens from being sampled.

3. **Top-P (Nucleus) Sampling**:
   - Dynamically selects the smallest set of top tokens whose cumulative probability reaches a specified threshold $P$ (e.g., $P = 0.90$).
   - Adapts the candidate pool size dynamically based on the model's confidence across different prediction steps.

---

## 🏗️ High-Level Architecture

The project follows a clean, modular pythonic design:

```
Text Generation Temperature Control/
├── app.py                  # Streamlit User Interface
├── generator/              # Model management & autoregressive generation loop
│   ├── __init__.py
│   ├── model.py            # GPT-2 model & tokenizer loader
│   └── generation.py       # Autoregressive generation pipeline
├── sampling/               # Modular sampling algorithms
│   ├── __init__.py
│   ├── temperature.py      # Temperature scaling logic
│   ├── top_k.py            # Top-K filtering algorithm
│   └── top_p.py            # Top-P (nucleus) filtering algorithm
├── utils/                  # Helper & formatting utilities
│   └── __init__.py
├── requirements.txt        # Python dependencies
└── README.md               # Project documentation
```

---

## 🔄 Planned Generation Pipeline

```
Prompt → Tokenizer → GPT-2 → Logits → Sampling → Next Token → Repeat → Generated Text
```

1. **Prompt**: User inputs a text prompt via the Streamlit interface.
2. **Tokenizer**: Converts the input prompt string into input token IDs.
3. **GPT-2**: Performs a forward pass to output raw logits for the next token position.
4. **Logits**: Raw unnormalized prediction scores for all vocabulary tokens.
5. **Sampling**: Logits are passed through configured sampling strategies:
   - **Temperature Scaling** $\rightarrow$ **Top-K Truncation** $\rightarrow$ **Top-P (Nucleus) Masking** $\rightarrow$ **Softmax & Categorical Sampling**
6. **Next Token**: Selects the next token ID from the filtered probability distribution.
7. **Repeat**: Appends the sampled token to input sequence and repeats autoregressively until max length or end-of-text token.
8. **Generated Text**: Decodes token IDs back into natural language for UI display.
