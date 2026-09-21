# Text Generation Temperature Control Playground

## 📌 Project Overview
This project is an interactive playground designed to demonstrate how different sampling strategies (**Temperature**, **Top-K**, and **Top-P / Nucleus Sampling**) influence text generation behavior using GPT-2 (`gpt2`). The application provides an intuitive interface (built with Streamlit) to inspect and experiment with token probability distributions and generation quality in real time.

---

## 🎛️ How Sampling Works

The sampling engine takes the raw logit outputs from GPT-2 and transforms them into a valid, candidate-filtered probability distribution before sampling a single next token:

```
GPT-2 final-position logits
            │
            ▼
   Temperature Scaling  (apply_temperature: scaled_logits = logits / T)
            │
            ▼
     Top-K Filtering    (apply_top_k: retains top K logits, masks rest with -inf)
            │
            ▼
     Top-P Filtering    (apply_top_p: retains nucleus until CumSum(P) >= top_p)
            │
            ▼
    Softmax Conversion  (logits_to_probs: P = softmax(logits))
            │
            ▼
  Categorical Sampling  (sample_categorical: torch.multinomial(P))
            │
            ▼
    Selected Token ID
```

### 1. Step-by-Step Sampling Flow
1. **GPT-2 Produces Logits**: A forward pass through GPT-2 yields raw prediction scores across the 50,257 vocabulary for every position.
2. **Final-Position Slicing**: We extract the slice `logits[:, -1, :]` corresponding to the final prompt token, which predicts the next token.
3. **Temperature Scaling**: Scales logits (`z / T`). Low $T$ sharpens differences (deterministic), while high $T$ flattens differences (creative/random).
4. **Top-K Filtering**: Truncates candidate set to a fixed rank cutoff $K$. Suppressed tokens are set to $-\infty$.
5. **Top-P Filtering**: Dynamically retains the top tokens whose cumulative probability reaches threshold $P$ (the "nucleus").
6. **Softmax Conversion**: Converts filtered logits into non-negative probabilities summing to $1.0$. Suppressed ($-\infty$) tokens become $0.0$ probability.
7. **Categorical Sampling**: Randomly samples a token ID according to the probability vector via `torch.multinomial`.
8. **Sequence Continuation**: The selected token ID is appended to the sequence during autoregressive text generation.

---

### 📊 Sampling Methods Comparison Table

| Method | What it changes | Main effect |
| :--- | :--- | :--- |
| **Temperature ($T$)** | Scales magnitude of raw logits ($z_i / T$) | Controls distribution entropy (low $T$ = deterministic & focused, high $T$ = diverse & random) |
| **Top-K** | Truncates candidate pool to fixed top $K$ tokens | Hard rank cutoff; prevents low-ranked nonsensical tokens from ever being sampled |
| **Top-P (Nucleus)** | Dynamically caps candidates by cumulative probability ($P$) | Flexible cutoff; contracts candidate pool when model is confident, expands when uncertain |

---

## 🔬 GPT-2 Inference Basics

```
Prompt → Tokenizer → input_ids → GPT-2 → logits → final-position logits → next-token prediction
```

1. **Prompt**: Raw text string provided by the user (e.g., `"The future of artificial intelligence"`).
2. **Tokenizer**: Translates prompt string into subword tokens using Byte-Pair Encoding (BPE).
3. **`input_ids`**: Integer tensor `[batch_size, sequence_length]` mapping each token to its index in the 50,257 vocabulary.
4. **GPT-2 Model**: Processes `input_ids` through multi-head self-attention transformer layers in evaluation mode (`model.eval()`).
5. **`logits`**: Unnormalized output prediction scores tensor `[batch_size, sequence_length, vocab_size]`.
6. **Final-Position Logits**: Extracted slice `logits[:, -1, :]` of shape `[batch_size, vocab_size]`.
7. **Next-Token Prediction**: Applying sampling transformations onto final-position logits before sampling next token ID.

---

## 🏗️ High-Level Architecture

```
Text Generation Temperature Control/
├── app.py                  # Streamlit User Interface
├── test_model.py           # Stage 2 GPT-2 inference & logits verification script
├── test_sampling.py        # Stage 3 sampling engine & numerical edge-case test suite
├── generator/              # Model management & autoregressive generation loop
│   ├── __init__.py
│   ├── model.py            # GPT-2 model & tokenizer loader
│   └── generation.py       # Autoregressive generation pipeline
├── sampling/               # Modular sampling algorithms
│   ├── __init__.py         # Package exports
│   ├── softmax.py          # Softmax probability converter (logits_to_probs)
│   ├── temperature.py      # Temperature scaling logic (apply_temperature)
│   ├── top_k.py            # Top-K filtering algorithm (apply_top_k)
│   ├── top_p.py            # Top-P (nucleus) filtering algorithm (apply_top_p)
│   └── sampler.py          # Categorical sampler & pipeline orchestrator (sample_next_token)
├── utils/                  # Helper & formatting utilities
│   └── __init__.py
├── requirements.txt        # Python dependencies
└── README.md               # Project documentation
```
