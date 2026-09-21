# Text Generation Temperature Control Playground

## 📌 Project Overview
This project is an interactive playground designed to demonstrate how different sampling strategies (**Temperature**, **Top-K**, and **Top-P / Nucleus Sampling**) influence text generation behavior using GPT-2 (`gpt2`). The application provides an intuitive interface (built with Streamlit) to inspect and experiment with token probability distributions and generation quality in real time.

---

## 💻 Streamlit Application

Launch the interactive web user interface by running:

```bash
streamlit run app.py
```

### 🎯 Available Interface Controls

```text
                    USER
                     │
                     ▼
              Streamlit UI
                app.py
                     │
                     │ parameters
                     ▼
             generate_text(...)
                     │
                     ▼
              Generation Engine
                     │
                     ▼
                  GPT-2
                     │
                     ▼
              Sampling Engine
                     │
                     ▼
               Generated Text
                     │
                     ▼
              Streamlit UI
```

1. **Prompt Input Area**: A text area accepting custom prompt strings (e.g., `"The future of artificial intelligence"`).
2. **Temperature Slider ($T \in [0.1, 2.0]$)**:
   - **Low Temperature ($T < 0.7$)**: Sharpens probabilities; makes generation predictable, focused, and deterministic.
   - **High Temperature ($T > 1.0$)**: Flattens probabilities; increases entropy, randomness, and output diversity.
3. **Top-K Truncation Slider ($K \in [0, 100]$)**:
   - Restricts sampling to the top $K$ highest-scoring tokens ($0$ disables Top-K filtering).
4. **Top-P Nucleus Slider ($P \in [0.1, 1.0]$)**:
   - Restricts sampling to the minimal set of tokens whose cumulative probability reaches $P$ ($1.0$ disables Top-P filtering).
5. **Maximum New Tokens ($1 - 200$)**: Sets the limit on autoregressively generated tokens.
6. **Use Fixed Seed (Checkbox & Number Input)**: When enabled, seeds the PyTorch `torch.Generator` once for 100% reproducible multi-step runs.

---

## 🔄 Autoregressive Generation Mechanics

Autoregressive text generation means generating text **one token at a time**, where each newly generated token is appended to the prompt sequence and fed back into GPT-2 to predict the subsequent token.

```
Prompt
  │
  ▼
Tokenizer (Encode prompt to input_ids [1, N])
  │
  ▼
┌────────────────────────────────────────────────────────┐
│               Autoregressive Loop                      │
│                                                        │
│  Input Token IDs [1, seq_len]                          │
│           │                                            │
│           ▼                                            │
│        GPT-2  (torch.inference_mode())                 │
│           │                                            │
│           ▼                                            │
│    Next-Token Logits  (slice: outputs.logits[:, -1, :])│
│           │                                            │
│           ▼                                            │
│   Sampling Engine (Temperature → Top-K → Top-P)        │
│           │                                            │
│           ▼                                            │
│   Softmax & Categorical Sampling (torch.multinomial)   │
│           │                                            │
│           ▼                                            │
│     Next Token ID (Scalar integer)                     │
│           │                                            │
│           ▼                                            │
│    EOS Check? ──► Stop if EOS token (50256)            │
│           │ No                                         │
│           ▼                                            │
│  Append Token ID to Input Sequence                     │
│  [1, seq_len + 1] ─────────────────────────────────────┘
└────────────────────────────────────────────────────────┘
  │
  ▼
Tokenizer Decode (token_ids → Output Text String)
```

### Key Concepts & Tensor Transformations

1. **Why GPT-2 Generates One Token at a Time**: Transformers generate text token-by-token because causal language models predict probability distributions strictly for the next position given all preceding context.
2. **Final-Position Logit Slicing (`logits[:, -1, :]`)**:
   - For an input sequence of length $N$, GPT-2 outputs logits tensor `[1, N, 50257]`.
   - The logits at position $i < N$ predict token $i+1$ based on prompt history up to position $i$.
   - We extract `logits[:, -1, :]` of shape `[1, 50257]` because position $-1$ represents the latest token, whose output logits predict the upcoming next token.
3. **Appending Token IDs**: The sampled token ID is converted to a tensor `[[next_token_id]]` and concatenated along the sequence dimension: `torch.cat([input_ids, next_token_tensor], dim=-1)`.
4. **Inference Mode (`torch.inference_mode()`)**: Disables autograd tracking and gradient computation, significantly reducing memory overhead and execution latency.
5. **Continuous Generator for Seed Reproducibility**: Seeded generation instantiates a single `torch.Generator` **once** at the start of `generate_text()`. Passing this `generator` across loop iterations maintains continuous random state without resetting the seed per token.
6. **EOS Early Stopping**: If the sampled token ID matches GPT-2's End-Of-Sequence token (`eos_token_id = 50256`), generation halts immediately.
7. **Context Length Safety**: GPT-2 supports up to 1,024 context tokens (`n_positions`). The loop enforces safety limits to prevent tensor overflow errors.

---

## 🎛️ How Sampling Works

The sampling engine takes raw logit outputs from GPT-2 and transforms them into a valid, candidate-filtered probability distribution before sampling a single next token:

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
├── app.py                  # Streamlit User Interface (Presentation Layer)
├── test_model.py           # Stage 2 GPT-2 inference & logits verification script
├── test_sampling.py        # Stage 3 sampling engine & numerical edge-case test suite
├── test_generation.py      # Stage 4 autoregressive text generation test suite
├── generator/              # Model management & autoregressive generation loop
│   ├── __init__.py         # Package exports
│   ├── model.py            # GPT-2 model & tokenizer loader
│   └── generation.py       # Explicit autoregressive text generation engine (generate_text)
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
