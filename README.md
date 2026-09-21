# Text Generation Temperature Control

An interactive visualization and educational tool for understanding **Temperature**, **Top-K**, and **Top-P (Nucleus)** sampling parameters in Large Language Model (LLM) text generation.

## 🌟 Overview

In text generation, **Temperature** ($T$) adjusts the sharpness of the probability distribution over the model's vocabulary prior to sampling:

$$P(w_i) = \frac{\exp(z_i / T)}{\sum_j \exp(z_j / T)}$$

- **Low Temperature ($T \to 0$)**: Makes the distribution more deterministic (focused on the most likely tokens). Ideal for coding, math, and factual Q&A.
- **Moderate Temperature ($T \approx 0.7 - 1.0$)**: Balances creativity and coherence. Great for general conversation and creative writing.
- **High Temperature ($T > 1.0$)**: Flattens the distribution, increasing output randomness, variety, and potential hallucinations.

## 🚀 Features

- **Interactive Token Probability Visualizer**: Adjust temperature dynamically to watch logit transformations and softmax probability updates in real-time.
- **Top-K & Top-P (Nucleus) Sampling Controls**: Observe how truncation strategies filter candidate tokens before sampling.
- **Interactive Text Generation Simulator**: Test different temperature settings with prompt completions and live token generation step-by-step.
- **Entropy & Perplexity Calculator**: Track real-time changes in distribution entropy as temperature fluctuates.

## 🛠️ Getting Started

Open `index.html` directly in any web browser or host locally using any static HTTP server.

```bash
# Optional local server
npx serve .
```

## 📜 License

MIT License
