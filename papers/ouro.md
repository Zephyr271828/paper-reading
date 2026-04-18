---
tags:
  - NLP
  - DEEP_LEARNING
arxiv: https://arxiv.org/abs/2510.25741
github: ""
website: http://ouro-llm.github.io
year: 2025
read: true
---

# Scaling Latent Reasoning via Looped Language Models

> **Links:** [arXiv](https://arxiv.org/abs/2510.25741) | [Website](http://ouro-llm.github.io)
> **Tags:** #NLP #DEEP_LEARNING

---

## Methodology

![](../assets/ouro_fig.png)

Ouro is a family of **Looped Language Models (LoopLM)** that replace depth with recurrence via weight-tied layers. A single $L$-layer transformer block is applied $T$ recurrent steps; step $t=1$ recovers the standard non-looped model:

$$F^{(t)}(\cdot) = \text{lmhead} \circ \underbrace{M^L \circ \cdots \circ M^L}_{t \text{ iterations}} \circ \text{emb}(\cdot)$$

- $\text{emb}(\cdot)$: token embedding layer; $\text{lmhead}$: vocabulary projection (tied to emb).
- $M^L$: the stack of $L$ transformer layers, with all weights *shared across recurrent steps*.
- $F^{(t)}$: forward pass that applies $M^L$ exactly $t$ times before the head; $t=1$ reduces to a standard $L$-layer transformer.
- $\circ$: function composition.

### Architecture

| Variant | Params | Layers | Hidden | Attn | Act | RoPE base |
|---|---|---|---|---|---|---|
| Ouro 1.4B | 1.4B | 24 | 2048 | MHA | SwiGLU | 10K → 1M |
| Ouro 2.6B | 2.6B | 48 | 2048 | MHA | SwiGLU | 10K → 1M |

Vocabulary size: 49,152. Max recurrent steps at inference: $T_\text{max} = 4$.

### Stage I — Entropy-Regularized Training

The model is trained with a mixture-of-depths objective over all exit steps $t \in \{1,\ldots,T_\text{max}\}$, with a learned discrete exit gate $p_\phi(t \mid x)$:

$$\mathcal{L} = \sum_t p_\phi(t \mid x)\,\mathcal{L}^{(t)} - \beta \cdot H\!\left(p_\phi(\cdot \mid x)\right)$$

- $p_\phi(t \mid x)$: learned exit-gate distribution over $t \in \{1, \ldots, T_\text{max}\}$, parameters $\phi$.
- $\mathcal{L}^{(t)}$: next-token cross-entropy loss computed at exit depth $t$ (i.e., from $F^{(t)}(x)$).
- $H(p_\phi(\cdot \mid x)) = -\sum_t p_\phi(t \mid x) \log p_\phi(t \mid x)$: Shannon entropy of the gate.
- $\beta \geq 0$: entropy coefficient.

This is an ELBO with a uniform prior over exit depths. The entropy term (coefficient $\beta$) prevents collapse to maximum depth and encourages exploration across depth levels.

### Stage II — Adaptive Gate Training

A specialized cross-entropy loss calibrates the exit gate against observed per-step loss improvements $I_i^{(t)}$:

$$\mathcal{L}_\text{adaptive} = -\frac{1}{M}\sum_i\!\left[w_i^{(t)}\log(1 - \lambda_i^{(t)}) + (1 - w_i^{(t)})\log\lambda_i^{(t)}\right]$$

where the ideal continuation probability is $w_i^{(t)} = \sigma\!\left(k \cdot (I_i^{(t)} - \gamma)\right)$ and $\lambda_i^{(t)}$ is the gate's exit probability at step $t$.

- $i$: example index; $M$: minibatch size.
- $I_i^{(t)} = \mathcal{L}_i^{(t-1)} - \mathcal{L}_i^{(t)}$: observed loss *improvement* from running one more recurrent step (positive ⇒ step was helpful).
- $\gamma$: threshold above which a step is deemed worth taking; $k > 0$: sigmoid slope (larger = harder decision boundary).
- $w_i^{(t)} \in [0,1]$: soft target label — "should we continue?" (close to $1$ when $I_i^{(t)} > \gamma$).
- $\lambda_i^{(t)}$: the gate's predicted *exit* probability at step $t$; the loss is binary cross-entropy between "continue" and "exit".

### Early Exit at Inference (Q-Exit)

The model exits at the first step where the cumulative exit CDF exceeds threshold $q \in [0,1]$:

$$t_\text{exit}(x) = \min\!\left\{m \in \{1,\ldots,T_\text{max}\} : \text{CDF}(m \mid x) \ge q\right\}$$

- $\text{CDF}(m \mid x) = \sum_{t \le m} p_\phi(t \mid x)$: cumulative exit probability through depth $m$.
- $q$: user-set confidence threshold.

Lower $q$ exits earlier (less compute); $q=1$ always runs all $T_\text{max}$ steps.

### KV Cache at Inference

A **last-step reuse** strategy stores only the final recurrent step's KV cache, reducing memory by $4\times$ with $\le 2\%$ performance loss on MATH-500.

---

## Experiment Setup

**Baselines (base):** Gemma3-1B/4B/12B, Llama3.2-1.2B/3B, Llama3.1-8B, Qwen2.5-1.5B/3B/7B, Qwen3-1.7B/4B/8B.

**Baselines (reasoning):** Qwen3-1.7B/4B/8B, DeepSeek-R1-Distill-Qwen-1.5B/7B.

**Benchmarks (base):** MMLU, MMLU-Pro, BBH, ARC-C, HellaSwag, Winogrande, GSM8K, MATH500, HumanEval(+), MBPP(+).

**Benchmarks (reasoning):** AIME24/25 (pass@1, pass@10), OlympiadBench, BeyondAIME, SuperGPQA, GPQA.

---

## Results

### Base Model — Ouro 1.4B vs 1–4B Baselines

| Model | Params | Tokens | MMLU | MMLU-Pro | BBH | GSM8K | MATH500 | HumanEval |
|---|---|---|---|---|---|---|---|---|
| Gemma3 1B | 1.0B | 2T | 39.85 | 11.31 | 30.26 | 2.05 | 41.00 | 6.70 |
| Llama3.2 1.2B | 1.0B | 9T | 45.46 | 11.80 | 30.72 | 7.05 | 7.40 | 19.50 |
| Qwen2.5 1.5B | 1.5B | 18T | 60.99 | 29.11 | 43.66 | 60.73 | 17.60 | 52.40 |
| Qwen3 1.7B | 1.7B | 36T | 62.46 | 37.27 | 53.51 | 70.28 | 25.80 | 66.50 |
| Qwen2.5 3B | 3.0B | 18T | 65.62 | 37.87 | 55.37 | 74.60 | 42.60 | 68.90 |
| Llama3.2 3B | 3.0B | 9T | 59.69 | 33.34 | 39.45 | 67.20 | 40.80 | 29.90 |
| Qwen3 4B | 4.0B | 36T | 73.19 | 51.40 | 70.95 | 72.86 | 59.60 | 77.40 |
| Gemma3 4B | 4.0B | 4T | 58.37 | 34.61 | 66.32 | 68.69 | 68.60 | 34.80 |
| **Ouro 1.4B R4** | **1.4B** | **7.7T** | **67.35** | **48.62** | **71.02** | **78.92** | **82.40** | **74.40** |

### Base Model — Ouro 2.6B vs 3–12B Baselines

| Model | Params | Tokens | MMLU | MMLU-Pro | BBH | GSM8K | MATH500 | HumanEval |
|---|---|---|---|---|---|---|---|---|
| Qwen2.5 3B | 3.0B | 18T | 65.62 | 37.87 | 55.37 | 74.60 | 42.60 | 68.90 |
| Llama3.2 3B | 3.0B | 9T | 59.69 | 33.34 | 39.45 | 67.20 | 40.80 | 29.90 |
| Qwen3 4B | 4.0B | 36T | 73.19 | 51.40 | 71.14 | 72.86 | 59.60 | 77.70 |
| Gemma3 4B | 4.0B | 4T | 58.37 | 34.61 | 66.32 | 68.69 | 68.60 | 34.80 |
| Qwen2.5 7B | 7.0B | 18T | 74.20 | 43.55 | 53.72 | 81.50 | 61.20 | 79.30 |
| Llama3.1 8B | 8.0B | 15T | 73.02 | 43.24 | 71.56 | 78.17 | 52.90 | 38.40 |
| Qwen3 8B | 8.0B | 36T | 76.63 | 53.72 | 77.65 | 83.09 | 62.30 | 84.80 |
| Gemma3 12B | 12.0B | 12T | 72.14 | 49.21 | 78.41 | 77.18 | 83.20 | 46.30 |
| **Ouro 2.6B R4** | **2.6B** | **7.7T** | **74.60** | **55.73** | **80.46** | **81.58** | **90.85** | **78.70** |

Ouro 1.4B matches standard 4B transformers; Ouro 2.6B matches/exceeds standard 8B models.

### Reasoning Model (Ouro-Thinking)

| Model | AIME24 pass@1 | AIME24 pass@10 | AIME25 pass@1 | AIME25 pass@10 | OlympiadBench | BeyondAIME | SuperGPQA | GPQA |
|---|---|---|---|---|---|---|---|---|
| Qwen3-1.7B | 32.0 | 55.6 | 22.0 | 33.3 | 56.4 | 15.0 | 35.9 | 34.0 |
| Qwen3-4B | 61.3 | 75.0 | 51.3 | 63.3 | 73.2 | 31.0 | 51.9 | 54.5 |
| Qwen3-8B | 73.0 | 86.7 | 66.7 | 81.3 | 75.3 | 38.0 | 48.0 | 59.1 |
| DeepSeek-R1-Distill-Qwen-1.5B | 29.6 | 66.7 | 23.0 | 43.33 | 56.44 | 9.0 | 26.5 | 33.2 |
| DeepSeek-R1-Distill-Qwen-7B | 57.3 | 83.3 | 36.0 | 73.3 | 72.0 | 30.0 | 46.6 | 51.0 |
| **Ouro-1.4B-Thinking-R4** | **65.0** | **83.3** | **46.3** | **73.3** | **71.6** | **34.0** | **47.4** | **45.5** |
| **Ouro-2.6B-Thinking-R4** | **64.7** | **90.0** | **50.3** | **76.7** | **76.4** | **39.0** | **53.7** | **52.7** |

### Early Exit (MMLU, varying average exit rounds)

| Strategy | Avg exit rounds | MMLU |
|---|---|---|
| Static baseline (R4) | 4.0 | ~66 |
| Adaptive gate (Stage II) | 2.5 | ~66 |
| Standard pre-trained gate | 2.5 | ~64 |
| Hidden-state diff threshold | 2.5 | ~65 |

Adaptive gate achieves full-depth accuracy at 2.5 average rounds.

### KV Cache Efficiency

| Cache strategy | GSM8K | MATH-500 | Memory vs. full |
|---|---|---|---|
| Full cache (R4) | 78.92 | 82.40 | 1× |
| Last-step reuse | 78.85 | 80.40 | 0.25× |

---

## Training Details

### Multi-Stage Pre-Training (7.7T tokens total)

| Stage | Purpose | Seq len | Tokens | LR | Recurrent steps |
|---|---|---|---|---|---|
| 1a | Stable training | 4K | 3T | 3×10⁻⁴ | 8 |
| 1b | Stable training | 4K | 3T | 3×10⁻⁴ | 4 |
| 2 | CT annealing | 16K | 1.4T | 3×10⁻⁵ | 4 |
| 3 | Long-context | 64K | 20B | 3×10⁻⁵ | 4 |
| 4 | Mid-training | 32K | 300B | 1×10⁻⁵ | 4 |

Optimizer: AdamW ($\beta_1=0.9$, $\beta_2=0.95$, weight decay 0.1, grad clip 1.0). Entropy coefficient $\beta = 0.1$ (stage 1a) → $0.05$ (all later stages).

### Stage 1 Data Mix (6T tokens)

| Source | Fraction |
|---|---|
| Nemotron-CC (web) | 73.4% |
| MAP-CC (web) | 13.0% |
| OpenCoder | 7.5% |
| MegaMath-web | 4.1% |
| Ultra-FineWeb-zh | 2.0% |

### Stage 2 Annealing Mix (1.4T tokens)

| Source | Fraction |
|---|---|
| High-quality Nemotron-CC | 66.5% |
| Math (Nemotron-CC-Math-v1 + MegaMath-HQ) | 19.6% |
| Code | 7.2% |
| SFT data | 9.6% |

### Supervised Fine-Tuning (8.3M examples)

| Domain | Count | Sources |
|---|---|---|
| Math | 3.5M | OpenThoughts3, AceReason-1.1-SFT |
| Code | 3.2M | AceReason, OpenCodeReasoning, Llama-Nemotron, OpenThoughts3 |
| Science | 808K | OpenThoughts3, Llama-Nemotron |
| Chat | 767K | OO1-Chat-747K, DeepWriting-20K |
