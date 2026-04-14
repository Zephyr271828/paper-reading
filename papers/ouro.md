---
tags:
  - NLP
  - DEEP_LEARNING
arxiv: "https://arxiv.org/abs/2510.25741"
github: ""
website: "http://ouro-llm.github.io"
year: 2025
read: false
---

# Ouro

> **Links:** [arXiv](https://arxiv.org/abs/2510.25741) | [Website](http://ouro-llm.github.io)
> **Tags:** #NLP #DEEP_LEARNING

---

## Methodology

![](../assets/ouro_fig.png)

Ouro is a family of **Looped Language Models (LoopLM)** that replace depth with recurrence via weight-tied layers. A single $L$-layer transformer block is applied $T$ recurrent steps; step $t=1$ recovers the standard non-looped model:

$$F^{(t)}(\cdot) = \text{lmhead} \circ \underbrace{M^L \circ \cdots \circ M^L}_{t \text{ iterations}} \circ \text{emb}(\cdot)$$

### Architecture

| Variant | Params | Layers | Hidden | Attn | Act | RoPE base |
|---|---|---|---|---|---|---|
| Ouro 1.4B | 1.4B | 24 | 2048 | MHA | SwiGLU | 10K → 1M |
| Ouro 2.6B | 2.6B | 48 | 2048 | MHA | SwiGLU | 10K → 1M |

Vocabulary size: 49,152. Max recurrent steps at inference: $T_\text{max} = 4$.

### Stage I — Entropy-Regularized Training

The model is trained with a mixture-of-depths objective over all exit steps $t \in \{1,\ldots,T_\text{max}\}$, with a learned discrete exit gate $p_\phi(t \mid x)$:

$$\mathcal{L} = \sum_t p_\phi(t \mid x)\,\mathcal{L}^{(t)} - \beta \cdot H\!\left(p_\phi(\cdot \mid x)\right)$$

This is an ELBO with a uniform prior over exit depths. The entropy term (coefficient $\beta$) prevents collapse to maximum depth and encourages exploration across depth levels.

### Stage II — Adaptive Gate Training

A specialized cross-entropy loss calibrates the exit gate against observed per-step loss improvements $I_i^{(t)}$:

$$\mathcal{L}_\text{adaptive} = -\frac{1}{M}\sum_i\!\left[w_i^{(t)}\log(1 - \lambda_i^{(t)}) + (1 - w_i^{(t)})\log\lambda_i^{(t)}\right]$$

where the ideal continuation probability is $w_i^{(t)} = \sigma\!\left(k \cdot (I_i^{(t)} - \gamma)\right)$ and $\lambda_i^{(t)}$ is the gate's exit probability at step $t$.

### Early Exit at Inference (Q-Exit)

The model exits at the first step where the cumulative exit CDF exceeds threshold $q \in [0,1]$:

$$t_\text{exit}(x) = \min\!\left\{m \in \{1,\ldots,T_\text{max}\} : \text{CDF}(m \mid x) \ge q\right\}$$

Lower $q$ exits earlier (less compute); $q=1$ always runs all $T_\text{max}$ steps.

### KV Cache at Inference

A **last-step reuse** strategy stores only the final recurrent step's KV cache, reducing memory by $4\times$ with $\le 2\%$ performance loss on MATH-500.

---

## Experiment Setup

**Baselines:** Qwen2.5-3B/7B, Llama-3.2-3B/3B-Instruct, SmolLM2-1.7B, OLMo-1B/7B, Pythia-1.4B, DeepSeek-R1-Distill-Qwen, Qwen3-1.7B/4B.

**Benchmarks (base):** MMLU, MMLU-Pro, BBH, GSM8K, MATH500, HumanEval.

**Benchmarks (reasoning):** AIME 2024, OlympiadBench, GPQA-Diamond.

---

## Results

### Base Model — Key Benchmarks

| Model | MMLU | MMLU-Pro | BBH | GSM8K | MATH500 | HumanEval |
|---|---|---|---|---|---|---|
| Ouro 1.4B (R4) | 67.35 | 48.62 | 71.02 | 78.92 | 82.40 | 74.40 |
| Ouro 2.6B (R4) | 74.60 | 55.73 | 80.46 | 81.58 | 90.85 | — |

Ouro 1.4B matches standard 4B transformers; Ouro 2.6B matches/exceeds standard 8B models.

### Reasoning Model (Ouro-Thinking)

| Model | AIME 2024 pass@1 | AIME 2024 pass@10 | OlympiadBench | GPQA-Diamond |
|---|---|---|---|---|
| Ouro-1.4B-Thinking-R4 | 65.0 | 83.3 | 71.6 | 47.4 |
| Ouro-2.6B-Thinking-R4 | 64.7 | 90.0 | 76.4 | 53.7 |

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
