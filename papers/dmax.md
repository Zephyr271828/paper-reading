---
tags:
  - DLM
  - SPEC_DECODING
arxiv: https://arxiv.org/abs/2604.08302
github: https://github.com/czg1225/DMax
website: ""
year: 2026
read: true
---

# DMax: Aggressive Parallel Decoding for dLLMs

> **Links:** [arXiv](https://arxiv.org/abs/2604.08302) | [GitHub](https://github.com/czg1225/DMax)
> **Tags:** #DLM #SPEC_DECODING

---

## Methodology

![](../assets/dmax_fig.png)

DMax enables aggressive parallel decoding for discrete diffusion language models (dLLMs) by addressing the train-inference gap through two components: **On-Policy Uniform Training (OPUT)** and **Soft Parallel Decoding (SPD)**.

### On-Policy Uniform Training (OPUT)

Standard diffusion training corrupts tokens by sampling from the vocabulary uniformly (random noise). OPUT instead samples noisy inputs from the model's own predictive distribution — bridging the gap between training-time corruption and inference-time self-correction errors.

**Training procedure:**
1. Sample corruption level $t \sim \text{Uniform}(t_l, t_h)$ with mask ratio $= 0.75$.
2. Create masked sequence $x_t^{(m)}$ by replacing tokens with `[MASK]` at probability $t$.
3. Feed $x_t^{(m)}$ to model; sample on-policy predictions $x_t^{(p)}$ from the model distribution.
4. Run two forward passes: one on $x_t^{(m)}$, one on $x_t^{(p)}$.
5. Apply cross-entropy loss on both outputs against the original clean sequence $x_0$:

$$\mathcal{L}_{\text{on-policy}} = \mathcal{L}_{\text{mask}} + \mathcal{L}_{\text{pred}}$$

$$\mathcal{L}_{\text{mask}} = -\sum_i \log p_\theta^{(m)}(x_0^i \mid x_t^{(m)}), \quad \mathcal{L}_{\text{pred}} = -\sum_i \log p_\theta^{(p)}(x_0^i \mid x_t^{(p)})$$

- $i$: token position; $x_0^i$: clean (ground-truth) token at position $i$.
- $x_t^{(m)}$: standard mask-corrupted input; $x_t^{(p)}$: on-policy corrupted input (tokens sampled from the model's own distribution at the same mask ratio $t$).
- $p_\theta^{(m)}, p_\theta^{(p)}$: the same network $\theta$ evaluated on $x_t^{(m)}$ and $x_t^{(p)}$ respectively — superscripts indicate *which input branch*, not separate parameters.

### Soft Parallel Decoding (SPD)

SPD operates on a block of 32 tokens and represents intermediate states as **interpolations in embedding space** between mask embeddings and token embeddings, rather than binary mask-to-token transitions. This allows iterative self-revision instead of hard commitment.

**Inference procedure per block:**
1. Initialize all positions in the block as `[MASK]`.
2. Compute predictions and per-token confidence scores $c_i$.
3. Promote the highest-confidence **contiguous prefix** (from left) to token positions using threshold $\tau_{\text{dec}}$ (0.5 for math, 0.65 for code).
4. For promoted positions, compute a **hybrid embedding**:

$$e_i = c_i \cdot e_{\hat{x}_i} + (1 - c_i) \cdot e_{\text{mask}}$$

where $\hat{x}_i$ is the predicted token and $e_{\text{mask}}$ is the mask embedding.

5. Iterate until convergence: predictions are consistent across steps **or** all confidences $> \tau_{\text{acc}} = 0.9$.

The contiguous-prefix constraint ensures causal coherence in autoregressive-style generation.

---

## Experiment Setup

- **Base model:** LLaDA-2.0-mini (fine-tuned, not LLaDA-2.0-8B)
- **Training data:** 0.7M math samples + 1.0M code samples via self-distillation from LLaDA-2.0-mini outputs
- **Optimizer:** AdamW, lr $= 2 \times 10^{-6}$, cosine schedule
- **Fine-tuning:** 2 epochs, batch size 8, on 8× H200 GPUs
- **Block size:** 32 tokens
- **Baselines:** LLaDA-2.0-mini (confidence threshold 0.95), Hierarchical Decoding, dParallel-SFT, Uniform Diffusion Training (UDLM objective)
- **Benchmarks:** GSM8K, MATH500, HumanEval-Instruct, MBPP-Instruct

---

## Results

### Main Results (Accuracy vs. Tokens-Per-Frame)

Tokens-per-frame (TPF) measures parallelism: how many tokens are committed per decoding step.

| Method | GSM8K TPF | GSM8K Acc. | MATH500 TPF | MATH500 Acc. | HumanEval TPF | HumanEval Acc. | MBPP TPF | MBPP Acc. |
|---|---|---|---|---|---|---|---|---|
| LLaDA-2.0-mini | 2.04 | 92.6% | 2.58 | 75.8% | 4.38 | 84.2% | 2.71 | 80.6% |
| DMax-Math | **5.48** | **92.1%** | **5.94** | **75.4%** | — | — | — | — |
| DMax-Coder | — | — | — | — | **7.36** | **83.5%** | **5.86** | **79.2%** |

- Throughput: **1,338 tokens/sec** on 2× H200 GPUs at batch size 1 (vs. 512 for baseline).
- Accuracy drop: $< 0.5\%$ on math; $< 1.5\%$ on code tasks.

### Ablation: Training and Inference Components (GSM8K, $\tau_{\text{dec}}=0.50$)

| On-Policy | Contiguous Prefix | Hybrid Embedding | TPF | Acc. |
|---|---|---|---|---|
| | | | 4.47 | 78.0% |
| ✓ | | | 5.14 | 90.1% |
| ✓ | ✓ | | 5.28 | 91.3% |
| ✓ | ✓ | ✓ | **5.48** | **92.1%** |

On-policy rollout is the core enabling factor; hybrid embeddings restore accuracy under extreme parallelism ($\tau_{\text{dec}} = 0.0$: 90.4% vs. 69.6% without).

### Ablation: Block Convergence Criteria ($\tau_{\text{dec}}=0.50$)

| Consistency | Confidence | GSM8K TPF | GSM8K Acc. | MBPP TPF | MBPP Acc. |
|---|---|---|---|---|---|
| ✓ | | 5.13 | 92.1% | 5.16 | 79.9% |
| | ✓ | 2.28 | 92.2% | 3.36 | 80.1% |
| ✓ | ✓ | **5.48** | **92.1%** | **5.86** | **79.2%** |

Consistency is the primary convergence signal; confidence acts as a secondary safety check.

---

## Related Papers

- [mdlm](mdlm.md)
- [sdar](sdar.md)