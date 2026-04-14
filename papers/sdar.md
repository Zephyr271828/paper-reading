---
tags:
  - DLM
  - SPEC_DECODING
arxiv: "https://arxiv.org/abs/2510.06303"
github: "https://github.com/JetAstra/SDAR"
website: ""
year: 2025
read: false
---

# SDAR

> **Links:** [arXiv](https://arxiv.org/abs/2510.06303) | [GitHub](https://github.com/JetAstra/SDAR)
> **Tags:** #DLM #SPEC_DECODING

---

## Methodology

![[assets/sdar_fig.png]]

SDAR converts a pretrained autoregressive (AR) model into a hybrid blockwise diffusion model via lightweight continued training (~30–50B tokens). Sequences are partitioned into $K$ non-overlapping blocks $b_1, b_2, \ldots, b_K$ of size $B$. Inter-block dependencies are handled autoregressively; intra-block tokens are decoded in parallel via masked diffusion.

### Paradigm Conversion (Training)

**Objective:** Replace the standard NLL loss with the blockwise NELBO:

$$\mathcal{L}_{\text{blockwise}}(\theta) = \mathbb{E}_{t,\, q(b_k^t \mid b_k^0)} \left[ -\frac{1}{t} \sum_{\ell=1}^{B} \mathbf{1}\!\left[x_t^{(k,\ell)} = \texttt{[MASK]}\right] \log p_\theta\!\left(x_0^{(k,\ell)} \mid b_k^t,\, b_{<k};\, \theta\right) \right]$$

The $1/t$ term is a time-dependent reweighting factor. Crucially, the conversion requires **no logits shift and no attention mask annealing**.

**Attention mask construction:** The perturbed ($b_k^t$) and clean ($b_{<k}$) sequences are concatenated into a single forward pass. Clean (preceding) blocks use block-wise causal attention; the corrupted block attends to itself and all preceding blocks.

**Forward process:** Tokens within each block are independently masked with probability $1 - \alpha_t$, where $\alpha_t$ follows a monotonically decreasing noise schedule.

### Hierarchical Inference

1. Generate blocks left-to-right (AR at the block level).
2. Within each block, run iterative masked diffusion denoising for $B$ steps (matching block size).
3. Apply a **remasking strategy** after each denoising step:
   - **Static:** unmask a fixed number of tokens per step regardless of confidence.
   - **Dynamic:** unmask all tokens exceeding a confidence threshold $\tau \in \{0.80, 0.85, 0.90, 0.95\}$, re-mask the rest; this achieves 2–4× speedup over static with minimal accuracy loss.

**Effective Tokens Per Forward Pass (TPF):** scales linearly with block size $B$, enabling a throughput–quality trade-off.

---

## Experiment Setup

**Models:** Qwen3 family (1.7B, 4B, 8B, 30B-A3B MoE); 2B Chat baseline experiments also included.

**Baselines:** AR-Chat (same Qwen3 weights, no conversion), MDLM, LLaDA-8B, Dream-7B.

**Conversion corpus:** 50B-token subset of a 1T-token general web + STEM + code corpus; context length 4096 with packed sequences.

**Block sizes tested:** $B \in \{4, 8, 16, 32, 64\}$.

**Inference hardware:** H200 GPU; throughput measured as tokens/sec.

---

## Results

### Table 1: 2B Chat Model Benchmarks (General)

| Model | BBH | MMLU | MATH | GSM8K | HumanEval | MBPP | GPQA |
|-------|-----|------|------|-------|-----------|------|------|
| AR-2B-Chat | 35.7 | 48.7 | 29.9 | 61.8 | 42.1 | 44.4 | 26.3 |
| MDLM-2B-Chat | 32.2 | 47.0 | 12.6 | 57.9 | 21.3 | 27.2 | 26.3 |
| AR-BD-2B-Chat-b16 | 35.9 | 50.9 | 26.8 | 59.4 | 40.0 | 42.9 | 28.2 |
| MDLM-BD-2B-Chat-b16 | 32.9 | 47.5 | 23.3 | 64.7 | 39.0 | 33.5 | 29.8 |

SDAR conversion (AR-BD) preserves AR accuracy while unlocking parallel inference; MDLM-BD recovers much of the performance lost by MDLM alone.

### Table 2: SDAR vs. Prior Diffusion Models (8B scale)

| Model | MMLU |
|-------|------|
| LLaDA-8B | 65.9 |
| Dream-7B | 69.5 |
| SDAR-8B-Chat | **78.6** |

### Table 3: Scientific Reasoning — 30B Models

| Benchmark | AR-30B-A3B-Sci | SDAR-30B-A3B-Sci | Δ |
|-----------|:--------------:|:-----------------:|:---:|
| ChemBench | 60.5 | **72.8** | +12.3 |
| GPQA-diamond | 61.2 | **66.0** | +4.8 |
| AIME-2024 | 74.9 | **76.7** | +1.8 |
| LiveMathBench-Hard | 55.4 | **58.7** | +3.3 |
| AIME-2025 | **60.7** | 59.2 | −1.5 |

### Test-Time Scaling (AIME-2024, Majority Vote)

| Method | AIME-2024 |
|--------|-----------|
| AR-30B-A3B | 74.9 |
| SDAR-30B-A3B (majority vote) | **86.7** (+11.8) |
| SDAR-30B-A3B (pass@k) | **93.3** (+18.4) |

### Ablations: Block Size vs. Throughput (H200)

| Block Size $B$ | Peak Throughput (tokens/sec) |
|:--------------:|:----------------------------:|
| 4 | **6,600** |
| 8 | ~4,800 |
| 16 | ~3,200 |
| 32 | ~2,100 |
| 64 | 1,509 |

Smaller blocks yield higher throughput (more AR steps, less diffusion parallelism); larger blocks enable more parallel decoding per step but increase latency per block.

---

## Related Papers

- [[mdlm]]
- [[rcd]]
- [[dmax]]
