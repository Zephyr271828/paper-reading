---
tags:
  - QUANT
  - MLSYS
  - THEORY
  - INFO_THEORY
arxiv: "https://arxiv.org/abs/2504.19874"
github: ""
website: ""
year: 2025
read: false
---

# TurboQuant

> **Links:** [arXiv](https://arxiv.org/abs/2504.19874)
> **Tags:** #QUANT #MLSYS #THEORY #INFO_THEORY

---

## Methodology

![](../assets/turboquant_fig.png)

TurboQuant is a data-oblivious online vector quantization algorithm that achieves near-optimal distortion rates for both MSE and inner product objectives. The core idea: apply a random rotation to concentrate coordinate magnitudes, then apply optimal scalar quantizers per coordinate independently.

### Random Rotation

Given input vector $x \in \mathbb{R}^d$ with $\|x\|_2 = 1$, apply a random rotation matrix $\Pi$ (e.g., randomized Hadamard transform). After rotation, each coordinate $(\Pi x)_i$ follows a Beta-like distribution:

$$f_X(t) = \frac{\Gamma(d/2)}{\sqrt{\pi}\,\Gamma\!\left(\frac{d-1}{2}\right)}\,(1-t^2)^{(d-3)/2}$$

which converges to $\mathcal{N}(0, 1/d)$ as $d \to \infty$. This concentration makes coordinate magnitudes nearly i.i.d., enabling independent scalar quantization with near-optimal rate-distortion.

### MSE Variant ($Q_\text{mse}$)

1. Rotate: $\tilde{x} = \Pi x$
2. Quantize each coordinate independently using optimal Lloyd-Max scalar quantizers with $2^b$ levels fit to the Beta distribution on $[-1,1]$
3. Store $b$ bits per coordinate as centroid indices
4. Decode: retrieve centroids, apply $\Pi^\top$

**MSE distortion upper bound:**

$$D_\text{mse} \leq \frac{\sqrt{3}\,\pi}{2} \cdot \frac{1}{4^b}$$

Information-theoretic lower bound: $D_\text{mse}(Q) \geq \frac{1}{4^b}$, so TurboQuant is within a constant factor (~2.7x) of optimal.

| Bits $b$ | TurboQuant $D_\text{mse}$ | Lower Bound |
|----------|--------------------------|-------------|
| 1 | 0.36 | 0.25 |
| 2 | 0.117 | 0.063 |
| 3 | 0.030 | 0.016 |
| 4 | 0.009 | 0.004 |

### Inner Product Variant ($Q_\text{prod}$)

Extends $Q_\text{mse}$ to unbiased inner product estimation using a two-stage scheme:

1. Apply $Q_\text{mse}$ with bit-width $(b-1)$ to get coarse quantized vector $\hat{x}$
2. Compute residual $r = x - \hat{x}$
3. Apply 1-bit Quantized Johnson-Lindenstrauss (QJL) transform to the residual: store $\text{sign}(Ar)$ and $\|r\|_2$
4. Storage: $(b-1)$-bit MSE indices + 1-bit QJL signs + scalar $\|r\|_2$

**Unbiasedness guarantee:** $\mathbb{E}[\langle y,\, Q_\text{prod}^{-1}(Q_\text{prod}(x))\rangle] = \langle y, x\rangle$

**Inner product distortion bound:**

$$D_\text{prod} \leq \frac{\sqrt{3}\,\pi^2\,\|y\|_2^2}{d} \cdot \frac{1}{4^b}$$

Lower bound: $D_\text{prod}(Q) \geq \frac{1}{d} \cdot \frac{1}{4^b}$

### KV Cache Quantization

TurboQuant is applied per key/value channel (across the sequence dimension). The random rotation $\Pi$ is shared across a layer (data-oblivious), enabling online single-pass quantization without offline training.

---

## Experiment Setup

**KV cache compression:**
- Models: Llama-3.1-8B-Instruct, Ministral-7B-Instruct
- Benchmarks: LongBench, Needle-in-a-Haystack
- Baselines: Full cache (FP16), KIVI, PolarQuant, SnapKV, PyramidKV
- Compression ratios: 2.5-bit and 3.5-bit per KV channel vs. FP16 (16-bit)

**Nearest neighbor search:**
- Datasets: GloVe ($d=200$), OpenAI3 ($d=1536$, $d=3072$)
- Baselines: Product Quantization (PQ), RabitQ
- Metric: Recall@10 at 4-bit compression

---

## Results

### LongBench (Llama-3.1-8B-Instruct)

| Method | KV Size (bits) | SingleQA | MultiQA | Summarization | Few-shot | Synthetic | Code | Avg |
|--------|---------------|----------|---------|---------------|----------|-----------|------|-----|
| Full Cache | 16 | 45.29 | 45.16 | 26.55 | 68.38 | 59.54 | 46.28 | 50.06 |
| KIVI | 3 | 43.38 | 37.99 | 27.16 | 68.38 | 59.50 | 44.68 | 48.50 |
| KIVI | 5 | 45.04 | 45.70 | 26.47 | 68.57 | 59.55 | 46.41 | 50.16 |
| PolarQuant | 3.9 | 45.18 | 44.48 | 26.23 | 68.25 | 60.07 | 45.24 | 49.78 |
| TurboQuant | 2.5 | 44.16 | 44.96 | 24.80 | 68.01 | 59.65 | 45.76 | 49.44 |
| TurboQuant | 3.5 | 45.01 | 45.31 | 26.00 | 68.63 | 59.95 | 46.17 | 50.06 |

### Needle-in-a-Haystack (4x compression)

| Method | Retrieval Score |
|--------|----------------|
| Full Cache | 0.997 |
| TurboQuant | 0.997 |
| PolarQuant | 0.995 |
| KIVI | 0.981 |
| PyramidKV | 0.895 |
| SnapKV | 0.858 |

### Nearest Neighbor Search — Recall@10 vs. Indexing Time

| Approach | Time d=200 | Time d=1536 | Time d=3072 |
|----------|------------|-------------|-------------|
| TurboQuant | 0.0007 s | 0.0013 s | 0.0021 s |
| Product Quantization | 37.04 s | 239.75 s | 494.42 s |
| RabitQ | 597.25 s | 2267.59 s | 3957.19 s |

TurboQuant indexes ~50,000x faster than PQ while achieving higher Recall@10 on OpenAI3 ($d=1536$: TurboQuant ~0.92 vs. PQ ~0.88, RabitQ ~0.85).

### Ablations

| Variant | Inner Product Bias | Notes |
|---------|-------------------|-------|
| $Q_\text{mse}$ | Biased | Simple, fast, near-optimal MSE |
| $Q_\text{prod}$ | Unbiased | Two-stage with 1-bit QJL residual |
| No rotation | High distortion | Coordinates not concentrated |
