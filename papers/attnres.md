---
tags:
  - DEEP_LEARNING
arxiv: https://arxiv.org/abs/2603.15031
github: https://github.com/MoonshotAI/Attention-Residuals
website: ""
year: 2026
read: false
---

# AttnRes

> **Links:** [arXiv](https://arxiv.org/abs/2603.15031) | [GitHub](https://github.com/MoonshotAI/Attention-Residuals) | [Website]()
> **Tags:** #DEEP_LEARNING

---

## Methodology

![[assets/attnres_fig.png]]

Standard PreNorm residual connections accumulate all layer outputs with **fixed unit weights**, causing uncontrolled hidden-state magnitude growth with depth (PreNorm dilution). AttnRes replaces this fixed accumulation with **softmax attention over preceding layer outputs**, allowing each layer to selectively aggregate earlier representations with learned, input-dependent weights.

### Full Attention Residuals

The core update rule for layer $l$:

$$\bm{h}_{l} = \sum_{i=0}^{l-1} \alpha_{i \to l} \cdot \bm{v}_{i}$$

where:
- $\bm{v}_0 = \bm{h}_1$ (token embedding), $\bm{v}_{i \geq 1} = f_i(\bm{h}_i)$ (layer output)
- $\bm{q}_l = \bm{w}_l$ — a **learned, input-independent** pseudo-query vector per layer (initialized to **zero** for stability)
- $\bm{k}_i = \bm{v}_i$ (keys equal values)
- Attention weights using RMSNorm on keys to prevent magnitude-dominated attention:

$$\alpha_{i \to l} = \frac{\exp\!\left(\bm{w}_l^\top \operatorname{RMSNorm}(\bm{k}_i)\right)}{\sum_{j=0}^{l-1} \exp\!\left(\bm{w}_l^\top \operatorname{RMSNorm}(\bm{k}_j)\right)}$$

Full AttnRes complexity: $O(L^2 d)$ arithmetic, $O(Ld)$ memory per token.

### Block Attention Residuals

To reduce the $O(Ld)$ memory and cross-stage communication overhead for large-scale training, layers are partitioned into $N$ blocks of $S = L/N$ layers each.

**Intra-block accumulation** — block $n$ sums its layer outputs:

$$\bm{b}_n = \sum_{j \in \mathcal{B}_n} f_j(\bm{h}_j)$$

with partial sums $\bm{b}_n^i$ tracking the running total after $i$ layers within the block.

**Inter-block attention** — for the $i$-th layer in block $n$, the value matrix is:

$$\mathbf{V} = \begin{cases} [\bm{b}_0, \bm{b}_1, \ldots, \bm{b}_{n-1}]^\top & i = 1 \\ [\bm{b}_0, \bm{b}_1, \ldots, \bm{b}_{n-1}, \bm{b}_n^{i-1}]^\top & i \geq 2 \end{cases}$$

where $\bm{b}_0 = \bm{h}_1$ (token embedding always included). Block AttnRes reduces memory/communication from $O(Ld)$ to $O(Nd)$. With $N \approx 8$, it recovers most of the gains of Full AttnRes.

### Pseudo-code (Block AttnRes)

```python
def block_attn_res(blocks, partial_block, proj, norm):
    V = torch.stack(blocks + [partial_block])        # [N+1, B, T, D]
    K = norm(V)
    logits = torch.einsum('d, n b t d -> n b t', proj.weight.squeeze(), K)
    h = torch.einsum('n b t, n b t d -> b t d', logits.softmax(0), V)
    return h
```

### Two-Phase Inference Strategy

**Phase 1 (parallel):** Batch all $S$ pseudo-queries in a block into a single matrix multiply against the cached block KV, returning outputs and log-sum-exp statistics.

**Phase 2 (sequential):** Compute intra-block partial-sum attention layer by layer; merge with Phase 1 via online softmax.

Per-layer memory access: $\left(\tfrac{N}{S}+5\right)d \approx 5.5d$ (typical: $L=128$, $N=8$, $S=16$), vs. $34d$ for mHC ($m=4$). End-to-end inference latency overhead: **< 2%**.

### Memory Access Comparison

| Method | Total I/O per token per layer | Typical ($L=128$, $N=8$) |
|--------|-------------------------------|--------------------------|
| Standard Residuals | $3d$ | $3d$ |
| mHC ($m=4$) | $(8m+2)d + 2m^2 + 4m$ | $34d$ |
| Full AttnRes (two-phase) | $(S+N)d$ | $24d$ |
| Block AttnRes (two-phase) | $\left(\tfrac{N}{S}+5\right)d$ | $5.5d$ |

---

## Experiment Setup

### Architecture

All models use the **Kimi Linear** architecture (MoE Transformer, Moonlight/DeepSeek-V3 style), interleaving Kimi Delta Attention (KDA) and MLA layers in a 3:1 ratio, each followed by an MoE FFN. AttnRes is the **only modification** to residual connections; all other components unchanged.

**Large-scale model:** 48B total / 3B activated parameters, 27 Transformer blocks (54 layers), 8 of 256 routed experts + 1 shared expert, Block AttnRes with $S=6$ layers/block → 9 blocks + embedding = 10 depth-wise sources.

### Training Recipe (1.4T run)

| Setting | Value |
|---------|-------|
| Pre-training tokens | 1T (WSD) + ~400B mid-training (annealing) |
| Context length | 4096 (pre-train), then 32K (extended) |
| Optimizer | Muon |
| LR schedule | WSD (Warmup-Stable-Decay) |
| Global batch size | 8M tokens |
| Pseudo-query init | **Zero** (ensures uniform weights at start) |

### Scaling Law Models

| Act. Params | Tokens | $L_b$ | $H$ | $d_\text{model}$ | $d_\text{ff}$ | lr | Batch |
|-------------|--------|--------|-----|------------------|---------------|----|-------|
| 194M | 38.7B | 12 | 12 | 896 | 400 | $2.99 \times 10^{-3}$ | 192 |
| 241M | 45.4B | 13 | 13 | 960 | 432 | $2.80 \times 10^{-3}$ | 256 |
| 296M | 62.1B | 14 | 14 | 1024 | 464 | $2.50 \times 10^{-3}$ | 320 |
| 436M | 87.9B | 16 | 16 | 1168 | 528 | $2.20 \times 10^{-3}$ | 384 |
| 528M | 119.0B | 17 | 17 | 1264 | 560 | $2.02 \times 10^{-3}$ | 432 |

Context length 8192, cosine LR schedule for all scaling runs.

---

## Results

### Scaling Laws

Fitted power-law $\mathcal{L} = A \times C^{-\alpha}$ (compute in PFLOP/s-days):

| Method | $A$ | $\alpha$ |
|--------|-----|----------|
| Baseline | 1.891 | 0.057 |
| Block AttnRes | 1.870 | 0.058 |
| Full AttnRes | 1.865 | 0.057 |

At 5.6 PFLOP/s-days: Block AttnRes reaches val loss 1.692 vs. baseline 1.714 — **1.25x compute advantage**.

### Validation Loss by Model Size

| Act. Params | Baseline | Block AttnRes | Full AttnRes | mHC(-lite) |
|-------------|----------|---------------|--------------|------------|
| 194M | 1.931 | 1.909 | **1.899** | 1.906 |
| 241M | 1.895 | 1.875 | 1.874 | **1.869** |
| 296M | 1.829 | 1.809 | **1.804** | 1.807 |
| 436M | 1.766 | 1.746 | **1.737** | 1.747 |
| 528M | 1.719 | 1.693 | **1.692** | 1.694 |

### Downstream Benchmarks (48B Model, 1.4T tokens)

| Category | Benchmark | Baseline | AttnRes |
|----------|-----------|----------|---------|
| General | MMLU | 73.5 | **74.6** |
| General | MMLU-Pro | **52.2** | **52.2** |
| General | GPQA-Diamond | 36.9 | **44.4** |
| General | BBH | 76.3 | **78.0** |
| General | ARC-Challenge | 64.6 | **65.7** |
| General | HellaSwag | 83.2 | **83.4** |
| General | TriviaQA | 69.9 | **71.8** |
| Math & Code | GSM8K | 81.7 | **82.4** |
| Math & Code | MGSM | 64.9 | **66.1** |
| Math & Code | Math (Minerva) | 53.5 | **57.1** |
| Math & Code | CMath | 84.7 | **85.1** |
| Math & Code | HumanEval | 59.1 | **62.2** |
| Math & Code | MBPP | 72.0 | **73.9** |
| Chinese | CMMLU | 82.0 | **82.9** |
| Chinese | C-Eval | 79.6 | **82.5** |

AttnRes matches or outperforms baseline on all 15 benchmarks. Largest gains: GPQA-Diamond (+7.5), Math (+3.6), HumanEval (+3.1).

### Ablations (436M model, val loss)

| Method / Variant | Val Loss |
|------------------|----------|
| Baseline (PreNorm) | 1.766 |
| DenseFormer (fixed weights) | 1.767 |
| SWA (window $W=8$) | 1.764 |
| mHC | 1.747 |
| Block AttnRes ($S=4$, $N\approx 8$) | 1.746 |
| Full AttnRes | **1.737** |
| Full AttnRes + input-dependent query | 1.731 |
| Full AttnRes, input-independent scalars | 1.749 |
| Full AttnRes, sigmoid instead of softmax | 1.741 |
| Full AttnRes, multihead ($H=16$) | 1.752 |
| Full AttnRes, no RMSNorm on keys | 1.743 |
| Block AttnRes, no RMSNorm on keys | 1.750 |

### Training Dynamics

- **Output magnitude:** Baseline shows monotonically growing hidden-state norms with depth (PreNorm dilution). Block AttnRes confines growth within each block, yielding a bounded periodic pattern.
- **Gradient magnitude:** Baseline has disproportionately large gradients in early layers. Block AttnRes yields substantially more uniform gradient distribution across depth.
- **Validation loss:** AttnRes achieves consistently lower loss throughout training; gap widens during the decay phase.

---

## Related Papers

- [[mhc]]
