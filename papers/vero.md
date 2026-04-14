---
tags:
  - RL
  - VISION
arxiv: "https://arxiv.org/abs/2604.04917"
github: "https://github.com/zlab-princeton/vero"
website: "https://vero-reasoning.github.io/"
year: 2025
read: false
---

# Vero

> **Links:** [arXiv](https://arxiv.org/abs/2604.04917) | [GitHub](https://github.com/zlab-princeton/vero) | [Website](https://vero-reasoning.github.io/)
> **Tags:** #RL #VISION

---

## Methodology

![[assets/vero_fig.png]]

Vero is a single-stage, fully open RL recipe for training vision-language models (VLMs) across diverse visual reasoning tasks. The key ingredients are:

1. **Vero-600K dataset** — 600K samples from 59 datasets across 6 task categories (100K each), curated with heuristic filtering, LLM-based ambiguity/verifiability filtering, and answer normalization.
2. **Task-routed multi-verifier reward** — ten specialized verifiers matched to answer format (string match, MC, numeric via math-verify, grounding IoU/F1, LLM-as-judge for open-ended).
3. **GSPO optimization** — replaces per-token importance ratios with a sequence-level ratio to improve training stability.

### Dataset: Vero-600K

| Category | # Datasets | Samples |
|---|---|---|
| Chart & OCR | 9 | ~100K |
| STEM | 13 | ~100K |
| Spatial & Action | 8 | ~100K |
| Knowledge & Recognition | 12 | ~100K |
| Grounding, Counting & Search | 11 | ~100K |
| Captioning & Instruction Following | 6 | ~100K |

**Data curation steps:**
- Heuristic screening: >1K examples, image resolution >200K pixels
- Manual quality inspection (error rate <5%)
- LLM-based filtering for ambiguity and verifiability
- Answer normalization (units, formatting, etc.)

### Reward Design

$$R(y,y^*) = (1-\alpha)\,R_\text{acc}(y,y^*) + \alpha\,R_\text{fmt}(y) + R_\text{overlong}(y)$$

with $\alpha = 0.2$.

- **$R_\text{acc}$**: task-routed verifier (string match, MC exact match, numeric via math-verify, grounding IoU/F1, LLM-judge for open-ended captions)
- **$R_\text{fmt}$**: binary reward for `<think>...</think><answer>...</answer>` format
- **$R_\text{overlong}$**: soft penalty that linearly ramps in the buffer zone $[L_\text{max}-2048,\, L_\text{max}]$

### GSPO Optimization

GSPO replaces GRPO's per-token importance ratios with a **sequence-level ratio**. The sequence-average log-probability difference is:

$$\bar{\Delta}_i = \frac{1}{|y_i|}\sum_t \left[\log\pi_\theta(y_{i,t}) - \log\pi_{\theta_\text{old}}(y_{i,t})\right]$$

This value is used to form token-level ratios for clipped policy optimization (no KL penalty, asymmetric clipping with $\varepsilon_\text{high} > \varepsilon_\text{low}$). GSPO produces more stable entropy than GRPO ($0.58\pm0.11$ vs $0.50\pm0.11$) and DAPO ($0.22\pm0.15$).

---

## Experiment Setup

- **Base models**: Qwen3-VL-8B-Instruct, Qwen3-VL-8B-Thinking, Qwen2.5-VL-7B-Instruct, MiMo-VL-7B-SFT
- **Training**: single-stage on-policy RL with uniform task category sampling
- **Evaluation suite**: VeroEval — 30 benchmarks across the 6 task categories (5 per category)
- **Baselines**: LLaVA-OV-1.5-RL, MiMo-VL-7B-RL, Qwen3-VL-8B-Thinking (proprietary RL recipe)

---

## Results

### Main Results (VeroEval — 30 benchmarks)

| Model | Chart & OCR | STEM | Spatial & Action | Knowledge & Recog. | Grounding & Count. | Caption & Instr. | Avg |
|---|---|---|---|---|---|---|---|
| Qwen3-VL-8B-Instruct (base) | 61.3 | 57.3 | 62.6 | 52.3 | 58.5 | 78.2 | 60.7 |
| Vero-Qwen3I-8B | 69.8 | 63.7 | 66.3 | 53.3 | 63.8 | 83.8 | 66.0 |
| Qwen3-VL-8B-Thinking (base) | — | — | — | — | — | — | 62.3 |
| Vero-Qwen3T-8B | — | — | — | — | — | — | 65.9 |
| Vero-Qwen25-7B | — | — | — | — | — | — | 57.9 |
| Vero-MiMo-7B | — | — | — | — | — | — | 63.3 |

Vero-Qwen3I-8B outperforms proprietary Qwen3-VL-8B-Thinking on 23/30 benchmarks; Vero-Qwen3T-8B on 24/30. Gains of +3.6 to +5.3 points are consistent across all four base models.

### Ablations

**Mixture weighting strategy** (Vero-Qwen25-7B, overall avg gain over base):

| Weighting | Avg Gain |
|---|---|
| Uniform | +5.8 |
| Difficulty-weighted | +5.2 |
| Size-weighted | +5.2 |
| Length-weighted | +4.8 |

**Reward design** (Chart & OCR category avg):

| Reward | Score |
|---|---|
| Task-routed multi-verifier | 61.9 |
| Generic math-verify only | 61.4 |

**RL algorithm** (entropy stability during training):

| Algorithm | Entropy |
|---|---|
| GSPO | $0.58\pm0.11$ |
| GRPO | $0.50\pm0.11$ |
| DAPO | $0.22\pm0.15$ |

Single-task vs. multi-task training: training exclusively on one category causes negative transfer of -15.8 to -35.5 points on Captioning & Instruction Following, confirming that data diversity is the critical ingredient.

---

## Related Papers

- [[deepcrl]]
