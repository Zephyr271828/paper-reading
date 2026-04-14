---
tags:
  - VISION
arxiv: https://arxiv.org/abs/2508.10104
github: ""
website: ""
year: 2025
read: false
---

# DINOv3

> **Links:** [arXiv](https://arxiv.org/abs/2508.10104) | [GitHub]() | [Website]()
> **Tags:** #VISION

---

## Methodology

![](../assets/dinov3_fig.png)

### Overview

DINOv3 is a 7B-parameter self-supervised vision foundation model that scales DINOv2 training along three axes: dataset curation (LVD-1689M), model scale (ViT-7B), and a novel **Gram anchoring** regularization to prevent dense feature degradation during extended training.

### Architecture: ViT-7B

| Property | DINOv2 (ViT-g) | DINOv3 (ViT-7B) |
|---|---|---|
| Parameters | 1.1B | 6.7B |
| Transformer blocks | 40 | 40 |
| Embedding dim | 1536 | 4096 |
| Patch size | 14 | 16 |
| Positional embeddings | Learnable | RoPE |
| Attention heads | 24 | 32 (128-dim each) |
| FFN hidden dim | 4096 | 8192 |
| Register tokens | 4 | 4 |
| DINO prototypes | 128k | 256k |

RoPE uses **box jittering**: the coordinate box $[-1,1]$ is randomly scaled to $[-s,s]$ with $s \sim [0.5, 2]$ per crop, improving resolution generalization.

### Dataset: LVD-1689M

Combines 1.689B images from three sources:
1. **Clustering-based curation**: internet images filtered by cluster membership
2. **Retrieval-based augmentation**: nearest-neighbor retrieval from curated seeds
3. **ImageNet-1k**: injected into 10% of training batches for global discriminability

Ablation (200k iters, ViT-g, linear probing):

| Dataset | IN1k k-NN | IN1k Linear | ObjectNet | iNat21 | Paris-H |
|---|---|---|---|---|---|
| Raw | 80.1 | 84.8 | 70.3 | 70.1 | 63.3 |
| Clustering | 79.4 | 85.4 | 72.3 | 81.3 | 85.2 |
| Retrieval | 84.0 | 86.7 | 70.7 | 86.0 | 82.7 |
| **LVD-1689M** | **84.6** | **87.2** | **72.8** | **87.0** | **85.9** |

### Training Pipeline

**Phase 1 — Pre-training (0–1M iterations):**

$$\mathcal{L}_\text{Pre} = \mathcal{L}_\text{DINO} + \mathcal{L}_\text{iBOT} + 0.1 \cdot \mathcal{L}_\text{Koleo}$$

- 256 GPUs, batch size 4096, constant LR schedule, AdamW optimizer
- Multi-crop: 2 global crops ($256^2$) + 8 local crops ($112^2$)

**Phase 2 — Gram Anchoring Refinement (1M+ iterations):**

After 1M iterations, segmentation performance degrades despite improving classification. Patch cosine similarity maps become noisy and poorly localized. Gram anchoring directly regularizes **pairwise patch similarity structure** without constraining individual features.

Let $\mathbf{X}_S \in \mathbb{R}^{P \times d}$ be the $\ell_2$-normalized patch features of the student and $\mathbf{X}_G \in \mathbb{R}^{P \times d}$ be those of a frozen **Gram teacher** (a snapshot of the EMA teacher from ~100k–200k iters):

$$\mathcal{L}_\text{Gram} = \left\| \mathbf{X}_S \mathbf{X}_S^\top - \mathbf{X}_G \mathbf{X}_G^\top \right\|_F^2$$

The refinement objective is:

$$\mathcal{L}_\text{Ref} = w_D \mathcal{L}_\text{DINO} + \mathcal{L}_\text{iBOT} + w_{DK} \mathcal{L}_\text{DKoleo} + w_\text{Gram} \mathcal{L}_\text{Gram}$$

- Gram teacher updated (reset to current EMA teacher) every 10k iterations
- Loss applied only on global crops

**High-Resolution Gram ($\mathcal{L}_\text{HRef}$):**

The Gram teacher processes images at **2x the normal resolution**, then bicubic-downsamples the feature map to match student output size. This distills finer patch consistency into training, yielding +2 mIoU on ADE20k over standard $\mathcal{L}_\text{Ref}$:

| Method | Teacher Iter | Res | IN1k Linear | ADE mIoU | NYU RMSE |
|---|---|---|---|---|---|
| Baseline | — | — | **88.2** | 50.3 | 0.307 |
| GRAM | 200k | x1 | 88.0 | 53.6 | 0.285 |
| **GRAM** | **200k** | **x2** | **88.0** | **55.7** | **0.281** |
| GRAM | 100k | x2 | 87.9 | 55.7 | 0.284 |
| GRAM | 1M | x2 | 88.1 | 54.9 | 0.290 |

**Phase 3 — Post-training:**
- **High-resolution adaptation**: 10k iterations with mixed resolutions {512, 768}
- **Distillation** to student family: 1M iterations + 250k cooldown per student

### Model Family (Distilled Students)

| Model | Params | GFLOPs @256px | GFLOPs @512px |
|---|---|---|---|
| ViT-S | 21M | 12 | 63 |
| ViT-S+ | 29M | 16 | 79 |
| ViT-B | 86M | 47 | 216 |
| ViT-L | 300M | 163 | 721 |
| ViT-H+ | 840M | 450 | 1903 |
| ViT-7B | 6716M | 3550 | 14515 |
| CNX-Tiny | 29M | 5 | 20 |
| CNX-Small | 50M | 11 | 46 |
| CNX-Base | 89M | 20 | 81 |
| CNX-Large | 198M | 38 | 152 |

---

## Experiment Setup

**Backbone:** ViT-7B frozen for all downstream evaluations (no backbone fine-tuning unless noted).

**Dense linear probing:** Single linear layer on frozen patch outputs; input adapted to 1024 patch tokens (512x512 for patch size 16).

**System-level tasks:** Frozen DINOv3 + task-specific decoder:
- Detection: Plain-DETR (100M trainable params), pre-trained on Objects365 then COCO, resolution 2048
- Segmentation: ViT-Adapter + Mask2Former (927M trainable), pre-trained on COCO-Stuff + Hypersim, then ADE20k at resolution 896
- Depth: DPT head following Depth Anything V2, trained on DAv2 synthetic data at 1024x768, backbone frozen

---

## Results

### Dense Linear Probing (Frozen Backbone)

| Method | ViT | ADE20k | Cityscapes | VOC | NYUv2 RMSE↓ | KITTI RMSE↓ |
|---|---|---|---|---|---|---|
| AM-RADIOv2.5 | g/14 | 53.0 | 78.4 | 85.4 | 0.340 | 2.918 |
| PEspatial | G/14 | 49.3 | 73.2 | 82.7 | 0.362 | 3.082 |
| SigLIP 2 | g/16 | 42.7 | 64.8 | 72.7 | 0.494 | 3.273 |
| DINOv2 | g/14 | 49.5 | 75.6 | 83.1 | 0.372 | 2.624 |
| Web-DINO | 7B/14 | 42.7 | 68.3 | 76.1 | 0.466 | 3.158 |
| **DINOv3** | **7B/16** | **55.9** | **81.1** | **86.6** | **0.309** | **2.346** |

### ImageNet-1k Linear Probing (Frozen Backbone)

| Method | ViT | Val | V2 | ReaL | R | S | A | C↓ | Obj. |
|---|---|---|---|---|---|---|---|---|---|
| PEcore | G/14 | **89.3** | **81.6** | 90.4 | **92.2** | **71.9** | **89.0** | 22.7 | **80.2** |
| SigLIP 2 | g/16 | 89.1 | 81.6 | **90.5** | **92.2** | 71.8 | 84.6 | 30.0 | 78.6 |
| DINOv2 | g/14 | 87.3 | 79.5 | 89.9 | 81.1 | 65.4 | 81.7 | 24.1 | 66.4 |
| **DINOv3** | **7B/16** | **88.4** | **81.4** | **90.4** | **91.1** | **71.3** | **86.9** | **19.6** | **79.0** |

### 3D Correspondence (Probe3D Protocol, Recall)

| Method | ViT | NAVI (Geometric) | SPair (Semantic) |
|---|---|---|---|
| AM-RADIOv2.5 | g/14 | 59.4 | 56.8 |
| DINOv2 | g/14 | 60.1 | 56.1 |
| **DINOv3** | **7B/16** | **64.4** | **58.7** |

### Unsupervised Object Discovery (TokenCut, CorLoc)

| Method | ViT | VOC07 | VOC12 | COCO |
|---|---|---|---|---|
| DINO | B/16 | 60.1 | 64.4 | 50.5 |
| DINOv2 | g/14 | 55.6 | 60.4 | 45.4 |
| **DINOv3** | **7B/16** | **66.1** | **69.5** | **55.1** |

### Video Segmentation Tracking (J&F-mean)

| Method | DAVIS-S | DAVIS-M | DAVIS-L | YT-VOS-L | MOSE-L |
|---|---|---|---|---|---|
| AM-RADIOv2.5 | 66.5 | 77.3 | 81.4 | 79.2 | 54.3 |
| DINOv2 | 63.9 | 73.6 | 76.6 | 74.6 | 48.5 |
| **DINOv3** | **71.1** | **79.7** | **83.3** | **80.7** | **55.6** |

### System-Level: Object Detection (Plain-DETR + Frozen Backbone)

| Model | Trainable Params | COCO Simple | COCO TTA | COCO-O mAP | COCO-O ER |
|---|---|---|---|---|---|
| EVA-02 + Co-DETR | 300M | 65.4 | 65.9 | 63.7 | 34.3 |
| PEspatial + DETA | 2B | 65.3 | 66.0 | 64.0 | 34.7 |
| **DINOv3 + Plain-DETR (frozen)** | **100M** | **65.6** | **66.1** | **66.4** | **36.8** |

### System-Level: Semantic Segmentation (ADE20k mIoU)

| Model | Trainable Params | Simple | TTA |
|---|---|---|---|
| BEIT3 | 1.6B | 62.0 | 62.8 |
| InternImage-H | 1.3B | 62.5 | 62.9 |
| ONE-PEACE | 2.2B | 62.0 | **63.0** |
| **DINOv3 (frozen backbone)** | **927M** | **62.6** | **63.0** |

### System-Level: Monocular Depth (DPT + Frozen Backbone)

| Method | NYUv2 ARel↓ | NYUv2 d1↑ | KITTI ARel↓ | KITTI d1↑ | ETH3D ARel↓ | ETH3D d1↑ |
|---|---|---|---|---|---|---|
| Marigold | 5.5 | 96.4 | 9.9 | 91.6 | 6.5 | 96.0 |
| DAv2 ViT-g (fine-tuned) | 4.4 | 97.9 | 7.5 | 94.7 | 13.1 | 86.5 |
| **DINOv3 (frozen)** | **4.3** | **98.0** | **7.3** | **96.7** | **5.4** | **97.5** |

### Distilled Family vs. Baselines

| Size | Model | IN-ReaL | IN-R | Obj. | Oxford-H | ADE20k | NYU↓ | DAVIS |
|---|---|---|---|---|---|---|---|---|
| S | DINOv2 | 87.3 | 54.0 | 47.8 | 39.5 | 45.5 | 0.446 | 73.6 |
| S | DINOv3 | 87.0 | 60.4 | 50.9 | 49.5 | 47.0 | 0.403 | 72.7 |
| B | DINOv2 | 89.0 | 68.4 | 57.3 | 51.0 | 48.4 | 0.416 | 72.9 |
| B | DINOv3 | 89.3 | 76.7 | 64.1 | 58.5 | 51.8 | 0.373 | 77.2 |
| L | DINOv2 | 89.7 | 79.1 | 64.7 | 55.7 | 48.8 | 0.394 | 73.4 |
| L | DINOv3 | **90.2** | **88.1** | **74.8** | **63.1** | **54.9** | **0.352** | **79.9** |
