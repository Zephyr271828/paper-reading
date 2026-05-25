---
tags:
  - MLSYS
arxiv: https://arxiv.org/abs/2309.06180
github: https://github.com/vllm-project/vllm
website: ""
year: 2023
read: false
---

# Efficient Memory Management for Large Language Model Serving with PagedAttention

> **Links:** [arXiv](https://arxiv.org/abs/2309.06180) | [GitHub](https://github.com/vllm-project/vllm)
> **Tags:** #MLSYS

---

## Methodology

![](../assets/vllm_fig.png)

### Core Problem

KV cache for LLM inference is large (up to 1.7 GB for a single LLaMA-13B sequence) and must be allocated for the full context length upfront. Prior systems waste **60–80% of GPU memory** due to:
1. **Internal fragmentation**: pre-allocated slots for the maximum possible sequence length.
2. **External fragmentation**: discontiguous free blocks across requests.
3. **Reservation waste**: memory held for potential future tokens that may never arrive.

### PagedAttention

Inspired by OS virtual memory paging. The KV cache for each sequence is divided into fixed-size **blocks** (default 16 tokens per block). Blocks need not be contiguous in GPU memory.

Each request has a **block table**: a mapping from logical block indices to physical block IDs.

$$\text{KV}[\text{token } t] = \text{PhysicalMem}\bigl[\text{block\_table}[\lfloor t/B \rfloor][\, t \bmod B \,]\bigr]$$

- $B$: block size (tokens per block, default 16).
- $\text{block\_table}[i]$: physical block ID for logical block $i$.
- $\lfloor t/B \rfloor$: which logical block token $t$ belongs to.
- $t \bmod B$: offset within that block.

During the attention kernel, the hardware gathers each block from its physical address via the block table. Attention is computed block-by-block over the paged cache.

**Memory waste after PagedAttention:** only the last (partially filled) block of each sequence wastes space — under **4% overhead** in practice.

### Memory Sharing via Copy-on-Write

**Parallel sampling** (multiple outputs from one prompt): all output sequences share the same physical blocks for the prompt portion. Each block has a **reference count**. When a sequence writes a new token that fills a shared block, the runtime triggers **copy-on-write** (CoW) to duplicate the block before modifying it.

**Prefix sharing**: system prompts or few-shot prefixes common across requests can share physical blocks, reducing recomputation and memory.

**Beam search**: candidate beams share parent-node blocks; only diverging tokens get new blocks.

### Scheduling and Preemption

vLLM uses **continuous batching** (iteration-level scheduling): requests are added/removed from the active batch at every decoding step, unlike static batching that waits for the whole batch to finish.

When memory is exhausted, the scheduler **preempts** requests using one of two policies:
- **Swapping**: copy KV blocks to CPU memory, restore later.
- **Recomputation**: evict KV blocks entirely, recompute the KV cache on demand (cheaper than swapping for short sequences).

The scheduler uses FCFS ordering to prevent starvation.

---

## Experiment Setup

- **Models**: OPT-13B, OPT-66B, LLaMA-13B
- **Hardware**: NVIDIA A100 (80 GB), A10G (24 GB)
- **Baselines**: HuggingFace Transformers (HF), HuggingFace TGI, Orca (OSDI '22)
- **Workloads**: ShareGPT conversation dataset (variable-length real traces), Alpaca dataset
- **Block size**: 16 tokens (default)
- **Metrics**: throughput (requests/sec at matched latency percentiles)

---

## Results

### Throughput vs. HuggingFace Transformers (A100, OPT-13B)

| Setting | vLLM speedup over HF Transformers |
|---|---|
| Single output (n=1) | 14–24× |
| Three parallel outputs (n=3) | 8.5–15× |

### Throughput vs. HuggingFace TGI (A100, OPT-13B)

| Setting | vLLM speedup over TGI |
|---|---|
| Single output (n=1) | 2.2–2.5× |
| Three parallel outputs (n=3) | 3.3–3.5× |

*Both tables: speedup measured at equal latency SLO; ranges reflect different request arrival rates.*

### Memory Efficiency

| System | KV Cache Memory Waste |
|---|---|
| Prior systems (static allocation) | 60–80% |
| vLLM (PagedAttention) | < 4% |

### Beam Search Throughput Benefit

Memory sharing in beam search reduces memory overhead by up to **55%**, yielding up to **2.2× throughput improvement** over systems without sharing.

### Real-World Deployment (LMSYS Chatbot Arena)

vLLM achieved up to **30× higher throughput** vs. the initial HuggingFace backend, reducing GPU count by **50%** at the same traffic level.

---

## Related Papers

- [flashattn](flashattn.md)
- [tvm](tvm.md)
- [mxnet](mxnet.md)
