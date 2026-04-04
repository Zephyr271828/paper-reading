# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Purpose

This is an Obsidian-based paper reading vault. Each paper gets a **paper card** — a concise, technically detailed Markdown note. The vault is read in Obsidian, which renders wiki-links (`[[name]]`) and the Graph View to visualize connections between papers.

## Repository Layout

```
papers/        ← one .md file per paper card
assets/        ← figures, screenshots, teaser images
tags/          ← one .md per tag category (DLM, RL, SPEC_DECODING, MLSYS, PRUNING)
templates/     ← paper_card_template.md
```

## Creating a Paper Card

### Naming Convention
Name the file after the **method abbreviation** (not the paper title abbreviation). Examples: `wino.md`, `llada.md`, `rcd.md`, `dflash.md`. Use lowercase. Place in `papers/`.

### Workflow for each new paper

1. **Retrieve metadata** — find the arXiv ID, GitHub repo, and project website. Put all three links at the top of the card even if some are missing.

2. **Fetch a figure** — use `WebFetch` or screenshot tools to retrieve the teaser figure or methodology diagram from the arXiv abstract page (`https://arxiv.org/abs/XXXX.XXXXX`) or the paper's website. Save the image to `assets/<method_name>_fig.png` and embed it with `![[assets/<method_name>_fig.png]]`.

3. **Fill the card** — follow `templates/paper_card_template.md`. Prioritize:
   - Concrete algorithmic steps (enough to reproduce)
   - Exact benchmark names and numbers for key results
   - Training/inference hyperparameters when relevant
   - Skip story-telling from the abstract; focus on what is technically novel

4. **Use LaTeX for all math** — inline: `$x$`, block: `$$\mathcal{L} = \ldots$$`. Never write math as plain text.

5. **Tags** — add YAML frontmatter tags and inline `#tags` from the set below. After adding the card, append a wiki-link to the relevant tag index files in `tags/`.

6. **Related papers** — at the bottom, add `[[method_name]]` wiki-links to other cards in `papers/` that are related, with a one-line note on the connection.

## Tag System

Tags live in two places:
- YAML frontmatter: `tags: [DLM, SPEC_DECODING]`
- Inline in the card body: `#DLM #SPEC_DECODING`
- Tag index files in `tags/` (e.g., `tags/DLM.md`) contain a bullet list of `[[method_name]]` wiki-links to all papers with that tag.

Current tags:

| Tag | Meaning |
|-----|---------|
| `DLM` | Discrete / diffusion language models |
| `RL` | Reinforcement learning for LLMs |
| `SPEC_DECODING` | Speculative decoding / inference acceleration |
| `MLSYS` | ML systems (memory, kernels, scheduling) |
| `PRUNING` | Pruning and sparsity |

**Adding new tags:** If a paper belongs to a research area not covered by the existing tags, create a new tag — but keep the tag set canonical. A new tag is justified when it represents a coherent, recurring research area that multiple papers could share (e.g., `QUANT` for quantization, `ARCH` for novel architecture design). Do not create one-off tags for a single paper. To add: create `tags/NEWTAG.md` following the same format, add the tag to the table above, and update this file.

## Obsidian Notes

- **Graph View** shows the relation map between papers via `[[wiki-links]]`.
- Tag index files (`tags/*.md`) act as hub nodes in the graph, clustering papers by topic.
- Keep wiki-link targets consistent: always link to the method abbreviation filename without extension (e.g., `[[llada]]` not `[[LLaDA]]`).
- The `assets/` folder should be set as an attachment folder in Obsidian settings so embedded images resolve correctly.
