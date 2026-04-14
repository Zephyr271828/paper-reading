# MEMORY.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Purpose

This is an Obsidian-based paper reading vault. Each paper gets a **paper card** — a concise, technically detailed Markdown note. The vault is read in Obsidian and browsed on GitHub, so links and image embeds use **standard Markdown syntax** (not Obsidian `[[wiki-links]]` or `![[embeds]]`) — both renderers support it, GitHub does not render wiki-links.

## Repository Layout

```
papers/        ← one .md file per paper card
assets/        ← figures, screenshots, teaser images
tags/          ← one .md per tag category (DLM, RL, SPEC_DECODING, MLSYS, PRUNING)
TAGS.md        ← canonical list of available tags to choose from
templates/     ← paper_card_template.md
```

## Creating a Paper Card

### Naming Convention
Name the file after the **method abbreviation** (not the paper title abbreviation). Examples: `wino.md`, `llada.md`, `rcd.md`, `dflash.md`. Use lowercase. Place in `papers/`.

### Title Line
The H1 title line inside the card (`# ...`) must be the **paper's original title** exactly as it appears on arXiv (including any subtitle after a colon), not the method abbreviation. The method abbreviation is only used for the filename and cross-references. Example: filename `wino.md`, H1 `# Wide-In, Narrow-Out: Revokable Decoding for Efficient and Effective DLLMs`.

### Workflow for each new paper

1. **Retrieve metadata** — find the arXiv ID, GitHub repo, and project website. Put all three links at the top of the card even if some are missing.

2. **Fetch a figure** — prefer the official teaser figure or methodology diagram from the arXiv HTML page, project website, GitHub repo, or another first-party source. If a direct asset is unavailable, take a **clean crop of the original paper figure** from the PDF / HTML. Do **not** create synthetic summary screenshots, redraw the figure, or use unrelated screenshots. Save the image to `assets/<method_name>_fig.png` and embed it with standard Markdown: `![](../assets/<method_name>_fig.png)` (relative path from `papers/`). If the figure is hosted in a GitHub repo, prefer fetching it through the GitHub file API as base64 and decode it locally instead of relying on raw GitHub downloads from the shell. If image extraction is blocked, fetch the arXiv source and inspect the figure file paths / `\includegraphics{...}` targets before falling back further.

3. **Fill the card** — follow `templates/paper_card_template.md`. Prioritize:
   - Concrete algorithmic steps (enough to reproduce)
   - Exact benchmark names and numbers for key results
   - Training/inference hyperparameters when relevant
   - Present results as Markdown tables whenever possible instead of prose bullets
   - Skip story-telling from the abstract; focus on what is technically novel
   - If table extraction is hard, fetch the arXiv source and use the LaTeX table source to reconstruct the tables; optionally also save a clean screenshot of the original table to `assets/<method_name>_table1.png`, `assets/<method_name>_table2.png`, etc.

4. **Use LaTeX for all math** — inline: `$x$`, block: `$$\mathcal{L} = \ldots$$`. Never write math as plain text.

5. **Tags** — read `TAGS.md` and select the matching tags from that file only. Add them in YAML frontmatter and as inline `#tags`. Do not create new tags or update `TAGS.md` as part of a paper-card run. After adding the card, append a Markdown link to the relevant tag index files in `tags/` when those index files already exist (e.g. `- [llada](../papers/llada.md)`).

6. **Related papers** — add a `Related Papers` section only if there are related cards already present in `papers/`. If the section is included, each bullet must be only the Markdown link, e.g. `- [llada](llada.md)` (relative to the current card, which is also in `papers/`). Do not add descriptions or commentary. If there are no related local cards, omit the entire section.

### Figure and Table Extraction Policy

- Preferred figure order: first-party image asset > clean crop from the paper PDF / arXiv HTML > figure path recovered from arXiv source.
- Preferred results-table order: reconstruct as Markdown table from the paper / arXiv source > optionally add a clean table screenshot as a supplement.
- For GitHub-hosted figure assets in automated runs, prefer connector-native file fetches with base64 content and decode locally into `assets/`. Avoid raw `curl` downloads unless outbound shell network access has already been confirmed.
- If screenshots are difficult or unreliable, fetch the arXiv source tarball and inspect:
  - LaTeX table definitions (`table`, `tabular`, `table*`) to recover the exact table contents.
  - Figure file paths (`\includegraphics{...}`) to locate the correct figure assets.
- Never replace a recoverable figure or table with a hand-made summary graphic.

## Tag System

Tags live in two places:
- YAML frontmatter: `tags: [DLM, SPEC_DECODING]`
- Inline in the card body: `#DLM #SPEC_DECODING`
- Tag index files in `tags/` (e.g., `tags/DLM.md`) contain a bullet list of `[method_name](../papers/method_name.md)` Markdown links to all papers with that tag.
- Available tags are defined in `TAGS.md`. Always read `TAGS.md` to find the current allowed tags instead of relying on a hardcoded list in this file.
- During a paper-card run, only choose from tags already listed in `TAGS.md`.
- Do not create new tags, modify `TAGS.md`, or invent near-duplicate tag names during a paper-card run.
- If a selected tag has a matching index file in `tags/`, append the paper Markdown link there. If no matching index file exists yet, leave it alone.

## Link & Embed Syntax

- Use **standard Markdown** everywhere so both Obsidian and GitHub render correctly:
  - Links: `[method](method.md)` for same-folder (papers → papers), `[method](../papers/method.md)` from `tags/` → `papers/`.
  - Image embeds: `![](../assets/method_fig.png)` from `papers/` → `assets/`.
- Do **not** use `[[wiki-links]]` or `![[embeds]]`; GitHub renders them as literal text.
- Keep link targets consistent: always use the lowercase method abbreviation filename (e.g., `[llada](llada.md)` not `[LLaDA](LLaDA.md)`).
- Obsidian Graph View still works with Markdown links, and tag index files (`tags/*.md`) still act as hub nodes clustering papers by topic.
- The `assets/` folder should be set as an attachment folder in Obsidian settings so embedded images resolve correctly.

## Automation Notes

- `new_paper.sh` launches the selected agent inside a detached `tmux` session so the run survives terminal shutdown and returns immediately.
- `new_paper.sh` validates the arXiv URL first, writes or appends logs at `logs/new_paper/<arxiv_id>.log`, and writes a final status file at `logs/new_paper/<arxiv_id>.status`. Watch progress with `tail -f`, inspect the status file, or reattach to the printed `tmux` session name. Override the default 30 minute timeout with `NEW_PAPER_TIMEOUT_SECONDS`.
