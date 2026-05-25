# MEMORY.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Terminal output

Do **not** print LaTeX math (`$...$`, `$$...$$`, `\mathbf{}`, etc.) in terminal/chat responses — it renders as unreadable raw source for the user. LaTeX is only acceptable when writing to `.md` files, where Obsidian/GitHub render it properly. In chat, use plain text, Unicode symbols (e.g. `Δ`, `∈`, `ℝ`, `≤`, `→`, `⊙`), or ASCII fallbacks instead.

**Be concise in terminal Q&A.** Conversations in this vault are typically many short back-and-forth turns — the user asks follow-ups rather than wanting one big answer. Default to a focused, direct answer that addresses exactly what was asked. Skip background recaps, multi-section deep dives, and exhaustive enumeration unless the user explicitly asks for depth ("explain in detail", "give me everything", "diagram", etc.). Tables, bullets, and headers are fine when they actually help — but a 2-3 sentence answer is often the right answer.

## Purpose

This is an Obsidian-based paper reading vault. Each paper gets a **paper card** — a concise, technically detailed Markdown note. The vault is read in Obsidian and browsed on GitHub, so links and image embeds use **standard Markdown syntax** (not Obsidian `[[wiki-links]]` or `![[embeds]]`) — both renderers support it, GitHub does not render wiki-links.

## Repository Layout

```
papers/        ← one .md file per paper card
blogs/         ← one .md file per blog card (technical blog posts)
assets/        ← figures, screenshots, teaser images (shared by papers/ and blogs/)
tags/          ← one .md per tag category (DLM, RL, SPEC_DECODING, MLSYS, PRUNING)
TAGS.md        ← canonical list of available tags to choose from
templates/     ← paper_card_template.md, blog_card_template.md
```

## Creating a Paper Card

### Naming Convention
Name the file after the **method abbreviation** (not the paper title abbreviation). Examples: `wino.md`, `llada.md`, `rcd.md`, `dflash.md`. Use lowercase. Place in `papers/`.

### Title Line
The H1 title line inside the card (`# ...`) must be the **paper's original title** exactly as it appears on arXiv (including any subtitle after a colon), not the method abbreviation. The method abbreviation is only used for the filename and cross-references. Example: filename `wino.md`, H1 `# Wide-In, Narrow-Out: Revokable Decoding for Efficient and Effective DLLMs`.

### Workflow for each new paper

1. **Retrieve metadata** — find the arXiv ID, GitHub repo, and project website. Put all three links at the top of the card even if some are missing.

2. **Fetch figures — prefer many, prefer figures over prose.** Pull every figure that meaningfully conveys the authors' idea: the method/architecture diagram, key results plots, important ablations, illustrative visualizations. Do not stop at one. Prefer first-party sources (arXiv HTML, project website, GitHub repo); fall back to a clean crop of the original paper figure from the PDF / HTML. Skip purely decorative or branding images. Do **not** create synthetic summary screenshots, redraw the figure, or use unrelated screenshots. Save the primary figure to `assets/<method_name>_fig.png` and additional figures to `assets/<method_name>_fig2.png`, `assets/<method_name>_fig3.png`, etc. Embed each one **inline next to the section it explains** (not all at the top) using standard Markdown: `![](../assets/<method_name>_fig.png)` (relative path from `papers/`). If the figure is hosted in a GitHub repo, prefer fetching it through the GitHub file API as base64 and decode it locally instead of relying on raw GitHub downloads from the shell. If image extraction is blocked, fetch the arXiv source and inspect the figure file paths / `\includegraphics{...}` targets before falling back further.

3. **Fill the card** — follow `templates/paper_card_template.md`. **Prefer figures and tables over long prose:** if a method diagram or results plot already shows it, lean on the figure and use prose only for what the figure cannot convey (precise numbers, hyperparameters, formulas, caveats). Replace what would be multi-paragraph verbal explanations with a figure plus a 1–3 sentence caption. Prioritize:
   - Concrete algorithmic steps (enough to reproduce)
   - Exact benchmark names and numbers for key results
   - Training/inference hyperparameters when relevant
   - Present results as Markdown tables whenever possible instead of prose bullets
   - Skip story-telling from the abstract; focus on what is technically novel
   - If table extraction is hard, fetch the arXiv source and use the LaTeX table source to reconstruct the tables; optionally also save a clean screenshot of the original table to `assets/<method_name>_table1.png`, `assets/<method_name>_table2.png`, etc.

4. **Use LaTeX for all math** — inline: `$x$`, block: `$$\mathcal{L} = \ldots$$`. Never write math as plain text.

5. **Tags** — read `TAGS.md` and select the matching tags from that file only. Add them in YAML frontmatter and as inline `#tags`. Do not create new tags or update `TAGS.md` as part of a paper-card run. After adding the card, append a Markdown link to the relevant tag index files in `tags/` when those index files already exist (e.g. `- [llada](../papers/llada.md)`).

6. **Related papers** — add a `Related Papers` section only if there are related cards already present in `papers/` or `blogs/`. Each bullet must be a bare Markdown link with no description. Cross-folder links are allowed and encouraged:
   - paper → paper: `- [llada](llada.md)`
   - paper → blog: `- [phrase-summary](../blogs/phrase-summary.md)`
   - blog → paper: `- [method](../papers/method.md)`
   - blog → blog: `- [other-slug](other-slug.md)`

   If there are no related local cards, omit the entire section.

### Figure and Table Extraction Policy

- **Use multiple figures whenever they help.** A card with one method diagram + one results chart + one ablation plot is usually clearer than the same content as three paragraphs of prose. Skip only purely decorative or branding images.
- **Place figures inline next to the section they illustrate**, not stacked at the top of the card.
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
- **When the user asks to add a new paper (typically with an arXiv URL), run `./new_paper.sh <url>` from the repo root rather than doing the work inline.** The script is the canonical entrypoint — it handles logging, timeouts, and the full paper-card workflow. Only write the card inline if the user explicitly opts out of the script.

## Creating a Blog Card

Blogs live in `blogs/` and use `templates/blog_card_template.md`. The workflow mirrors the paper workflow with a few differences:

- **Filename**: lowercase kebab-case **phrase summary** of the blog's contents (2-6 words), not a URL slug. E.g. `why-attention-is-quadratic.md`, `gradient-checkpointing-tradeoffs.md`. Place in `blogs/`.
- **Title line**: the post's original title verbatim.
- **Tags**: choose from `TAGS.md` only — same canonical set as papers. Same rule: do not create new tags during a blog-card run.
- **Figures**: same policy as papers — pull every figure that meaningfully conveys the author's idea (method diagrams, key result charts, illustrative tradeoffs), not just the hero image. If the blog itself only has a teaser + decorative branding, follow its links to the canonical docs/repo/paper and pull the real method figures from there. Save to `assets/<slug>_fig.png`, `<slug>_fig2.png`, etc., and embed inline next to the relevant section with `![](../assets/<slug>_fig.png)`. Skip purely decorative or branding images. Do not synthesize substitutes if extraction fails.
- **Related section**: cross-folder links allowed (see paper-card rule above).
- **`new_blog.sh`** is the canonical entrypoint, mirroring `new_paper.sh`. It accepts any HTTP(S) URL, runs in a detached `tmux` session, and writes logs/status to `logs/new_blog/`. Override the default 30 minute timeout with `NEW_BLOG_TIMEOUT_SECONDS`.
- **When the user asks to add a new blog (typically with a URL that is not arXiv), run `./new_blog.sh <url>` from the repo root rather than doing the work inline.** Only write the card inline if the user explicitly opts out of the script.
