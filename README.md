# paper-reading

A personal paper reading vault. View with [Obsidian](https://obsidian.md) for graph-based navigation between papers.

## Installation

**1. Install Claude Code**

```bash
curl -fsSL https://claude.ai/install.sh | bash
```

**2. Clone the repo and make the script executable**

```bash
git clone <repo_url>
cd paper-reading
chmod +x new_paper.sh
```

**3. Open in Obsidian**

Download [Obsidian](https://obsidian.md), then: **Open folder as vault** → select this directory.

## Adding a paper

```bash
# Using Codex (default)
./new_paper.sh https://arxiv.org/abs/XXXX.XXXXX

# Using Claude Code
./new_paper.sh --claude https://arxiv.org/abs/XXXX.XXXXX
```

The script runs in the background and streams output to `logs/new_paper/<timestamp>.log`. The chosen engine generates a paper card in `papers/`, fetches a figure into `assets/`, and updates the tag index in `tags/`.

To use Codex, install it first:

```bash
npm install -g @openai/codex
```

## Viewing

Open the vault in Obsidian. Use **Graph View** to explore connections between papers and tag hubs.
