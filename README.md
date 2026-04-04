# paper-reading

A personal paper reading vault. View with [Obsidian](https://obsidian.md) for graph-based navigation between papers.

## Installation

**1. Install Claude Code**

```bash
curl -fsSL https://claude.ai/install.sh | bash
```

**2. Install tmux**

```bash
brew install tmux
```

**3. Clone the repo and make the script executable**

```bash
git clone <repo_url>
cd paper-reading
chmod +x new_paper.sh
```

**4. Open in Obsidian**

Download [Obsidian](https://obsidian.md), then: **Open folder as vault** → select this directory.

## Adding a paper

```bash
# Using Codex (default)
./new_paper.sh https://arxiv.org/abs/XXXX.XXXXX

# Using Claude Code
./new_paper.sh --claude https://arxiv.org/abs/XXXX.XXXXX
```

The script validates the arXiv URL first, then starts a detached `tmux` session and streams output to `logs/new_paper/<arxiv_id>.log`. Reattach with `tmux attach -t <session_name>`, inspect progress with `tail -f logs/new_paper/<arxiv_id>.log`, or read the final result from the printed `logs/new_paper/<arxiv_id>.<run_id>.status` path. If the log file already exists for that arXiv ID, the script prints a warning and appends to the existing log.

Set `NEW_PAPER_TIMEOUT_SECONDS` to override the default 30 minute timeout.

The chosen engine generates a paper card in `papers/`, fetches a figure into `assets/`, and updates the tag index in `tags/`.

To use Codex, install it first:

```bash
npm install -g @openai/codex
```

## Viewing

Open the vault in Obsidian. Use **Graph View** to explore connections between papers and tag hubs.
