#!/usr/bin/env bash
# Usage: ./new_paper.sh [--codex|--claude] <arxiv_url>
# Example: ./new_paper.sh https://arxiv.org/abs/2406.11931
#          ./new_paper.sh --claude https://arxiv.org/abs/2406.11931
set -euo pipefail

REPO_DIR="$(cd "$(dirname "$0")" && pwd)"
ENGINE="codex"

# Parse args
while [[ $# -gt 0 ]]; do
  case $1 in
    --codex) ENGINE="codex"; shift ;;
    --claude) ENGINE="claude"; shift ;;
    *) ARXIV_URL="$1"; shift ;;
  esac
done
: "${ARXIV_URL:?Usage: $0 [--codex|--claude] <arxiv_url>}"

LOG_DIR="$REPO_DIR/logs/new_paper"
mkdir -p "$LOG_DIR"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
LOG_FILE="$LOG_DIR/${TIMESTAMP}.log"

cd "$REPO_DIR"

PROMPT="
Generate a paper card for this paper: $ARXIV_URL

Follow the 6-step workflow in MEMORY.md exactly:

1. Fetch the arXiv page (and project website if one exists) to determine the method name, GitHub repo URL, and project website URL.

2. Retrieve the teaser figure or methodology diagram from the arXiv HTML page (https://arxiv.org/html/<ID>) or the project website. Save it to assets/<method>_fig.png. Embed it in the card with ![[assets/<method>_fig.png]].

3. Create papers/<method>.md using templates/paper_card_template.md as the starting point. The card must focus on technical details useful for reproduction — skip abstract-level story-telling. Include: core algorithmic steps, training/inference details (loss, optimizer, key hyperparameters, dataset), and key quantitative results with exact benchmark names and numbers.

4. Use LaTeX for all math. Inline: \$x\$. Block: \$\$\\mathcal{L} = \\ldots\$\$.

5. Assign tags from: DLM, RL, SPEC_DECODING, MLSYS, PRUNING. Add them in YAML frontmatter (tags: [...]) and as inline #hashtags. Append a [[method_name]] wiki-link to each matching tags/*.md index file.

6. Add a Related Papers section at the bottom with [[wiki-links]] to existing papers in papers/ that are conceptually related, with a one-line note per link.
"

echo "Starting paper card generation for: $ARXIV_URL (engine: $ENGINE)" | tee "$LOG_FILE"
echo "Log: $LOG_FILE"

if [[ "$ENGINE" == "codex" ]]; then
  codex --approval-mode full-auto "$PROMPT" >> "$LOG_FILE" 2>&1 &
else
  claude --dangerously-skip-permissions -p "$PROMPT" >> "$LOG_FILE" 2>&1 &
fi

echo "Running in background (PID $!). Follow progress with:"
echo "  tail -f $LOG_FILE"
