#!/usr/bin/env bash
# Usage: ./new_paper.sh [--codex|--claude] <arxiv_url>
# Example: ./new_paper.sh https://arxiv.org/abs/2406.11931
#          ./new_paper.sh --claude https://arxiv.org/abs/2406.11931
set -euo pipefail

REPO_DIR="$(cd "$(dirname "$0")" && pwd)"
ENGINE="codex"
TIMEOUT_SECONDS="${NEW_PAPER_TIMEOUT_SECONDS:-1800}"
POLL_INTERVAL_SECONDS=2

# Parse args
while [[ $# -gt 0 ]]; do
  case $1 in
    --codex) ENGINE="codex"; shift ;;
    --claude) ENGINE="claude"; shift ;;
    *) ARXIV_URL="$1"; shift ;;
  esac
done
: "${ARXIV_URL:?Usage: $0 [--codex|--claude] <arxiv_url>}"

case "$TIMEOUT_SECONDS" in
  ''|*[!0-9]*)
    echo "Error: NEW_PAPER_TIMEOUT_SECONDS must be a positive integer." >&2
    exit 1
    ;;
esac

if (( TIMEOUT_SECONDS <= 0 )); then
  echo "Error: NEW_PAPER_TIMEOUT_SECONDS must be greater than 0." >&2
  exit 1
fi

LOG_DIR="$REPO_DIR/logs/new_paper"
mkdir -p "$LOG_DIR"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
LOG_FILE="$LOG_DIR/${TIMESTAMP}.log"
STATUS_FILE="$LOG_DIR/${TIMESTAMP}.status"
START_MARKER="$LOG_DIR/${TIMESTAMP}.start"
RUNNER_FILE="$LOG_DIR/${TIMESTAMP}.runner.sh"
SESSION_NAME="new_paper_${ENGINE}_${TIMESTAMP}_$$"

cd "$REPO_DIR"
touch "$START_MARKER"

PROMPT="
Generate a paper card for this paper: $ARXIV_URL

Follow the 6-step workflow in MEMORY.md exactly:

1. Fetch the arXiv page (and project website if one exists) to determine the method name, GitHub repo URL, and project website URL.

2. Retrieve the official teaser figure or methodology diagram from the arXiv HTML page (https://arxiv.org/html/<ID>), project website, GitHub repo, or another first-party source. If needed, use a clean crop of the original paper figure. Do not create synthetic summary screenshots. Save it to assets/<method>_fig.png and embed it in the card with ![[assets/<method>_fig.png]]. If screenshotting / extraction is difficult, fetch the arXiv source and inspect figure file paths / \\includegraphics targets.

3. Create papers/<method>.md using templates/paper_card_template.md as the starting point. The card must focus on technical details useful for reproduction — skip abstract-level story-telling. Include: core algorithmic steps, training/inference details (loss, optimizer, key hyperparameters, dataset), and key quantitative results with exact benchmark names and numbers. Present results as Markdown tables whenever possible. If table extraction is hard, fetch the arXiv source and use the LaTeX table source to reconstruct the tables; you may also save a clean screenshot of the original table to assets/<method>_table1.png, etc.

4. Use LaTeX for all math. Inline: \$x\$. Block: \$\$\\mathcal{L} = \\ldots\$\$.

5. Assign tags from: DLM, RL, SPEC_DECODING, MLSYS, PRUNING. Add them in YAML frontmatter (tags: [...]) and as inline #hashtags. Append a [[method_name]] wiki-link to each matching tags/*.md index file.

6. Add a Related Papers section only if there are related papers already present in papers/. If included, each bullet should be only the bare [[wiki-link]] with no extra description. If there are no related local papers, omit the entire section.

After all file writes are complete, print exactly one line in this format:
RESULT_PAPER_CARD=papers/<method>.md

If you also saved a figure, print exactly one line in this format:
RESULT_FIGURE=assets/<method>_fig.png

Do not print the RESULT_ lines until the files exist on disk.
"

echo "Starting paper card generation for: $ARXIV_URL (engine: $ENGINE)" | tee "$LOG_FILE"
echo "Log: $LOG_FILE"
echo "Status: $STATUS_FILE"

if ! command -v tmux >/dev/null 2>&1; then
  echo "Error: tmux is required so the run survives terminal shutdown." >&2
  exit 1
fi

if [[ "$ENGINE" == "codex" ]]; then
  RUN_CMD=(codex exec --full-auto "$PROMPT")
else
  RUN_CMD=(claude --dangerously-skip-permissions -p "$PROMPT")
fi

if ! command -v "${RUN_CMD[0]}" >/dev/null 2>&1; then
  echo "Error: '${RUN_CMD[0]}' is not installed or not on PATH." >&2
  exit 1
fi

RUN_CMD_LITERAL="$(printf '%q ' "${RUN_CMD[@]}")"

cat > "$RUNNER_FILE" <<EOF
#!/usr/bin/env bash
set -euo pipefail

REPO_DIR=$(printf '%q' "$REPO_DIR")
LOG_FILE=$(printf '%q' "$LOG_FILE")
STATUS_FILE=$(printf '%q' "$STATUS_FILE")
START_MARKER=$(printf '%q' "$START_MARKER")
TIMEOUT_SECONDS=$(printf '%q' "$TIMEOUT_SECONDS")
RUN_CMD=($RUN_CMD_LITERAL)

cd "\$REPO_DIR"

STATUS_TMP="\${STATUS_FILE}.tmp"
TIMEOUT_MARKER="\${STATUS_FILE}.timeout"
rm -f "\$STATUS_TMP" "\$TIMEOUT_MARKER"

"\${RUN_CMD[@]}" >> "\$LOG_FILE" 2>&1 &
AGENT_PID=\$!

(
  sleep "\$TIMEOUT_SECONDS"
  if kill -0 "\$AGENT_PID" 2>/dev/null; then
    : > "\$TIMEOUT_MARKER"
    printf 'Timeout after %ss; terminating agent.\n' "\$TIMEOUT_SECONDS" >> "\$LOG_FILE"
    kill "\$AGENT_PID" 2>/dev/null || true
    sleep 5
    kill -9 "\$AGENT_PID" 2>/dev/null || true
  fi
) &
WATCHDOG_PID=\$!

AGENT_EXIT_CODE=0
if wait "\$AGENT_PID"; then
  AGENT_EXIT_CODE=0
else
  AGENT_EXIT_CODE=\$?
fi

kill "\$WATCHDOG_PID" 2>/dev/null || true
wait "\$WATCHDOG_PID" 2>/dev/null || true

TIMED_OUT=0
if [[ -e "\$TIMEOUT_MARKER" ]]; then
  TIMED_OUT=1
fi

RESULT_PAPER_CARD=""
RESULT_FIGURE=""
if [[ -f "\$LOG_FILE" ]]; then
  RESULT_PAPER_CARD="\$(awk -F= '/^RESULT_PAPER_CARD=/{print \$2}' "\$LOG_FILE" | tail -n 1 | tr -d '\r')"
  RESULT_FIGURE="\$(awk -F= '/^RESULT_FIGURE=/{print \$2}' "\$LOG_FILE" | tail -n 1 | tr -d '\r')"
fi

if [[ -n "\$RESULT_PAPER_CARD" ]]; then
  if [[ "\$RESULT_PAPER_CARD" != papers/*.md ]] || [[ ! -s "\$REPO_DIR/\$RESULT_PAPER_CARD" ]]; then
    RESULT_PAPER_CARD=""
  fi
fi

if [[ -z "\$RESULT_PAPER_CARD" ]]; then
  changed_cards=()
  while IFS= read -r card_path; do
    changed_cards+=("\${card_path#\$REPO_DIR/}")
  done < <(find "\$REPO_DIR/papers" -maxdepth 1 -type f -name '*.md' ! -name '.gitkeep' -newer "\$START_MARKER" | sort)

  if (( \${#changed_cards[@]} == 1 )) && [[ -s "\$REPO_DIR/\${changed_cards[0]}" ]]; then
    RESULT_PAPER_CARD="\${changed_cards[0]}"
  fi
fi

EXIT_CODE=0
MESSAGE="Paper card generation completed."

if (( TIMED_OUT )); then
  EXIT_CODE=124
  MESSAGE="Timed out after \$TIMEOUT_SECONDS seconds."
elif (( AGENT_EXIT_CODE != 0 )); then
  EXIT_CODE=\$AGENT_EXIT_CODE
  MESSAGE="Agent exited with code \$AGENT_EXIT_CODE."
fi

if [[ -z "\$RESULT_PAPER_CARD" ]]; then
  if (( EXIT_CODE == 0 )); then
    EXIT_CODE=1
    MESSAGE="No paper card was created in papers/."
  else
    MESSAGE="\$MESSAGE No paper card was created in papers/."
  fi
fi

{
  printf 'EXIT_CODE=%q\n' "\$EXIT_CODE"
  printf 'TIMED_OUT=%q\n' "\$TIMED_OUT"
  printf 'AGENT_EXIT_CODE=%q\n' "\$AGENT_EXIT_CODE"
  printf 'PAPER_CARD_PATH=%q\n' "\$RESULT_PAPER_CARD"
  printf 'FIGURE_PATH=%q\n' "\$RESULT_FIGURE"
  printf 'MESSAGE=%q\n' "\$MESSAGE"
} > "\$STATUS_TMP"

mv "\$STATUS_TMP" "\$STATUS_FILE"
rm -f "\$TIMEOUT_MARKER"
EOF

chmod +x "$RUNNER_FILE"
TMUX_CMD="$(printf '%q' "$RUNNER_FILE")"

tmux new-session -d -s "$SESSION_NAME" "$TMUX_CMD"

echo "Running in detached tmux session: $SESSION_NAME"
echo "Reattach with:"
echo "  tmux attach -t $SESSION_NAME"
echo "List active sessions with:"
echo "  tmux ls"
echo "Follow progress with:"
echo "  tail -f $LOG_FILE"

cleanup() {
  if [[ -f "$STATUS_FILE" ]]; then
    rm -f "$RUNNER_FILE" "$START_MARKER"
  fi
}

interrupt() {
  echo
  echo "Interrupted. The tmux session is still running: $SESSION_NAME" >&2
  echo "Check progress with: tail -f $LOG_FILE" >&2
  exit 130
}

trap interrupt INT TERM
trap cleanup EXIT

echo "Waiting for completion..."

while [[ ! -f "$STATUS_FILE" ]]; do
  if ! tmux has-session -t "$SESSION_NAME" 2>/dev/null; then
    sleep 1
    if [[ ! -f "$STATUS_FILE" ]]; then
      {
        printf 'EXIT_CODE=%q\n' 1
        printf 'TIMED_OUT=%q\n' 0
        printf 'AGENT_EXIT_CODE=%q\n' 1
        printf 'PAPER_CARD_PATH=%q\n' ""
        printf 'FIGURE_PATH=%q\n' ""
        printf 'MESSAGE=%q\n' "tmux session exited before writing a final status."
      } > "$STATUS_FILE"
    fi
    break
  fi
  sleep "$POLL_INTERVAL_SECONDS"
done

# shellcheck disable=SC1090
source "$STATUS_FILE"

echo "$MESSAGE"
if [[ -n "$PAPER_CARD_PATH" ]]; then
  echo "Paper card: $PAPER_CARD_PATH"
fi
if [[ -n "$FIGURE_PATH" ]]; then
  echo "Figure: $FIGURE_PATH"
fi

exit "$EXIT_CODE"
