#!/usr/bin/env bash
# Usage: ./new_paper.sh [--codex|--claude] <arxiv_url>
# Example: ./new_paper.sh https://arxiv.org/abs/2406.11931
#          ./new_paper.sh --claude https://arxiv.org/abs/2406.11931
set -euo pipefail

REPO_DIR="$(cd "$(dirname "$0")" && pwd)"
ENGINE="claude"
TIMEOUT_SECONDS="${NEW_PAPER_TIMEOUT_SECONDS:-1800}"
ARXIV_ID=""
ARXIV_LOG_BASENAME=""

# Parse args
while [[ $# -gt 0 ]]; do
  case $1 in
    --codex) ENGINE="codex"; shift ;;
    --claude) ENGINE="claude"; shift ;;
    *) ARXIV_URL="$1"; shift ;;
  esac
done
: "${ARXIV_URL:?Usage: $0 [--codex|--claude] <arxiv_url>}"

parse_arxiv_url() {
  local arxiv_url="$1"
  local url_no_fragment url_no_query path_kind arxiv_id

  url_no_fragment="${arxiv_url%%#*}"
  url_no_query="${url_no_fragment%%\?*}"

  if [[ "$url_no_query" =~ ^https?://(www\.)?arxiv\.org/(abs|html|pdf)/(.*)$ ]]; then
    path_kind="${BASH_REMATCH[2]}"
    arxiv_id="${BASH_REMATCH[3]}"
  else
    echo "Error: Invalid arXiv URL format: $arxiv_url" >&2
    echo "Expected https://arxiv.org/abs/<id>, https://arxiv.org/html/<id>, or https://arxiv.org/pdf/<id>.pdf." >&2
    exit 1
  fi

  if [[ -z "$arxiv_id" || "$arxiv_id" == */ || "$arxiv_id" == *" "* ]]; then
    echo "Error: Invalid arXiv URL format: $arxiv_url" >&2
    exit 1
  fi

  if [[ "$path_kind" == "pdf" ]]; then
    if [[ "$arxiv_id" != *.pdf ]]; then
      echo "Error: Invalid arXiv PDF URL format: $arxiv_url" >&2
      exit 1
    fi
    arxiv_id="${arxiv_id%.pdf}"
  elif [[ "$arxiv_id" == *.pdf ]]; then
    echo "Error: Invalid arXiv URL format: $arxiv_url" >&2
    exit 1
  fi

  if [[ "$arxiv_id" =~ ^[0-9]{4}\.[0-9]{4,5}(v[0-9]+)?$ ]] || [[ "$arxiv_id" =~ ^[[:alnum:]-]+/[0-9]{7}(v[0-9]+)?$ ]]; then
    ARXIV_ID="$arxiv_id"
    ARXIV_LOG_BASENAME="${arxiv_id//\//_}"
    return 0
  fi

  echo "Error: Invalid arXiv ID in URL: $arxiv_url" >&2
  exit 1
}

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

parse_arxiv_url "$ARXIV_URL"

LOG_DIR="$REPO_DIR/logs/new_paper"
mkdir -p "$LOG_DIR"
RUN_ID="$(date +%Y%m%d_%H%M%S)_$$"
LOG_FILE="$LOG_DIR/${ARXIV_LOG_BASENAME}.log"
STATUS_FILE="$LOG_DIR/${ARXIV_LOG_BASENAME}.status"
START_MARKER="$LOG_DIR/${ARXIV_LOG_BASENAME}.${RUN_ID}.start"
RUNNER_FILE="$LOG_DIR/${ARXIV_LOG_BASENAME}.${RUN_ID}.runner.sh"
RUN_LOG_MARKER="===== new_paper run ${RUN_ID} ====="
SESSION_SAFE_ID="${ARXIV_LOG_BASENAME//[^[:alnum:]_.-]/_}"
SESSION_NAME="new_paper_${ENGINE}_${SESSION_SAFE_ID}_${RUN_ID}_$$"

cd "$REPO_DIR"
touch "$START_MARKER"

PROMPT="
Generate a paper card for this paper: $ARXIV_URL

IMPORTANT: You are running in fully automated non-interactive mode. All tool permissions are pre-approved. Do NOT ask for permission or wait for approval — just call the tools directly (Read, Write, Edit, Glob, Grep, Bash, WebFetch, etc.). Proceed through every step without stopping.

Follow the 6-step workflow in MEMORY.md exactly:

1. Fetch the arXiv page (and project website if one exists) to determine the method name, GitHub repo URL, and project website URL.

2. Retrieve the official teaser figure or methodology diagram from the arXiv HTML page (https://arxiv.org/html/<ID>), project website, GitHub repo, or another first-party source. If needed, use a clean crop of the original paper figure. Do not create synthetic summary screenshots. Save it to assets/<method>_fig.png and embed it in the card with ![[assets/<method>_fig.png]]. If the figure is hosted in a GitHub repo, prefer the GitHub connector file API with base64 content and decode it locally into assets/; do not rely on raw GitHub curl downloads unless shell network access has already been verified. If screenshotting / extraction is difficult, fetch the arXiv source and inspect figure file paths / \\includegraphics targets.

3. Create papers/<method>.md using templates/paper_card_template.md as the starting point. The card must focus on technical details useful for reproduction — skip abstract-level story-telling. Include: core algorithmic steps, training/inference details (loss, optimizer, key hyperparameters, dataset), and key quantitative results with exact benchmark names and numbers. Present results as Markdown tables whenever possible. If table extraction is hard, fetch the arXiv source and use the LaTeX table source to reconstruct the tables; you may also save a clean screenshot of the original table to assets/<method>_table1.png, etc.

4. Use LaTeX for all math. Inline: \$x\$. Block: \$\$\\mathcal{L} = \\ldots\$\$.

5. Read TAGS.md to find the available tags, then assign only the tags from TAGS.md that fit the paper. Add them in YAML frontmatter (tags: [...]) and as inline #hashtags. Do not create new tags, do not edit TAGS.md, and do not invent tags that are not already listed there. Append a [[method_name]] wiki-link to each matching tags/*.md index file only when that index file already exists.

6. Add a Related Papers section only if there are related papers already present in papers/. If included, each bullet should be only the bare [[wiki-link]] with no extra description. If there are no related local papers, omit the entire section.

After all file writes are complete, print exactly one line in this format:
RESULT_PAPER_CARD=papers/<method>.md

If you also saved a figure, print exactly one line in this format:
RESULT_FIGURE=assets/<method>_fig.png

Do not print the RESULT_ lines until the files exist on disk. Validate that any RESULT_FIGURE path is a non-empty local file before printing it.
"

if [[ -e "$LOG_FILE" ]]; then
  echo "Warning: Log file already exists; appending to: $LOG_FILE" >&2
fi

CLAUDE_MODEL="claude-sonnet-4-6"
CLAUDE_EFFORT="medium"

{
  printf '\n'
  printf '%s\n' "$RUN_LOG_MARKER"
  printf 'Started at: %s\n' "$(date '+%Y-%m-%d %H:%M:%S %z')"
  printf 'Starting paper card generation for: %s (engine: %s, arXiv ID: %s, model: %s, effort: %s)\n' "$ARXIV_URL" "$ENGINE" "$ARXIV_ID" "$CLAUDE_MODEL" "$CLAUDE_EFFORT"
} | tee -a "$LOG_FILE"
echo "Log: $LOG_FILE"
echo "Status: $STATUS_FILE"

if ! command -v tmux >/dev/null 2>&1; then
  echo "Error: tmux is required so the run survives terminal shutdown." >&2
  exit 1
fi

if [[ "$ENGINE" == "codex" ]]; then
  RUN_CMD=(codex exec --full-auto "$PROMPT")
else
  RUN_CMD=(claude --dangerously-skip-permissions --model "$CLAUDE_MODEL" --effort "$CLAUDE_EFFORT" -p "$PROMPT")
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
RUNNER_FILE=$(printf '%q' "$RUNNER_FILE")
RUN_LOG_MARKER=$(printf '%q' "$RUN_LOG_MARKER")
RUN_ID=$(printf '%q' "$RUN_ID")
TIMEOUT_SECONDS=$(printf '%q' "$TIMEOUT_SECONDS")
RUN_CMD=($RUN_CMD_LITERAL)

cd "\$REPO_DIR"

STATUS_TMP="\${STATUS_FILE}.\${RUN_ID}.tmp"
TIMEOUT_MARKER="\${STATUS_FILE}.\${RUN_ID}.timeout"
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
  RESULT_PAPER_CARD="\$(awk -F= -v marker="\$RUN_LOG_MARKER" '
    \$0 == marker { in_run=1; next }
    in_run && /^RESULT_PAPER_CARD=/ { print \$2 }
  ' "\$LOG_FILE" | tail -n 1 | tr -d '\r')"
  RESULT_FIGURE="\$(awk -F= -v marker="\$RUN_LOG_MARKER" '
    \$0 == marker { in_run=1; next }
    in_run && /^RESULT_FIGURE=/ { print \$2 }
  ' "\$LOG_FILE" | tail -n 1 | tr -d '\r')"
fi

if [[ -n "\$RESULT_PAPER_CARD" ]]; then
  if [[ "\$RESULT_PAPER_CARD" != papers/*.md ]] || [[ ! -s "\$REPO_DIR/\$RESULT_PAPER_CARD" ]]; then
    RESULT_PAPER_CARD=""
  fi
fi

if [[ -n "\$RESULT_FIGURE" ]]; then
  if [[ "\$RESULT_FIGURE" != assets/* ]] || [[ ! -s "\$REPO_DIR/\$RESULT_FIGURE" ]]; then
    RESULT_FIGURE=""
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

if [[ -z "\$RESULT_FIGURE" ]]; then
  changed_figures=()
  while IFS= read -r figure_path; do
    changed_figures+=("\${figure_path#\$REPO_DIR/}")
  done < <(find "\$REPO_DIR/assets" -maxdepth 1 -type f \
    \( -name '*_fig.png' -o -name '*_fig.jpg' -o -name '*_fig.jpeg' -o -name '*_fig.webp' -o -name '*_fig.gif' \) \
    ! -name '.gitkeep' -newer "\$START_MARKER" | sort)

  if (( \${#changed_figures[@]} == 1 )) && [[ -s "\$REPO_DIR/\${changed_figures[0]}" ]]; then
    RESULT_FIGURE="\${changed_figures[0]}"
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
rm -f "\$RUNNER_FILE" "\$START_MARKER"
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
echo "Check final status with:"
echo "  cat $STATUS_FILE"
