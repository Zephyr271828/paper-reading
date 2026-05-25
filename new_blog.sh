#!/usr/bin/env bash
# Usage: ./new_blog.sh [--codex|--claude] <blog_url>
# Example: ./new_blog.sh https://lilianweng.github.io/posts/2023-06-23-agent/
#          ./new_blog.sh --claude https://karpathy.github.io/2015/05/21/rnn-effectiveness/
set -euo pipefail

REPO_DIR="$(cd "$(dirname "$0")" && pwd)"
ENGINE="claude"
TIMEOUT_SECONDS="${NEW_BLOG_TIMEOUT_SECONDS:-1800}"
BLOG_URL=""
BLOG_LOG_BASENAME=""

# Parse args
while [[ $# -gt 0 ]]; do
  case $1 in
    --codex) ENGINE="codex"; shift ;;
    --claude) ENGINE="claude"; shift ;;
    *) BLOG_URL="$1"; shift ;;
  esac
done
: "${BLOG_URL:?Usage: $0 [--codex|--claude] <blog_url>}"

parse_blog_url() {
  local blog_url="$1"
  local url_no_fragment url_no_query host path slug

  url_no_fragment="${blog_url%%#*}"
  url_no_query="${url_no_fragment%%\?*}"

  if [[ ! "$url_no_query" =~ ^https?://([^/]+)(/.*)?$ ]]; then
    echo "Error: Invalid blog URL format: $blog_url" >&2
    echo "Expected http(s)://host/path." >&2
    exit 1
  fi
  host="${BASH_REMATCH[1]}"
  path="${BASH_REMATCH[2]:-/}"

  # Derive a stable log basename: host + last non-empty path segment.
  # This is ONLY for log/status filenames; the actual blog-card filename
  # is chosen by the agent as a phrase summary of the contents.
  host="${host#www.}"
  path="${path%/}"
  slug="${path##*/}"
  if [[ -z "$slug" ]]; then
    slug="$(printf '%s' "$path" | tr '/' '_')"
    slug="${slug#_}"
  fi
  if [[ -z "$slug" ]]; then
    slug="index"
  fi

  BLOG_LOG_BASENAME="$(printf '%s_%s' "$host" "$slug" | tr '/' '_' | tr -c '[:alnum:]_.-' '_' )"
  BLOG_LOG_BASENAME="${BLOG_LOG_BASENAME##_}"
  BLOG_LOG_BASENAME="${BLOG_LOG_BASENAME%%_}"
  if [[ -z "$BLOG_LOG_BASENAME" ]]; then
    echo "Error: Could not derive log basename from URL: $blog_url" >&2
    exit 1
  fi
}

case "$TIMEOUT_SECONDS" in
  ''|*[!0-9]*)
    echo "Error: NEW_BLOG_TIMEOUT_SECONDS must be a positive integer." >&2
    exit 1
    ;;
esac

if (( TIMEOUT_SECONDS <= 0 )); then
  echo "Error: NEW_BLOG_TIMEOUT_SECONDS must be greater than 0." >&2
  exit 1
fi

parse_blog_url "$BLOG_URL"

LOG_DIR="$REPO_DIR/logs/new_blog"
mkdir -p "$LOG_DIR"
RUN_ID="$(date +%Y%m%d_%H%M%S)_$$"
LOG_FILE="$LOG_DIR/${BLOG_LOG_BASENAME}.log"
STATUS_FILE="$LOG_DIR/${BLOG_LOG_BASENAME}.status"
START_MARKER="$LOG_DIR/${BLOG_LOG_BASENAME}.${RUN_ID}.start"
RUNNER_FILE="$LOG_DIR/${BLOG_LOG_BASENAME}.${RUN_ID}.runner.sh"
RUN_LOG_MARKER="===== new_blog run ${RUN_ID} ====="
SESSION_SAFE_ID="${BLOG_LOG_BASENAME//[^[:alnum:]_.-]/_}"
SESSION_NAME="new_blog_${ENGINE}_${SESSION_SAFE_ID}_${RUN_ID}_$$"

cd "$REPO_DIR"
touch "$START_MARKER"

PROMPT="
Generate a blog card for this blog post: $BLOG_URL

IMPORTANT: You are running in fully automated non-interactive mode. All tool permissions are pre-approved. Do NOT ask for permission or wait for approval — just call the tools directly (Read, Write, Edit, Glob, Grep, Bash, WebFetch, etc.). Proceed through every step without stopping.

Follow this workflow:

1. Fetch the blog page at $BLOG_URL. Identify: the post's original title (verbatim, as printed on the page), the author/site, the publication year, and the technical topic.

2. Choose a filename. The filename must be a lowercase kebab-case PHRASE SUMMARY of the blog's contents (NOT just a slug of the URL). Aim for 2-6 words capturing the technical idea. Examples: 'why-attention-is-quadratic.md', 'training-llms-on-tpus.md', 'gradient-checkpointing-tradeoffs.md'. Place the file at blogs/<phrase-summary>.md. If a file with that name already exists, pick a more specific variant.

3. Extract embedded images from the post AND from any first-party explainer pages it links to (e.g. official docs, project sites). PREFER FIGURES OVER PROSE: include EVERY figure that meaningfully conveys the author's idea — method/architecture diagrams, key result charts, illustrative tradeoffs — not just the hero image. If the blog itself has only a teaser and decorative branding, follow the links to the canonical docs/repo to find the real method diagrams. Save them to assets/<phrase-summary>_fig.png, assets/<phrase-summary>_fig2.png, assets/<phrase-summary>_fig3.png, ... and embed each one INLINE next to the section it explains using standard Markdown: ![](../assets/<phrase-summary>_fig.png). Do NOT pile all figures at the top. Skip purely decorative or branding images. Do NOT use Obsidian ![[embed]] syntax — GitHub does not render it. Do not synthesize new figures. If image extraction fails, omit images rather than fabricating substitutes.

4. Create blogs/<phrase-summary>.md using templates/blog_card_template.md as the starting point. The H1 title line inside the card must be the post's ORIGINAL TITLE verbatim, NOT the phrase-summary filename. The card should focus on the technical substance — algorithms, numbers, system designs, formulas — not the author's narrative voice. PRIORITIZE FIGURES AND TABLES OVER LONG PROSE: if a method diagram or results chart shows it clearly, lean on the figure and use prose only for what it cannot convey (numbers, formulas, caveats). Replace what would be multi-paragraph verbal explanations with a figure plus a 1-3 sentence caption. Use the template's sections (Summary, Key Points, optional Notable Quotes, Related). Use LaTeX for any math: inline \$x\$, block \$\$\\mathcal{L} = \\ldots\$\$.

5. Read TAGS.md to find the available tags, then assign only the tags from TAGS.md that fit the post. Add them in YAML frontmatter (tags: [...]) and as inline #hashtags. Do not create new tags, do not edit TAGS.md, and do not invent tags not already listed there. Append a Markdown link like - [phrase-summary](../blogs/phrase-summary.md) to each matching tags/*.md index file only when that index file already exists. Do NOT use [[wiki-links]].

6. Related section: include only if there are related local cards already present in papers/ or blogs/. Cross-folder links are allowed and encouraged:
     - to another blog:   - [other-slug](other-slug.md)
     - to a paper card:   - [method](../papers/method.md)
   Bare Markdown links only, no descriptions. If no related local cards exist, omit the entire section.

After all file writes are complete, print exactly one line in this format:
RESULT_BLOG_CARD=blogs/<phrase-summary>.md

If you also saved at least one figure, print exactly one line in this format (the primary/first figure):
RESULT_FIGURE=assets/<phrase-summary>_fig.png

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
  printf 'Starting blog card generation for: %s (engine: %s, model: %s, effort: %s)\n' "$BLOG_URL" "$ENGINE" "$CLAUDE_MODEL" "$CLAUDE_EFFORT"
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

RESULT_BLOG_CARD=""
RESULT_FIGURE=""
if [[ -f "\$LOG_FILE" ]]; then
  RESULT_BLOG_CARD="\$(awk -F= -v marker="\$RUN_LOG_MARKER" '
    \$0 == marker { in_run=1; next }
    in_run && /^RESULT_BLOG_CARD=/ { print \$2 }
  ' "\$LOG_FILE" | tail -n 1 | tr -d '\r')"
  RESULT_FIGURE="\$(awk -F= -v marker="\$RUN_LOG_MARKER" '
    \$0 == marker { in_run=1; next }
    in_run && /^RESULT_FIGURE=/ { print \$2 }
  ' "\$LOG_FILE" | tail -n 1 | tr -d '\r')"
fi

if [[ -n "\$RESULT_BLOG_CARD" ]]; then
  if [[ "\$RESULT_BLOG_CARD" != blogs/*.md ]] || [[ ! -s "\$REPO_DIR/\$RESULT_BLOG_CARD" ]]; then
    RESULT_BLOG_CARD=""
  fi
fi

if [[ -n "\$RESULT_FIGURE" ]]; then
  if [[ "\$RESULT_FIGURE" != assets/* ]] || [[ ! -s "\$REPO_DIR/\$RESULT_FIGURE" ]]; then
    RESULT_FIGURE=""
  fi
fi

if [[ -z "\$RESULT_BLOG_CARD" ]]; then
  changed_cards=()
  while IFS= read -r card_path; do
    changed_cards+=("\${card_path#\$REPO_DIR/}")
  done < <(find "\$REPO_DIR/blogs" -maxdepth 1 -type f -name '*.md' ! -name '.gitkeep' -newer "\$START_MARKER" | sort)

  if (( \${#changed_cards[@]} == 1 )) && [[ -s "\$REPO_DIR/\${changed_cards[0]}" ]]; then
    RESULT_BLOG_CARD="\${changed_cards[0]}"
  fi
fi

if [[ -z "\$RESULT_FIGURE" ]]; then
  changed_figures=()
  while IFS= read -r figure_path; do
    changed_figures+=("\${figure_path#\$REPO_DIR/}")
  done < <(find "\$REPO_DIR/assets" -maxdepth 1 -type f \
    \( -name '*_fig.png' -o -name '*_fig.jpg' -o -name '*_fig.jpeg' -o -name '*_fig.webp' -o -name '*_fig.gif' \) \
    ! -name '.gitkeep' -newer "\$START_MARKER" | sort)

  if (( \${#changed_figures[@]} >= 1 )) && [[ -s "\$REPO_DIR/\${changed_figures[0]}" ]]; then
    RESULT_FIGURE="\${changed_figures[0]}"
  fi
fi

EXIT_CODE=0
MESSAGE="Blog card generation completed."

if (( TIMED_OUT )); then
  EXIT_CODE=124
  MESSAGE="Timed out after \$TIMEOUT_SECONDS seconds."
elif (( AGENT_EXIT_CODE != 0 )); then
  EXIT_CODE=\$AGENT_EXIT_CODE
  MESSAGE="Agent exited with code \$AGENT_EXIT_CODE."
fi

if [[ -z "\$RESULT_BLOG_CARD" ]]; then
  if (( EXIT_CODE == 0 )); then
    EXIT_CODE=1
    MESSAGE="No blog card was created in blogs/."
  else
    MESSAGE="\$MESSAGE No blog card was created in blogs/."
  fi
fi

{
  printf 'EXIT_CODE=%q\n' "\$EXIT_CODE"
  printf 'TIMED_OUT=%q\n' "\$TIMED_OUT"
  printf 'AGENT_EXIT_CODE=%q\n' "\$AGENT_EXIT_CODE"
  printf 'BLOG_CARD_PATH=%q\n' "\$RESULT_BLOG_CARD"
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
