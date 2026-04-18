#!/usr/bin/env bash
# Launch Claude Code remote-control in a detached tmux session.
set -euo pipefail

SESSION="claude-rc"
VAULT_DIR="$(cd "$(dirname "$0")" && pwd)"

if tmux has-session -t "$SESSION" 2>/dev/null; then
  echo "Already running. Attach with: tmux attach -t $SESSION"
  exit 0
fi

tmux new-session -d -s "$SESSION" -c "$VAULT_DIR" "claude remote-control --spawn=same-dir"

# Wait for the session URL to appear, then print it.
for _ in {1..30}; do
  URL=$(tmux capture-pane -t "$SESSION" -p | grep -oE 'https://claude\.ai/code\?environment=[A-Za-z0-9_]+' | head -n1 || true)
  [[ -n "$URL" ]] && break
  sleep 1
done

echo "Started. Session: $SESSION"
[[ -n "${URL:-}" ]] && echo "Open on phone: $URL" || echo "URL not captured yet — run: tmux attach -t $SESSION"
echo "Stop with: tmux kill-session -t $SESSION"
