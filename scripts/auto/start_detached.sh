#!/usr/bin/env bash
# Start TriAudit auto-implementation detached from SSH/laptop.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
MONITOR="$ROOT/project_monitor"
mkdir -p "$MONITOR"
cd "$ROOT"

PID_FILE="$MONITOR/orchestrator.pid"
OUT="$MONITOR/orchestrator.nohup.out"
STATUS="$MONITOR/orchestrator.status"

if [[ -f "$PID_FILE" ]]; then
  old="$(cat "$PID_FILE" || true)"
  if [[ -n "${old:-}" ]] && kill -0 "$old" 2>/dev/null; then
    echo "Orchestrator already running pid=$old"
    exit 0
  fi
fi

# Default: resume from first unverified phase if not set
export TRIAUDIT_START_PHASE="${TRIAUDIT_START_PHASE:-2}"
export TRIAUDIT_END_PHASE="${TRIAUDIT_END_PHASE:-9}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTHONPATH="$ROOT/src:${PYTHONPATH:-}"
# Multi-restart cost control (full suite still runs FGSM/PGD-20 on all test images)
export TRIAUDIT_MR_MAX_BATCHES="${TRIAUDIT_MR_MAX_BATCHES:-40}"

echo "starting phases ${TRIAUDIT_START_PHASE}..${TRIAUDIT_END_PHASE} on GPU ${CUDA_VISIBLE_DEVICES}"
echo "running" > "$STATUS"

if command -v tmux >/dev/null 2>&1; then
  SESSION="triaudit_auto"
  tmux kill-session -t "$SESSION" 2>/dev/null || true
  tmux new-session -d -s "$SESSION" "cd '$ROOT' && bash '$ROOT/scripts/auto/orchestrate.sh' 2>&1 | tee -a '$MONITOR/orchestrator.log'"
  echo "started tmux session=$SESSION"
  echo "attach: tmux attach -t $SESSION"
else
  nohup bash "$ROOT/scripts/auto/orchestrate.sh" >"$OUT" 2>&1 &
  echo $! >"$PID_FILE"
  echo "started nohup pid=$(cat "$PID_FILE")"
  echo "logs: $OUT and $MONITOR/orchestrator.log"
fi
