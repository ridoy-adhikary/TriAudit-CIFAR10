#!/usr/bin/env bash
# TriAudit unattended orchestrator — survives laptop/SSH disconnect (nohup/tmux).
# Env lock: ridoy only. Hard gate: previous phase must verify before next starts.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
RIDOY_PYTHON="/home/sda/Users/temp_user/.conda/envs/ridoy/bin/python"
MONITOR="$ROOT/project_monitor"
LOG="$MONITOR/orchestrator.log"
STATE_HELPER="$ROOT/scripts/auto/phase_state.py"
RUN_PHASE="$ROOT/scripts/auto/run_phase.py"

mkdir -p "$MONITOR"
cd "$ROOT"
export PYTHONPATH="$ROOT/src:${PYTHONPATH:-}"
# Prefer GPU 0 unless user overrides
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

ts() { date '+%Y-%m-%d %H:%M:%S'; }
log() {
  echo "[$(ts)] ORCH | $*" | tee -a "$LOG" | tee -a "$MONITOR/project_activity.log"
}

if [[ ! -f "$RIDOY_PYTHON" ]]; then
  log "FAIL ridoy python missing: $RIDOY_PYTHON"
  exit 2
fi

case "$(readlink -f "$RIDOY_PYTHON" 2>/dev/null || echo "$RIDOY_PYTHON")" in
  */.conda/envs/ridoy/*) ;;
  *) log "FAIL unexpected python path"; exit 2 ;;
esac

START_PHASE="${TRIAUDIT_START_PHASE:-0}"
END_PHASE="${TRIAUDIT_END_PHASE:-9}"

log "start orchestrator root=$ROOT phases=${START_PHASE}..${END_PHASE} python=$RIDOY_PYTHON CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"

# Init state only if missing (do NOT wipe verified gates)
if [[ ! -f "$MONITOR/phase_state.json" ]]; then
  "$RIDOY_PYTHON" "$STATE_HELPER" init >/dev/null
fi

# CUDA preflight for training-heavy runs
if ! "$RIDOY_PYTHON" -c "import torch; raise SystemExit(0 if torch.cuda.is_available() else 1)"; then
  log "WARN CUDA not visible at orchestrator start"
fi

for phase in $(seq "$START_PHASE" "$END_PHASE"); do
  # Skip already verified phases
  if "$RIDOY_PYTHON" - <<PY
import json
from pathlib import Path
s=json.loads(Path('$MONITOR/phase_state.json').read_text())
p=s['phases'].get(str($phase), {})
raise SystemExit(0 if p.get('verified') else 1)
PY
  then
    log "SKIP phase $phase (already verified)"
    continue
  fi

  log "===== BEGIN PHASE $phase ====="

  if [[ "$phase" -gt 0 ]]; then
    if ! "$RIDOY_PYTHON" "$STATE_HELPER" can-start "$phase"; then
      log "BLOCKED phase $phase — previous phase not verified. Stopping."
      exit 10
    fi
  fi

  set +e
  "$RIDOY_PYTHON" "$RUN_PHASE" --phase "$phase"
  rc=$?
  set -e

  if [[ "$rc" -ne 0 ]]; then
    log "STOP at phase $phase with rc=$rc (will not continue)"
    echo "failed_phase_$phase" > "$MONITOR/orchestrator.status"
    exit "$rc"
  fi

  log "===== PHASE $phase VERIFIED — continuing ====="
done

log "ALL REQUESTED PHASES COMPLETE (${START_PHASE}..${END_PHASE})"
echo "completed" > "$MONITOR/orchestrator.status"
exit 0
