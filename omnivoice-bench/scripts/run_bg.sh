#!/usr/bin/env bash
# Helper: launch a python -m module fully detached so ssh can disconnect.
# Usage: bash scripts/run_bg.sh <log_file> <module> [args...]
set -e
LOG="$1"; shift
mkdir -p "$(dirname "$LOG")"
cd /workspace/omnivoice-bench
nohup python3 -m "$@" >"$LOG" 2>&1 </dev/null &
disown
echo $! > "$LOG.pid"
echo "launched pid=$(cat "$LOG.pid") -> $LOG"
