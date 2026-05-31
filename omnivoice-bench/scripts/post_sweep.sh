#!/usr/bin/env bash
# Wait for the sweep to finish, then run the WER quality check.
# Designed to be launched in the background so the agent can detach.
set -euo pipefail
SWEEP_PID_FILE="/tmp/sweep.log.pid"
QUALITY_LOG="/tmp/quality.log"
cd /workspace/omnivoice-bench

if [ -f "$SWEEP_PID_FILE" ]; then
  PID=$(cat "$SWEEP_PID_FILE")
  echo "[post] waiting for sweep pid=$PID ..."
  while kill -0 "$PID" 2>/dev/null; do sleep 10; done
  echo "[post] sweep done"
fi

# Make sure no leftover server is holding the GPU.
LPIDS=$(lsof -ti tcp:8000 2>/dev/null || true)
if [ -n "$LPIDS" ]; then echo "[post] killing leftover servers $LPIDS"; kill -9 $LPIDS || true; fi

echo "[post] starting quality_check ..."
python3 -m bench.quality_check > "$QUALITY_LOG" 2>&1
echo "[post] quality_check done, log -> $QUALITY_LOG"

echo "[post] starting analyze ..."
python3 -m bench.analyze --results-dir results/raw 2>&1
echo "[post] MAIN_DONE"

# Secondary: c1..c20 RPS sweep at 5 min/cell using the best (mode, num_step)
# picked from the main sweep. Total ~100 min.
LPIDS=$(lsof -ti tcp:8000 2>/dev/null || true)
if [ -n "$LPIDS" ]; then echo "[post] killing leftover servers $LPIDS"; kill -9 $LPIDS || true; fi

echo "[post] starting RPS sweep (c1..c20, 300s/cell) ..."
python3 -m bench.rps_sweep --results-dir results/rps --duration-s 300 --warmup 5 \
  --c-min 1 --c-max 20 > /tmp/rps.log 2>&1
echo "[post] RPS sweep done, log -> /tmp/rps.log"

echo "[post] starting rps_analyze ..."
python3 -m bench.rps_analyze --rps-dir results/rps --report results/rps_report.md 2>&1
echo "[post] ALL_DONE"
