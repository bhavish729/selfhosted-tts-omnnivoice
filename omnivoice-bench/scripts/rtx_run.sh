#!/usr/bin/env bash
# Launch the compiled OpenAI endpoint + run the burst sweep on the RTX Pro 6000.
# Run ON the GPU box from repo root after scripts/rtx_setup.sh.
#
#   bash scripts/rtx_run.sh            # localhost burst (server + client same box)
#
# Produces results/burst/report.html + results/burst/audio/*.wav
set -e
PORT=${PORT:-8000}

echo "=== free port $PORT ==="
L=$(/usr/bin/lsof -ti tcp:$PORT 2>/dev/null || true); [ -n "$L" ] && kill -9 $L; sleep 1

echo "=== launch compiled batched server (instruct mode, num_step=8) ==="
setsid bash -c "OMNIVOICE_COMPILE=1 OMNIVOICE_INSTRUCT=1 OMNIVOICE_WARM_STEPS=8 OMNIVOICE_API_NUM_STEP=8 \
  .venv/bin/python -m server.batched_server --host 0.0.0.0 --port $PORT --max-batch-size 8 \
  > /tmp/rtx_server.log 2>&1" </dev/null >/dev/null 2>&1 &
echo "server launching (compile warmup ~10-60s)..."

echo "=== wait for health ==="
for i in $(seq 1 120); do
  curl -s http://127.0.0.1:$PORT/health 2>/dev/null | grep -q true && { echo "HEALTH_OK after ${i}x3s"; break; }
  sleep 3
done
curl -s http://127.0.0.1:$PORT/health | grep -q true || { echo "HEALTH FAILED"; tail -30 /tmp/rtx_server.log; exit 1; }

echo "=== burst sweep N=10..100 ==="
.venv/bin/python -m bench.burst_bench \
  --server-url http://127.0.0.1:$PORT/v1/audio/speech \
  --prompts corpus/burst_prompts.jsonl \
  --batches 10,20,30,40,50,60,70,80,90,100 \
  --out-dir results/burst --ttfb-target-ms 200 --warmup 8

echo "=== render HTML report ==="
.venv/bin/python -m bench.burst_report --results results/burst/burst_results.json --out results/burst/report.html

echo "=== stop server ==="
L=$(/usr/bin/lsof -ti tcp:$PORT 2>/dev/null || true); [ -n "$L" ] && kill -9 $L
echo "=== RUN_DONE -> results/burst/report.html ==="
