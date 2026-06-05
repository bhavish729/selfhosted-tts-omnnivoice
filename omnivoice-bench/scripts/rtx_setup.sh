#!/usr/bin/env bash
# One-shot provisioning for the RTX Pro 6000 (Blackwell, sm_120) burst benchmark.
# Run ON the GPU box from the repo root (omnivoice-bench/).
#
#   bash scripts/rtx_setup.sh
#
# Then launch the endpoint + burst sweep (see scripts/rtx_run.sh).
set -e

echo "=== GPU ==="
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader

echo "=== system deps ==="
sudo DEBIAN_FRONTEND=noninteractive apt-get update -qq
sudo DEBIAN_FRONTEND=noninteractive apt-get install -y -qq python3-venv python3-pip ffmpeg

echo "=== venv ==="
python3 -m venv .venv
.venv/bin/python -m pip install --upgrade pip

echo "=== torch (cu128 — Blackwell sm_120 needs CUDA 12.8+) ==="
.venv/bin/python -m pip install torch==2.8.0+cu128 torchaudio==2.8.0+cu128 \
    --extra-index-url https://download.pytorch.org/whl/cu128
.venv/bin/python -m pip install omnivoice fastapi "uvicorn[standard]" httpx \
    pandas numpy soundfile pynvml pydantic tqdm

echo "=== CRITICAL: does torch.compile work on this GPU? (the 4.7x lever) ==="
.venv/bin/python - <<'PY'
import torch
print("torch", torch.__version__, "cuda", torch.cuda.is_available())
print("device", torch.cuda.get_device_name(0))
print("capability", torch.cuda.get_device_capability(0))  # Blackwell = (12, 0)
# tiny compile smoke test
m = torch.nn.Sequential(torch.nn.Linear(512,512), torch.nn.GELU(), torch.nn.Linear(512,512)).cuda().half()
mc = torch.compile(m, mode="reduce-overhead")
x = torch.randn(8,512, device="cuda", dtype=torch.half)
for _ in range(3): y = mc(x)
torch.cuda.synchronize()
print("COMPILE_SMOKE_OK", tuple(y.shape))
PY

echo "=== build prompt pool (100-300 chars, 60/lang, unique) ==="
.venv/bin/python -m bench.burst_prompts --n-per-lang 60 --out corpus/burst_prompts.jsonl

echo "=== SETUP_DONE ==="
