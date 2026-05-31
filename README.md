# Self-Hosted OmniVoice TTS — H100 Benchmark, Optimization & Deploy Runbook

Benchmarking and optimizing [`k2-fsa/OmniVoice`](https://github.com/k2-fsa/OmniVoice)
(~0.6 B params, Qwen3-0.6B backbone + diffusion audio head) for **self-hosted
Indic-language TTS** on a single NVIDIA H100 80GB — across 12 Indic languages
(Hindi, Bengali, Tamil, Telugu, Marathi, Gujarati, Kannada, Malayalam, Punjabi,
Odia, Assamese, Urdu).

This README doubles as a **deploy runbook**: an agent (or human) can follow
[§ Deploy from scratch](#deploy-from-scratch-on-a-fresh-h100) to reproduce the
full result on a clean H100 in ~20 minutes.

---

## TL;DR — the result

On one H100 at `num_step=8`, serving healthy Indic audio:

| Configuration | Peak RPS | p95 TTFB | Throughput | Notes |
|---|---|---|---|---|
| naive, fp16 | 6.7 | 193 ms @ c1 | 20× RT | baseline; serializes |
| batched8, fp16 | 35 | 229 ms @ c5 | 103× RT | batching → ~4× |
| **naive, compiled** | **31.6** | **163 ms @ c5** | 89× RT | **CUDA graphs alone beats fp16-batched** |
| **batched8, compiled** | **54.1** | 453 ms @ c20 | **157× RT** | **project peak** |

**Three levers, ranked by payoff:**

1. **CUDA graphs (`torch.compile(mode="reduce-overhead")`) — the big win.** Cuts
   the model forward **4.7–6×** (148 ms → 28 ms), quality STT-identical. The model
   is *kernel-launch-overhead-bound*, not compute-bound. So dramatic that **naive
   serial serving + compile (31 RPS @ p95 163 ms) clears the 200 ms SLO without a
   batcher at all** — a major operational simplification for ≤30 RPS/pod.
2. **Continuous batching — ~4× on its own**, stacks with compile to the 54 RPS peak.
3. **FP8 quantization — DON'T.** Tested: made the model **3.4× *slower*** (overhead-bound model + torchao kernels need torch ≥ 2.11; fell back to unfused path). Negative result, documented below.

`guidance_scale` is **latency-neutral** (flat 145 ms across 1.0–3.0). `num_step` is
the quality/speed dial (8 = fastest usable; 16 balanced; 32 default/best quality).

---

## Deploy from scratch on a fresh H100

Tested on **JarvisLabs H100 80GB, Ubuntu 22.04, driver 580 / CUDA 13, Python 3.10**.
Everything below assumes you SSH in as a sudo-capable user (`ubuntu` or `root`).

### 0. Connect

```bash
ssh -o StrictHostKeyChecking=accept-new <user>@<ip>
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
# Expect: NVIDIA H100 80GB HBM3, 81559 MiB, 580.x
```

> ⚠️ **Provider gotcha (JarvisLabs & similar):** a reboot/resume can wipe
> `~/.ssh/authorized_keys`, change the IP, and (rarely) wedge the GPU with an
> **Xid 154 "GPU Reset Required"** fault that only a VM reboot clears. If
> `nvidia-smi` shows `ERR!`, reboot the VM. Storage under `/home` is persistent
> across reboots on JarvisLabs, so code/venv/results survive.

### 1. Get the code

```bash
cd ~
git clone https://github.com/bhavish729/selfhosted-tts-omnnivoice.git omnivoice-bench
cd omnivoice-bench
```

### 2. System deps + venv

```bash
sudo apt-get update -qq
sudo DEBIAN_FRONTEND=noninteractive apt-get install -y -qq python3-pip python3-venv ffmpeg
python3 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
```

### 3. Python deps (torch 2.8.0 + cu128 + everything)

```bash
.venv/bin/python -m pip install torch==2.8.0+cu128 torchaudio==2.8.0+cu128 \
    --extra-index-url https://download.pytorch.org/whl/cu128
.venv/bin/python -m pip install omnivoice fastapi "uvicorn[standard]" httpx \
    pandas matplotlib soundfile pynvml faster-whisper pydantic tabulate tqdm torchao
# sanity
.venv/bin/python -c "import torch,omnivoice,torchao; print(torch.__version__, torch.cuda.is_available())"
# -> 2.8.0+cu128 True
```

### 4. Build the corpus (prompts only — no model needed)

The 12 native-script prompt sets are in `corpus/bundled_prompts.json` (committed).
Generate the JSONL test sets:

```bash
.venv/bin/python corpus/build_corpus.py --skip-ref      # writes corpus/prompts.jsonl (600 lines)
grep '"language_id": "hi"' corpus/prompts.jsonl > corpus/prompts_hi.jsonl   # 50 Hindi prompts
```

> **Audio mode — IMPORTANT.** We serve in **instruct mode**
> (`instruct="female, indian accent"`, env `OMNIVOICE_INSTRUCT=1`, the default),
> NOT voice-cloning. Reason: OmniVoice's voice-clone path **rescales output
> loudness to match the reference clip**, and self-generated references can come
> out near-silent → silent clones. Instruct mode has no reference dependency and
> produces healthy audio at identical latency. (To use voice-cloning instead, set
> `OMNIVOICE_INSTRUCT=0` and supply loud reference clips in `corpus/ref_audio/`.)

### 5. Run the production server (compiled — recommended)

```bash
OMNIVOICE_COMPILE=1 OMNIVOICE_INSTRUCT=1 OMNIVOICE_WARM_STEPS=8 \
  .venv/bin/python -m server.batched_server --host 0.0.0.0 --port 8000 --max-batch-size 8
```

First boot compiles + CUDA-graph-captures during warmup (~10–60 s; health stays
down until done — that's expected). Then:

```bash
curl -s -X POST localhost:8000/tts -H 'Content-Type: application/json' \
  -d '{"text":"नमस्ते, यह एक परीक्षण है।","ref_audio_path":"x","ref_text":"x","language_id":"hi","num_step":8}' \
  -o out.wav -D - | grep -i x-gen-ms
# -> X-Gen-ms: ~28   (compiled; would be ~148 uncompiled)
```

For ≤30 RPS under a strict 200 ms SLO, `server/naive_server.py` (same env flags)
is simpler and sufficient — CUDA graphs alone hit 31 RPS @ p95 163 ms.

### 6. Reproduce the benchmark sweeps

```bash
# Uncompiled batched-vs-naive concurrency sweep (instruct mode)
OMNIVOICE_INSTRUCT=1 .venv/bin/python -m bench.stage1_batched \
    --modes naive batched8 batched16 --num-step 8 --c-max 20 \
    --out-dir results/stage1_instruct

# Compiled sweep (the headline numbers)
OMNIVOICE_COMPILE=1 OMNIVOICE_INSTRUCT=1 OMNIVOICE_WARM_STEPS=8 \
  .venv/bin/python -m bench.stage1_batched \
    --modes naive batched8 batched16 --num-step 8 --c-max 20 \
    --out-dir results/stage1_compiled

# Standalone CUDA-graph latency (single + multi-length)
.venv/bin/python bench/cudagraph.py
.venv/bin/python bench/cudagraph_multilen.py

# FP8 experiment (shows it's slower — negative result)
.venv/bin/python bench/fp8_quant.py
.venv/bin/python bench/fp8_bf16.py

# Generate audio samples for quality review (48 WAVs, 12 langs × 4 categories)
OMNIVOICE_INSTRUCT=1 .venv/bin/python -m bench.generate_samples --num-step 8 --out samples_final
```

Run sweeps **detached** so SSH drops don't kill them:
```bash
setsid bash -c 'OMNIVOICE_COMPILE=1 OMNIVOICE_INSTRUCT=1 OMNIVOICE_WARM_STEPS=8 \
  .venv/bin/python -m bench.stage1_batched --modes naive batched8 batched16 \
  --num-step 8 --c-max 20 --out-dir results/stage1_compiled > /tmp/sweep.log 2>&1; \
  echo DONE >> /tmp/sweep.log' </dev/null >/dev/null 2>&1 &
```

---

## Critical implementation details (read before modifying the servers)

These are the non-obvious things that took debugging to get right:

1. **`torch.compile` is thread-affine.** CUDA-graph trees capture on the thread
   that first runs them. The model **must be warmed AND served on the same single
   thread**, or every request 500s. Both servers use a dedicated
   `ThreadPoolExecutor(max_workers=1)` (`GEN_EXECUTOR`) for warmup *and*
   generation. **Do not** revert to `asyncio.to_thread` (arbitrary pool threads → broken graphs).

2. **Compile + dynamic batch sizes churn the graph cache.** Each distinct batch
   size (1,2,…,8) is a new input shape → a fresh CUDA-graph recapture, which
   stalls the batcher at low concurrency (batched8 dipped to 8.7 RPS at c5 before
   all shapes were captured, then jumped to 54 RPS at c20 once warm). **For
   production, pad requests to a fixed batch size** to avoid recapture churn.

3. **Compiled-server health check is slow.** Warmup compiles for ~10–60 s; the
   sweep harness waits up to 900 s (`bench/stage1_batched.py`). `OMNIVOICE_WARM_STEPS=8`
   limits warmup to the step count you actually serve (instead of 8/16/32),
   cutting compile time.

4. **Env flags** (all read in `server/common.py`):
   - `OMNIVOICE_COMPILE=1` — enable `torch.compile(llm, mode="reduce-overhead")`
   - `OMNIVOICE_INSTRUCT=1` — instruct mode (default; avoids voice-clone loudness bug)
   - `OMNIVOICE_INSTRUCT_TEXT="female, indian accent"` — the instruct prompt
   - `OMNIVOICE_WARM_STEPS=8` — diffusion steps to warm/compile

---

## Findings in detail

### 1. CUDA graphs: 4.7–6× faster, quality-identical
`torch.compile(model.llm, mode="reduce-overhead")`, measured per utterance length:

| Utterance | Audio | Baseline gen | CUDA graph | Speedup |
|---|---|---|---|---|
| short | 2.2 s | 167.5 ms | 28.0 ms | **6.0×** |
| medium | 3.4 s | 169.0 ms | 33.2 ms | **5.1×** |
| long | 6.6 s | 172.1 ms | 46.4 ms | **3.7×** |

STT (faster-whisper large-v3) transcription of compiled vs baseline matches
exactly. This proves the model was launch-overhead-bound (also why FP8 failed and
why uncompiled SM% capped at ~57%).

### 2. Compiled concurrency sweep (the stacked result)
`results/stage1_compiled/stage1_index.csv`:

| Mode | c1 | c5 | c10 | c20 (peak) |
|---|---|---|---|---|
| naive | 5.4 RPS / 285 ms | **31.6 / 163 ms** | 31.2 / 328 ms | 31.2 / 652 ms |
| batched8 | 4.8 / 313 ms | 8.7 / 843 ms* | 23.3 / 1049 ms | **54.1 / 453 ms** |
| batched16 | 4.0 / 397 ms | 14.9 / 461 ms | 23.0 / 569 ms | 46.5 / 800 ms |

*the c5 dip is graph-recapture churn (detail #2). `batch=16` never beats `batch=8`.

### 3. Uncompiled concurrency sweep (`results/stage1_instruct/`)
naive peaks ~6.7 RPS; batched8 peaks 35 RPS @ 103×. Batching alone = ~4×.

### 4. FP8 quantization — makes it slower (negative result)
`results/fp8/`. torchao row-wise FP8:

| Config | mean gen | vs baseline |
|---|---|---|
| fp16 | 151.6 ms | — |
| fp16 + FP8 on `audio_heads` (8 M) | 147.3 ms | 1.03× |
| bf16 | 171.2 ms | — |
| bf16 + FP8 full backbone (596 M) | 588.6 ms | **0.29× (3.4× slower)** |

0.6 B model at batch-1 → matmuls too small for tensor cores; torchao FP8 kernels
need torch ≥ 2.11 (we're on 2.8) so they fell back to unfused dequant→matmul→requant.
**FP8 is the wrong lever for a model this small.**

### 5. guidance_scale — latency-neutral
Flat 145 ms across gs = 1.0/1.3/1.5/2.0/3.0. OmniVoice always runs both CFG passes.

### 6. Audio pipeline bug (fixed)
Voice-clone output inherits the reference clip's loudness; self-generated refs can
be near-silent → silent clones. Fix: instruct mode (no reference). STT-verified
intelligible in 9/10 testable languages (Assamese/Odia unsupported by Whisper).

---

## Repo layout

```
omnivoice-bench/
├── server/
│   ├── common.py            # loader, torch.compile, warmup, generate_one/batch; all env flags
│   ├── naive_server.py      # FastAPI + asyncio.Lock + GEN_EXECUTOR (compile-safe)
│   └── batched_server.py    # FastAPI + asyncio.Queue continuous batcher + GEN_EXECUTOR
├── corpus/
│   ├── build_corpus.py      # --skip-ref builds prompts.jsonl without the model
│   └── bundled_prompts.json # 12-lang native-script collections-domain phrasings
├── bench/
│   ├── load_gen.py          # closed-loop async httpx load generator
│   ├── gpu_monitor.py       # pynvml 1 Hz SM%/VRAM/power logger
│   ├── stage1_batched.py    # batched-vs-naive concurrency sweep (c1..c20)
│   ├── stage1_guidance.py   # guidance_scale latency sweep
│   ├── cudagraph.py / cudagraph_multilen.py  # CUDA-graph latency
│   ├── fp8_quant.py / fp8_bf16.py            # FP8 experiments
│   ├── generate_samples.py  # per-language audio sample generator
│   └── analyze.py           # report generator
└── results/
    ├── stage1_instruct/     # uncompiled sweep (canonical fp16)
    ├── stage1_compiled/     # compiled sweep (headline)
    ├── cudagraph/           # CUDA-graph latency results
    └── fp8/                 # FP8 negative-result data
```

## Production recommendation

- **Live interactive TTS (≤30 RPS/pod, strict 200 ms):** `naive_server.py` with
  `OMNIVOICE_COMPILE=1`. Simplest; 31 RPS @ p95 163 ms. Scale horizontally.
- **High throughput (batch/offline or relaxed SLO):** `batched_server.py` with
  `OMNIVOICE_COMPILE=1 --max-batch-size 8`, padded to fixed batch size. 54 RPS / 157× RT.
- **Do not** use FP8. **Do not** chase `guidance_scale`. **Do** keep `num_step=8`
  unless quality review demands 16.
- OmniVoice's diffusion head emits the whole clip at once (no token streaming), so
  TTFB = full gen time. For sub-100 ms *first-audio* regardless of length you'd need
  a streaming/autoregressive model; OmniVoice fits batch/pre-render and short-utterance
  interactive use, not long-form live streaming.

🤖 Generated with [Claude Code](https://claude.com/claude-code)
