# Self-Hosted OmniVoice TTS — H100 Benchmark & Optimization

Benchmarking and optimizing [`k2-fsa/OmniVoice`](https://github.com/k2-fsa/OmniVoice)
(~0.6 B params, Qwen3-0.6B backbone + diffusion audio head) for **self-hosted
Indic-language TTS** on a single NVIDIA H100 80 GB.

The goal: find the operating point that maximizes **concurrency and throughput**
while keeping **p95 TTFB under 200 ms**, across 12 Indic languages (Hindi,
Bengali, Tamil, Telugu, Marathi, Gujarati, Kannada, Malayalam, Punjabi, Odia,
Assamese, Urdu).

> **TL;DR** — On one H100, OmniVoice at `num_step=8`:
> - **Naive serving:** ~5.8 req/s at p95 TTFB **193 ms** (only config strictly < 200 ms).
> - **Continuous batching (batch=8):** **23 req/s at p95 229 ms** (4× naive), peaking at
>   **35 req/s / 103× real-time** at higher concurrency.
> - The 200 ms wall is set by the **143 ms model forward** (diffusion, whole-clip),
>   not by the serving code (~20 ms overhead). The GPU is only **~57 % utilized at
>   peak**, so there is headroom that batching alone can't reach — motivating
>   CUDA-graph / FP8 work.

---

## Key findings

### 1. Batching is the single biggest lever (4–5×)
OmniVoice's `generate()` accepts **lists natively**, so a request-queue batcher does
real GPU-level batching. Result (num_step=8, Hindi, instruct mode, healthy audio):

| Mode | Concurrency | p95 TTFB | RPS | Throughput (audio-s/wall-s) | GPU SM% |
|---|---|---|---|---|---|
| naive | 1 | **193 ms** | 5.8 | 15× | 24 % |
| batched8 | 2 | 209 ms | 10.2 | 27× | 23 % |
| batched8 | 5 | **229 ms** | **23.1** | 65× | 32 % |
| batched8 | 10 | 457 ms | 33.0 | 95× | 50 % |
| batched8 | 20 | 693 ms | **35.3** | **103×** | 57 % |

`batch=16` performs the same as `batch=8` — the 20 ms batch-wait window never
assembles batches larger than ~8 at these arrival rates, so **batch=8 is the sweet spot**.

### 2. The 200 ms floor is the diffusion forward, not the server
Steady-state latency breakdown (num_step=8, ~3 s utterance):

| Component | Latency |
|---|---|
| Pure model `generate()` | **143 ms** |
| + server overhead (WAV-encode, asyncio) | +20 ms → 163 ms |
| + localhost network | +2 ms → 165 ms |
| Request #1 cold-start (excluded) | ~210 ms |

To go meaningfully below 200 ms at higher concurrency you must cut the **143 ms
model forward** — hence the FP8 / CUDA-graph experiments.

### 3. `guidance_scale` is latency-neutral
Sweeping CFG scale (1.0 → 3.0) showed **flat 145 ms** throughout. OmniVoice always
runs both conditional + unconditional passes; changing the scale just reweights
them. No free 2× there. *(Clean negative result.)*

### 4. `num_step` is the quality/speed dial
- `num_step=32` (default): ~620 ms gen — best quality.
- `num_step=16`: ~330 ms gen — balanced.
- `num_step=8`: ~143 ms gen — fastest; the only setting that approaches 200 ms.
  STT spot-check (faster-whisper large-v3) confirmed intelligible Hindi/Indic output.

### 5. Audio pipeline gotcha (fixed)
OmniVoice **voice-cloning rescales output loudness to match the reference clip.**
A self-generated reference that comes out quiet → every cloned utterance is quiet.
We switched to **instruct mode** (`instruct="female, indian accent"`, no reference
clip) which produces healthy audio at identical latency, verified via STT in 9/10
testable languages. Toggle with `OMNIVOICE_INSTRUCT=0/1`.

---

## Repo layout

```
omnivoice-bench/
├── server/
│   ├── common.py            # model loader, warmup, generate_one/batch (instruct + clone modes)
│   ├── naive_server.py      # FastAPI + asyncio.Lock (one-at-a-time baseline)
│   └── batched_server.py    # FastAPI + asyncio.Queue continuous batcher
├── corpus/
│   ├── build_corpus.py      # generates prompts.jsonl + ref clips + speaker cache
│   └── bundled_prompts.json # 12-language collections-domain phrasings (native script)
├── bench/
│   ├── load_gen.py          # closed-loop async httpx load generator
│   ├── gpu_monitor.py       # pynvml 1 Hz SM%/VRAM/power logger
│   ├── run_sweep.py         # full mode × num_step × concurrency matrix
│   ├── stage1_batched.py    # batched-vs-naive concurrency sweep (c1..c20)
│   ├── stage1_guidance.py   # guidance_scale latency sweep
│   ├── mini_sweep.py        # quick c1..c5 sanity sweep
│   ├── generate_samples.py  # per-language audio sample generator (quality review)
│   └── analyze.py           # report.md generator
├── results/
│   ├── stage1/              # original sweep (voice-clone) index
│   └── stage1_instruct/     # clean sweep (instruct mode) index  ← canonical
└── Makefile
```

## Running it

On a fresh H100 (Ubuntu 22.04, CUDA 12.8 driver ≥ 550):

```bash
cd omnivoice-bench
python3 -m venv .venv && source .venv/bin/activate
pip install torch==2.8.0+cu128 torchaudio==2.8.0+cu128 \
    --extra-index-url https://download.pytorch.org/whl/cu128
pip install omnivoice fastapi "uvicorn[standard]" httpx pandas matplotlib \
    soundfile pynvml faster-whisper pydantic tabulate tqdm

# Build the corpus (ref clips + prompts)
python -m corpus.build_corpus

# Clean batched-vs-naive concurrency sweep at num_step=8 (instruct mode)
OMNIVOICE_INSTRUCT=1 python -m bench.stage1_batched \
    --modes naive batched8 batched16 --num-step 8 --c-max 20 \
    --out-dir results/stage1_instruct
```

Each server can also be run standalone:

```bash
python -m server.batched_server --host 0.0.0.0 --port 8000 --max-batch-size 8
curl -X POST localhost:8000/tts -H 'Content-Type: application/json' \
  -d '{"text":"नमस्ते","ref_audio_path":"corpus/ref_audio/hi.wav","ref_text":"x","language_id":"hi","num_step":8}' \
  -o out.wav
```

## Methodology notes / caveats

- **Closed-loop load** (N in-flight held constant), not open-loop RPS — the correct
  model for "how many concurrent calls can one GPU support."
- **TTFB = request received → first audio byte.** OmniVoice's diffusion head emits
  the whole clip at once (no token streaming), so TTFB = total gen time for short
  utterances. True sub-100 ms first-audio would require a streaming/autoregressive
  model; OmniVoice is best suited to **batch / pre-render**, not live conversational TTS.
- Reference clips and the bundled prompt set are assistant-generated, not
  native-speaker reviewed — quality signal is **indicative** (STT-based), not a final MOS.
- Kannada returned blank-on-STT despite loud audio (likely a Whisper artifact); flagged.
- All numbers from a single H100 80 GB HBM3, no STT/LLM co-tenant on the GPU.

## Status

- ✅ Stage 1a — batched vs naive concurrency sweep (4–5× throughput from batching)
- ✅ Stage 1b — guidance_scale latency sweep (latency-neutral)
- ✅ Audio pipeline fix (instruct mode) + STT verification
- 🔬 Stage 2 — FP8 quantization to cut the 143 ms model forward (in progress)

🤖 Generated with [Claude Code](https://claude.com/claude-code)
