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
> - **CUDA graphs are the big unlock: `torch.compile(mode="reduce-overhead")` cuts the
>   model forward 4.7× (148 ms → 32 ms), quality STT-identical.** The model was almost
>   entirely kernel-launch-overhead-bound — which is also why FP8 *didn't* help (it
>   made things 3.4× slower). With graphs, even a 6.6-second utterance generates in 46 ms,
>   so every length clears 200 ms at the raw-model level with huge concurrency headroom.

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

### 5. FP8 quantization makes it *slower* — the model is overhead-bound, not compute-bound
We tested FP8 (torchao `Float8DynamicActivationFloat8WeightConfig`, row-wise) on the
H100 tensor cores at num_step=8:

| Config | mean gen | vs its baseline | quality (STT) |
|---|---|---|---|
| fp16 baseline | 151.6 ms | — | correct |
| fp16 + FP8 on `audio_heads` only (8 M) | 147.3 ms | 1.03× | identical |
| fp16 + FP8 on `llm` (596 M) | **failed** | — | (row-wise needs bf16 weights) |
| bf16 baseline | 171.2 ms | — | correct |
| **bf16 + FP8 on full backbone** | **588.6 ms** | **0.29× (3.4× slower!)** | correct |

Quantizing the full 596 M backbone to FP8 was **3.4× slower**, not faster. Reasons:
1. **Tiny matmuls.** A 0.6 B model at batch-1 has matmuls far too small to saturate
   tensor cores; wall-time is **kernel-launch + Python-dispatch overhead and the
   sequential 8-step diffusion loop**, not FLOPs. This matches the **~57 % SM ceiling**
   in batched serving — the GPU is never compute-bound.
2. **Unfused fallback.** torchao's compiled FP8 kernels require torch ≥ 2.11 ("Skipping
   import of cpp extensions ... found 2.8.0"), so FP8 ran as unfused
   dequant→matmul→requant per layer — adding overhead with no tensor-core payoff.

**Conclusion: FP8 is the wrong lever for a model this small.** The remaining latency
lever is **CUDA-graph capture** (eliminating per-step kernel-launch overhead), not
numeric precision. Reproduce: `bench/fp8_quant.py`, `bench/fp8_bf16.py` (need `torchao`).

### 6. CUDA graphs are the real lever — 4.7× faster, quality-identical
`torch.compile(model.llm, mode="reduce-overhead")` (which uses CUDA-graph trees to
replay each diffusion step as one captured graph instead of thousands of individual
kernel launches):

| Utterance | Audio | Baseline gen | CUDA graph | Speedup |
|---|---|---|---|---|
| short | 2.2 s | 167.5 ms | 28.0 ms | **6.0×** |
| medium | 3.4 s | 169.0 ms | 33.2 ms | **5.1×** |
| long | 6.6 s | 172.1 ms | 46.4 ms | **3.7×** |

(Single-length run measured 148 ms → 31.7 ms = 4.7× on a 3 s clip.)

- **Compiled RTF is 0.007–0.013** — i.e. **75–140× real-time** for a single stream,
  vs. ~0.05 (≈18×) uncompiled.
- **Quality is identical** — STT transcription of compiled vs. baseline audio matches
  exactly ("आपके खाते में 12,000 रुपए का भुगतान बकाया है").
- **This is the proof the model was overhead-bound, not compute-bound** — exactly why
  FP8 (finding 5) failed and the SM ceiling sat at ~57 %. Remove the per-step kernel
  launches and ~80 % of wall-time disappears.
- **Implication:** with graphs, even a 6.6 s utterance generates in 46 ms, so *every*
  production length clears 200 ms at the raw-model level with large concurrency headroom.
  One-time cost: ~19 s compile/warmup at startup; CUDA-graph trees re-capture per new
  input shape (a few extra warmups across the length range).

Reproduce: `bench/cudagraph.py` (single length), `bench/cudagraph_multilen.py` (sweep).

### 7. Audio pipeline gotcha (fixed)
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
- ✅ Stage 2 — FP8 quantization (tested; makes it 3.4× *slower* — model is overhead-bound, not compute-bound)
- ✅ Stage 2 — **CUDA-graph capture: 4.7× faster (148→32 ms), quality-identical — the winning lever**
- ⬜ Next — integrate `torch.compile` into the batched server + re-run the concurrency sweep on top of it

🤖 Generated with [Claude Code](https://claude.com/claude-code)
