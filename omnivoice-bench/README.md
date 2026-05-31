# OmniVoice H100 Concurrency & Throughput Benchmark

Benchmark of [`k2-fsa/OmniVoice`](https://github.com/k2-fsa/OmniVoice) on a single
NVIDIA H100 80GB for Indic-language TTS, answering:

1. **Max concurrent requests where p95 TTFB stays under 200 ms** at each
   diffusion-step setting.
2. **Sustained throughput** (seconds of audio generated per second of
   wall-clock time) at that operating point.

Plus a side-by-side of naive serial vs. continuous-batching serving.

## Layout

```
omnivoice-bench/
├── server/
│   ├── common.py           # OmniVoice loader, warmup, per-call + batched helpers
│   ├── naive_server.py     # FastAPI, asyncio.Lock around generate()
│   └── batched_server.py   # FastAPI + asyncio.Queue continuous batcher
├── corpus/
│   ├── build_corpus.py     # generates prompts.jsonl + ref WAVs + speaker cache
│   ├── bundled_prompts.json # native-script phrasings, 28/lang x 12 langs
│   └── ref_audio/          # built at corpus time
├── bench/
│   ├── load_gen.py         # closed-loop async httpx client
│   ├── gpu_monitor.py      # nvml 1Hz logger
│   ├── run_sweep.py        # 4 modes x 3 num_step x 10 conc = 120 cells
│   ├── quality_check.py    # WER via faster-whisper, num_step in {8,16,32}
│   └── analyze.py          # report.md generator
├── results/
│   ├── raw/                # per-cell CSVs + summary.json + sweep_index.csv
│   ├── plots/              # generated PNGs
│   ├── quality_samples/    # WAVs for the WER check
│   ├── quality.csv         # per-(lang, num_step) mean WER
│   └── report.md           # final deliverable
└── Makefile
```

## One-command re-run (on the H100 pod)

```bash
make install            # creates venv-style deps (caller is responsible for the venv)
make corpus             # ~5 min: generates ref audio + speaker cache + prompts.jsonl
make smoke              # ~2 min: validates the loop on one cell before committing 90 min
make sweep              # ~90 min: 120 cells, all server modes
make quality            # ~10 min: WER spot-check across num_step
make report             # renders results/report.md
```

`make all` runs `corpus → sweep → quality → report` end to end.

## Key methodology choices

- **Closed-loop load** (semaphore-bounded in-flight count), not open-loop RPS.
  This is the right model for "how many concurrent calls can I support."
- **TTFB = HTTP request received → first audio byte ready.** OmniVoice's
  diffusion head is one-shot, so TTFB == total generation time for short
  utterances. The 200 ms target is aggressive.
- **`num_step` is the dominant lever.** Defaults to 32; dropping to 16 roughly
  halves latency. 8 is borderline-usable, measured via WER.
- **`load_asr=False`** at model load — we always pass `ref_text`, so Whisper is
  not in the measured path.
- **Speaker prompts pre-cached** via `create_voice_clone_prompt(...)` and
  reused via `voice_clone_prompt=`. This is the production pattern.
- **Real batching at the model level**: `OmniVoice.generate()` accepts list
  inputs natively, so the batched server is doing GPU-level batching, not
  loop-pretend.
- **Warmup excluded**: 20 warmup requests per cell are sent and discarded
  before any measurement.

## Caveats

- Bundled prompts (`corpus/bundled_prompts.json`) were assembled from the
  assistant's multilingual knowledge — readable by native speakers but not
  editorially reviewed. To swap in production-quality lines, drop a file at
  `corpus/source_prompts_indic.jsonl` (one JSON object per line:
  `{"language_id": "hi", "text": "..."}`) and pass `--source ...` to
  `build_corpus.py`.
- Reference audio is self-generated via OmniVoice's voice-design mode. Per the
  upstream docs, `instruct` attributes were trained on EN/ZH data only —
  so per-language voice character is off-distribution even though `language=`
  drives correct phonetics. Swap in native-speaker reference WAVs for higher-
  fidelity benchmarks.
- WER uses `faster-whisper large-v3` locally (`as`/Assamese: no Whisper
  support, marked NaN in the report).
- Client and server are on `localhost`. Real-world TTFB will add ~10–50 ms of
  network jitter that this bench does not capture.

## SSH access

This repo lives both on the user's Mac (canonical) and at `/workspace/omnivoice-bench`
on the RunPod H100. To sync changes:

```bash
rsync -av --delete --exclude=results --exclude=corpus/ref_audio \
  --exclude=corpus/speaker_cache.pkl \
  omnivoice-bench/ root@103.207.149.105:/workspace/omnivoice-bench/ \
  -e "ssh -p 10883 -i ~/.ssh/runpod_claude"
```

And to pull results back:

```bash
rsync -av root@103.207.149.105:/workspace/omnivoice-bench/results/ \
  omnivoice-bench/results/ \
  -e "ssh -p 10883 -i ~/.ssh/runpod_claude"
```
