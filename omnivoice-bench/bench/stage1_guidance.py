"""Stage 1b: guidance_scale sweep at num_step=8, Hindi.

Classifier-free guidance runs TWO forward passes per diffusion step (conditional
+ unconditional). Lowering guidance_scale (or gs=1.0, where the uncond term may
be skippable) could be a ~2x latency lever. This script:

  1. For each guidance_scale value, times generation on the same N Hindi prompts.
  2. Saves WAVs per guidance value so quality can be A/B compared.

Output:
  results/guidance/gs_<val>/*.wav + *.txt
  results/guidance/guidance_latency.csv
"""
from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

from server import common


def gen(ctx, text, lang, prompt_obj, num_step, gs):
    cfg = ctx.gen_cfg_cls(num_step=num_step, guidance_scale=gs)
    t0 = time.perf_counter()
    audios = ctx.model.generate(
        text=text, language=lang,
        voice_clone_prompt=prompt_obj,
        generation_config=cfg,
    )
    torch.cuda.synchronize()
    return audios[0], (time.perf_counter() - t0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--guidance-values", type=float, nargs="+",
                    default=[1.0, 1.3, 1.5, 2.0, 3.0])
    ap.add_argument("--num-step", type=int, default=8)
    ap.add_argument("--prompts", default="corpus/prompts_hi.jsonl")
    ap.add_argument("--n-prompts", type=int, default=6)
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--out-dir", default="results/guidance")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    prompts = []
    with open(args.prompts) as f:
        for line in f:
            prompts.append(json.loads(line))
            if len(prompts) >= args.n_prompts:
                break

    ctx = common.load_model(load_asr=False)
    common.load_speaker_cache(ctx, Path("corpus/speaker_cache.pkl"))
    sp = ctx.speaker_cache["hi"]

    # warmup
    for _ in range(2):
        gen(ctx, prompts[0]["text"], "hi", sp, args.num_step, 2.0)

    rows = []
    for gs in args.guidance_values:
        gs_dir = out_dir / f"gs_{gs}"
        gs_dir.mkdir(parents=True, exist_ok=True)
        times_ms = []
        for pr in prompts:
            audio, _ = gen(ctx, pr["text"], "hi", sp, args.num_step, gs)  # warm this prompt
            for _ in range(args.reps):
                audio, dt = gen(ctx, pr["text"], "hi", sp, args.num_step, gs)
                times_ms.append(dt * 1000.0)
            sf.write(gs_dir / f"{pr['id']}.wav", audio, ctx.sampling_rate)
            (gs_dir / f"{pr['id']}.txt").write_text(pr["text"] + "\n")
        arr = np.array(times_ms)
        row = {
            "guidance_scale": gs, "num_step": args.num_step,
            "n_samples": len(times_ms),
            "gen_ms_mean": round(float(arr.mean()), 1),
            "gen_ms_p50": round(float(np.percentile(arr, 50)), 1),
            "gen_ms_p95": round(float(np.percentile(arr, 95)), 1),
            "gen_ms_min": round(float(arr.min()), 1),
        }
        rows.append(row)
        print(f"[guidance] gs={gs}: mean={row['gen_ms_mean']}ms p95={row['gen_ms_p95']}ms",
              flush=True)

    csv_path = out_dir / "guidance_latency.csv"
    with csv_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"[guidance] wrote {csv_path}", flush=True)
    print("[guidance] DONE", flush=True)


if __name__ == "__main__":
    main()
