"""Stage 1a: batched vs naive concurrency sweep at num_step=8, Hindi.

For each mode in {naive, batched8, batched16}, starts ONE server, runs load_gen
across c=1..C_MAX with sustained per-worker request counts, and records p50/p95/p99
TTFB, RPS, audio-sec/wall-sec throughput, and SM% utilization.

Driven from one process so it survives SSH disconnects when run detached.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
import time
from pathlib import Path

from bench.run_sweep import (
    SERVER_PORT, SERVER_URL,
    start_server, stop_proc, kill_port, wait_health,
)


def run_load(concurrency, total, warmup, num_step, prompts, out_csv):
    cmd = [
        sys.executable, "-m", "bench.load_gen",
        "--server-url", SERVER_URL,
        "--concurrency", str(concurrency),
        "--total-requests", str(total),
        "--num-step", str(num_step),
        "--warmup", str(warmup),
        "--prompts-file", prompts,
        "--output", str(out_csv),
    ]
    subprocess.check_call(cmd)
    with out_csv.with_suffix(".summary.json").open() as f:
        return json.load(f)


def gpu_stats(gpu_csv: Path):
    if not gpu_csv.exists():
        return "", "", ""
    import pandas as pd
    import numpy as np
    g = pd.read_csv(gpu_csv)
    if g.empty:
        return "", "", ""
    return (round(g["sm_pct"].mean(), 1),
            round(np.percentile(g["sm_pct"], 95), 1),
            int(g["vram_used_mb"].max()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--modes", nargs="+", default=["naive", "batched8", "batched16"])
    ap.add_argument("--num-step", type=int, default=8)
    ap.add_argument("--c-max", type=int, default=20)
    ap.add_argument("--total-per-worker", type=int, default=6,
                    help="total requests per cell = max(8, c * this)")
    ap.add_argument("--warmup", type=int, default=4)
    ap.add_argument("--prompts", default="corpus/prompts_hi.jsonl")
    ap.add_argument("--out-dir", default="results/stage1")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    index_path = out_dir / "stage1_index.csv"
    fields = [
        "mode", "num_step", "concurrency", "requests_ok", "requests_err",
        "wall_s", "rps", "audio_total_s", "throughput_audio_per_wall",
        "ttfb_ms_p50", "ttfb_ms_p95", "ttfb_ms_p99",
        "sm_pct_mean", "sm_pct_p95", "vram_mb_max", "err_rate",
    ]
    idx_f = index_path.open("w", newline="")
    idx_w = csv.DictWriter(idx_f, fieldnames=fields, extrasaction="ignore")
    idx_w.writeheader(); idx_f.flush()

    concurrencies = list(range(1, args.c_max + 1))
    print(f"[stage1] modes={args.modes} num_step={args.num_step} c=1..{args.c_max}",
          flush=True)

    for mode in args.modes:
        print(f"\n########## MODE: {mode} ##########", flush=True)
        kill_port(SERVER_PORT)
        proc = start_server(mode, out_dir / f"{mode}_server.log",
                            env_extra={"OMP_NUM_THREADS": "8"})
        try:
            # torch.compile + CUDA-graph capture during warmup can take minutes,
            # so allow a generous health window.
            if not wait_health(SERVER_URL, timeout_s=900.0):
                print(f"[stage1] {mode}: health timeout, skipping", flush=True)
                continue
            time.sleep(2.0)
            for c in concurrencies:
                total = max(8, c * args.total_per_worker)
                run_id = f"{mode}_step{args.num_step}_c{c}"
                raw = out_dir / f"{run_id}.csv"
                gpu = out_dir / f"{run_id}_gpu.csv"
                mon = subprocess.Popen(
                    [sys.executable, "-m", "bench.gpu_monitor", "--output", str(gpu)],
                    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                    preexec_fn=os.setsid,
                )
                try:
                    s = run_load(c, total, args.warmup, args.num_step, args.prompts, raw)
                finally:
                    stop_proc(mon, "mon", timeout=8.0)
                sm_mean, sm_p95, vram_max = gpu_stats(gpu)
                row = {
                    "mode": mode, "num_step": args.num_step, "concurrency": c,
                    "requests_ok": s["requests_ok"], "requests_err": s["requests_err"],
                    "wall_s": s["wall_s"], "rps": s["rps"],
                    "audio_total_s": s["audio_total_s"],
                    "throughput_audio_per_wall": s["throughput_audio_per_wall"],
                    "ttfb_ms_p50": s["ttfb_ms_p50"], "ttfb_ms_p95": s["ttfb_ms_p95"],
                    "ttfb_ms_p99": s["ttfb_ms_p99"],
                    "sm_pct_mean": sm_mean, "sm_pct_p95": sm_p95,
                    "vram_mb_max": vram_max, "err_rate": s["err_rate"],
                }
                idx_w.writerow(row); idx_f.flush()
                print(f"[stage1] {run_id}: p95={s['ttfb_ms_p95']}ms rps={s['rps']} "
                      f"thr={s['throughput_audio_per_wall']}x sm%={sm_mean} "
                      f"err={s['err_rate']}", flush=True)
        finally:
            stop_proc(proc, "server")
            kill_port(SERVER_PORT)

    idx_f.close()
    print("\n[stage1] DONE", flush=True)


if __name__ == "__main__":
    main()
