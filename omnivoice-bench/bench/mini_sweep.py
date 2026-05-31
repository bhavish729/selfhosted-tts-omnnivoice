"""Tiny Hindi-only sanity sweep: naive, num_step=32, c=1..5, N reqs/cell.

Shares ONE naive server across all cells (same mode + num_step), runs load_gen
sequentially for each concurrency, then prints a per-request timing table and
a per-cell aggregate so you can eyeball the spread before committing to a
larger sweep.

Usage:
    python -m bench.mini_sweep \
        --c-min 1 --c-max 5 --total 3 --num-step 32
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import pandas as pd

from bench.run_sweep import (
    SERVER_PORT, SERVER_URL,
    start_server, stop_proc, kill_port, wait_health,
)


def run_load(concurrency: int, total: int, warmup: int, num_step: int,
             prompts: str, out_csv: Path) -> dict:
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
    print(f"  -> {' '.join(cmd)}", flush=True)
    subprocess.check_call(cmd)
    sj = out_csv.with_suffix(".summary.json")
    with sj.open() as f:
        return json.load(f)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="naive")
    ap.add_argument("--num-step", type=int, default=32)
    ap.add_argument("--c-min", type=int, default=1)
    ap.add_argument("--c-max", type=int, default=5)
    ap.add_argument("--total", type=int, default=3, help="requests per cell (overridden by --total-per-worker if >0)")
    ap.add_argument("--total-per-worker", type=int, default=0,
                    help="If >0, total = max(--total, c * this) so each worker gets N reqs.")
    ap.add_argument("--warmup", type=int, default=2)
    ap.add_argument("--prompts", default="corpus/prompts_hi.jsonl")
    ap.add_argument("--out-dir", default="results/mini")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    server_log = out_dir / "mini_server.log"

    kill_port(SERVER_PORT)
    proc = start_server(args.mode, server_log, env_extra={"OMP_NUM_THREADS": "8"})
    try:
        if not wait_health(SERVER_URL, timeout_s=300.0):
            print("[mini] FAIL: server health timeout", flush=True)
            return
        time.sleep(2.0)

        summaries = []
        per_request_rows = []
        for c in range(args.c_min, args.c_max + 1):
            if args.total_per_worker > 0:
                total = max(args.total, c * args.total_per_worker)
            else:
                total = args.total
            print(f"\n=== c={c}, total={total}, num_step={args.num_step} ===",
                  flush=True)
            out_csv = out_dir / f"c{c}.csv"
            s = run_load(c, total, args.warmup, args.num_step,
                         args.prompts, out_csv)
            s["concurrency"] = c
            summaries.append(s)
            # Per-request rows for the per-cell table.
            with out_csv.open() as f:
                for row in csv.DictReader(f):
                    if int(row["status"]) != 200:
                        continue
                    per_request_rows.append({
                        "c": c,
                        "rid": int(row["request_id"]),
                        "ttfb_ms": float(row["ttfb_ms"]),
                        "total_ms": float(row["total_latency_ms"]),
                        "gen_ms": float(row["gen_ms_server"]),
                        "queue_ms": float(row["queue_wait_ms"]),
                        "audio_s": float(row["audio_dur_s"]),
                        "lang": row["lang"],
                    })

        # Print summary tables.
        agg = pd.DataFrame(summaries)
        cols = ["concurrency", "requests_ok", "wall_s",
                "ttfb_ms_p50", "ttfb_ms_p95",
                "audio_total_s", "throughput_audio_per_wall",
                "err_rate"]
        # rps may not exist if --duration-s was 0 (it does — load_gen always writes it).
        if "rps" in agg.columns:
            cols.insert(3, "rps")
        print("\n========== AGGREGATE ==========", flush=True)
        print(agg[cols].round(2).to_string(index=False), flush=True)

        per = pd.DataFrame(per_request_rows)
        print("\n========== PER-REQUEST ==========", flush=True)
        print(per.round(1).to_string(index=False), flush=True)

        # Save aggregates.
        agg.to_csv(out_dir / "mini_summary.csv", index=False)
        per.to_csv(out_dir / "mini_per_request.csv", index=False)
        print(f"\n[mini] saved -> {out_dir}/", flush=True)

    finally:
        stop_proc(proc, "server")
        kill_port(SERVER_PORT)


if __name__ == "__main__":
    main()
