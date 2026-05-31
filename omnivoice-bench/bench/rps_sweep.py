"""Granular concurrency RPS sweep: c=1..20, single (mode, num_step) combo,
5 min of sustained load per cell. Designed to find the exact concurrency
inflection point and the max sustained RPS the server can hold.

Picks the (mode, num_step) automatically from the main sweep's
results/raw/sweep_index.csv if --auto-pick is set (default: best
throughput_audio_per_wall while requests_ok > 0). Otherwise pass --mode +
--num-step explicitly.
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
import datetime as dt
from pathlib import Path

import pandas as pd

from bench.run_sweep import (
    SERVER_PORT, SERVER_URL,
    start_server, stop_proc, kill_port, wait_health,
)


def pick_best(sweep_index: Path) -> tuple[str, int]:
    df = pd.read_csv(sweep_index)
    df["throughput_audio_per_wall"] = pd.to_numeric(
        df["throughput_audio_per_wall"], errors="coerce"
    )
    df["err_rate"] = pd.to_numeric(df["err_rate"], errors="coerce").fillna(0)
    ok = df[(df["err_rate"] <= 0.05) & (df["requests_ok"] > 0)]
    if ok.empty:
        raise SystemExit(f"No usable rows in {sweep_index}")
    best = ok.sort_values("throughput_audio_per_wall", ascending=False).iloc[0]
    return str(best["server_mode"]), int(best["num_step"])


def run_cell(results_dir: Path, mode: str, num_step: int, concurrency: int,
             duration_s: float, warmup: int) -> dict:
    ts = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_id = f"{ts}_rps_{mode}_step{num_step}_conc{concurrency}"
    raw_path = results_dir / f"{run_id}.csv"
    gpu_path = results_dir / f"{run_id}_gpu.csv"
    server_log = results_dir / f"{run_id}_server.log"
    monitor_log = results_dir / f"{run_id}_monitor.log"

    kill_port(SERVER_PORT)
    server_proc = start_server(mode, server_log, env_extra={"OMP_NUM_THREADS": "8"})
    summary = {
        "run_id": run_id, "server_mode": mode,
        "num_step": num_step, "concurrency": concurrency, "error": "",
    }
    try:
        if not wait_health(SERVER_URL, timeout_s=300.0):
            summary["error"] = "server_health_timeout"
            print(f"[rps] FAIL {run_id}: health timeout", flush=True)
            return summary
        time.sleep(2.0)
        mon = subprocess.Popen(
            [sys.executable, "-m", "bench.gpu_monitor", "--output", str(gpu_path)],
            stdout=monitor_log.open("w"), stderr=subprocess.STDOUT,
            preexec_fn=os.setsid,
        )
        try:
            cmd = [
                sys.executable, "-m", "bench.load_gen",
                "--server-url", SERVER_URL,
                "--concurrency", str(concurrency),
                "--duration-s", str(duration_s),
                "--num-step", str(num_step),
                "--warmup", str(warmup),
                "--output", str(raw_path),
            ]
            print(f"[rps] RUN {run_id}", flush=True)
            subprocess.check_call(cmd)
        finally:
            stop_proc(mon, "gpu_monitor", timeout=10.0)
        sp = results_dir / f"{run_id}.summary.json"
        if sp.exists():
            with sp.open() as f:
                summary.update(json.load(f))
    except subprocess.CalledProcessError as e:
        summary["error"] = f"load_gen_exit_{e.returncode}"
        print(f"[rps] FAIL {run_id}: {summary['error']}", flush=True)
    finally:
        stop_proc(server_proc, "server")
        kill_port(SERVER_PORT)
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", default="results/rps")
    ap.add_argument("--duration-s", type=float, default=300.0)
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--mode", default=None,
                    help="Server mode. If omitted, picked from --main-index.")
    ap.add_argument("--num-step", type=int, default=None,
                    help="num_step. If omitted, picked from --main-index.")
    ap.add_argument("--main-index", default="results/raw/sweep_index.csv")
    ap.add_argument("--c-min", type=int, default=1)
    ap.add_argument("--c-max", type=int, default=20)
    args = ap.parse_args()

    if args.mode is None or args.num_step is None:
        mode, num_step = pick_best(Path(args.main_index))
        print(f"[rps] auto-picked: mode={mode} num_step={num_step}", flush=True)
    else:
        mode, num_step = args.mode, args.num_step

    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    index_path = results_dir / "rps_index.csv"
    fields = [
        "run_id", "server_mode", "num_step", "concurrency",
        "requests_ok", "requests_err", "wall_s", "rps",
        "audio_total_s", "throughput_audio_per_wall",
        "ttfb_ms_p50", "ttfb_ms_p95", "ttfb_ms_p99",
        "total_ms_p50", "total_ms_p95", "total_ms_p99",
        "err_rate", "error",
    ]
    write_header = not index_path.exists()
    idx_f = index_path.open("a", newline="")
    idx_w = csv.DictWriter(idx_f, fieldnames=fields, extrasaction="ignore")
    if write_header:
        idx_w.writeheader(); idx_f.flush()

    concurrencies = list(range(args.c_min, args.c_max + 1))
    print(f"[rps] {len(concurrencies)} cells: {mode} step={num_step} "
          f"c={concurrencies[0]}..{concurrencies[-1]} duration={args.duration_s}s",
          flush=True)
    t_start = time.time()
    for i, c in enumerate(concurrencies, 1):
        print(f"\n=== [{i}/{len(concurrencies)}] {mode} step={num_step} conc={c} ===",
              flush=True)
        s = run_cell(results_dir, mode, num_step, c, args.duration_s, args.warmup)
        idx_w.writerow(s); idx_f.flush()
    idx_f.close()
    print(f"\n[rps] done in {(time.time()-t_start)/60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
