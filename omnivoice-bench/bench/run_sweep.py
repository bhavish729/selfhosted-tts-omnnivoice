"""Sweep orchestrator: 4 server modes x 3 num_step x 10 concurrency = 120 cells.

Per cell:
  1. Start the right server fresh on a clean port (kills any prior one).
  2. Wait for /health to return ready.
  3. Start gpu_monitor.
  4. Run load_gen (warmup + measured).
  5. Stop gpu_monitor + server.
  6. Append a row to results/raw/sweep_index.csv.

Continues on per-cell failure (logs >5% errors or p95 >5s). Resumable: skips a
cell if results/raw/<run_id>.summary.json already exists, unless --force is set.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import shlex
import signal
import subprocess
import sys
import time
import datetime as dt
from pathlib import Path


SERVER_MODES = ["naive", "batched4", "batched8", "batched16"]
NUM_STEPS = [8, 16, 32]
CONCURRENCIES = [1, 2, 4, 8, 12, 16, 24, 32, 48, 64]

SERVER_PORT = 8000
SERVER_URL = f"http://127.0.0.1:{SERVER_PORT}"


def mode_to_args(mode: str) -> tuple[str, list[str]]:
    """Return (module, extra_args) to launch the appropriate server."""
    if mode == "naive":
        return ("server.naive_server", [])
    if mode.startswith("batched"):
        bs = int(mode.removeprefix("batched"))
        return ("server.batched_server", ["--max-batch-size", str(bs)])
    raise ValueError(mode)


def wait_health(url: str, timeout_s: float = 180.0) -> bool:
    import httpx
    t0 = time.time()
    while time.time() - t0 < timeout_s:
        try:
            r = httpx.get(url + "/health", timeout=5.0)
            if r.status_code == 200:
                return True
        except Exception:
            pass
        time.sleep(2.0)
    return False


def start_server(mode: str, log_path: Path, env_extra: dict) -> subprocess.Popen:
    module, extra = mode_to_args(mode)
    cmd = [
        sys.executable, "-m", module,
        "--host", "127.0.0.1",
        "--port", str(SERVER_PORT),
        *extra,
    ]
    env = os.environ.copy()
    env.update(env_extra)
    log_f = log_path.open("w")
    proc = subprocess.Popen(cmd, stdout=log_f, stderr=subprocess.STDOUT, env=env,
                            preexec_fn=os.setsid)
    print(f"[sweep] started {mode} pid={proc.pid} log={log_path}", flush=True)
    return proc


def stop_proc(proc: subprocess.Popen, name: str, timeout: float = 30.0):
    if proc.poll() is not None:
        return
    try:
        os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        proc.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        print(f"[sweep] {name} SIGKILL", flush=True)
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        except ProcessLookupError:
            pass
        proc.wait(timeout=5.0)


def kill_port(port: int):
    """Kill any process holding the port (lsof + kill)."""
    try:
        out = subprocess.check_output(["lsof", "-ti", f"tcp:{port}"], text=True).strip()
    except subprocess.CalledProcessError:
        return
    for pid in out.split():
        try:
            os.kill(int(pid), signal.SIGKILL)
        except Exception:
            pass


def run_cell(
    results_dir: Path,
    mode: str,
    num_step: int,
    concurrency: int,
    total_requests: int,
    warmup: int,
    force: bool,
    settle_s: float,
) -> dict:
    ts = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_id = f"{ts}_{mode}_step{num_step}_conc{concurrency}"
    summary_path = results_dir / f"{run_id}.summary.json"
    if summary_path.exists() and not force:
        with summary_path.open() as f:
            s = json.load(f)
        print(f"[sweep] SKIP {run_id} (already done)", flush=True)
        s["run_id"] = run_id
        s["server_mode"] = mode
        s["skipped"] = True
        return s

    raw_path = results_dir / f"{run_id}.csv"
    gpu_path = results_dir / f"{run_id}_gpu.csv"
    server_log = results_dir / f"{run_id}_server.log"
    monitor_log = results_dir / f"{run_id}_monitor.log"

    kill_port(SERVER_PORT)
    server_proc = start_server(mode, server_log, env_extra={"OMP_NUM_THREADS": "8"})

    summary = {
        "run_id": run_id, "server_mode": mode, "num_step": num_step,
        "concurrency": concurrency, "error": "",
    }

    try:
        if not wait_health(SERVER_URL, timeout_s=300.0):
            summary["error"] = "server_health_timeout"
            print(f"[sweep] FAIL {run_id}: health timeout", flush=True)
            return summary
        time.sleep(settle_s)

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
                "--total-requests", str(total_requests),
                "--num-step", str(num_step),
                "--warmup", str(warmup),
                "--output", str(raw_path),
            ]
            print(f"[sweep] RUN {run_id}", flush=True)
            subprocess.check_call(cmd)
        finally:
            stop_proc(mon, "gpu_monitor", timeout=10.0)

        if summary_path.exists():
            with summary_path.open() as f:
                file_summary = json.load(f)
            summary.update(file_summary)

    except subprocess.CalledProcessError as e:
        summary["error"] = f"load_gen_exit_{e.returncode}"
        print(f"[sweep] FAIL {run_id}: {summary['error']}", flush=True)
    finally:
        stop_proc(server_proc, "server")
        kill_port(SERVER_PORT)

    if summary.get("err_rate", 0) > 0.05:
        print(f"[sweep] WARN {run_id} err_rate={summary['err_rate']:.2%}", flush=True)
    if summary.get("total_ms_p95", 0) > 5000:
        print(f"[sweep] WARN {run_id} p95={summary['total_ms_p95']:.0f} ms (>5s)", flush=True)

    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", default="results/raw")
    ap.add_argument("--total-requests", type=int, default=300)
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--settle-s", type=float, default=2.0,
                    help="Sleep after /health passes, before starting load")
    ap.add_argument("--modes", nargs="+", default=SERVER_MODES, choices=SERVER_MODES)
    ap.add_argument("--num-steps", type=int, nargs="+", default=NUM_STEPS)
    ap.add_argument("--concurrencies", type=int, nargs="+", default=CONCURRENCIES)
    ap.add_argument("--force", action="store_true", help="Re-run cells that already have a summary")
    ap.add_argument("--smoke", action="store_true",
                    help="Tiny smoke matrix: naive only, num_step=16, conc=[1,4]")
    args = ap.parse_args()

    if args.smoke:
        args.modes = ["naive"]
        args.num_steps = [16]
        args.concurrencies = [1, 4]
        args.total_requests = 20
        args.warmup = 4

    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    index_path = results_dir / "sweep_index.csv"
    fields = [
        "run_id", "server_mode", "num_step", "concurrency",
        "requests_ok", "requests_err", "wall_s", "audio_total_s",
        "throughput_audio_per_wall",
        "ttfb_ms_p50", "ttfb_ms_p95", "ttfb_ms_p99",
        "total_ms_p50", "total_ms_p95", "total_ms_p99",
        "err_rate", "error",
    ]
    write_header = not index_path.exists()
    idx_f = index_path.open("a", newline="")
    idx_w = csv.DictWriter(idx_f, fieldnames=fields, extrasaction="ignore")
    if write_header:
        idx_w.writeheader()
        idx_f.flush()

    total_cells = len(args.modes) * len(args.num_steps) * len(args.concurrencies)
    print(f"[sweep] {total_cells} cells: modes={args.modes} steps={args.num_steps} conc={args.concurrencies}",
          flush=True)
    t_start = time.time()
    done = 0
    for mode in args.modes:
        for ns in args.num_steps:
            for conc in args.concurrencies:
                done += 1
                print(f"\n=== [{done}/{total_cells}] {mode} step={ns} conc={conc} ===",
                      flush=True)
                summary = run_cell(
                    results_dir, mode, ns, conc,
                    total_requests=args.total_requests,
                    warmup=args.warmup,
                    force=args.force,
                    settle_s=args.settle_s,
                )
                idx_w.writerow(summary)
                idx_f.flush()
    idx_f.close()
    elapsed = time.time() - t_start
    print(f"\n[sweep] done in {elapsed/60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
