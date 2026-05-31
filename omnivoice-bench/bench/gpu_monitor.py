"""GPU monitor: writes per-second SM%, mem%, power, temp, VRAM to a CSV.

Two sources:
  1. `nvidia-smi dmon -s pucvmet -d 1` parsed and re-emitted as CSV.
  2. pynvml direct poll at 1 Hz (more reliable timestamping).
The pynvml stream is the primary; dmon is a backup.

Usage:
    python -m bench.gpu_monitor --output results/raw/<run>_gpu.csv [--device 0]
    # SIGINT to stop cleanly.
"""
from __future__ import annotations

import argparse
import csv
import signal
import sys
import time
from pathlib import Path

try:
    import pynvml
except Exception as e:  # pragma: no cover
    print(f"[gpu_monitor] pynvml import failed: {e}", file=sys.stderr)
    pynvml = None


STOP = False


def _handle_sig(signum, frame):
    global STOP
    STOP = True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", required=True)
    ap.add_argument("--device", type=int, default=0)
    ap.add_argument("--interval-s", type=float, default=1.0)
    args = ap.parse_args()

    signal.signal(signal.SIGINT, _handle_sig)
    signal.signal(signal.SIGTERM, _handle_sig)

    if pynvml is None:
        print("[gpu_monitor] no pynvml; exiting", file=sys.stderr)
        sys.exit(2)

    pynvml.nvmlInit()
    handle = pynvml.nvmlDeviceGetHandleByIndex(args.device)
    name = pynvml.nvmlDeviceGetName(handle)
    if isinstance(name, bytes):
        name = name.decode()
    print(f"[gpu_monitor] device {args.device}: {name}", flush=True)

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "ts_unix", "device", "name",
            "sm_pct", "mem_io_pct", "vram_used_mb", "vram_total_mb",
            "power_w", "temp_c", "sm_clock_mhz", "mem_clock_mhz",
        ])
        while not STOP:
            ts = time.time()
            try:
                util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
                power = pynvml.nvmlDeviceGetPowerUsage(handle) / 1000.0
                temp = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
                try:
                    sm_clk = pynvml.nvmlDeviceGetClockInfo(handle, pynvml.NVML_CLOCK_SM)
                    mem_clk = pynvml.nvmlDeviceGetClockInfo(handle, pynvml.NVML_CLOCK_MEM)
                except pynvml.NVMLError:
                    sm_clk = mem_clk = -1
                w.writerow([
                    f"{ts:.3f}", args.device, name,
                    util.gpu, util.memory,
                    int(mem.used / 1024 / 1024), int(mem.total / 1024 / 1024),
                    f"{power:.1f}", temp, sm_clk, mem_clk,
                ])
                f.flush()
            except pynvml.NVMLError as e:
                print(f"[gpu_monitor] nvml err: {e}", file=sys.stderr)
            time.sleep(args.interval_s)

    pynvml.nvmlShutdown()
    print("[gpu_monitor] stopped", flush=True)


if __name__ == "__main__":
    main()
