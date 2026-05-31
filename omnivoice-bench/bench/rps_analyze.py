"""Generate results/rps_report.md from results/rps/rps_index.csv.

Headline questions, in order:
  1. Max sustained RPS the server can hold (regardless of TTFB).
  2. Max RPS at p95 TTFB < TARGET_MS (default 200 ms).
  3. The RPS/latency tradeoff curve: at what concurrency does each break?
  4. Audio-sec/wall-sec throughput at the operating point.
  5. GPU utilization at the operating point.

Plots: RPS vs concurrency, p95 TTFB vs concurrency, throughput vs concurrency.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def load_gpu(results_dir: Path, run_id: str) -> pd.DataFrame:
    p = results_dir / f"{run_id}_gpu.csv"
    return pd.read_csv(p) if p.exists() else pd.DataFrame()


def gpu_summary(results_dir: Path, run_id: str) -> dict:
    gpu = load_gpu(results_dir, run_id)
    if gpu.empty:
        return {}
    return {
        "sm_pct_mean": round(gpu["sm_pct"].mean(), 1),
        "sm_pct_p95": round(np.percentile(gpu["sm_pct"], 95), 1),
        "vram_used_mb_max": int(gpu["vram_used_mb"].max()),
        "power_w_mean": round(gpu["power_w"].mean(), 1),
    }


def plot_one(df: pd.DataFrame, y: str, ylabel: str, out: Path,
             hline: float | None = None, hline_label: str = ""):
    plt.figure(figsize=(9, 5))
    plt.plot(df["concurrency"], df[y], marker="o", linewidth=2)
    if hline is not None:
        plt.axhline(hline, linestyle="--", color="red", linewidth=1, label=hline_label)
        plt.legend()
    plt.xlabel("Concurrency (in-flight requests)")
    plt.ylabel(ylabel)
    plt.title(f"{ylabel} vs concurrency")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out, dpi=120)
    plt.close()


def make_report(rps_dir: Path, report_path: Path, target_ttfb_ms: float):
    idx = pd.read_csv(rps_dir / "rps_index.csv")
    for c in ("rps", "ttfb_ms_p95", "throughput_audio_per_wall", "err_rate"):
        idx[c] = pd.to_numeric(idx[c], errors="coerce")
    idx = idx.sort_values("concurrency").reset_index(drop=True)

    plots = rps_dir.parent / "rps_plots"
    plots.mkdir(parents=True, exist_ok=True)
    plot_one(idx, "rps", "RPS (req/s)", plots / "rps_vs_conc.png")
    plot_one(idx, "ttfb_ms_p95", "p95 TTFB (ms)", plots / "p95_vs_conc.png",
             hline=target_ttfb_ms, hline_label=f"target = {target_ttfb_ms:.0f} ms")
    plot_one(idx, "throughput_audio_per_wall",
             "Throughput (audio-s / wall-s)", plots / "throughput_vs_conc.png")

    mode = idx["server_mode"].iloc[0]
    num_step = int(idx["num_step"].iloc[0])

    max_rps_row = idx.loc[idx["rps"].idxmax()]
    under_target = idx[(idx["ttfb_ms_p95"] <= target_ttfb_ms) & (idx["err_rate"] <= 0.05)]
    if not under_target.empty:
        best_constrained = under_target.sort_values("rps", ascending=False).iloc[0]
    else:
        best_constrained = None

    gpu_at_max = gpu_summary(rps_dir, max_rps_row["run_id"])

    L = [f"# OmniVoice c1..c20 RPS Benchmark — {mode}, num_step={num_step}\n"]
    L.append(f"Sustained load: **300 seconds per concurrency**, 12-language Indic mix, "
             f"closed-loop client, no STT/LLM co-tenants on the GPU.\n")
    L.append("\n## Headline\n")
    L.append(f"- **Peak sustained RPS:** **{max_rps_row['rps']:.2f}** at concurrency "
             f"{int(max_rps_row['concurrency'])} (p95 TTFB "
             f"{max_rps_row['ttfb_ms_p95']:.0f} ms, throughput "
             f"{max_rps_row['throughput_audio_per_wall']:.2f} audio-s/wall-s).\n")
    if best_constrained is not None:
        L.append(f"- **Max RPS under p95 TTFB < {target_ttfb_ms:.0f} ms:** "
                 f"**{best_constrained['rps']:.2f}** at concurrency "
                 f"{int(best_constrained['concurrency'])}.\n")
    else:
        L.append(f"- **No concurrency held p95 TTFB < {target_ttfb_ms:.0f} ms** "
                 f"at sustained 5-min load.\n")

    L.append("\n## Full table\n")
    cols = ["concurrency", "rps", "throughput_audio_per_wall",
            "ttfb_ms_p50", "ttfb_ms_p95", "ttfb_ms_p99",
            "requests_ok", "requests_err", "err_rate", "wall_s"]
    L.append(idx[cols].round(2).to_markdown(index=False) + "\n")

    L.append("\n## Plots\n")
    L.append("![RPS vs concurrency](rps_plots/rps_vs_conc.png)\n")
    L.append("![p95 TTFB vs concurrency](rps_plots/p95_vs_conc.png)\n")
    L.append("![Throughput vs concurrency](rps_plots/throughput_vs_conc.png)\n")

    if gpu_at_max:
        L.append(f"\n## GPU at peak-RPS cell (conc={int(max_rps_row['concurrency'])})\n")
        L.append(f"- SM% mean: **{gpu_at_max['sm_pct_mean']}**\n")
        L.append(f"- SM% p95: **{gpu_at_max['sm_pct_p95']}**\n")
        L.append(f"- VRAM peak: **{gpu_at_max['vram_used_mb_max']} MB**\n")
        L.append(f"- Power mean: **{gpu_at_max['power_w_mean']} W**\n")
        if gpu_at_max["sm_pct_p95"] < 60:
            L.append("\n_SM% p95 < 60 — compute is not saturated. Bottleneck is "
                     "elsewhere (Python overhead, kernel launches, or batcher "
                     "tuning headroom)._\n")

    L.append("\n## Caveats\n"
             "- Closed-loop load; in-flight count held constant. Real arrivals "
             "are typically open-loop Poisson — the numbers here characterize "
             "server capacity, not user-arrival behavior.\n"
             "- TTFB measured client-to-first-byte on `localhost`; production "
             "RTT will add ~10–50 ms.\n"
             "- Single 5-minute window per concurrency. Thermal/clock-throttle "
             "behavior over multiple hours not characterized.\n"
             "- Single (mode, num_step) combo. For other combos see the main "
             "report `results/report.md`.\n")

    report_path.write_text("\n".join(L))
    print(f"[rps_analyze] wrote {report_path}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rps-dir", default="results/rps")
    ap.add_argument("--report", default="results/rps_report.md")
    ap.add_argument("--target-ttfb-ms", type=float, default=200.0)
    args = ap.parse_args()
    make_report(Path(args.rps_dir), Path(args.report), args.target_ttfb_ms)


if __name__ == "__main__":
    main()
