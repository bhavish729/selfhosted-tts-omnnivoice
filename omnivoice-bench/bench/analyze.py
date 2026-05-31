"""Read results/raw/ and produce results/report.md.

The report answers the seven questions from CLAUDE.md Step 7 in order:
  1. Max concurrency where p95 TTFB < 200 ms, per (server mode, num_step).
  2. Sustained throughput at that operating point.
  3. Per-language p95 TTFB at the best operating point.
  4. WER vs. num_step (from results/quality.csv if present).
  5. Naive vs. batched: is batching worth building?
  6. GPU utilization at the operating point.
  7. Recommended operating point.

Plots written to results/plots/.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


TARGET_P95_TTFB_MS = 200.0


def load_sweep(results_dir: Path) -> pd.DataFrame:
    idx = pd.read_csv(results_dir / "sweep_index.csv")
    # Filter out failed cells for headline numbers but keep them for the appendix.
    return idx


def load_raw(results_dir: Path, run_id: str) -> pd.DataFrame:
    p = results_dir / f"{run_id}.csv"
    if not p.exists():
        return pd.DataFrame()
    return pd.read_csv(p)


def load_gpu(results_dir: Path, run_id: str) -> pd.DataFrame:
    p = results_dir / f"{run_id}_gpu.csv"
    if not p.exists():
        return pd.DataFrame()
    return pd.read_csv(p)


def max_concurrency_under_target(df: pd.DataFrame, target_ms: float) -> pd.DataFrame:
    """For each (server_mode, num_step), return the row with the highest concurrency
    whose p95 TTFB is still <= target_ms and err_rate <= 5%."""
    ok = df[(df["ttfb_ms_p95"] <= target_ms) & (df["err_rate"].fillna(0) <= 0.05)]
    if ok.empty:
        return pd.DataFrame(columns=df.columns)
    return (
        ok.sort_values("concurrency", ascending=False)
        .groupby(["server_mode", "num_step"], as_index=False)
        .first()
    )


def plot_p95_vs_concurrency(df: pd.DataFrame, out_path: Path):
    plt.figure(figsize=(9, 6))
    for (mode, ns), grp in df.groupby(["server_mode", "num_step"]):
        grp = grp.sort_values("concurrency")
        plt.plot(grp["concurrency"], grp["ttfb_ms_p95"],
                 marker="o", label=f"{mode}, step={ns}")
    plt.axhline(TARGET_P95_TTFB_MS, linestyle="--", color="red", linewidth=1,
                label=f"target p95 = {TARGET_P95_TTFB_MS:.0f} ms")
    plt.xscale("log", base=2)
    plt.xlabel("Concurrency (in-flight requests)")
    plt.ylabel("p95 TTFB (ms)")
    plt.title("p95 TTFB vs. concurrency")
    plt.legend(fontsize=8, ncol=2)
    plt.grid(True, which="both", alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=120)
    plt.close()


def plot_throughput_vs_concurrency(df: pd.DataFrame, out_path: Path):
    plt.figure(figsize=(9, 6))
    for (mode, ns), grp in df.groupby(["server_mode", "num_step"]):
        grp = grp.sort_values("concurrency")
        plt.plot(grp["concurrency"], grp["throughput_audio_per_wall"],
                 marker="o", label=f"{mode}, step={ns}")
    plt.xscale("log", base=2)
    plt.xlabel("Concurrency")
    plt.ylabel("Throughput (audio-seconds / wall-second)")
    plt.title("Sustained throughput vs. concurrency")
    plt.legend(fontsize=8, ncol=2)
    plt.grid(True, which="both", alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=120)
    plt.close()


def per_language_p95(results_dir: Path, run_id: str) -> pd.DataFrame:
    raw = load_raw(results_dir, run_id)
    if raw.empty:
        return pd.DataFrame()
    raw = raw[raw["status"] == 200]
    g = raw.groupby("lang")["ttfb_ms"].agg(
        p50=lambda s: np.percentile(s, 50),
        p95=lambda s: np.percentile(s, 95),
        p99=lambda s: np.percentile(s, 99),
        count="count",
    )
    return g.reset_index().sort_values("p95", ascending=False)


def gpu_summary_for_run(results_dir: Path, run_id: str) -> dict:
    gpu = load_gpu(results_dir, run_id)
    if gpu.empty:
        return {}
    return {
        "sm_pct_mean": round(gpu["sm_pct"].mean(), 1),
        "sm_pct_p95": round(np.percentile(gpu["sm_pct"], 95), 1),
        "vram_used_mb_max": int(gpu["vram_used_mb"].max()),
        "power_w_mean": round(gpu["power_w"].mean(), 1),
    }


def make_report(results_dir: Path, report_path: Path, quality_csv: Path | None):
    plots_dir = results_dir.parent / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    df = load_sweep(results_dir)
    df["ttfb_ms_p95"] = pd.to_numeric(df["ttfb_ms_p95"], errors="coerce")
    df["throughput_audio_per_wall"] = pd.to_numeric(df["throughput_audio_per_wall"], errors="coerce")
    df["err_rate"] = pd.to_numeric(df["err_rate"], errors="coerce").fillna(0)

    plot_p95_vs_concurrency(df, plots_dir / "p95_ttfb_vs_concurrency.png")
    plot_throughput_vs_concurrency(df, plots_dir / "throughput_vs_concurrency.png")

    max_conc = max_concurrency_under_target(df, TARGET_P95_TTFB_MS)

    # Pivot: rows=num_step, cols=server_mode, cells=max concurrency.
    if not max_conc.empty:
        conc_pivot = max_conc.pivot(index="num_step", columns="server_mode", values="concurrency").fillna(0).astype(int)
        thr_pivot = max_conc.pivot(index="num_step", columns="server_mode", values="throughput_audio_per_wall").round(2)
    else:
        conc_pivot = pd.DataFrame()
        thr_pivot = pd.DataFrame()

    # Pick the single best operating point: highest throughput among cells meeting target.
    best = None
    if not max_conc.empty:
        best = max_conc.sort_values("throughput_audio_per_wall", ascending=False).iloc[0].to_dict()

    # Per-language breakdown at the best operating point.
    lang_table = pd.DataFrame()
    gpu_summary = {}
    if best:
        lang_table = per_language_p95(results_dir, best["run_id"])
        gpu_summary = gpu_summary_for_run(results_dir, best["run_id"])

    # Quality / WER table if available.
    wer_table = pd.DataFrame()
    if quality_csv and quality_csv.exists():
        wer_table = pd.read_csv(quality_csv)

    # Naive vs batched delta.
    naive_best = max_conc[max_conc["server_mode"] == "naive"].sort_values(
        "throughput_audio_per_wall", ascending=False
    )
    batched_best = max_conc[max_conc["server_mode"].str.startswith("batched")].sort_values(
        "throughput_audio_per_wall", ascending=False
    )
    naive_thr = naive_best.iloc[0]["throughput_audio_per_wall"] if not naive_best.empty else float("nan")
    batched_thr = batched_best.iloc[0]["throughput_audio_per_wall"] if not batched_best.empty else float("nan")
    uplift = (batched_thr / naive_thr - 1.0) * 100 if naive_thr and naive_thr > 0 else float("nan")

    # Build the markdown.
    L = []
    L.append("# OmniVoice H100 Concurrency & Throughput Benchmark — Results\n")
    if best:
        L.append(
            f"## Headline\n"
            f"**On a single H100, OmniVoice supports {int(best['concurrency'])} concurrent Indic-language TTS "
            f"requests at p95 TTFB {best['ttfb_ms_p95']:.0f} ms, producing "
            f"{best['throughput_audio_per_wall']:.2f} seconds of audio per second of wall-clock** "
            f"(server={best['server_mode']}, num_step={int(best['num_step'])}).\n"
        )
    else:
        L.append("## Headline\n**No cell met the p95 TTFB < 200 ms target.** See breakdown below.\n")

    L.append("\n## 1. Max concurrency at p95 TTFB < 200 ms\n")
    if conc_pivot.empty:
        L.append("_No cell met the target._\n")
    else:
        L.append(conc_pivot.to_markdown() + "\n")
    L.append("\n![p95 TTFB vs. concurrency](plots/p95_ttfb_vs_concurrency.png)\n")

    L.append("\n## 2. Sustained throughput at that operating point\n")
    L.append("Audio-seconds generated per wall-clock second:\n\n")
    if thr_pivot.empty:
        L.append("_No data._\n")
    else:
        L.append(thr_pivot.to_markdown() + "\n")
    L.append("\n![Throughput vs. concurrency](plots/throughput_vs_concurrency.png)\n")

    L.append("\n## 3. Per-language picture (at the best operating point)\n")
    if not lang_table.empty:
        L.append(lang_table.round(1).to_markdown(index=False) + "\n")
        outliers = lang_table[lang_table["p95"] > lang_table["p95"].median() * 1.5]
        if not outliers.empty:
            L.append("\n**Outlier languages (p95 > 1.5x median):** "
                     + ", ".join(outliers["lang"]) + "\n")
    else:
        L.append("_No per-language data available (no qualifying run)._\n")

    L.append("\n## 4. Quality vs. speed (WER by num_step)\n")
    if not wer_table.empty:
        L.append(wer_table.round(3).to_markdown(index=False) + "\n")
    else:
        L.append("_results/quality.csv not found — run the quality spot-check (Step 6)._\n")

    L.append("\n## 5. Naive vs. batched: is batching worth building?\n")
    if not np.isnan(uplift):
        L.append(f"Best naive throughput: **{naive_thr:.2f}** audio-s/wall-s.\n")
        L.append(f"Best batched throughput: **{batched_thr:.2f}** audio-s/wall-s.\n")
        L.append(f"**Uplift: {uplift:+.1f}%.**\n\n")
        if uplift < 30:
            L.append(
                "Recommendation: **stay with naive serving** and scale horizontally. "
                "Batched mode gives <30% uplift — the engineering cost of a continuous batcher is not justified here.\n"
            )
        else:
            L.append(
                "Recommendation: **build the batched server.** The uplift is meaningful and the "
                "batcher is ~150 LoC.\n"
            )
    else:
        L.append("_Not enough data to compare._\n")

    L.append("\n## 6. GPU utilization at the operating point\n")
    if gpu_summary:
        L.append(f"- SM% mean: **{gpu_summary['sm_pct_mean']}**\n")
        L.append(f"- SM% p95: **{gpu_summary['sm_pct_p95']}**\n")
        L.append(f"- VRAM peak: **{gpu_summary['vram_used_mb_max']} MB**\n")
        L.append(f"- Power mean: **{gpu_summary['power_w_mean']} W**\n")
        if gpu_summary["sm_pct_p95"] < 60:
            L.append(
                "\n**SM% p95 < 60 — compute is NOT the bottleneck.** Likely Python overhead, kernel "
                "launch latency, or lack of true batched inference at the model level. There is "
                "performance left on the table that this benchmark configuration cannot unlock.\n"
            )
    else:
        L.append("_No GPU data for best run._\n")

    L.append("\n## 7. Recommended operating point for TaraVoice production\n")
    if best:
        L.append(
            f"- **Server:** `{best['server_mode']}`\n"
            f"- **`num_step`:** {int(best['num_step'])}\n"
            f"- **Concurrent calls supported:** {int(best['concurrency'])}\n"
            f"- **p95 TTFB:** {best['ttfb_ms_p95']:.0f} ms (target was 200 ms)\n"
            f"- **Sustained throughput:** {best['throughput_audio_per_wall']:.2f} audio-s/wall-s\n"
        )
        if not wer_table.empty:
            L.append("- **WER on the spot-check set:** see Section 4.\n")
        L.append(
            "\n**Assumptions:** utterance length 3–8 s; balanced 12-language Indic mix; "
            "single H100 with no STT/LLM co-tenant; closed-loop client model; reference audio "
            "pre-cached.\n"
        )
    else:
        L.append("_No cell met the target. See Section 1._\n")

    L.append("\n## Caveats and what this benchmark does NOT measure\n")
    L.append(
        "- **Cold-start latency**: warmup is excluded; the first request after a deploy will be slower.\n"
        "- **Multi-tenancy**: STT and LLM were not co-resident on the GPU.\n"
        "- **Sustained 1-hour load**: each cell runs ~300 requests; thermal/clock-throttle behavior over hours is not characterized.\n"
        "- **Network jitter**: client and server are on `localhost`; real client RTT will add to TTFB.\n"
        "- **Voice quality**: only WER was checked. No MOS, no naturalness evaluation, no native-speaker listening.\n"
        "- **Reference audio**: self-generated via OmniVoice voice-design mode (no native-speaker samples in the bench corpus).\n"
    )

    report_path.write_text("\n".join(L))
    print(f"[analyze] wrote {report_path}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", default="results/raw")
    ap.add_argument("--report", default="results/report.md")
    ap.add_argument("--quality", default="results/quality.csv")
    args = ap.parse_args()
    make_report(Path(args.results_dir), Path(args.report), Path(args.quality))


if __name__ == "__main__":
    main()
