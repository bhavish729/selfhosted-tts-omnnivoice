"""WER spot-check for the num_step quality tradeoff.

For each num_step in {8, 16, 32}, generate the SAME 2 prompts per language
(24 total) and save WAVs under results/quality_samples/num_step_<N>/.
Then transcribe with faster-whisper large-v3 and compute mean WER per
(language, num_step). Outputs results/quality.csv.

Note: WER on synthesized speech with a different language's writing system
is sensitive to the STT's tokenization choices. The numbers here are
indicative — a steep WER jump from num_step=16 to num_step=8 is the signal
the report should care about, not absolute WER levels.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf

from server import common


WS_RE = re.compile(r"\s+")


def _normalize(s: str) -> list[str]:
    s = s.strip().lower()
    s = re.sub(r"[।,.!?;:\"'\(\)\[\]।؟،]", " ", s)
    s = WS_RE.sub(" ", s).strip()
    return s.split()


def wer(ref: str, hyp: str) -> float:
    r = _normalize(ref)
    h = _normalize(hyp)
    if not r:
        return 0.0
    # Standard Levenshtein-on-words.
    n, m = len(r), len(h)
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1): dp[i][0] = i
    for j in range(m + 1): dp[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = 0 if r[i-1] == h[j-1] else 1
            dp[i][j] = min(
                dp[i-1][j] + 1,
                dp[i][j-1] + 1,
                dp[i-1][j-1] + cost,
            )
    return dp[n][m] / n


# Mapping: our internal codes -> faster-whisper language codes.
# Whisper supports most of ours; "ory" is "or" in Whisper, "as" not officially
# supported (fall back to "bn" which uses similar script — flagged as caveat).
WHISPER_LANG = {
    "hi": "hi", "bn": "bn", "ta": "ta", "te": "te", "mr": "mr",
    "gu": "gu", "kn": "kn", "ml": "ml", "pa": "pa",
    "ory": "or", "as": None, "ur": "ur",
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompts-file", default="corpus/prompts.jsonl")
    ap.add_argument("--out-dir", default="results/quality_samples")
    ap.add_argument("--per-lang", type=int, default=2)
    ap.add_argument("--num-steps", type=int, nargs="+", default=[8, 16, 32])
    ap.add_argument("--out-csv", default="results/quality.csv")
    ap.add_argument("--whisper-model", default="large-v3")
    args = ap.parse_args()

    # 1) Pick 2 prompts per language (deterministic: first two).
    prompts: dict[str, list[dict]] = {}
    with open(args.prompts_file) as f:
        for line in f:
            row = json.loads(line)
            prompts.setdefault(row["language_id"], []).append(row)
    test_set: list[dict] = []
    for lang, rows in prompts.items():
        test_set.extend(rows[: args.per_lang])
    print(f"[quality] {len(test_set)} prompts across {len(prompts)} langs", flush=True)

    # 2) Generate for each num_step, save WAVs.
    ctx = common.load_model(load_asr=False)
    common.load_speaker_cache(ctx, Path("corpus/speaker_cache.pkl"))
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for ns in args.num_steps:
        ns_dir = out_dir / f"num_step_{ns}"
        ns_dir.mkdir(parents=True, exist_ok=True)
        for row in test_set:
            wav_path = ns_dir / f"{row['id']}.wav"
            if wav_path.exists():
                continue
            audio, _ = common.generate_one(
                ctx, row["text"], row["language_id"],
                row["ref_audio"], row["ref_text"], ns,
            )
            sf.write(wav_path, audio, ctx.sampling_rate)
        print(f"[quality] num_step={ns}: {len(test_set)} WAVs in {ns_dir}", flush=True)

    # 3) Transcribe + WER.
    print(f"[quality] loading faster-whisper {args.whisper_model} ...", flush=True)
    from faster_whisper import WhisperModel  # type: ignore
    whisper = WhisperModel(args.whisper_model, device="cuda", compute_type="float16")

    rows = []
    for ns in args.num_steps:
        ns_dir = out_dir / f"num_step_{ns}"
        for prompt in test_set:
            wav_path = ns_dir / f"{prompt['id']}.wav"
            lang = prompt["language_id"]
            whisper_lang = WHISPER_LANG.get(lang)
            if whisper_lang is None:
                rows.append({
                    "language_id": lang, "num_step": ns, "wer": float("nan"),
                    "note": "no_whisper_support",
                })
                continue
            segments, _ = whisper.transcribe(
                str(wav_path), language=whisper_lang, beam_size=1,
            )
            hyp = " ".join(s.text for s in segments)
            w = wer(prompt["text"], hyp)
            rows.append({
                "language_id": lang, "num_step": ns, "wer": w,
                "hyp": hyp[:200], "ref": prompt["text"][:200],
            })
            print(f"[quality] {lang} step={ns} WER={w:.3f}", flush=True)

    df = pd.DataFrame(rows)
    # Mean WER per (lang, num_step), excluding nans.
    agg = (
        df.dropna(subset=["wer"])
        .groupby(["language_id", "num_step"])["wer"]
        .agg(["mean", "count"])
        .reset_index()
        .rename(columns={"mean": "wer_mean", "count": "n"})
    )
    agg.to_csv(args.out_csv, index=False)
    df.to_csv(Path(args.out_csv).with_suffix(".raw.csv"), index=False)
    print(f"[quality] wrote {args.out_csv}", flush=True)


if __name__ == "__main__":
    main()
