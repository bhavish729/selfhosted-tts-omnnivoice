"""Generate quality-review audio samples across all 12 Indic languages.

For each language, generates one sample per category (greeting, payment_reminder,
ptp, close) so you can hear a diverse acoustic + content set per language.
Outputs to `samples/<lang>/<lang>_<category>.wav` with the prompt text saved
alongside in `samples/<lang>/<lang>_<category>.txt` so you can verify the
spoken content matches.

Usage:
    python -m bench.generate_samples [--num-step 32] [--out samples]
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import soundfile as sf

from server import common


# Diverse category set to surface different acoustic content per language.
CATEGORIES = ["greeting", "payment_reminder", "ptp", "close"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bundled", default="corpus/bundled_prompts.json")
    ap.add_argument("--ref-dir", default="corpus/ref_audio")
    ap.add_argument("--cache-path", default="corpus/speaker_cache.pkl")
    ap.add_argument("--num-step", type=int, default=32,
                    help="32 = max quality (slowest), 16 = balanced, 8 = fastest")
    ap.add_argument("--out", default="samples")
    ap.add_argument("--also-regenerate-refs", action="store_true",
                    help="Also re-generate every reference WAV from voice-design.")
    args = ap.parse_args()

    out_root = Path(args.out)
    out_root.mkdir(parents=True, exist_ok=True)

    with open(args.bundled) as f:
        bundled = json.load(f)
    bundled.pop("_meta", None)

    print(f"[samples] loading model (num_step={args.num_step}) ...", flush=True)
    ctx = common.load_model(load_asr=False)
    common.load_speaker_cache(ctx, Path(args.cache_path))
    gen_cfg = common.make_gen_cfg(ctx, num_step=args.num_step)

    if args.also_regenerate_refs:
        print("[samples] regenerating reference WAVs ...", flush=True)
        from corpus.build_corpus import REF_TEMPLATE, generate_ref_audio
        ref_dir = Path(args.ref_dir)
        ref_dir.mkdir(parents=True, exist_ok=True)
        for lang in common.INDIC_LANGS:
            generate_ref_audio(ctx, lang, ref_dir / f"{lang}.wav", gen_cfg)
        # Re-prime the speaker cache from the fresh refs.
        ctx.speaker_cache.clear()

    manifest = []
    t0 = time.time()
    for lang in common.INDIC_LANGS:
        lang_dir = out_root / lang
        lang_dir.mkdir(parents=True, exist_ok=True)
        if lang not in bundled:
            print(f"[samples] skip {lang}: no bundled prompts", flush=True)
            continue
        ref_audio = str(Path(args.ref_dir) / f"{lang}.wav")
        # Pull ref_text from corpus build templates.
        from corpus.build_corpus import REF_TEMPLATE
        ref_text = REF_TEMPLATE[lang]

        for cat in CATEGORIES:
            phrasings = bundled[lang].get(cat) or bundled[lang].get("mixed") or []
            if not phrasings:
                continue
            text = phrasings[0]  # First phrasing is the canonical one.
            t_start = time.perf_counter()
            audio, gen_s = common.generate_one(
                ctx, text, lang, ref_audio, ref_text, args.num_step,
            )
            wav_path = lang_dir / f"{lang}_{cat}.wav"
            sf.write(wav_path, audio, ctx.sampling_rate)
            txt_path = wav_path.with_suffix(".txt")
            txt_path.write_text(text + "\n")
            dur = len(audio) / ctx.sampling_rate
            print(f"[samples] {lang}/{cat:18s} dur={dur:.2f}s gen={gen_s*1000:.0f}ms "
                  f"-> {wav_path}", flush=True)
            manifest.append({
                "language_id": lang, "category": cat, "text": text,
                "wav": str(wav_path), "duration_s": round(dur, 2),
                "gen_ms": round(gen_s * 1000, 1), "num_step": args.num_step,
            })

    manifest_path = out_root / "MANIFEST.jsonl"
    with manifest_path.open("w") as f:
        for row in manifest:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"\n[samples] generated {len(manifest)} samples in {time.time()-t0:.1f}s "
          f"-> {out_root}/", flush=True)
    print(f"[samples] manifest: {manifest_path}", flush=True)


if __name__ == "__main__":
    main()
