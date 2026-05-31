"""Build the benchmark corpus.

Outputs:
  corpus/ref_audio/<lang>.wav         — one ~5s ref clip per language
  corpus/speaker_cache.pkl            — pickled dict[lang -> VoiceClonePrompt]
  corpus/prompts.jsonl                — N prompts (default 50) per lang x 12 langs

Prompts cover 7 collections-domain categories: greeting, payment_reminder,
due_date, ptp, no_money_rebuttal, confirmation, close.

The bundled per-language phrasings live in corpus/bundled_prompts.json — they
are short, native-script lines (4 per category x 7 categories = 28 per lang),
sampled with replacement to reach N per language. Caveat in the final report:
these were produced from the assistant's multilingual knowledge and have not
been editorially reviewed by native speakers. Use --source to override with a
JSONL of production-quality lines: {language_id, text} per row.

Reference audio is generated via OmniVoice's voice-design mode (instruct=...).
Per the model docs, `instruct` attributes were trained on EN/ZH only, but the
`language` parameter still drives phonetics, so the per-language refs are valid
acoustic anchors even if voice character is off-distribution.
"""
from __future__ import annotations

import argparse
import json
import pickle
import random
import time
from pathlib import Path

import numpy as np
import soundfile as sf

from server import common


# Common short greeting per language — used for the reference clip text.
REF_TEMPLATE = {
    "hi":  "नमस्ते, मैं आपकी मदद के लिए हूँ। कृपया एक मिनट का समय दीजिए।",
    "bn":  "নমস্কার, আমি আপনাকে সাহায্য করতে এসেছি। দয়া করে এক মিনিট সময় দিন।",
    "ta":  "வணக்கம், நான் உங்களுக்கு உதவ வந்துள்ளேன். தயவுசெய்து ஒரு நிமிடம் தாருங்கள்.",
    "te":  "నమస్తే, నేను మీకు సహాయం చేయడానికి వచ్చాను. దయచేసి ఒక నిమిషం సమయం ఇవ్వండి.",
    "mr":  "नमस्कार, मी तुम्हाला मदत करण्यासाठी आहे. कृपया एक मिनिट वेळ द्या.",
    "gu":  "નમસ્તે, હું તમારી મદદ માટે છું. કૃપા કરી એક મિનિટ સમય આપો.",
    "kn":  "ನಮಸ್ಕಾರ, ನಾನು ನಿಮಗೆ ಸಹಾಯ ಮಾಡಲು ಬಂದಿದ್ದೇನೆ. ದಯವಿಟ್ಟು ಒಂದು ನಿಮಿಷ ಸಮಯ ಕೊಡಿ.",
    "ml":  "നമസ്കാരം, ഞാൻ നിങ്ങളെ സഹായിക്കാൻ വന്നതാണ്. ദയവായി ഒരു മിനിറ്റ് സമയം തരൂ.",
    "pa":  "ਸਤ ਸ੍ਰੀ ਅਕਾਲ, ਮੈਂ ਤੁਹਾਡੀ ਮਦਦ ਲਈ ਆਈ ਹਾਂ। ਕਿਰਪਾ ਕਰਕੇ ਇੱਕ ਮਿੰਟ ਸਮਾਂ ਦਿਓ।",
    "ory": "ନମସ୍କାର, ମୁଁ ଆପଣଙ୍କ ସାହାଯ୍ୟ ପାଇଁ ଆସିଛି। ଦୟାକରି ଗୋଟିଏ ମିନିଟ ସମୟ ଦିଅନ୍ତୁ।",
    "as":  "নমস্কাৰ, মই আপোনাক সহায় কৰিবলৈ আহিছোঁ। অনুগ্ৰহ কৰি এক মিনিট সময় দিয়ক।",
    "ur":  "السلام علیکم، میں آپ کی مدد کے لیے حاضر ہوں۔ براہ کرم ایک منٹ وقت دیں۔",
}


def load_bundled(path: Path) -> dict[str, dict[str, list[str]]]:
    with path.open() as f:
        d = json.load(f)
    d.pop("_meta", None)
    return d


def expand_prompts(category_map: dict[str, list[str]], lang: str, n: int,
                   rng: random.Random) -> list[dict]:
    """Sample `n` prompts evenly across categories with replacement."""
    items: list[dict] = []
    cats = list(category_map.keys())
    flat: list[tuple[str, str]] = [(cat, text) for cat in cats for text in category_map[cat]]
    if not flat:
        return []
    while len(items) < n:
        cat, text = rng.choice(flat)
        items.append({"category": cat, "text": text, "language_id": lang})
    return items


def generate_ref_audio(ctx, lang: str, out_path: Path, gen_cfg) -> None:
    text = REF_TEMPLATE[lang]
    print(f"[corpus] ref audio: {lang} -> {out_path}", flush=True)
    audios = ctx.model.generate(
        text=text,
        language=lang,
        instruct="female, indian accent",
        generation_config=gen_cfg,
    )
    sf.write(out_path, audios[0], ctx.sampling_rate)


def precompute_speaker_cache(ctx, langs: list[str], ref_dir: Path,
                              cache_path: Path) -> None:
    cache: dict = {}
    for lang in langs:
        ref_audio = ref_dir / f"{lang}.wav"
        ref_text = REF_TEMPLATE[lang]
        print(f"[corpus] precompute voice prompt: {lang}", flush=True)
        prompt = ctx.model.create_voice_clone_prompt(
            ref_audio=str(ref_audio), ref_text=ref_text, preprocess_prompt=True,
        )
        for attr in ("ref_audio_tokens",):
            t = getattr(prompt, attr, None)
            if t is not None and hasattr(t, "cpu"):
                setattr(prompt, attr, t.cpu())
        cache[lang] = prompt
    with cache_path.open("wb") as f:
        pickle.dump(cache, f)
    print(f"[corpus] wrote {cache_path} ({len(cache)} entries)", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-lang", type=int, default=50)
    ap.add_argument("--source", default=None,
                    help="Optional JSONL of production lines: {language_id, text}")
    ap.add_argument("--bundled", default="corpus/bundled_prompts.json")
    ap.add_argument("--ref-dir", default="corpus/ref_audio")
    ap.add_argument("--cache-path", default="corpus/speaker_cache.pkl")
    ap.add_argument("--out", default="corpus/prompts.jsonl")
    ap.add_argument("--skip-ref", action="store_true",
                    help="Skip ref audio + speaker cache (don't load model).")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    ref_dir = Path(args.ref_dir); ref_dir.mkdir(parents=True, exist_ok=True)
    out_path = Path(args.out); out_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path = Path(args.cache_path)

    if args.source:
        per_lang_pool: dict[str, dict[str, list[str]]] = {}
        grouped: dict[str, list[str]] = {}
        with open(args.source) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                grouped.setdefault(row["language_id"], []).append(row["text"])
        for lang, texts in grouped.items():
            per_lang_pool[lang] = {"mixed": texts}
        print(f"[corpus] source: {args.source}, langs={sorted(per_lang_pool)}", flush=True)
    else:
        per_lang_pool = load_bundled(Path(args.bundled))
        print(f"[corpus] bundled prompts: langs={sorted(per_lang_pool)}", flush=True)

    written = 0
    with out_path.open("w") as f:
        for lang in common.INDIC_LANGS:
            if lang not in per_lang_pool:
                print(f"[corpus] skip {lang}: no source prompts", flush=True)
                continue
            cats = per_lang_pool[lang]
            items = expand_prompts(cats, lang, args.per_lang, rng)
            for i, item in enumerate(items):
                row = {
                    "id": f"{lang}_{i:04d}",
                    "language_id": lang,
                    "text": item["text"],
                    "category": item["category"],
                    "ref_audio": str(ref_dir / f"{lang}.wav"),
                    "ref_text": REF_TEMPLATE[lang],
                }
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
                written += 1
    print(f"[corpus] wrote {written} prompts to {out_path}", flush=True)

    if args.skip_ref:
        print("[corpus] --skip-ref: not generating ref audio / speaker cache.", flush=True)
        return

    print("[corpus] loading OmniVoice for ref audio + speaker cache ...", flush=True)
    ctx = common.load_model(load_asr=False)
    gen_cfg = common.make_gen_cfg(ctx, num_step=32)

    for lang in common.INDIC_LANGS:
        target = ref_dir / f"{lang}.wav"
        if not target.exists():
            generate_ref_audio(ctx, lang, target, gen_cfg)

    precompute_speaker_cache(ctx, common.INDIC_LANGS, ref_dir, cache_path)
    print("[corpus] done.", flush=True)


if __name__ == "__main__":
    main()
