"""Build a pool of UNIQUE multilingual prompts in the 100-300 char range.

The bundled phrases are short (18-63 chars), so we concatenate randomly-ordered
phrases within a language until the target length is reached. Every prompt is
deduplicated so no two requests across the whole burst sweep are identical
(prevents trivially-repeated inputs; note CUDA-graph recapture is keyed on token
*length* not text, so we also get length variety from the 100-300 range).
"""
from __future__ import annotations

import json
import random
from pathlib import Path

LANGS = ["hi", "bn", "ta", "te", "mr", "gu", "kn", "ml", "pa", "ory", "as", "ur"]


def load_phrase_pool(bundled_path: str = "corpus/bundled_prompts.json") -> dict[str, list[str]]:
    d = json.load(open(bundled_path))
    d.pop("_meta", None)
    pool = {}
    for lang in LANGS:
        if lang in d:
            pool[lang] = [t for cat in d[lang].values() for t in cat]
    return pool


def make_prompt(phrases: list[str], rng: random.Random,
                min_chars: int = 100, max_chars: int = 300) -> str:
    """Concatenate random phrases (with spaces) until within [min,max] chars."""
    target = rng.randint(min_chars, max_chars)
    parts: list[str] = []
    total = 0
    # shuffle a working copy so order varies
    bag = phrases[:]
    rng.shuffle(bag)
    i = 0
    while total < target:
        p = bag[i % len(bag)]
        parts.append(p)
        total += len(p) + 1
        i += 1
        if i > 200:  # safety
            break
    text = " ".join(parts)
    # trim to max_chars on a word/phrase boundary if we overshot a lot
    if len(text) > max_chars:
        # cut at last space before max_chars
        cut = text.rfind(" ", 0, max_chars)
        text = text[:cut] if cut > min_chars else text[:max_chars]
    return text.strip()


def build_pool(n_per_lang: int, seed: int = 0,
               min_chars: int = 100, max_chars: int = 300,
               bundled_path: str = "corpus/bundled_prompts.json") -> list[dict]:
    """Return a list of {language_id, text, char_len} dicts, unique texts."""
    rng = random.Random(seed)
    phrase_pool = load_phrase_pool(bundled_path)
    out: list[dict] = []
    seen: set[str] = set()
    for lang in LANGS:
        phrases = phrase_pool[lang]
        made = 0
        attempts = 0
        while made < n_per_lang and attempts < n_per_lang * 50:
            attempts += 1
            text = make_prompt(phrases, rng, min_chars, max_chars)
            if text in seen or len(text) < min_chars:
                continue
            seen.add(text)
            out.append({"language_id": lang, "text": text, "char_len": len(text)})
            made += 1
    rng.shuffle(out)
    return out


if __name__ == "__main__":
    import argparse, statistics
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-per-lang", type=int, default=60)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="corpus/burst_prompts.jsonl")
    args = ap.parse_args()
    pool = build_pool(args.n_per_lang, seed=args.seed)
    with open(args.out, "w") as f:
        for row in pool:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    lens = [r["char_len"] for r in pool]
    print(f"wrote {len(pool)} prompts -> {args.out}")
    print(f"char_len: min={min(lens)} max={max(lens)} mean={statistics.mean(lens):.0f}")
    from collections import Counter
    print("per lang:", dict(Counter(r["language_id"] for r in pool)))
