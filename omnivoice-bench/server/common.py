"""Shared OmniVoice model loader, warmup, and per-request helpers.

Key facts about the model API (verified against the master branch):
  - Class: `omnivoice.OmniVoice`. Loader: `OmniVoice.from_pretrained(...)`.
  - generate() accepts `text` as str OR list[str]; returns list[np.ndarray].
  - Per-call decoding knobs live on OmniVoiceGenerationConfig, passed via
    `generation_config=...`. `num_step` is the diffusion-step count.
  - `create_voice_clone_prompt(ref_audio, ref_text)` returns a
    VoiceClonePrompt that can be reused via `voice_clone_prompt=...` to skip
    re-tokenising the reference on every call. This is the production pattern
    and is what the bench measures.
  - Sampling rate: `model.sampling_rate` (24000 in practice).
  - `load_asr=False` skips loading Whisper since we always supply ref_text.
"""
from __future__ import annotations

import io
import json
import os
import pickle
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import soundfile as sf
import torch


# Indic language IDs we benchmark. "hi" is also used for Hinglish prompts since
# OmniVoice has no dedicated code-switch tag.
INDIC_LANGS = [
    "hi", "bn", "ta", "te", "mr", "gu",
    "kn", "ml", "pa", "ory", "as", "ur",
]


@dataclass
class ServerCtx:
    model: object
    gen_cfg_cls: type
    voice_prompt_cls: type
    speaker_cache: dict  # lang_id -> VoiceClonePrompt
    sampling_rate: int


def load_model(model_id: str = "k2-fsa/OmniVoice",
               device: str = "cuda:0",
               dtype=torch.float16,
               load_asr: bool = False) -> ServerCtx:
    """Load OmniVoice into GPU memory and return a server context."""
    from omnivoice import OmniVoice  # type: ignore
    from omnivoice.models.omnivoice import (  # type: ignore
        OmniVoiceGenerationConfig,
        VoiceClonePrompt,
    )
    print(f"[common] loading {model_id} on {device} dtype={dtype} load_asr={load_asr}", flush=True)
    t0 = time.perf_counter()
    model = OmniVoice.from_pretrained(
        model_id, device_map=device, dtype=dtype, load_asr=load_asr,
    )
    t1 = time.perf_counter()
    sr = getattr(model, "sampling_rate", 24000)
    print(f"[common] model loaded in {t1-t0:.1f}s (sr={sr})", flush=True)
    return ServerCtx(
        model=model,
        gen_cfg_cls=OmniVoiceGenerationConfig,
        voice_prompt_cls=VoiceClonePrompt,
        speaker_cache={},
        sampling_rate=sr,
    )


def load_speaker_cache(ctx: ServerCtx, cache_path: Path) -> None:
    """Load precomputed VoiceClonePrompt objects (built by corpus/build_corpus.py)
    and attach to ctx.speaker_cache. Falls back to building on-the-fly from
    ref_audio paths if no cache file exists."""
    if not cache_path.exists():
        print(f"[common] no speaker cache at {cache_path}; will build per-call", flush=True)
        return
    with cache_path.open("rb") as f:
        cache = pickle.load(f)
    # Move tensors to GPU once.
    for lang, prompt in cache.items():
        for attr in ("ref_audio_tokens",):
            t = getattr(prompt, attr, None)
            if isinstance(t, torch.Tensor):
                setattr(prompt, attr, t.to(ctx.model.device))
        ctx.speaker_cache[lang] = prompt
    print(f"[common] speaker cache: {sorted(ctx.speaker_cache)}", flush=True)


def ensure_voice_prompt(ctx: ServerCtx, lang: str, ref_audio: str, ref_text: str):
    """Return a cached VoiceClonePrompt for `lang`, building+caching it if needed."""
    if lang in ctx.speaker_cache:
        return ctx.speaker_cache[lang]
    prompt = ctx.model.create_voice_clone_prompt(
        ref_audio=ref_audio, ref_text=ref_text, preprocess_prompt=True,
    )
    ctx.speaker_cache[lang] = prompt
    return prompt


def make_gen_cfg(ctx: ServerCtx, num_step: int):
    return ctx.gen_cfg_cls(num_step=num_step)


# Instruct mode: when enabled, generation uses voice-design `instruct=` instead
# of voice-cloning from a reference clip. This avoids the loudness-inheritance
# bug (self-generated reference clips can be near-silent, making cloned output
# silent). Instruct mode produces healthy audio at identical latency. Toggle via
# the OMNIVOICE_INSTRUCT env var (set to "0" to use voice-clone instead).
INSTRUCT_MODE = os.environ.get("OMNIVOICE_INSTRUCT", "1") != "0"
INSTRUCT_TEXT = os.environ.get("OMNIVOICE_INSTRUCT_TEXT", "female, indian accent")


def warmup(ctx: ServerCtx, ref_audio_dir: Path | None = None,
           steps_to_warm: tuple[int, ...] = (8, 16, 32)) -> None:
    """Run a handful of generations with mixed num_step + langs to settle cuDNN/SDPA.
    Discards outputs. If no ref audio exists yet, uses voice-design mode."""
    print("[common] warmup ...", flush=True)
    t0 = time.perf_counter()
    sample_text = {
        "hi": "नमस्ते, मेरा नाम आरती है।",
        "ta": "வணக்கம், என் பெயர் ஆர்த்தி.",
        "bn": "নমস্কার, আমার নাম আরতি।",
    }
    for step in steps_to_warm:
        cfg = make_gen_cfg(ctx, step)
        for lang, text in sample_text.items():
            try:
                ref_wav = ref_audio_dir / f"{lang}.wav" if ref_audio_dir else None
                if ref_wav and ref_wav.exists() and lang in ctx.speaker_cache:
                    _ = ctx.model.generate(
                        text=text,
                        language=lang,
                        voice_clone_prompt=ctx.speaker_cache[lang],
                        generation_config=cfg,
                    )
                else:
                    _ = ctx.model.generate(
                        text=text,
                        language=lang,
                        instruct="female, indian accent",
                        generation_config=cfg,
                    )
            except Exception as e:
                print(f"[common] warmup skip lang={lang} step={step}: {e}", flush=True)
    torch.cuda.synchronize()
    print(f"[common] warmup done in {time.perf_counter()-t0:.1f}s", flush=True)


def generate_one(ctx: ServerCtx, text: str, lang: str,
                 ref_audio: str, ref_text: str,
                 num_step: int) -> tuple[np.ndarray, float]:
    """Generate a single utterance. Returns (audio_np, wall_seconds)."""
    cfg = make_gen_cfg(ctx, num_step)
    t0 = time.perf_counter()
    if INSTRUCT_MODE:
        audios = ctx.model.generate(
            text=text,
            language=lang,
            instruct=INSTRUCT_TEXT,
            generation_config=cfg,
        )
    else:
        prompt = ensure_voice_prompt(ctx, lang, ref_audio, ref_text)
        audios = ctx.model.generate(
            text=text,
            language=lang,
            voice_clone_prompt=prompt,
            generation_config=cfg,
        )
    torch.cuda.synchronize()
    t1 = time.perf_counter()
    audio = audios[0]
    return audio, (t1 - t0)


def generate_batch(ctx: ServerCtx, items: list[dict], num_step: int
                   ) -> tuple[list[np.ndarray], float]:
    """Batched generation. `items` is a list of dicts with keys
    text/lang/ref_audio/ref_text. Returns (list_of_audio_np, wall_seconds)."""
    texts, langs = [], []
    for it in items:
        texts.append(it["text"])
        langs.append(it["lang"])
    cfg = make_gen_cfg(ctx, num_step)
    t0 = time.perf_counter()
    if INSTRUCT_MODE:
        audios = ctx.model.generate(
            text=texts,
            language=langs,
            instruct=[INSTRUCT_TEXT] * len(texts),
            generation_config=cfg,
        )
    else:
        prompts = [ensure_voice_prompt(ctx, it["lang"], it["ref_audio"], it["ref_text"])
                   for it in items]
        audios = ctx.model.generate(
            text=texts,
            language=langs,
            voice_clone_prompt=prompts,
            generation_config=cfg,
        )
    torch.cuda.synchronize()
    t1 = time.perf_counter()
    return audios, (t1 - t0)


def audio_to_wav_bytes(audio: np.ndarray, sample_rate: int) -> bytes:
    buf = io.BytesIO()
    sf.write(buf, audio, sample_rate, format="WAV", subtype="PCM_16")
    return buf.getvalue()


def audio_duration_s(audio: np.ndarray, sample_rate: int) -> float:
    return float(len(audio)) / float(sample_rate)
