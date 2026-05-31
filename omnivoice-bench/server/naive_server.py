"""Naive FastAPI server: one model, one global asyncio.Lock around generate().

Concurrent HTTP requests queue up; the GPU sees one request at a time. This is
the baseline. Run as:

    python -m server.naive_server --host 0.0.0.0 --port 8000
"""
from __future__ import annotations

import argparse
import asyncio
import time
from pathlib import Path

from fastapi import FastAPI, HTTPException, Response
from pydantic import BaseModel
import uvicorn

from server import common


from concurrent.futures import ThreadPoolExecutor

app = FastAPI()
CTX: common.ServerCtx | None = None
LOCK = asyncio.Lock()
REF_AUDIO_DIR = Path("corpus/ref_audio")

# Single dedicated worker thread: torch.compile's Triton codecache is
# thread-affine in torch 2.8, so generation must run on the same thread the
# model was warmed/compiled on. See batched_server.py for the same fix.
GEN_EXECUTOR = ThreadPoolExecutor(max_workers=1, thread_name_prefix="gen")


class TTSRequest(BaseModel):
    text: str
    ref_audio_path: str
    ref_text: str
    language_id: str
    num_step: int = 32


@app.get("/health")
async def health():
    return {"ok": CTX is not None}


@app.post("/tts")
async def tts(req: TTSRequest):
    if CTX is None:
        raise HTTPException(503, "model not loaded")
    t_recv = time.perf_counter()
    async with LOCK:
        t_start = time.perf_counter()
        queue_wait_ms = (t_start - t_recv) * 1000.0
        try:
            loop = asyncio.get_event_loop()
            audio, gen_s = await loop.run_in_executor(
                GEN_EXECUTOR, common.generate_one,
                CTX, req.text, req.language_id,
                req.ref_audio_path, req.ref_text, req.num_step,
            )
        except Exception as e:
            raise HTTPException(500, f"gen_failed: {type(e).__name__}: {e}")
    audio_dur = common.audio_duration_s(audio, CTX.sampling_rate)
    wav_bytes = common.audio_to_wav_bytes(audio, CTX.sampling_rate)
    t_done = time.perf_counter()
    ttfb_ms = (t_done - t_recv) * 1000.0  # whole response ready

    headers = {
        "X-TTFB-ms": f"{ttfb_ms:.2f}",
        "X-Gen-ms": f"{gen_s*1000:.2f}",
        "X-Queue-Wait-ms": f"{queue_wait_ms:.2f}",
        "X-Audio-Duration-s": f"{audio_dur:.3f}",
    }
    return Response(content=wav_bytes, media_type="audio/wav", headers=headers)


def _startup(speaker_cache_path: Path):
    global CTX
    CTX = common.load_model()
    if speaker_cache_path.exists():
        common.load_speaker_cache(CTX, speaker_cache_path)
    # Warm/compile on the same dedicated thread that serves generation, so
    # torch.compile's thread-affine Triton kernels are valid at request time.
    fut = GEN_EXECUTOR.submit(common.warmup, CTX, REF_AUDIO_DIR)
    fut.result()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--speaker-cache", default="corpus/speaker_cache.pkl")
    args = ap.parse_args()
    _startup(Path(args.speaker_cache))
    uvicorn.run(app, host=args.host, port=args.port, workers=1, log_level="warning")


if __name__ == "__main__":
    main()
