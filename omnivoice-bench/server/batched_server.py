"""Continuous-batching FastAPI server for OmniVoice.

Same /tts contract as naive_server. A background coroutine pulls pending
requests off an asyncio.Queue, waits up to MAX_BATCH_WAIT_MS to accumulate
up to MAX_BATCH_SIZE items, then calls model.generate(text=[...], ...) once.

OmniVoice does support list inputs to generate() natively, so this is real
GPU-level batching (not loop-pretend).
"""
from __future__ import annotations

import argparse
import asyncio
import time
from dataclasses import dataclass, field
from pathlib import Path

from fastapi import FastAPI, HTTPException, Response
from pydantic import BaseModel
import uvicorn

from server import common


from concurrent.futures import ThreadPoolExecutor

app = FastAPI()
CTX: common.ServerCtx | None = None
QUEUE: asyncio.Queue | None = None
BATCHER_TASK: asyncio.Task | None = None
MAX_BATCH_SIZE = 8
MAX_BATCH_WAIT_MS = 20.0
REF_AUDIO_DIR = Path("corpus/ref_audio")

# Single dedicated worker thread for all model generation. torch.compile's
# Triton codecache is thread-affine in torch 2.8, so generation MUST run on the
# same thread the model was warmed/compiled on. Using one pinned executor thread
# (instead of asyncio.to_thread's arbitrary pool threads) guarantees this and
# also serialises GPU access correctly for the single model instance.
GEN_EXECUTOR = ThreadPoolExecutor(max_workers=1, thread_name_prefix="gen")


class TTSRequest(BaseModel):
    text: str
    ref_audio_path: str
    ref_text: str
    language_id: str
    num_step: int = 32


@dataclass
class Pending:
    req: TTSRequest
    t_recv: float
    future: asyncio.Future = field(default_factory=lambda: asyncio.get_event_loop().create_future())


async def batcher_loop():
    """Pull pending requests, batch by num_step (the diffusion-step count must
    match across a batch), call generate_batch, deliver per-request audio."""
    assert QUEUE is not None and CTX is not None
    loop = asyncio.get_event_loop()
    while True:
        first: Pending = await QUEUE.get()
        batch: list[Pending] = [first]
        deadline = loop.time() + (MAX_BATCH_WAIT_MS / 1000.0)
        while len(batch) < MAX_BATCH_SIZE:
            remaining = deadline - loop.time()
            if remaining <= 0:
                break
            try:
                p = await asyncio.wait_for(QUEUE.get(), timeout=remaining)
            except asyncio.TimeoutError:
                break
            # Group strictly by num_step: same generation_config across batch.
            if p.req.num_step != first.req.num_step:
                # Put the wrong-step request back at the front by scheduling its
                # own immediate-batch path: send it back to the queue.
                QUEUE.put_nowait(p)
                break
            batch.append(p)

        t_batch_start = time.perf_counter()
        try:
            items = [
                {
                    "text": p.req.text,
                    "lang": p.req.language_id,
                    "ref_audio": p.req.ref_audio_path,
                    "ref_text": p.req.ref_text,
                }
                for p in batch
            ]
            loop = asyncio.get_event_loop()
            audios, gen_s = await loop.run_in_executor(
                GEN_EXECUTOR, common.generate_batch, CTX, items, first.req.num_step,
            )
            t_done = time.perf_counter()
            for p, audio in zip(batch, audios):
                if p.future.done():
                    continue
                queue_wait_ms = (t_batch_start - p.t_recv) * 1000.0
                total_ms = (t_done - p.t_recv) * 1000.0
                audio_dur = common.audio_duration_s(audio, CTX.sampling_rate)
                p.future.set_result({
                    "audio": audio,
                    "gen_ms": gen_s * 1000.0,
                    "queue_wait_ms": queue_wait_ms,
                    "ttfb_ms": total_ms,
                    "audio_dur_s": audio_dur,
                    "batch_size": len(batch),
                })
        except Exception as e:
            for p in batch:
                if not p.future.done():
                    p.future.set_exception(e)


@app.get("/health")
async def health():
    return {"ok": CTX is not None and BATCHER_TASK is not None}


@app.get("/config")
async def config():
    return {"max_batch_size": MAX_BATCH_SIZE, "max_batch_wait_ms": MAX_BATCH_WAIT_MS}


@app.post("/tts")
async def tts(req: TTSRequest):
    if CTX is None or QUEUE is None:
        raise HTTPException(503, "model not loaded")
    p = Pending(req=req, t_recv=time.perf_counter())
    QUEUE.put_nowait(p)
    try:
        result = await p.future
    except Exception as e:
        raise HTTPException(500, f"gen_failed: {type(e).__name__}: {e}")
    wav_bytes = common.audio_to_wav_bytes(result["audio"], CTX.sampling_rate)
    headers = {
        "X-TTFB-ms": f"{result['ttfb_ms']:.2f}",
        "X-Gen-ms": f"{result['gen_ms']:.2f}",
        "X-Queue-Wait-ms": f"{result['queue_wait_ms']:.2f}",
        "X-Audio-Duration-s": f"{result['audio_dur_s']:.3f}",
        "X-Batch-Size": str(result["batch_size"]),
    }
    return Response(content=wav_bytes, media_type="audio/wav", headers=headers)


@app.on_event("startup")
async def _on_startup():
    global QUEUE, BATCHER_TASK
    QUEUE = asyncio.Queue()
    BATCHER_TASK = asyncio.create_task(batcher_loop())


@app.on_event("shutdown")
async def _on_shutdown():
    if BATCHER_TASK:
        BATCHER_TASK.cancel()


def _startup_sync(speaker_cache_path: Path):
    global CTX
    CTX = common.load_model()
    if speaker_cache_path.exists():
        common.load_speaker_cache(CTX, speaker_cache_path)
    # Warm/compile ON the same dedicated thread that will serve generation, so
    # torch.compile's thread-affine Triton kernels are valid at request time.
    fut = GEN_EXECUTOR.submit(common.warmup, CTX, REF_AUDIO_DIR)
    fut.result()


def main():
    global MAX_BATCH_SIZE, MAX_BATCH_WAIT_MS
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--max-batch-size", type=int, default=8)
    ap.add_argument("--max-batch-wait-ms", type=float, default=20.0)
    ap.add_argument("--speaker-cache", default="corpus/speaker_cache.pkl")
    args = ap.parse_args()
    MAX_BATCH_SIZE = args.max_batch_size
    MAX_BATCH_WAIT_MS = args.max_batch_wait_ms
    _startup_sync(Path(args.speaker_cache))
    uvicorn.run(app, host=args.host, port=args.port, workers=1, log_level="warning")


if __name__ == "__main__":
    main()
