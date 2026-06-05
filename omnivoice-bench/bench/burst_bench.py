"""Burst benchmark: fire N requests SIMULTANEOUSLY at the OpenAI /v1/audio/speech
endpoint, for N in a sweep (default 10,20,...,100). Records per-request TTFB and
saves every response's audio. Emits a JSON results file consumed by burst_report.py.

This is OPEN-burst (all N launched at the same instant), not closed-loop — it
answers "if the server receives N at once, what TTFB does each of the N get?"

Each request uses a unique prompt, random language, random 100-300 char length
(from corpus/burst_prompts.jsonl). Prompts are consumed without replacement across
the whole sweep so nothing repeats.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import time
import wave
from pathlib import Path

import httpx


def load_prompts(path: str) -> list[dict]:
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def pcm_to_wav(pcm: bytes, path: Path, sample_rate: int = 24000):
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)        # s16le
        w.setframerate(sample_rate)
        w.writeframes(pcm)


async def one_request(client: httpx.AsyncClient, url: str, prompt: dict,
                      idx: int, batch: int, audio_dir: Path) -> dict:
    payload = {
        "model": "omnivoice",
        "input": prompt["text"],
        "voice": prompt["language_id"],
        "response_format": "pcm",
    }
    t_send = time.perf_counter()
    rec = {
        "batch": batch, "idx": idx,
        "language_id": prompt["language_id"],
        "char_len": prompt["char_len"],
        "text": prompt["text"],
    }
    try:
        async with client.stream("POST", url, json=payload) as resp:
            t_first = None
            chunks = []
            async for chunk in resp.aiter_bytes():
                if t_first is None:
                    t_first = time.perf_counter()   # first audio byte
                chunks.append(chunk)
            t_done = time.perf_counter()
            status = resp.status_code
            pcm = b"".join(chunks)
            sr = int(resp.headers.get("x-sample-rate", "24000"))
            gen_ms = float(resp.headers.get("x-gen-ms", 0.0))
            rec["status"] = status
            rec["ttfb_ms"] = (t_first - t_send) * 1000.0 if t_first else (t_done - t_send) * 1000.0
            rec["total_ms"] = (t_done - t_send) * 1000.0
            rec["server_gen_ms"] = gen_ms
            rec["bytes"] = len(pcm)
            if status == 200 and pcm:
                audio_dur = len(pcm) / 2 / sr
                rec["audio_dur_s"] = round(audio_dur, 3)
                wav_name = f"b{batch:03d}_{idx:03d}_{prompt['language_id']}.wav"
                pcm_to_wav(pcm, audio_dir / wav_name, sr)
                rec["wav"] = wav_name
            else:
                rec["error"] = f"http_{status}"
    except Exception as e:
        t_done = time.perf_counter()
        rec["status"] = 0
        rec["ttfb_ms"] = (t_done - t_send) * 1000.0
        rec["total_ms"] = (t_done - t_send) * 1000.0
        rec["error"] = f"{type(e).__name__}: {str(e)[:120]}"
    rec["t_send_rel"] = t_send
    return rec


async def run_batch(url: str, prompts: list[dict], batch: int,
                    audio_dir: Path, timeout: float) -> list[dict]:
    """Fire all len(prompts) requests at once."""
    limits = httpx.Limits(max_connections=len(prompts) + 10,
                          max_keepalive_connections=0)
    async with httpx.AsyncClient(timeout=timeout, limits=limits) as client:
        t0 = time.perf_counter()
        tasks = [one_request(client, url, p, i, batch, audio_dir)
                 for i, p in enumerate(prompts)]
        results = await asyncio.gather(*tasks)
        wall = time.perf_counter() - t0
    # normalize t_send relative to batch start
    for r in results:
        r["t_send_rel"] = round((r["t_send_rel"] - t0) * 1000.0, 2)
    return results, wall


def pct(vals, p):
    if not vals:
        return float("nan")
    vals = sorted(vals)
    k = int(round((p / 100) * (len(vals) - 1)))
    return vals[k]


async def main_async(args):
    prompts = load_prompts(args.prompts)
    audio_dir = Path(args.out_dir) / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)

    batches = [int(x) for x in args.batches.split(",")]
    needed = sum(batches)
    if needed > len(prompts):
        raise SystemExit(f"Need {needed} unique prompts but pool has {len(prompts)}. "
                         f"Regenerate with more --n-per-lang.")

    # Optional warmup so first-real-batch isn't paying cold graph capture
    cursor = 0
    all_batches = []
    if args.warmup > 0:
        wp = prompts[cursor:cursor + args.warmup]; cursor += args.warmup
        print(f"[burst] warmup {args.warmup} reqs ...", flush=True)
        await run_batch(args.server_url, wp, 0, audio_dir / "_warmup", args.timeout)

    for n in batches:
        bp = prompts[cursor:cursor + n]; cursor += n
        print(f"[burst] firing N={n} simultaneously ...", flush=True)
        results, wall = await run_batch(args.server_url, bp, n, audio_dir, args.timeout)
        ok = [r for r in results if r["status"] == 200]
        ttfbs = [r["ttfb_ms"] for r in ok]
        under = sum(1 for t in ttfbs if t <= args.ttfb_target_ms)
        summary = {
            "batch": n,
            "requests": len(results),
            "ok": len(ok),
            "errors": len(results) - len(ok),
            "wall_s": round(wall, 3),
            "ttfb_p50": round(pct(ttfbs, 50), 1) if ttfbs else None,
            "ttfb_p95": round(pct(ttfbs, 95), 1) if ttfbs else None,
            "ttfb_p99": round(pct(ttfbs, 99), 1) if ttfbs else None,
            "ttfb_max": round(max(ttfbs), 1) if ttfbs else None,
            "under_target": under,
            "under_target_pct": round(100 * under / len(ok), 1) if ok else 0.0,
            "effective_rps": round(len(ok) / wall, 2) if wall > 0 else 0.0,
            "requests_detail": results,
        }
        all_batches.append(summary)
        print(f"[burst] N={n}: ok={len(ok)}/{len(results)} "
              f"p50={summary['ttfb_p50']} p95={summary['ttfb_p95']} "
              f"max={summary['ttfb_max']} under{args.ttfb_target_ms}ms="
              f"{under}/{len(ok)} rps={summary['effective_rps']}", flush=True)

    out = {
        "meta": {
            "server_url": args.server_url,
            "ttfb_target_ms": args.ttfb_target_ms,
            "batches": batches,
            "total_requests": sum(batches),
        },
        "batches": all_batches,
    }
    out_json = Path(args.out_dir) / "burst_results.json"
    with out_json.open("w") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"[burst] wrote {out_json}", flush=True)
    print("[burst] DONE", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--server-url", required=True,
                    help="full URL e.g. http://HOST:8000/v1/audio/speech")
    ap.add_argument("--prompts", default="corpus/burst_prompts.jsonl")
    ap.add_argument("--batches", default="10,20,30,40,50,60,70,80,90,100")
    ap.add_argument("--out-dir", default="results/burst")
    ap.add_argument("--ttfb-target-ms", type=float, default=200.0)
    ap.add_argument("--warmup", type=int, default=8)
    ap.add_argument("--timeout", type=float, default=120.0)
    args = ap.parse_args()
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
