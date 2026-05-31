"""Closed-loop async load generator for OmniVoice TTS servers.

Holds exactly N requests in flight via a semaphore, samples prompts uniformly
across the corpus, and records per-request timing to CSV. Prints p50/p95/p99
TTFB + total latency and audio-sec/wall-sec throughput at the end.
"""
from __future__ import annotations

import argparse
import asyncio
import csv
import json
import random
import statistics
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional

import httpx


@dataclass
class RequestRecord:
    request_id: int
    lang: str
    text_len: int
    num_step: int
    ttfb_ms: float
    total_latency_ms: float
    gen_ms_server: float
    queue_wait_ms: float
    audio_dur_s: float
    status: int
    t_send: float
    t_first_byte: float
    t_done: float
    error: str = ""


def load_prompts(path: Path) -> list[dict]:
    prompts = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                prompts.append(json.loads(line))
    if not prompts:
        raise SystemExit(f"No prompts in {path}")
    return prompts


async def one_request(
    client: httpx.AsyncClient,
    server_url: str,
    prompt: dict,
    num_step: int,
    request_id: int,
) -> RequestRecord:
    payload = {
        "text": prompt["text"],
        "ref_audio_path": prompt["ref_audio"],
        "ref_text": prompt["ref_text"],
        "language_id": prompt.get("language_id"),
        "num_step": num_step,
    }
    t_send = time.perf_counter()
    t_first_byte = t_send
    try:
        async with client.stream("POST", f"{server_url}/tts", json=payload) as resp:
            # First byte timing: read 1 chunk then drain.
            got_first = False
            total_bytes = 0
            async for chunk in resp.aiter_bytes():
                if not got_first:
                    t_first_byte = time.perf_counter()
                    got_first = True
                total_bytes += len(chunk)
            t_done = time.perf_counter()
            status = resp.status_code
            # Server reports timings in response headers.
            gen_ms = float(resp.headers.get("X-Gen-ms", 0.0))
            queue_ms = float(resp.headers.get("X-Queue-Wait-ms", 0.0))
            audio_dur = float(resp.headers.get("X-Audio-Duration-s", 0.0))
            ttfb_ms_hdr = float(resp.headers.get("X-TTFB-ms", 0.0))
            # Wall TTFB from client perspective (more honest than server-reported).
            ttfb_ms = (t_first_byte - t_send) * 1000.0
            if ttfb_ms_hdr and abs(ttfb_ms - ttfb_ms_hdr) > 50:
                # Client and server disagree by >50 ms; trust client side for TTFB.
                pass
            total_ms = (t_done - t_send) * 1000.0
            err = "" if status == 200 else f"http_{status}"
            return RequestRecord(
                request_id=request_id,
                lang=prompt.get("language_id", "?"),
                text_len=len(prompt["text"]),
                num_step=num_step,
                ttfb_ms=ttfb_ms,
                total_latency_ms=total_ms,
                gen_ms_server=gen_ms,
                queue_wait_ms=queue_ms,
                audio_dur_s=audio_dur,
                status=status,
                t_send=t_send,
                t_first_byte=t_first_byte,
                t_done=t_done,
                error=err,
            )
    except Exception as e:
        t_done = time.perf_counter()
        return RequestRecord(
            request_id=request_id,
            lang=prompt.get("language_id", "?"),
            text_len=len(prompt["text"]),
            num_step=num_step,
            ttfb_ms=(t_done - t_send) * 1000.0,
            total_latency_ms=(t_done - t_send) * 1000.0,
            gen_ms_server=0.0,
            queue_wait_ms=0.0,
            audio_dur_s=0.0,
            status=0,
            t_send=t_send,
            t_first_byte=t_send,
            t_done=t_done,
            error=type(e).__name__ + ":" + str(e)[:120],
        )


async def closed_loop_worker(
    sem: asyncio.Semaphore,
    client: httpx.AsyncClient,
    server_url: str,
    prompts: list[dict],
    num_step: int,
    counter: list[int],
    total: int,
    out: list[RequestRecord],
    rng: random.Random,
    deadline: float | None = None,
):
    """If `deadline` (perf_counter time) is set, stop after deadline regardless
    of `total`. If `total` is set (and deadline is None), stop after `total`
    requests have been claimed. Exactly one of these two stop conditions is
    expected to bind."""
    while True:
        async with sem:
            if deadline is not None and time.perf_counter() >= deadline:
                return
            if deadline is None and counter[0] >= total:
                return
            rid = counter[0]
            counter[0] += 1
            prompt = rng.choice(prompts)
            rec = await one_request(client, server_url, prompt, num_step, rid)
            out.append(rec)


async def run(args):
    prompts = load_prompts(Path(args.prompts_file))
    rng = random.Random(args.seed)

    # Warmup
    warm_records: list[RequestRecord] = []
    async with httpx.AsyncClient(timeout=args.timeout) as client:
        if args.warmup > 0:
            print(f"[warmup] {args.warmup} requests at concurrency=1 ...", flush=True)
            sem = asyncio.Semaphore(1)
            counter = [0]
            await asyncio.gather(
                *(
                    closed_loop_worker(
                        sem, client, args.server_url, prompts, args.num_step,
                        counter, args.warmup, warm_records, rng,
                    )
                    for _ in range(1)
                )
            )
            print(f"[warmup] done ({len(warm_records)} returned)", flush=True)

        # Measured run
        use_duration = args.duration_s > 0
        if use_duration:
            print(
                f"[run ] concurrency={args.concurrency} duration_s={args.duration_s} "
                f"num_step={args.num_step}",
                flush=True,
            )
        else:
            print(
                f"[run ] concurrency={args.concurrency} total={args.total_requests} "
                f"num_step={args.num_step}",
                flush=True,
            )
        records: list[RequestRecord] = []
        sem = asyncio.Semaphore(args.concurrency)
        counter = [0]
        t0 = time.perf_counter()
        deadline = (t0 + args.duration_s) if use_duration else None
        await asyncio.gather(
            *(
                closed_loop_worker(
                    sem, client, args.server_url, prompts, args.num_step,
                    counter, args.total_requests, records, rng,
                    deadline=deadline,
                )
                for _ in range(args.concurrency)
            )
        )
        t1 = time.perf_counter()

    wall_s = t1 - t0
    ok = [r for r in records if r.status == 200]
    errs = [r for r in records if r.status != 200]
    audio_total = sum(r.audio_dur_s for r in ok)
    throughput = audio_total / wall_s if wall_s > 0 else 0.0

    def pct(vals, p):
        if not vals:
            return float("nan")
        vals = sorted(vals)
        k = int(round((p / 100) * (len(vals) - 1)))
        return vals[k]

    ttfb = [r.ttfb_ms for r in ok]
    tot = [r.total_latency_ms for r in ok]

    summary = {
        "concurrency": args.concurrency,
        "num_step": args.num_step,
        "requests_ok": len(ok),
        "requests_err": len(errs),
        "wall_s": round(wall_s, 3),
        "rps": round(len(ok) / wall_s, 3) if wall_s > 0 else 0.0,
        "audio_total_s": round(audio_total, 3),
        "throughput_audio_per_wall": round(throughput, 3),
        "ttfb_ms_p50": round(pct(ttfb, 50), 1),
        "ttfb_ms_p95": round(pct(ttfb, 95), 1),
        "ttfb_ms_p99": round(pct(ttfb, 99), 1),
        "total_ms_p50": round(pct(tot, 50), 1),
        "total_ms_p95": round(pct(tot, 95), 1),
        "total_ms_p99": round(pct(tot, 99), 1),
        "err_rate": round(len(errs) / max(len(records), 1), 4),
    }
    print("[summary]", json.dumps(summary), flush=True)

    out_csv = Path(args.output)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(asdict(records[0] if records else RequestRecord(0,"",0,0,0,0,0,0,0,0,0,0,0,"")).keys()))
        w.writeheader()
        for r in records:
            w.writerow(asdict(r))

    out_json = out_csv.with_suffix(".summary.json")
    with out_json.open("w") as f:
        json.dump(summary, f, indent=2)

    if errs:
        print(f"[warn] {len(errs)} errors. First: {errs[0].error}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--server-url", required=True)
    ap.add_argument("--concurrency", type=int, required=True)
    ap.add_argument("--total-requests", type=int, default=300)
    ap.add_argument("--duration-s", type=float, default=0.0,
                    help="If >0, run for this many seconds instead of --total-requests.")
    ap.add_argument("--prompts-file", default="corpus/prompts.jsonl")
    ap.add_argument("--num-step", type=int, choices=[8, 16, 32], required=True)
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--output", required=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--timeout", type=float, default=120.0)
    args = ap.parse_args()
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
