#!/usr/bin/env python3
"""Measure what the service actually does under load.

Two modes:

* ``--local`` (default) drives the engine directly, so you see model latency
  without HTTP in the way, at several batch sizes.
* ``--url http://host:port`` fires concurrent requests at a running server and
  reports end-to-end percentiles plus throughput.

    python scripts/benchmark.py
    python scripts/benchmark.py --url http://127.0.0.1:8000 --concurrency 16 -n 500
"""

from __future__ import annotations

import argparse
import asyncio
import statistics
import sys
import time
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

SENTENCES = [
    "i feel so happy and excited about the news",
    "i am completely drained and hopeless tonight",
    "i cannot believe they cancelled it again",
    "walking home in the dark makes me nervous",
    "what a wonderful surprise this turned out to be",
    "i feel a quiet affection for this old house",
    "everything feels heavy and grey lately",
    "i am thrilled that the results finally came through",
]


def percentiles(samples: list[float]) -> dict[str, float]:
    ordered = sorted(samples)

    def at(fraction: float) -> float:
        return ordered[min(len(ordered) - 1, max(0, round(fraction * (len(ordered) - 1))))]

    return {
        "mean": round(statistics.fmean(ordered), 2),
        "p50": round(at(0.50), 2),
        "p90": round(at(0.90), 2),
        "p99": round(at(0.99), 2),
        "max": round(ordered[-1], 2),
    }


def bench_local(args: argparse.Namespace) -> None:
    from app.config import get_settings
    from app.engine import EmotionEngine

    settings = get_settings()
    if args.artifact:
        settings = settings.model_copy(update={"artifact_path": args.artifact})
    if args.quantize:
        settings = settings.model_copy(update={"quantize": True})

    engine = EmotionEngine.load(settings)
    print(f"model {engine.version} on {engine.device}, {engine.info()['threads']} threads")
    print(f"{'batch':>6} {'iters':>6} {'ms/batch':>10} {'ms/item':>9} {'items/s':>9}")

    for batch_size in args.batch_sizes:
        texts = [SENTENCES[i % len(SENTENCES)] for i in range(batch_size)]
        engine.predict_many(texts, use_cache=False)  # warm

        timings: list[float] = []
        for _ in range(args.iterations):
            started = time.perf_counter()
            engine.predict_many(texts, use_cache=False)
            timings.append((time.perf_counter() - started) * 1000)

        mean_ms = statistics.fmean(timings)
        print(
            f"{batch_size:>6} {args.iterations:>6} {mean_ms:>10.2f} "
            f"{mean_ms / batch_size:>9.2f} {1000 * batch_size / mean_ms:>9.0f}"
        )

    cached = []
    for _ in range(200):
        started = time.perf_counter()
        engine.predict_many([SENTENCES[0]])
        cached.append((time.perf_counter() - started) * 1000)
    print(f"\ncache hit: {percentiles(cached)} ms")


async def bench_http(args: argparse.Namespace) -> None:
    import httpx

    limits = httpx.Limits(max_connections=args.concurrency * 2)
    timings: list[float] = []
    failures = 0
    semaphore = asyncio.Semaphore(args.concurrency)

    async with httpx.AsyncClient(base_url=args.url, timeout=30, limits=limits) as client:
        health = await client.get("/ready")
        health.raise_for_status()
        print(f"server ready: {health.json()}")

        # A per-run salt keeps the server-side cache from answering a repeat run
        # and flattering the numbers.
        salt = uuid.uuid4().hex[:6]

        async def one(index: int) -> None:
            nonlocal failures
            # Vary the text so the cache does not answer everything.
            text = f"{SENTENCES[index % len(SENTENCES)]} {salt} {index}"
            async with semaphore:
                started = time.perf_counter()
                try:
                    response = await client.post("/predict", json={"text": text})
                    response.raise_for_status()
                except Exception:
                    failures += 1
                    return
                timings.append((time.perf_counter() - started) * 1000)

        started = time.perf_counter()
        await asyncio.gather(*(one(index) for index in range(args.requests)))
        wall = time.perf_counter() - started

    print(f"\n{args.requests} requests, concurrency {args.concurrency}")
    print(f"  latency ms : {percentiles(timings)}")
    print(f"  throughput : {len(timings) / wall:.0f} req/s over {wall:.2f}s")
    print(f"  failures   : {failures}")

    async with httpx.AsyncClient(base_url=args.url, timeout=10) as client:
        metrics = (await client.get("/metrics")).json()
    print(
        f"  server-side: mean batch {metrics['mean_batch_size']}, "
        f"cache hit rate {metrics['cache_hit_rate']}, latency {metrics['latency_ms']}"
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--url", help="benchmark a running server instead of the local engine")
    parser.add_argument("--artifact", type=Path, default=None)
    parser.add_argument("--quantize", action="store_true")
    parser.add_argument("--batch-sizes", type=int, nargs="*", default=[1, 4, 8, 16, 32, 64])
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("-n", "--requests", type=int, default=200)
    parser.add_argument("-c", "--concurrency", type=int, default=8)
    args = parser.parse_args()

    if args.url:
        asyncio.run(bench_http(args))
    else:
        bench_local(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
