"""Tiny, dependency-free runtime metrics.

Enough to answer "is it slow, and why" without dragging in Prometheus: counters
plus a bounded latency window that yields p50/p90/p99.
"""

from __future__ import annotations

import threading
import time
from collections import deque
from statistics import fmean


class Metrics:
    __slots__ = (
        "_latencies",
        "_lock",
        "_started",
        "batched_items",
        "batches",
        "cache_hits",
        "cache_misses",
        "errors",
        "predictions",
    )

    def __init__(self, window: int = 2048) -> None:
        self._lock = threading.Lock()
        self._started = time.monotonic()
        self._latencies: deque[float] = deque(maxlen=window)
        self.predictions = 0
        self.cache_hits = 0
        self.cache_misses = 0
        self.batches = 0
        self.batched_items = 0
        self.errors = 0

    def reset(self) -> None:
        """Drop everything recorded so far (used after warm-up)."""
        with self._lock:
            self._started = time.monotonic()
            self._latencies.clear()
            self.predictions = 0
            self.cache_hits = 0
            self.cache_misses = 0
            self.batches = 0
            self.batched_items = 0
            self.errors = 0

    # -- recording -------------------------------------------------------

    def record_batch(self, size: int, latency_ms: float) -> None:
        with self._lock:
            self.batches += 1
            self.batched_items += size
            self._latencies.append(latency_ms)

    def record_predictions(self, hits: int, misses: int) -> None:
        with self._lock:
            self.predictions += hits + misses
            self.cache_hits += hits
            self.cache_misses += misses

    def record_error(self) -> None:
        with self._lock:
            self.errors += 1

    # -- reporting -------------------------------------------------------

    @property
    def uptime_seconds(self) -> float:
        return time.monotonic() - self._started

    def _percentile(self, samples: list[float], fraction: float) -> float:
        if not samples:
            return 0.0
        index = min(len(samples) - 1, max(0, round(fraction * (len(samples) - 1))))
        return samples[index]

    def snapshot(self, cache_size: int = 0) -> dict[str, object]:
        with self._lock:
            samples = sorted(self._latencies)
            total_lookups = self.cache_hits + self.cache_misses
            return {
                "uptime_seconds": round(self.uptime_seconds, 1),
                "predictions": self.predictions,
                "cache_hits": self.cache_hits,
                "cache_misses": self.cache_misses,
                "cache_hit_rate": round(self.cache_hits / total_lookups, 4)
                if total_lookups
                else 0.0,
                "cache_size": cache_size,
                "batches": self.batches,
                "mean_batch_size": round(self.batched_items / self.batches, 2)
                if self.batches
                else 0.0,
                "latency_ms": {
                    "mean": round(fmean(samples), 2) if samples else 0.0,
                    "p50": round(self._percentile(samples, 0.50), 2),
                    "p90": round(self._percentile(samples, 0.90), 2),
                    "p99": round(self._percentile(samples, 0.99), 2),
                    "max": round(samples[-1], 2) if samples else 0.0,
                    "samples": len(samples),
                },
                "errors": self.errors,
            }


class RateLimiter:
    """Fixed-window per-client limiter. In-memory, so per-process by design."""

    __slots__ = ("_hits", "_limit", "_lock", "_window")

    def __init__(self, limit_per_minute: int, window_seconds: float = 60.0) -> None:
        self._limit = limit_per_minute
        self._window = window_seconds
        self._lock = threading.Lock()
        self._hits: dict[str, tuple[float, int]] = {}

    @property
    def enabled(self) -> bool:
        return self._limit > 0

    def allow(self, client: str) -> bool:
        if not self.enabled:
            return True
        now = time.monotonic()
        with self._lock:
            if len(self._hits) > 10_000:  # cheap guard against unbounded growth
                self._hits = {
                    key: value for key, value in self._hits.items() if now - value[0] < self._window
                }
            window_start, count = self._hits.get(client, (now, 0))
            if now - window_start >= self._window:
                window_start, count = now, 0
            if count >= self._limit:
                return False
            self._hits[client] = (window_start, count + 1)
            return True
