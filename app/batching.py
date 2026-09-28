"""Adaptive micro-batching for single-text requests.

A GRU forward pass is dominated by 50 sequential time steps; the per-step matmul
is tiny. Running one sentence costs almost exactly what running thirty-two costs,
so under concurrent load the useful thing to do is collect whatever arrived in
the last few milliseconds and push it through as one tensor.

The batching here is *opportunistic*: the worker never idles waiting for company.
It takes the first request, sweeps up anything already queued behind it (bounded
by ``batch_max_size`` and a ``batch_max_wait_ms`` safety window), and dispatches.
Because a batch is dispatched before the next one is collected, requests that
arrive while a forward pass is running pile up and leave together. A lone user on
an idle server therefore sees no added latency at all, while throughput under
concurrency rises several-fold.

Inference itself runs on a dedicated single-thread executor: it is CPU-bound and
would otherwise block the event loop, and letting several passes interleave only
makes them contend for the same cores.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field

from .engine import EmotionEngine, Prediction

logger = logging.getLogger(__name__)

_SHUTDOWN = object()


@dataclass
class _Request:
    text: str
    future: asyncio.Future[Prediction] = field(repr=False)


class DynamicBatcher:
    """Groups concurrent single-text requests into one forward pass."""

    def __init__(
        self,
        engine: EmotionEngine,
        *,
        max_size: int = 32,
        max_wait_ms: float = 6.0,
        workers: int = 1,
    ) -> None:
        self.engine = engine
        self.max_size = max_size
        self.max_wait = max_wait_ms / 1000.0
        self._queue: asyncio.Queue = asyncio.Queue()
        self._executor = ThreadPoolExecutor(max_workers=workers, thread_name_prefix="infer")
        self._worker: asyncio.Task | None = None
        self._closing = False

    # -- lifecycle -------------------------------------------------------

    async def start(self) -> None:
        if self._worker is None:
            self._worker = asyncio.create_task(self._run(), name="moodline-batcher")

    async def stop(self) -> None:
        self._closing = True
        if self._worker is not None:
            await self._queue.put(_SHUTDOWN)
            try:
                await asyncio.wait_for(self._worker, timeout=5)
            except (asyncio.TimeoutError, asyncio.CancelledError):  # pragma: no cover
                self._worker.cancel()
            self._worker = None
        self._executor.shutdown(wait=True, cancel_futures=True)

    # -- public API ------------------------------------------------------

    async def predict(self, text: str) -> Prediction:
        cached = self.engine.peek_cache(text)
        if cached is not None:
            return cached
        if self._closing:
            raise RuntimeError("batcher is shutting down")

        loop = asyncio.get_running_loop()
        future: asyncio.Future[Prediction] = loop.create_future()
        await self._queue.put(_Request(text, future))
        return await future

    async def predict_many(self, texts: Sequence[str]) -> list[Prediction]:
        """Batch endpoints already arrive grouped; run them directly."""
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            self._executor, lambda: self.engine.predict_many(list(texts))
        )

    # -- worker ----------------------------------------------------------

    async def _run(self) -> None:
        loop = asyncio.get_running_loop()
        pending: asyncio.Future | None = None

        while True:
            if pending is None:
                pending = asyncio.ensure_future(self._queue.get())
            item = await pending
            pending = None
            if item is _SHUTDOWN:
                break

            batch: list[_Request] = [item]
            deadline = loop.time() + self.max_wait

            # Only wait for company if somebody is already queued behind us;
            # an idle server should not add latency to a lone request.
            while len(batch) < self.max_size and not self._queue.empty():
                remaining = deadline - loop.time()
                if remaining <= 0:
                    break
                if pending is None:
                    pending = asyncio.ensure_future(self._queue.get())
                done, _ = await asyncio.wait({pending}, timeout=remaining)
                if not done:
                    break  # keep the in-flight get for the next round
                nxt = pending.result()
                pending = None
                if nxt is _SHUTDOWN:
                    await self._dispatch(batch)
                    return
                batch.append(nxt)

            await self._dispatch(batch)

        if pending is not None:
            pending.cancel()
        self._drain()

    async def _dispatch(self, batch: list[_Request]) -> None:
        live = [request for request in batch if not request.future.cancelled()]
        if not live:
            return
        loop = asyncio.get_running_loop()
        texts = [request.text for request in live]
        try:
            predictions = await loop.run_in_executor(
                self._executor, lambda: self.engine.predict_many(texts)
            )
        except Exception as exc:  # pragma: no cover - defensive
            logger.exception("inference failed for a batch of %d", len(live))
            self.engine.metrics.record_error()
            for request in live:
                if not request.future.done():
                    request.future.set_exception(exc)
            return

        for request, prediction in zip(live, predictions, strict=True):
            if not request.future.done():
                request.future.set_result(prediction)

    def _drain(self) -> None:
        while not self._queue.empty():
            item = self._queue.get_nowait()
            if isinstance(item, _Request) and not item.future.done():
                item.future.set_exception(RuntimeError("server is shutting down"))
