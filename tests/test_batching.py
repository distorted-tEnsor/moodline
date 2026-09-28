"""The micro-batcher must be invisible: same answers, fewer forward passes."""

from __future__ import annotations

import asyncio

import pytest

from tests.conftest import requires_artifact

pytestmark = [requires_artifact, pytest.mark.asyncio]

SENTENCES = [
    "i feel so alone and hopeless",
    "i am furious about the delay",
    "i feel terrified of the dark",
    "what a delightful surprise",
    "i love my grandmother dearly",
    "everything is wonderful today",
]


@pytest.fixture
async def batcher(engine):
    from app.batching import DynamicBatcher

    instance = DynamicBatcher(engine, max_size=32, max_wait_ms=5)
    await instance.start()
    try:
        yield instance
    finally:
        await instance.stop()


async def test_concurrent_requests_get_their_own_answers(batcher, engine) -> None:
    engine.clear_cache()
    expected = {
        text: prediction.label
        for text, prediction in zip(
            SENTENCES, engine.predict_many(SENTENCES, use_cache=False), strict=True
        )
    }

    engine.clear_cache()
    results = await asyncio.gather(*(batcher.predict(text) for text in SENTENCES))

    assert [result.text for result in results] == SENTENCES
    for result in results:
        assert result.label == expected[result.text]


async def test_concurrency_produces_fewer_batches_than_requests(batcher, engine) -> None:
    engine.clear_cache()
    engine.metrics.reset()

    texts = [f"a distinct sentence number {index} about feeling" for index in range(24)]
    await asyncio.gather(*(batcher.predict(text) for text in texts))

    snapshot = engine.metrics.snapshot()
    assert snapshot["predictions"] == 24
    assert snapshot["batches"] < 24, "requests were not grouped at all"
    assert snapshot["mean_batch_size"] > 1


async def test_cache_hits_skip_the_queue(batcher, engine) -> None:
    engine.clear_cache()
    await batcher.predict("a sentence used for the cache test")
    engine.metrics.reset()

    result = await batcher.predict("A SENTENCE used for the cache test!")
    assert result.cached is True
    assert engine.metrics.snapshot()["batches"] == 0


async def test_batch_helper_matches_individual_calls(batcher) -> None:
    together = await batcher.predict_many(SENTENCES)
    apart = [await batcher.predict(text) for text in SENTENCES]
    assert [item.label for item in together] == [item.label for item in apart]


async def test_predict_after_stop_raises(engine) -> None:
    from app.batching import DynamicBatcher

    instance = DynamicBatcher(engine, max_size=4, max_wait_ms=1)
    await instance.start()
    await instance.stop()
    engine.clear_cache()
    with pytest.raises(RuntimeError):
        await instance.predict("anything at all")
