"""Model loading and synchronous inference.

Design notes
------------
* The artifact is loaded once per process with ``weights_only=True``; it carries
  the architecture, the weights and the vocabulary, so nothing else is read from
  disk at start-up.
* Inference runs under ``torch.inference_mode()`` with autograd globally off and
  a bounded thread count. Torch's default is one OS thread per core, which on a
  small container means threads fighting each other for a 3.4M-parameter model.
* Results are memoised on the *normalised* text, so "I'm happy!" and "im happy"
  share a cache entry.
* ``predict_many`` is the only inference entry point: one padded tensor, one
  forward pass, regardless of how the callers arrived.
"""

from __future__ import annotations

import hashlib
import logging
import os
import threading
import time
from collections import OrderedDict
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from .config import Settings
from .metrics import Metrics
from .model import BiGRUEmotionClassifier, Bundle
from .tokenizer import TextEncoder, normalise

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class Prediction:
    text: str
    label: str
    probabilities: dict[str, float]
    tokens: int
    coverage: float
    cached: bool = False


class _LRUCache:
    __slots__ = ("_capacity", "_data", "_lock")

    def __init__(self, capacity: int) -> None:
        self._data: OrderedDict[str, Prediction] = OrderedDict()
        self._capacity = capacity
        self._lock = threading.Lock()

    def __len__(self) -> int:
        return len(self._data)

    @property
    def enabled(self) -> bool:
        return self._capacity > 0

    def get(self, key: str) -> Prediction | None:
        if not self.enabled:
            return None
        with self._lock:
            value = self._data.get(key)
            if value is not None:
                self._data.move_to_end(key)
            return value

    def put(self, key: str, value: Prediction) -> None:
        if not self.enabled:
            return
        with self._lock:
            self._data[key] = value
            self._data.move_to_end(key)
            while len(self._data) > self._capacity:
                self._data.popitem(last=False)

    def clear(self) -> None:
        with self._lock:
            self._data.clear()


def resolve_device(preference: str) -> torch.device:
    if preference != "auto":
        return torch.device(preference)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class EmotionEngine:
    """Owns the model, the tokenizer and the cache for one process."""

    def __init__(
        self,
        *,
        model: BiGRUEmotionClassifier,
        encoder: TextEncoder,
        labels: Sequence[str],
        device: torch.device,
        version: str,
        metrics: Metrics,
        cache_size: int = 2048,
        meta: dict | None = None,
    ) -> None:
        self.model = model
        self.encoder = encoder
        self.labels = list(labels)
        self.device = device
        self.version = version
        self.metrics = metrics
        self.meta = meta or {}
        self._cache = _LRUCache(cache_size)

    # -- construction ----------------------------------------------------

    @classmethod
    def load(cls, settings: Settings, metrics: Metrics | None = None) -> EmotionEngine:
        path = Path(settings.artifact_path)
        if not path.exists():
            raise FileNotFoundError(
                f"model artifact not found at {path}. "
                "Run: python scripts/convert_keras_to_pytorch.py"
            )

        threads = settings.torch_threads or max(1, min(4, os.cpu_count() or 1))
        torch.set_num_threads(threads)
        torch.set_grad_enabled(False)

        started = time.perf_counter()
        payload = torch.load(path, map_location="cpu", weights_only=True)
        bundle = Bundle.from_payload(payload)

        device = resolve_device(settings.device)
        model = BiGRUEmotionClassifier(bundle.config)
        model.load_state_dict(bundle.state_dict, strict=False)
        model.eval().to(device)

        if settings.quantize:
            if device.type != "cpu":
                logger.warning("quantization requested but device is %s; skipping", device)
            else:
                import warnings

                with warnings.catch_warnings():
                    # torch's own dynamic-quantization path emits a deprecation
                    # warning from inside the RNN module; nothing we can act on.
                    warnings.filterwarnings("ignore", category=UserWarning, module="torch.ao.*")
                    model = torch.ao.quantization.quantize_dynamic(
                        model, {torch.nn.GRU, torch.nn.Linear}, dtype=torch.qint8
                    )
                logger.info("applied int8 dynamic quantization")

        encoder = TextEncoder.from_bundle(bundle.vocab, bundle.tokenizer)
        version = cls._version_for(path, bundle)

        engine = cls(
            model=model,
            encoder=encoder,
            labels=bundle.labels,
            device=device,
            version=version,
            metrics=metrics or Metrics(),
            cache_size=settings.cache_size,
            meta=bundle.meta,
        )

        for _ in range(settings.warmup_iterations):
            engine.predict_many(["warmup sentence for the graph"], use_cache=False)
        engine._cache.clear()
        engine.metrics.reset()  # warm-up latencies would skew the percentiles

        logger.info(
            "model %s ready on %s in %.2fs (%d threads, %d params, vocab %d)",
            version,
            device,
            time.perf_counter() - started,
            threads,
            sum(p.numel() for p in model.parameters()),
            len(encoder.vocab),
        )
        return engine

    @staticmethod
    def _version_for(path: Path, bundle: Bundle) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1 << 20), b""):
                digest.update(chunk)
        stamp = str(bundle.meta.get("converted_at") or bundle.meta.get("trained_at") or "")[
            :10
        ].replace("-", "")
        return f"bigru-{stamp or 'local'}-{digest.hexdigest()[:8]}"

    # -- inference -------------------------------------------------------

    def predict_many(self, texts: Sequence[str], *, use_cache: bool = True) -> list[Prediction]:
        if not texts:
            return []

        keys = [normalise(text) for text in texts]
        results: list[Prediction | None] = [None] * len(texts)
        pending: list[int] = []

        if use_cache:
            for index, key in enumerate(keys):
                hit = self._cache.get(key)
                if hit is not None:
                    results[index] = Prediction(
                        text=texts[index],
                        label=hit.label,
                        probabilities=hit.probabilities,
                        tokens=hit.tokens,
                        coverage=hit.coverage,
                        cached=True,
                    )
                else:
                    pending.append(index)
        else:
            pending = list(range(len(texts)))

        if pending:
            batch = [texts[index] for index in pending]
            started = time.perf_counter()
            fresh = self._infer(batch)
            self.metrics.record_batch(len(batch), (time.perf_counter() - started) * 1000)
            for index, prediction in zip(pending, fresh, strict=True):
                results[index] = prediction
                if use_cache:
                    self._cache.put(keys[index], prediction)

        self.metrics.record_predictions(hits=len(texts) - len(pending), misses=len(pending))
        return [result for result in results if result is not None]

    def predict(self, text: str) -> Prediction:
        return self.predict_many([text])[0]

    def peek_cache(self, text: str) -> Prediction | None:
        """Cache lookup without touching the model. Lets callers skip the queue."""
        hit = self._cache.get(normalise(text))
        if hit is None:
            self.metrics.record_predictions(hits=0, misses=0)
            return None
        self.metrics.record_predictions(hits=1, misses=0)
        return Prediction(
            text=text,
            label=hit.label,
            probabilities=hit.probabilities,
            tokens=hit.tokens,
            coverage=hit.coverage,
            cached=True,
        )

    @torch.inference_mode()
    def _infer(self, texts: Sequence[str]) -> list[Prediction]:
        tokens, lengths = self.encoder.encode_batch(texts)
        token_tensor = torch.from_numpy(tokens).to(self.device, non_blocking=True)
        length_tensor = torch.from_numpy(lengths)

        logits = self.model(token_tensor, length_tensor if self.model.config.masked else None)
        probabilities = torch.softmax(logits.float(), dim=-1).cpu().numpy()

        predictions: list[Prediction] = []
        for row, text in enumerate(texts):
            scores = probabilities[row]
            top = int(np.argmax(scores))
            predictions.append(
                Prediction(
                    text=text,
                    label=self.labels[top],
                    probabilities={
                        label: float(score)
                        for label, score in zip(self.labels, scores, strict=True)
                    },
                    tokens=int(lengths[row]),
                    coverage=self.encoder.coverage(text),
                    cached=False,
                )
            )
        return predictions

    # -- introspection ---------------------------------------------------

    @property
    def cache_size(self) -> int:
        return len(self._cache)

    def clear_cache(self) -> None:
        self._cache.clear()

    def info(self) -> dict[str, object]:
        return {
            "version": self.version,
            "device": str(self.device),
            "labels": self.labels,
            "max_sequence_length": self.encoder.max_length,
            "vocab_size": len(self.encoder.vocab),
            "parameters": sum(p.numel() for p in self.model.parameters()),
            "threads": torch.get_num_threads(),
            "torch": torch.__version__,
            **{k: v for k, v in self.meta.items() if k != "state_dict"},
        }
