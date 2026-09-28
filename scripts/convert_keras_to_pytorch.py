#!/usr/bin/env python3
"""One-shot conversion of the trained Keras BiGRU into a PyTorch artifact.

Why
---
``BiGRU_Model.keras`` is 41 MB, two thirds of which is Adam optimizer state, and
reading it requires TensorFlow (~450 MB of RSS, several seconds of import time)
inside every web worker. ``tokenizer.pkl`` has the same problem: it is a pickled
Keras object, so unpickling it needs Keras importable.

This script reads both *without TensorFlow* -- a ``.keras`` file is a zip of
``config.json`` + ``model.weights.h5``, and the tokenizer's state is a plain dict
behind the class -- and writes a single self-contained ``.pt`` bundle holding the
architecture config, the weights, the labels and the vocabulary.

The GRU gate mapping
--------------------
Keras stores GRU gates in the order ``[z, r, h]``; PyTorch uses ``[r, z, n]``.
Both compute ``h_t = z * h_{t-1} + (1 - z) * h~`` and both use the
``reset_after=True`` formulation (the reset gate is applied *after* the
recurrent matmul), so the conversion is a pure re-ordering of slices -- no maths
changes. Keras' ``(2, 3u)`` bias maps to torch's ``bias_ih`` / ``bias_hh``.

Usage
-----
    python scripts/convert_keras_to_pytorch.py                  # convert + verify
    python scripts/convert_keras_to_pytorch.py --no-verify      # skip TF check
    python scripts/convert_keras_to_pytorch.py --out artifacts/emotion_bigru.pt
"""

from __future__ import annotations

import argparse
import io
import json
import pickle
import sys
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.model import (
    DEFAULT_LABELS,
    BiGRUEmotionClassifier,
    Bundle,
    ModelConfig,
)
from app.tokenizer import TextEncoder

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_KERAS = ROOT / "artifacts" / "BiGRU_Model.keras"
DEFAULT_TOKENIZER = ROOT / "artifacts" / "tokenizer.pkl"
DEFAULT_OUT = ROOT / "artifacts" / "emotion_bigru.pt"

SAMPLE_TEXTS = [
    "i feel so happy and excited about this",
    "i feel alone and hopeless today",
    "i am furious that they cancelled the trip",
    "i feel terrified walking home in the dark",
    "i was completely shocked by the gift",
    "i feel a deep affection for my grandmother",
    "",  # all-padding edge case
    "zzzz qqqq unknowable gibberish tokens",  # pure OOV edge case
]


# --------------------------------------------------------------------------
# Reading Keras artifacts without TensorFlow
# --------------------------------------------------------------------------


class _KerasShim:
    """Stand-in for any Keras class met while unpickling; keeps only its state."""

    def __setstate__(self, state: dict[str, Any]) -> None:
        self.__dict__.update(state)


class _TolerantUnpickler(pickle.Unpickler):
    def find_class(self, module: str, name: str):  # type: ignore[override]
        if module.startswith(("keras", "tensorflow", "tf_keras")):
            return _KerasShim
        return super().find_class(module, name)


def read_tokenizer(path: Path) -> dict[str, Any]:
    with path.open("rb") as handle:
        obj = _TolerantUnpickler(handle).load()
    state = getattr(obj, "__dict__", obj)
    if not isinstance(state, dict) or "word_index" not in state:
        raise ValueError(f"{path} does not look like a Keras Tokenizer pickle")
    return state


def read_keras_archive(path: Path) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    """Returns ``(config, {h5_path: array})`` for the layer weights only."""
    with zipfile.ZipFile(path) as archive:
        config = json.loads(archive.read("config.json"))
        raw_weights = archive.read("model.weights.h5")

    weights: dict[str, np.ndarray] = {}
    with h5py.File(io.BytesIO(raw_weights), "r") as handle:

        def visit(name: str, node: Any) -> None:
            if isinstance(node, h5py.Dataset) and name.startswith("layers/"):
                weights[name] = np.asarray(node)

        handle.visititems(visit)
    return config, weights


# --------------------------------------------------------------------------
# Weight surgery
# --------------------------------------------------------------------------


def _split_gates(array: np.ndarray, units: int) -> tuple[np.ndarray, ...]:
    """Keras packs gates on the last axis as [z, r, h]."""
    return array[..., :units], array[..., units : 2 * units], array[..., 2 * units :]


def convert_gru_direction(
    kernel: np.ndarray, recurrent: np.ndarray, bias: np.ndarray, units: int
) -> dict[str, torch.Tensor]:
    """Keras GRU direction -> torch ``weight_ih/weight_hh/bias_ih/bias_hh``."""
    z_k, r_k, h_k = _split_gates(kernel, units)
    z_r, r_r, h_r = _split_gates(recurrent, units)
    z_bi, r_bi, h_bi = _split_gates(bias[0], units)
    z_bh, r_bh, h_bh = _split_gates(bias[1], units)

    return {
        "weight_ih": torch.from_numpy(
            np.ascontiguousarray(np.concatenate([r_k, z_k, h_k], axis=-1).T)
        ),
        "weight_hh": torch.from_numpy(
            np.ascontiguousarray(np.concatenate([r_r, z_r, h_r], axis=-1).T)
        ),
        "bias_ih": torch.from_numpy(np.ascontiguousarray(np.concatenate([r_bi, z_bi, h_bi]))),
        "bias_hh": torch.from_numpy(np.ascontiguousarray(np.concatenate([r_bh, z_bh, h_bh]))),
    }


def build_state_dict(
    weights: dict[str, np.ndarray], units: tuple[int, int]
) -> dict[str, torch.Tensor]:
    state: dict[str, torch.Tensor] = {}

    embedding = weights["layers/embedding/vars/0"]
    state["embedding.weight"] = torch.from_numpy(np.ascontiguousarray(embedding))

    for index, (keras_layer, torch_layer) in enumerate(
        (("bidirectional", "gru1"), ("bidirectional_1", "gru2"))
    ):
        for direction, suffix in (("forward_layer", "_l0"), ("backward_layer", "_l0_reverse")):
            prefix = f"layers/{keras_layer}/{direction}/cell/vars"
            converted = convert_gru_direction(
                weights[f"{prefix}/0"],
                weights[f"{prefix}/1"],
                weights[f"{prefix}/2"],
                units[index],
            )
            for name, tensor in converted.items():
                state[f"{torch_layer}.{name}{suffix}"] = tensor

    dense_kernel = weights["layers/dense/vars/0"]
    dense_bias = weights["layers/dense/vars/1"]
    state["classifier.weight"] = torch.from_numpy(np.ascontiguousarray(dense_kernel.T))
    state["classifier.bias"] = torch.from_numpy(np.ascontiguousarray(dense_bias))
    return state


def infer_config(keras_config: dict[str, Any]) -> ModelConfig:
    layers = keras_config["config"]["layers"]
    embedding = next(layer for layer in layers if layer["class_name"] == "Embedding")["config"]
    bidirectionals = [layer for layer in layers if layer["class_name"] == "Bidirectional"]
    dropouts = [layer for layer in layers if layer["class_name"] == "Dropout"]
    dense = next(layer for layer in layers if layer["class_name"] == "Dense")["config"]
    input_layer = next((layer for layer in layers if layer["class_name"] == "InputLayer"), None)

    max_len = 50
    if input_layer is not None:
        shape = input_layer["config"].get("batch_shape") or []
        if len(shape) == 2 and shape[1]:
            max_len = int(shape[1])

    for layer in bidirectionals:
        if not layer["config"]["layer"]["config"].get("reset_after", True):
            raise ValueError("GRU was trained with reset_after=False; gate mapping differs")
    if embedding.get("mask_zero"):
        raise ValueError("model was trained with masking; use training/train.py instead")

    return ModelConfig(
        vocab_size=int(embedding["input_dim"]),
        embedding_dim=int(embedding["output_dim"]),
        hidden_sizes=tuple(
            int(layer["config"]["layer"]["config"]["units"]) for layer in bidirectionals
        ),
        num_classes=int(dense["units"]),
        dropout=float(dropouts[0]["config"]["rate"]) if dropouts else 0.5,
        max_sequence_length=max_len,
        masked=False,
        pooling="last",
    )


# --------------------------------------------------------------------------
# Verification
# --------------------------------------------------------------------------


def verify_against_keras(
    keras_path: Path, model: BiGRUEmotionClassifier, encoder: TextEncoder
) -> float:
    """Load the original in Keras and compare probabilities. Returns max abs diff."""
    import os

    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
    try:
        import keras
    except Exception as exc:  # pragma: no cover - optional path
        print(f"  ! skipping verification, Keras unavailable ({exc})")
        return float("nan")

    reference = keras.models.load_model(keras_path, compile=False)
    tokens, _ = encoder.encode_batch(SAMPLE_TEXTS)

    keras_probs = np.asarray(reference.predict(tokens, verbose=0), dtype=np.float64)
    torch_probs = model.predict_proba(torch.from_numpy(tokens)).numpy().astype(np.float64)

    diff = float(np.max(np.abs(keras_probs - torch_probs)))
    agree = int(np.sum(keras_probs.argmax(1) == torch_probs.argmax(1)))
    print(
        f"  keras vs torch : max |Δp| = {diff:.3e}   argmax agreement {agree}/{len(SAMPLE_TEXTS)}"
    )

    rng = np.random.default_rng(0)
    random_tokens = rng.integers(0, 10_000, size=(64, encoder.max_length)).astype(np.int64)
    keras_rand = np.asarray(reference.predict(random_tokens, verbose=0), dtype=np.float64)
    torch_rand = model.predict_proba(torch.from_numpy(random_tokens)).numpy().astype(np.float64)
    rand_diff = float(np.max(np.abs(keras_rand - torch_rand)))
    print(f"  64 random rows : max |Δp| = {rand_diff:.3e}")
    return max(diff, rand_diff)


# --------------------------------------------------------------------------
# Entry point
# --------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--keras", type=Path, default=DEFAULT_KERAS)
    parser.add_argument("--tokenizer", type=Path, default=DEFAULT_TOKENIZER)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--labels", nargs="*", default=list(DEFAULT_LABELS))
    parser.add_argument("--no-verify", action="store_true", help="skip the TensorFlow parity check")
    parser.add_argument("--tolerance", type=float, default=1e-4)
    args = parser.parse_args()

    for path in (args.keras, args.tokenizer):
        if not path.exists():
            parser.error(f"missing input: {path}")

    print(f"reading {args.keras.name} ({args.keras.stat().st_size / 1e6:.1f} MB)")
    keras_config, weights = read_keras_archive(args.keras)
    config = infer_config(keras_config)
    print(f"  architecture   : {config.to_dict()}")

    print(f"reading {args.tokenizer.name}")
    tok_state = read_tokenizer(args.tokenizer)
    num_words = int(tok_state.get("num_words") or config.vocab_size)
    vocab = {
        word: int(index)
        for word, index in tok_state["word_index"].items()
        if int(index) < num_words
    }
    print(
        f"  vocabulary     : {len(tok_state['word_index'])} words seen, {len(vocab)} kept (num_words={num_words})"
    )

    model = BiGRUEmotionClassifier(config)
    state_dict = build_state_dict(weights, config.hidden_sizes)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    unexpected = list(unexpected)
    missing = [k for k in missing if not k.startswith("pool.")]
    if missing or unexpected:
        raise SystemExit(f"state_dict mismatch: missing={missing} unexpected={unexpected}")
    model.eval()
    print(f"  parameters     : {model.num_parameters():,}")

    tokenizer_meta = {
        "num_words": num_words,
        "oov_token": tok_state.get("oov_token", "<unk>"),
        "lower": bool(tok_state.get("lower", True)),
        "filters": tok_state.get("filters", ""),
        "split": tok_state.get("split", " "),
        "max_length": config.max_sequence_length,
        "padding": "post",
        "truncating": "post",
        "strip_apostrophes": True,
    }
    encoder = TextEncoder.from_bundle(vocab, tokenizer_meta)

    max_diff = float("nan")
    if not args.no_verify:
        print("verifying against the original Keras graph")
        max_diff = verify_against_keras(args.keras, model, encoder)
        if max_diff == max_diff and max_diff > args.tolerance:  # not NaN and too big
            raise SystemExit(f"parity check failed: {max_diff:.3e} > {args.tolerance:.1e}")

    bundle = Bundle(
        config=config,
        labels=list(args.labels),
        vocab=vocab,
        tokenizer=tokenizer_meta,
        state_dict=model.state_dict(),
        meta={
            "source": args.keras.name,
            "converted_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            # str() matters: torch.__version__ is a TorchVersion instance, which
            # torch.load(weights_only=True) refuses to unpickle.
            "torch_version": str(torch.__version__),
            "keras_version": "3.13.2 (training)",
            "max_abs_probability_delta": None if max_diff != max_diff else max_diff,
            "parameters": model.num_parameters(),
        },
    )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(bundle.to_payload(), args.out)
    size_mb = args.out.stat().st_size / 1e6
    print(f"wrote {args.out} ({size_mb:.1f} MB, was {args.keras.stat().st_size / 1e6:.1f} MB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
