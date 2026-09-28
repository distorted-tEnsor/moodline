"""Guards the Keras -> PyTorch gate mapping.

Builds a small random Keras BiGRU, converts it with the production helpers and
compares outputs. Independent of the shipped artifact, so a regression in the
mapping is caught even if the checked-in weights are fine. Skipped when
TensorFlow is not installed (it is not a runtime dependency).
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

keras = pytest.importorskip("keras", reason="TensorFlow/Keras only needed for conversion")

from app.model import BiGRUEmotionClassifier, ModelConfig
from scripts.convert_keras_to_pytorch import build_state_dict, convert_gru_direction

VOCAB, DIM, H1, H2, CLASSES, LENGTH = 40, 8, 6, 4, 3, 7


@pytest.fixture(scope="module")
def reference():
    keras.utils.set_random_seed(7)
    model = keras.Sequential(
        [
            keras.layers.Input(shape=(LENGTH,)),
            keras.layers.Embedding(VOCAB, DIM),
            keras.layers.Bidirectional(keras.layers.GRU(H1, return_sequences=True)),
            keras.layers.Dropout(0.5),
            keras.layers.Bidirectional(keras.layers.GRU(H2)),
            keras.layers.Dropout(0.5),
            keras.layers.Dense(CLASSES, activation="softmax"),
        ]
    )
    # Push the weights away from their symmetric initial values so a wrong gate
    # order cannot coincidentally match.
    for weight in model.weights:
        weight.assign(np.random.default_rng(0).normal(0, 0.6, weight.shape).astype("float32"))
    return model


def _weights_by_path(model) -> dict[str, np.ndarray]:
    """Rebuild the ``layers/...`` names that the HDF5 file uses."""
    embedding, first, _, second, _, dense = model.layers
    weights = {"layers/embedding/vars/0": embedding.weights[0].numpy()}
    for name, layer in (("bidirectional", first), ("bidirectional_1", second)):
        for direction, inner in (
            ("forward_layer", layer.forward_layer),
            ("backward_layer", layer.backward_layer),
        ):
            for index, weight in enumerate(inner.weights):
                weights[f"layers/{name}/{direction}/cell/vars/{index}"] = weight.numpy()
    weights["layers/dense/vars/0"] = dense.weights[0].numpy()
    weights["layers/dense/vars/1"] = dense.weights[1].numpy()
    return weights


def test_converted_model_matches_keras(reference) -> None:
    config = ModelConfig(
        vocab_size=VOCAB,
        embedding_dim=DIM,
        hidden_sizes=(H1, H2),
        num_classes=CLASSES,
        dropout=0.5,
        max_sequence_length=LENGTH,
    )
    model = BiGRUEmotionClassifier(config)
    model.load_state_dict(
        build_state_dict(_weights_by_path(reference), config.hidden_sizes), strict=False
    )
    model.eval()

    tokens = np.random.default_rng(3).integers(0, VOCAB, size=(16, LENGTH)).astype(np.int64)
    expected = np.asarray(reference.predict(tokens, verbose=0), dtype=np.float64)
    actual = model.predict_proba(torch.from_numpy(tokens)).numpy().astype(np.float64)

    assert np.max(np.abs(expected - actual)) < 1e-5
    assert np.array_equal(expected.argmax(1), actual.argmax(1))


def test_gate_reordering_is_not_the_identity() -> None:
    """Sanity check on the test itself: [z,r,h] and [r,z,n] really do differ."""
    rng = np.random.default_rng(1)
    units = 5
    kernel = rng.normal(size=(3, 3 * units))
    recurrent = rng.normal(size=(units, 3 * units))
    bias = rng.normal(size=(2, 3 * units))

    converted = convert_gru_direction(kernel, recurrent, bias, units)
    assert converted["weight_ih"].shape == (3 * units, 3)
    assert not np.allclose(converted["weight_ih"].numpy(), kernel.T)


def test_bundle_round_trips(tmp_path) -> None:
    from app.model import Bundle

    config = ModelConfig(
        vocab_size=VOCAB, embedding_dim=DIM, hidden_sizes=(H1, H2), num_classes=CLASSES
    )
    model = BiGRUEmotionClassifier(config)
    bundle = Bundle(
        config=config,
        labels=["a", "b", "c"],
        vocab={"<unk>": 1, "word": 2},
        tokenizer={"num_words": VOCAB, "oov_token": "<unk>", "max_length": LENGTH},
        state_dict=model.state_dict(),
        meta={"source": "test"},
    )
    path = tmp_path / "bundle.pt"
    torch.save(bundle.to_payload(), path)

    # weights_only=True is what the service uses; the payload must stay plain.
    restored = Bundle.from_payload(torch.load(path, map_location="cpu", weights_only=True))
    assert restored.config == config
    assert restored.labels == ["a", "b", "c"]
    assert set(restored.state_dict) == set(bundle.state_dict)
