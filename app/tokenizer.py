"""Dependency-free reimplementation of the Keras ``Tokenizer`` used in training.

The original service unpickled a ``keras.preprocessing.text.Tokenizer``, which
dragged the whole of TensorFlow into the web process just to look words up in a
dictionary. This module reproduces ``texts_to_sequences`` + ``pad_sequences``
exactly, from a plain ``{word: index}`` map, using only the standard library and
NumPy.

Keras' lookup rule, which this mirrors:

* a known word with ``index < num_words``      -> its index
* a known word with ``index >= num_words``     -> OOV index (because an
  ``oov_token`` was configured at training time)
* an unknown word                              -> OOV index
* anything left over                           -> dropped

Text normalisation matches what the deployed API did before tokenising
(lowercase, drop apostrophes, punctuation to spaces), which is also what the
training corpus looks like: ``"I can't"`` -> ``["i", "cant"]``.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Sequence
from typing import Any

import numpy as np

_APOSTROPHES = re.compile(r"['‘’ʼ`]")
_NON_ALNUM = re.compile(r"[^a-z0-9\s]+")
_WHITESPACE = re.compile(r"\s+")


def normalise(text: str) -> str:
    """Lowercase, strip apostrophes, punctuation -> space, collapse whitespace."""
    text = _APOSTROPHES.sub("", text.lower())
    text = _NON_ALNUM.sub(" ", text)
    return _WHITESPACE.sub(" ", text).strip()


class TextEncoder:
    """Turns raw strings into padded integer matrices."""

    __slots__ = (
        "_cache",
        "max_length",
        "num_words",
        "oov_index",
        "padding",
        "truncating",
        "vocab",
    )

    def __init__(
        self,
        vocab: dict[str, int],
        *,
        num_words: int = 10_000,
        oov_token: str = "<unk>",
        max_length: int = 50,
        padding: str = "post",
        truncating: str = "post",
    ) -> None:
        # Drop anything the model can never see; keeps the map small and the
        # lookup branch-free.
        self.vocab = {w: i for w, i in vocab.items() if i < num_words}
        self.num_words = num_words
        self.oov_index = self.vocab.get(oov_token, 1)
        self.max_length = max_length
        self.padding = padding
        self.truncating = truncating

    # -- construction ----------------------------------------------------

    @classmethod
    def from_bundle(cls, vocab: dict[str, int], tokenizer: dict[str, Any]) -> TextEncoder:
        return cls(
            vocab,
            num_words=int(tokenizer.get("num_words", 10_000)),
            oov_token=str(tokenizer.get("oov_token", "<unk>")),
            max_length=int(tokenizer.get("max_length", 50)),
            padding=str(tokenizer.get("padding", "post")),
            truncating=str(tokenizer.get("truncating", "post")),
        )

    # -- encoding --------------------------------------------------------

    def tokenize(self, text: str) -> list[str]:
        cleaned = normalise(text)
        return cleaned.split(" ") if cleaned else []

    def encode(self, text: str) -> list[int]:
        """Words -> ids, with Keras' OOV semantics. No padding applied."""
        get = self.vocab.get
        oov = self.oov_index
        return [get(word, oov) for word in self.tokenize(text)]

    def encode_batch(self, texts: Sequence[str]) -> tuple[np.ndarray, np.ndarray]:
        """Returns ``(padded_ids (B, T) int64, true_lengths (B,) int64)``."""
        sequences = [self.encode(text) for text in texts]
        return self.pad(sequences)

    def pad(self, sequences: Iterable[Sequence[int]]) -> tuple[np.ndarray, np.ndarray]:
        sequences = list(sequences)
        max_len = self.max_length
        matrix = np.zeros((len(sequences), max_len), dtype=np.int64)
        lengths = np.zeros(len(sequences), dtype=np.int64)

        for row, sequence in enumerate(sequences):
            if len(sequence) > max_len:
                sequence = sequence[:max_len] if self.truncating == "post" else sequence[-max_len:]
            n = len(sequence)
            lengths[row] = n
            if n == 0:
                continue
            if self.padding == "post":
                matrix[row, :n] = sequence
            else:
                matrix[row, max_len - n :] = sequence

        # A GRU over a zero-length sequence is undefined for packing; clamp to 1
        # (the row is all padding, which is exactly what the legacy model saw).
        np.maximum(lengths, 1, out=lengths)
        return matrix, lengths

    # -- introspection ---------------------------------------------------

    @property
    def vocab_size(self) -> int:
        return self.num_words

    def coverage(self, text: str) -> float:
        """Share of words that are in-vocabulary. Useful for low-confidence hints."""
        tokens = self.tokenize(text)
        if not tokens:
            return 0.0
        known = sum(1 for word in tokens if word in self.vocab)
        return known / len(tokens)
