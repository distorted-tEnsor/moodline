"""PyTorch definition of the Moodline emotion classifier.

This is a faithful re-implementation of the Keras model that was trained in
``notebooks/final_clean.ipynb``:

    Embedding(10000, 300)
    Bidirectional(GRU(128, return_sequences=True))
    Dropout(0.5)
    Bidirectional(GRU(64))
    Dropout(0.5)
    Dense(6, softmax)

Two details matter for numerical parity with the original Keras weights and are
deliberately *not* "cleaned up":

1. ``padding_idx`` is **not** set on the embedding. The model was trained with
   ``mask_zero=False``, so the padding row was a normal trainable vector that
   received gradients. Forcing it to zero here would silently change every
   prediction.
2. The sequence length is fixed (50) and padding is *post*. Because training had
   no masking, trailing pad steps are part of what the GRU learned to expect;
   trimming batches to their longest member would be faster but would move the
   model off its training distribution.

Both constraints disappear if you retrain with ``training/train.py``, which uses
packed sequences and proper masking (see ``masked`` in the config).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

import torch
import torch.nn as nn


@dataclass(frozen=True)
class ModelConfig:
    """Everything needed to rebuild the network from weights alone."""

    vocab_size: int = 10_000
    embedding_dim: int = 300
    hidden_sizes: tuple[int, ...] = (128, 64)
    num_classes: int = 6
    dropout: float = 0.5
    max_sequence_length: int = 50
    masked: bool = False
    pooling: str = "last"  # "last" (legacy parity) or "attention" (retrained)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ModelConfig:
        known = set(cls.__dataclass_fields__)
        payload = {k: v for k, v in data.items() if k in known}
        if "hidden_sizes" in payload:
            payload["hidden_sizes"] = tuple(payload["hidden_sizes"])
        return cls(**payload)

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["hidden_sizes"] = list(self.hidden_sizes)
        return data


class MaskedAttentionPool(nn.Module):
    """Additive attention over time steps, used by retrained (masked) models."""

    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.score = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Tanh(),
            nn.Linear(hidden_dim // 2, 1, bias=False),
        )

    def forward(self, sequence: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        # sequence: (B, T, H); mask: (B, T) with True on real tokens.
        scores = self.score(sequence).squeeze(-1)  # (B, T)
        scores = scores.masked_fill(~mask, torch.finfo(scores.dtype).min)
        weights = torch.softmax(scores, dim=1).unsqueeze(-1)  # (B, T, 1)
        return torch.sum(sequence * weights, dim=1)


class BiGRUEmotionClassifier(nn.Module):
    """Stacked bidirectional GRU classifier returning raw logits."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config
        h1, h2 = config.hidden_sizes

        self.embedding = nn.Embedding(config.vocab_size, config.embedding_dim)
        self.gru1 = nn.GRU(config.embedding_dim, h1, batch_first=True, bidirectional=True)
        self.dropout1 = nn.Dropout(config.dropout)
        self.gru2 = nn.GRU(h1 * 2, h2, batch_first=True, bidirectional=True)
        self.dropout2 = nn.Dropout(config.dropout)
        self.pool = MaskedAttentionPool(h2 * 2) if config.pooling == "attention" else None
        self.classifier = nn.Linear(h2 * 2, config.num_classes)

    def forward(self, tokens: torch.Tensor, lengths: torch.Tensor | None = None) -> torch.Tensor:
        """tokens: (B, T) int64 -> logits: (B, num_classes)."""
        embedded = self.embedding(tokens)

        if self.config.masked and lengths is not None:
            return self._forward_masked(embedded, tokens, lengths)

        sequence, _ = self.gru1(embedded)
        sequence = self.dropout1(sequence)
        _, hidden = self.gru2(sequence)  # hidden: (2, B, h2)
        # Keras Bidirectional(merge_mode="concat") emits [forward_last, backward_last];
        # torch's h_n rows are ordered the same way.
        pooled = torch.cat((hidden[0], hidden[1]), dim=-1)
        return self.classifier(self.dropout2(pooled))

    def _forward_masked(
        self, embedded: torch.Tensor, tokens: torch.Tensor, lengths: torch.Tensor
    ) -> torch.Tensor:
        from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

        packed = pack_padded_sequence(
            embedded, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        packed_out, _ = self.gru1(packed)
        sequence, _ = pad_packed_sequence(packed_out, batch_first=True, total_length=tokens.size(1))
        sequence = self.dropout1(sequence)

        packed = pack_padded_sequence(
            sequence, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        packed_out, hidden = self.gru2(packed)

        if self.pool is not None:
            sequence, _ = pad_packed_sequence(
                packed_out, batch_first=True, total_length=tokens.size(1)
            )
            mask = torch.arange(tokens.size(1), device=tokens.device)[None, :] < lengths[:, None]
            pooled = self.pool(sequence, mask)
        else:
            pooled = torch.cat((hidden[0], hidden[1]), dim=-1)

        return self.classifier(self.dropout2(pooled))

    @torch.inference_mode()
    def predict_proba(
        self, tokens: torch.Tensor, lengths: torch.Tensor | None = None
    ) -> torch.Tensor:
        return torch.softmax(self(tokens, lengths), dim=-1)

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())


DEFAULT_LABELS: tuple[str, ...] = (
    "sadness",
    "joy",
    "love",
    "anger",
    "fear",
    "surprise",
)

EMOTION_EMOJIS: dict[str, str] = {
    "sadness": "😢",
    "joy": "😄",
    "love": "❤️",
    "anger": "😠",
    "fear": "😨",
    "surprise": "😲",
}


@dataclass
class Bundle:
    """A self-contained artifact: architecture + weights + vocabulary."""

    config: ModelConfig
    labels: list[str]
    vocab: dict[str, int]
    tokenizer: dict[str, Any]
    state_dict: dict[str, torch.Tensor]
    meta: dict[str, Any] = field(default_factory=dict)

    def to_payload(self) -> dict[str, Any]:
        return {
            "format_version": 2,
            "config": self.config.to_dict(),
            "labels": list(self.labels),
            "vocab": self.vocab,
            "tokenizer": self.tokenizer,
            "state_dict": {k: v.cpu() for k, v in self.state_dict.items()},
            "meta": self.meta,
        }

    @classmethod
    def from_payload(cls, payload: dict[str, Any]) -> Bundle:
        return cls(
            config=ModelConfig.from_dict(payload["config"]),
            labels=list(payload["labels"]),
            vocab=payload["vocab"],
            tokenizer=payload["tokenizer"],
            state_dict=payload["state_dict"],
            meta=payload.get("meta", {}),
        )
