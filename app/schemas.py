"""Request and response models.

Note on ``all_probabilites``: the original API shipped that (misspelled) field and
the front-end reads it. Responses now carry the correctly spelled
``probabilities`` *and* keep the old key as a deprecated alias so existing
clients keep working.
"""

from __future__ import annotations

from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field, field_validator

from .model import EMOTION_EMOJIS

TextField = Annotated[
    str,
    Field(
        min_length=1,
        max_length=2000,
        description="The sentence to analyse",
    ),
]


class TextInput(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={"example": {"text": "I feel so happy and excited"}}
    )

    text: TextField

    @field_validator("text")
    @classmethod
    def _not_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("text must contain at least one non-whitespace character")
        return value


class BatchInput(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={
            "example": {"texts": ["I feel great today", "everything is falling apart"]}
        }
    )

    texts: list[TextField] = Field(min_length=1, max_length=64)

    @field_validator("texts")
    @classmethod
    def _not_all_blank(cls, value: list[str]) -> list[str]:
        if any(not item.strip() for item in value):
            raise ValueError("every text must contain at least one non-whitespace character")
        return value


class PredictionResponse(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    text: str
    predicted_emotion: str
    emoji: str
    confidence: float = Field(ge=0.0, le=1.0)
    probabilities: dict[str, float]
    all_probabilites: dict[str, float] = Field(
        description="Deprecated alias of `probabilities`, kept for older clients",
        deprecated=True,
    )
    tokens: int = Field(description="In-vocabulary tokens fed to the model")
    vocab_coverage: float = Field(
        ge=0.0, le=1.0, description="Share of words the model has actually seen in training"
    )
    low_confidence: bool = Field(
        description="True when the prediction is weak or the text is mostly out-of-vocabulary"
    )
    cached: bool = False
    model_version: str
    took_ms: float

    @classmethod
    def build(
        cls,
        *,
        text: str,
        label: str,
        probabilities: dict[str, float],
        tokens: int,
        coverage: float,
        cached: bool,
        model_version: str,
        took_ms: float,
    ) -> PredictionResponse:
        confidence = probabilities[label]
        return cls(
            text=text,
            predicted_emotion=label,
            emoji=EMOTION_EMOJIS.get(label, "🙂"),
            confidence=confidence,
            probabilities=probabilities,
            all_probabilites=probabilities,
            tokens=tokens,
            vocab_coverage=coverage,
            low_confidence=confidence < 0.5 or coverage < 0.5,
            cached=cached,
            model_version=model_version,
            took_ms=round(took_ms, 2),
        )


class BatchResponse(BaseModel):
    results: list[PredictionResponse]
    took_ms: float


class LabelInfo(BaseModel):
    name: str
    emoji: str
    index: int


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model_version: str | None = None
    device: str | None = None
    uptime_seconds: float | None = None


class MetricsResponse(BaseModel):
    uptime_seconds: float
    predictions: int
    cache_hits: int
    cache_misses: int
    cache_hit_rate: float
    cache_size: int
    batches: int
    mean_batch_size: float
    latency_ms: dict[str, float]
    errors: int


class ErrorResponse(BaseModel):
    detail: str
    request_id: str | None = None
