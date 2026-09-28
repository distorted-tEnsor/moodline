"""Typed, environment-driven configuration.

Everything that used to be a module-level constant in ``main.py`` lives here and
can be overridden with ``MOODLINE_*`` environment variables or a ``.env`` file,
so the same image runs locally, in CI and in production without edits.
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

ROOT = Path(__file__).resolve().parents[1]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="MOODLINE_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # -- artifacts ------------------------------------------------------
    artifact_path: Path = Field(default=ROOT / "artifacts" / "emotion_bigru.pt")
    static_dir: Path = Field(default=ROOT / "static")

    # -- runtime --------------------------------------------------------
    device: str = Field(default="auto", description="auto | cpu | cuda | mps")
    torch_threads: int = Field(default=0, ge=0, description="0 = pick automatically")
    quantize: bool = Field(
        default=False,
        description="int8 dynamic quantization: ~2x faster on CPU, tiny accuracy drift",
    )
    warmup_iterations: int = Field(default=2, ge=0)

    # -- request handling ----------------------------------------------
    max_text_length: int = Field(default=2000, ge=1)
    max_batch_items: int = Field(default=64, ge=1, description="items per /predict/batch call")
    cache_size: int = Field(default=2048, ge=0, description="0 disables the result cache")

    # -- micro-batching -------------------------------------------------
    batching_enabled: bool = True
    batch_max_size: int = Field(default=32, ge=1)
    batch_max_wait_ms: float = Field(default=6.0, ge=0.0)
    inference_workers: int = Field(
        default=1,
        ge=1,
        description="Threads running forward passes. 1 is right for CPU: torch "
        "already parallelises inside a single pass.",
    )

    # -- service --------------------------------------------------------
    host: str = "0.0.0.0"
    port: int = Field(default_factory=lambda: int(os.getenv("PORT", "8000")))
    cors_origins: list[str] = Field(default_factory=lambda: ["*"])
    api_key: str | None = Field(default=None, description="if set, /predict* require X-API-Key")
    rate_limit_per_minute: int = Field(default=240, ge=0, description="0 disables")
    log_level: str = "INFO"
    log_json: bool = Field(default=False, description="structured logs for aggregators")
    docs_enabled: bool = True

    @field_validator("cors_origins", mode="before")
    @classmethod
    def _split_origins(cls, value: object) -> object:
        if isinstance(value, str):
            return [origin.strip() for origin in value.split(",") if origin.strip()]
        return value

    @field_validator("device")
    @classmethod
    def _check_device(cls, value: str) -> str:
        allowed = {"auto", "cpu", "cuda", "mps"}
        if value not in allowed:
            raise ValueError(f"device must be one of {sorted(allowed)}")
        return value


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
