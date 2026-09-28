from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

ARTIFACT = ROOT / "artifacts" / "emotion_bigru.pt"

requires_artifact = pytest.mark.skipif(
    not ARTIFACT.exists(),
    reason="run scripts/convert_keras_to_pytorch.py first",
)


@pytest.fixture(scope="session")
def settings():
    from app.config import Settings

    return Settings(
        artifact_path=ARTIFACT,
        warmup_iterations=0,
        rate_limit_per_minute=0,  # off, so load tests do not trip it
        log_level="WARNING",
    )


@pytest.fixture(scope="session")
def engine(settings):
    from app.engine import EmotionEngine

    return EmotionEngine.load(settings)


@pytest.fixture
def client(settings):
    from fastapi.testclient import TestClient

    from app.main import create_app

    with TestClient(create_app(settings)) as test_client:
        yield test_client
