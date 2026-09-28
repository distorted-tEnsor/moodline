"""HTTP contract: status codes, schema, headers, auth, limits."""

from __future__ import annotations

import math

import pytest

from tests.conftest import requires_artifact

pytestmark = requires_artifact


class TestOps:
    def test_health_is_up(self, client) -> None:
        body = client.get("/health").json()
        assert body["model_loaded"] is True
        assert body["model_version"].startswith("bigru-")

    def test_ready(self, client) -> None:
        assert client.get("/ready").status_code == 200

    def test_labels(self, client) -> None:
        labels = client.get("/labels").json()
        assert [item["name"] for item in labels] == [
            "sadness",
            "joy",
            "love",
            "anger",
            "fear",
            "surprise",
        ]
        assert all(item["emoji"] for item in labels)

    def test_metrics_shape(self, client) -> None:
        client.post("/predict", json={"text": "i feel fine"})
        body = client.get("/metrics").json()
        assert body["predictions"] >= 1
        assert set(body["latency_ms"]) >= {"mean", "p50", "p90", "p99"}

    def test_ui_is_served(self, client) -> None:
        response = client.get("/")
        assert response.status_code == 200
        assert "text/html" in response.headers["content-type"]

    def test_request_id_header(self, client) -> None:
        response = client.get("/health", headers={"X-Request-ID": "abc123"})
        assert response.headers["X-Request-ID"] == "abc123"
        assert float(response.headers["X-Process-Time-Ms"]) >= 0


class TestPredict:
    def test_happy_path(self, client) -> None:
        response = client.post("/predict", json={"text": "i feel so alone and hopeless"})
        assert response.status_code == 200
        body = response.json()
        assert body["predicted_emotion"] == "sadness"
        assert body["emoji"]
        assert math.isclose(sum(body["probabilities"].values()), 1.0, rel_tol=1e-5)
        assert body["confidence"] == body["probabilities"][body["predicted_emotion"]]
        assert body["took_ms"] >= 0
        assert body["tokens"] > 0

    def test_deprecated_alias_still_present(self, client) -> None:
        body = client.post("/predict", json={"text": "i am so happy"}).json()
        assert body["all_probabilites"] == body["probabilities"]

    def test_low_confidence_flag(self, client) -> None:
        body = client.post("/predict", json={"text": "qwertyuiop asdfgh zxcvbn"}).json()
        assert body["vocab_coverage"] == 0.0
        assert body["low_confidence"] is True

    def test_cached_flag_on_repeat(self, client) -> None:
        payload = {"text": "the very same sentence twice over"}
        assert client.post("/predict", json=payload).json()["cached"] is False
        assert client.post("/predict", json=payload).json()["cached"] is True

    @pytest.mark.parametrize(
        "payload",
        [
            {},
            {"text": ""},
            {"text": "   "},
            {"text": "x" * 2001},
            {"text": 42},
            {"texts": ["wrong field"]},
        ],
    )
    def test_rejects_bad_input(self, client, payload) -> None:
        response = client.post("/predict", json=payload)
        assert response.status_code == 422
        assert "detail" in response.json()

    def test_accepts_maximum_length(self, client) -> None:
        assert client.post("/predict", json={"text": "a" * 2000}).status_code == 200


class TestBatch:
    def test_returns_one_result_per_input_in_order(self, client) -> None:
        texts = ["i am furious", "i feel loved", "i am terrified"]
        body = client.post("/predict/batch", json={"texts": texts}).json()
        assert [item["text"] for item in body["results"]] == texts
        assert [item["predicted_emotion"] for item in body["results"]] == ["anger", "love", "fear"]

    def test_matches_single_predictions(self, client) -> None:
        text = "i feel a quiet joy this morning"
        single = client.post("/predict", json={"text": text}).json()
        batched = client.post("/predict/batch", json={"texts": [text, "unrelated filler"]}).json()
        assert batched["results"][0]["predicted_emotion"] == single["predicted_emotion"]
        assert batched["results"][0]["confidence"] == pytest.approx(single["confidence"], abs=1e-6)

    def test_rejects_empty_and_oversized(self, client) -> None:
        assert client.post("/predict/batch", json={"texts": []}).status_code == 422
        assert client.post("/predict/batch", json={"texts": ["ok"] * 65}).status_code == 422


class TestSecurity:
    def test_api_key_required_when_configured(self, settings) -> None:
        from fastapi.testclient import TestClient

        from app.main import create_app

        secured = settings.model_copy(update={"api_key": "s3cret"})
        with TestClient(create_app(secured)) as client:
            assert client.post("/predict", json={"text": "hello"}).status_code == 401
            assert client.get("/health").status_code == 200
            ok = client.post("/predict", json={"text": "hello"}, headers={"X-API-Key": "s3cret"})
            assert ok.status_code == 200

    def test_rate_limit(self, settings) -> None:
        from fastapi.testclient import TestClient

        from app.main import create_app

        limited = settings.model_copy(update={"rate_limit_per_minute": 3})
        with TestClient(create_app(limited)) as client:
            codes = [
                client.post("/predict", json={"text": f"sentence number {i}"}).status_code
                for i in range(5)
            ]
        assert codes[:3] == [200, 200, 200]
        assert codes[3:] == [429, 429]

    def test_unknown_route_is_404(self, client) -> None:
        assert client.get("/nope").status_code == 404
