"""Moodline — emotion classification API.

FastAPI service around a PyTorch bidirectional GRU. The model, tokenizer and
batching live in their own modules; this file is only wiring: lifespan, routes,
middleware and error handling.

Run it with:
    uvicorn app.main:app --reload
    python -m app.main
"""

from __future__ import annotations

import logging
import time
import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, Header, HTTPException, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from .batching import DynamicBatcher
from .config import Settings, get_settings
from .engine import EmotionEngine, Prediction
from .logging_conf import configure_logging
from .metrics import Metrics, RateLimiter
from .model import EMOTION_EMOJIS
from .schemas import (
    BatchInput,
    BatchResponse,
    ErrorResponse,
    HealthResponse,
    LabelInfo,
    MetricsResponse,
    PredictionResponse,
    TextInput,
)

logger = logging.getLogger("moodline")

DESCRIPTION = """
Reads the emotion in a sentence — *sadness, joy, love, anger, fear, surprise* —
with a bidirectional GRU trained on `dair-ai/emotion`.

* `POST /predict` — one sentence
* `POST /predict/batch` — up to 64 sentences in a single forward pass
* `GET /metrics` — latency percentiles, cache hit rate, batch sizes
"""


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    settings: Settings = app.state.settings
    configure_logging(settings.log_level, settings.log_json)

    metrics = Metrics()
    app.state.metrics = metrics
    app.state.started_at = time.monotonic()
    app.state.engine = None
    app.state.batcher = None

    logger.info("loading model from %s", settings.artifact_path)
    engine = EmotionEngine.load(settings, metrics)
    app.state.engine = engine

    if settings.batching_enabled:
        batcher = DynamicBatcher(
            engine,
            max_size=settings.batch_max_size,
            max_wait_ms=settings.batch_max_wait_ms,
            workers=settings.inference_workers,
        )
        await batcher.start()
        app.state.batcher = batcher

    try:
        yield
    finally:
        if app.state.batcher is not None:
            await app.state.batcher.stop()
        app.state.engine = None
        logger.info("shutdown complete")


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or get_settings()

    app = FastAPI(
        title="Moodline",
        description=DESCRIPTION,
        version="2.0.0",
        lifespan=lifespan,
        docs_url="/docs" if settings.docs_enabled else None,
        redoc_url=None,
        openapi_url="/openapi.json" if settings.docs_enabled else None,
        responses={500: {"model": ErrorResponse}},
    )
    app.state.settings = settings
    app.state.limiter = RateLimiter(settings.rate_limit_per_minute)

    app.add_middleware(GZipMiddleware, minimum_size=512)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        # Credentials cannot be combined with a wildcard origin; browsers reject it.
        allow_credentials="*" not in settings.cors_origins,
        allow_methods=["GET", "POST", "OPTIONS"],
        allow_headers=["*"],
        max_age=3600,
    )

    _register_middleware(app)
    _register_errors(app)
    _register_routes(app, settings)

    if settings.static_dir.is_dir():
        app.mount("/static", StaticFiles(directory=settings.static_dir), name="static")
    else:  # pragma: no cover - only in stripped deployments
        logger.warning("static directory %s not found; UI disabled", settings.static_dir)

    return app


# ---------------------------------------------------------------------------
# Cross-cutting concerns
# ---------------------------------------------------------------------------


def _register_middleware(app: FastAPI) -> None:
    @app.middleware("http")
    async def request_context(request: Request, call_next):
        request_id = request.headers.get("x-request-id") or uuid.uuid4().hex[:12]
        request.state.request_id = request_id

        limiter: RateLimiter = app.state.limiter
        if limiter.enabled and request.url.path.startswith("/predict"):
            client = request.headers.get("x-forwarded-for", "").split(",")[0].strip() or (
                request.client.host if request.client else "unknown"
            )
            if not limiter.allow(client):
                return JSONResponse(
                    status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                    content={
                        "detail": "Too many requests. Slow down a little.",
                        "request_id": request_id,
                    },
                    headers={"Retry-After": "30", "X-Request-ID": request_id},
                )

        started = time.perf_counter()
        response = await call_next(request)
        elapsed_ms = (time.perf_counter() - started) * 1000
        response.headers["X-Request-ID"] = request_id
        response.headers["X-Process-Time-Ms"] = f"{elapsed_ms:.2f}"
        if request.url.path.startswith("/predict"):
            logger.info(
                "%s %s -> %s in %.1fms",
                request.method,
                request.url.path,
                response.status_code,
                elapsed_ms,
                extra={"request_id": request_id},
            )
        return response


def _register_errors(app: FastAPI) -> None:
    @app.exception_handler(RequestValidationError)
    async def validation_error(request: Request, exc: RequestValidationError):
        first = exc.errors()[0] if exc.errors() else {}
        field = ".".join(str(part) for part in first.get("loc", ())[1:]) or "request"
        return JSONResponse(
            status_code=422,  # the named constant was renamed across Starlette versions
            content={
                "detail": f"{field}: {first.get('msg', 'invalid request')}",
                "request_id": getattr(request.state, "request_id", None),
            },
        )

    @app.exception_handler(HTTPException)
    async def http_error(request: Request, exc: HTTPException):
        return JSONResponse(
            status_code=exc.status_code,
            content={
                "detail": exc.detail,
                "request_id": getattr(request.state, "request_id", None),
            },
            headers=getattr(exc, "headers", None),
        )

    @app.exception_handler(Exception)
    async def unhandled_error(request: Request, exc: Exception):
        request_id = getattr(request.state, "request_id", None)
        logger.exception("unhandled error", extra={"request_id": request_id})
        metrics: Metrics | None = getattr(app.state, "metrics", None)
        if metrics is not None:
            metrics.record_error()
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={
                "detail": "Internal error while analysing the text.",
                "request_id": request_id,
            },
        )


# ---------------------------------------------------------------------------
# Dependencies
# ---------------------------------------------------------------------------


def get_engine(request: Request) -> EmotionEngine:
    engine: EmotionEngine | None = getattr(request.app.state, "engine", None)
    if engine is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model is still loading. Try again in a moment.",
            headers={"Retry-After": "5"},
        )
    return engine


async def require_api_key(request: Request, x_api_key: str | None = Header(default=None)) -> None:
    expected: str | None = request.app.state.settings.api_key
    if expected and x_api_key != expected:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid or missing API key"
        )


def _to_response(prediction: Prediction, version: str, took_ms: float) -> PredictionResponse:
    return PredictionResponse.build(
        text=prediction.text,
        label=prediction.label,
        probabilities=prediction.probabilities,
        tokens=prediction.tokens,
        coverage=prediction.coverage,
        cached=prediction.cached,
        model_version=version,
        took_ms=took_ms,
    )


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


def _register_routes(app: FastAPI, settings: Settings) -> None:
    index_file = settings.static_dir / "index.html"

    @app.get("/", include_in_schema=False)
    async def server_ui():
        if not index_file.exists():
            raise HTTPException(status_code=404, detail="UI is not bundled in this deployment")
        return FileResponse(index_file, headers={"Cache-Control": "no-cache"})

    @app.get("/favicon.ico", include_in_schema=False)
    async def favicon():
        icon = settings.static_dir / "favicon.svg"
        if icon.exists():
            return FileResponse(icon)
        return JSONResponse(status_code=204, content=None)

    @app.get("/health", response_model=HealthResponse, tags=["ops"])
    async def health(request: Request):
        """Liveness: answers even while the model is still loading."""
        engine: EmotionEngine | None = getattr(request.app.state, "engine", None)
        return HealthResponse(
            status="Server is running",
            model_loaded=engine is not None,
            model_version=engine.version if engine else None,
            device=str(engine.device) if engine else None,
            uptime_seconds=round(time.monotonic() - request.app.state.started_at, 1),
        )

    @app.get("/ready", response_model=HealthResponse, tags=["ops"])
    async def ready(request: Request):
        """Readiness: 503 until the model can actually serve traffic."""
        engine: EmotionEngine | None = getattr(request.app.state, "engine", None)
        if engine is None:
            raise HTTPException(status_code=503, detail="Model is still loading")
        return HealthResponse(
            status="ready",
            model_loaded=True,
            model_version=engine.version,
            device=str(engine.device),
            uptime_seconds=round(time.monotonic() - request.app.state.started_at, 1),
        )

    @app.get("/labels", response_model=list[LabelInfo], tags=["ops"])
    async def labels(engine: EmotionEngine = Depends(get_engine)):
        return [
            LabelInfo(name=name, emoji=EMOTION_EMOJIS.get(name, "🙂"), index=index)
            for index, name in enumerate(engine.labels)
        ]

    @app.get("/model", tags=["ops"])
    async def model_info(engine: EmotionEngine = Depends(get_engine)):
        return engine.info()

    @app.get("/metrics", response_model=MetricsResponse, tags=["ops"])
    async def metrics(request: Request):
        engine: EmotionEngine | None = getattr(request.app.state, "engine", None)
        snapshot = request.app.state.metrics.snapshot(engine.cache_size if engine else 0)
        return MetricsResponse(**snapshot)

    @app.post(
        "/predict",
        response_model=PredictionResponse,
        tags=["inference"],
        dependencies=[Depends(require_api_key)],
        responses={
            422: {"model": ErrorResponse},
            429: {"model": ErrorResponse},
            503: {"model": ErrorResponse},
        },
    )
    async def predict_emotion(
        payload: TextInput, request: Request, engine: EmotionEngine = Depends(get_engine)
    ):
        """Classify one sentence.

        Concurrent calls are merged into a single forward pass by the batcher.
        """
        started = time.perf_counter()
        batcher: DynamicBatcher | None = getattr(request.app.state, "batcher", None)
        if batcher is not None:
            prediction = await batcher.predict(payload.text)
        else:
            prediction = engine.predict(payload.text)
        return _to_response(prediction, engine.version, (time.perf_counter() - started) * 1000)

    @app.post(
        "/predict/batch",
        response_model=BatchResponse,
        tags=["inference"],
        dependencies=[Depends(require_api_key)],
        responses={422: {"model": ErrorResponse}, 503: {"model": ErrorResponse}},
    )
    async def predict_batch(
        payload: BatchInput, request: Request, engine: EmotionEngine = Depends(get_engine)
    ):
        """Classify up to `max_batch_items` sentences in one pass."""
        if len(payload.texts) > settings.max_batch_items:
            raise HTTPException(
                status_code=422,
                detail=f"At most {settings.max_batch_items} texts per request",
            )

        started = time.perf_counter()
        batcher: DynamicBatcher | None = getattr(request.app.state, "batcher", None)
        if batcher is not None:
            predictions = await batcher.predict_many(payload.texts)
        else:
            predictions = engine.predict_many(payload.texts)
        took = (time.perf_counter() - started) * 1000
        return BatchResponse(
            results=[_to_response(prediction, engine.version, took) for prediction in predictions],
            took_ms=round(took, 2),
        )


app = create_app()


if __name__ == "__main__":  # pragma: no cover
    import uvicorn

    config = get_settings()
    uvicorn.run(
        "app.main:app",
        host=config.host,
        port=config.port,
        log_config=None,
        workers=1,
    )
