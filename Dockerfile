# syntax=docker/dockerfile:1
#
# Build:  docker build -t moodline .
# Run:    docker run -p 8000:8000 moodline
#
# Two stages so the CPU torch wheel is resolved once and the final image carries
# no build tooling. Result is ~900 MB against ~3.5 GB for the TensorFlow image
# the first version needed.

FROM python:3.11-slim AS builder

ENV PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /build
COPY requirements.txt .
RUN python -m venv /opt/venv && \
    /opt/venv/bin/pip install --upgrade pip && \
    /opt/venv/bin/pip install -r requirements.txt


FROM python:3.11-slim AS runtime

ENV PATH="/opt/venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    # torch spawns one thread per core by default and then fights itself
    # inside a container with a CPU quota.
    OMP_NUM_THREADS=2 \
    MOODLINE_TORCH_THREADS=2 \
    MOODLINE_LOG_JSON=true

RUN useradd --create-home --uid 10001 moodline
COPY --from=builder /opt/venv /opt/venv

WORKDIR /app
COPY --chown=moodline:moodline app/ ./app/
COPY --chown=moodline:moodline static/ ./static/
COPY --chown=moodline:moodline artifacts/emotion_bigru.pt ./artifacts/emotion_bigru.pt

USER moodline
EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/ready', timeout=4).status==200 else 1)"

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
