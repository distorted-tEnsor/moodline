# Moodline

Reads the emotion in a sentence — **sadness, joy, love, anger, fear, surprise** — with a
stacked bidirectional GRU, served by FastAPI behind a small web UI.

The model was trained in Keras. It now **runs in PyTorch**: the trained weights were
converted once, offline, and the serving path has no TensorFlow in it at all.

```
                    ┌──────────────────────────────────────────┐
  BiGRU_Model.keras │ scripts/convert_keras_to_pytorch.py      │  one-shot, offline
  tokenizer.pkl  ──►│  · reads the .keras zip + HDF5 directly  │──►  emotion_bigru.pt
      (42 MB)       │  · maps Keras [z,r,h] gates to torch      │        (14 MB)
                    │    [r,z,n]                                │
                    │  · verifies max |Δp| < 1e-4 vs Keras      │
                    └──────────────────────────────────────────┘
                                                                        │
  browser ──► FastAPI ──► micro-batcher ──► PyTorch BiGRU ◄─────────────┘
                 │              │
                 └── LRU cache ─┘
```

---

## Quick start

```bash
pip install -r requirements.txt
uvicorn app.main:app --reload
# open http://127.0.0.1:8000
```

`artifacts/emotion_bigru.pt` is the only file the service needs. If it is missing, rebuild
it from the Keras model:

```bash
pip install -r requirements-convert.txt
python scripts/convert_keras_to_pytorch.py        # prints the parity check
```

Everything else:

```bash
make help
```

---

## What changed, and why

### 1. TensorFlow is gone from the serving path

The old `main.py` did `from keras.models import load_model` and unpickled a Keras
`Tokenizer`. That pulled TensorFlow into every web worker: ~450 MB of resident memory and
several seconds of import time, to run a 3.4M-parameter model and look words up in a
dictionary.

The conversion script reads the `.keras` file itself — it is a zip of `config.json` and
`model.weights.h5` — and reads the tokenizer pickle through a shim class, so neither Keras
nor TensorFlow is needed even to convert. TensorFlow is optional there, used only to verify
the result.

| | before | after |
|---|---|---|
| serving dependencies | tensorflow-cpu, h5py, keras | torch, numpy |
| model artifact | 41.5 MB (⅔ Adam optimizer state) | **14.0 MB** |
| container image | ~3.5 GB | **~900 MB** |
| cold start (model ready) | ~10 s | **~0.1 s** |

Conversion is exact: max absolute difference in output probabilities against the original
Keras graph is **1.1 × 10⁻⁶** — float32 rounding — across the sample sentences and 64 random
token matrices. `tests/test_conversion.py` rebuilds a small random Keras BiGRU and re-checks
the mapping on every run, so a regression there fails CI rather than silently shifting every
prediction.

<details>
<summary>The gate mapping, in case you ever touch it</summary>

Keras packs GRU gates as `[z, r, h]`; PyTorch uses `[r, z, n]`. Both compute
`h_t = z·h_{t-1} + (1-z)·h̃` and both use the `reset_after=True` formulation, so the
conversion is a pure re-ordering of slices — the arithmetic is identical. Keras' `(2, 3u)`
bias splits into torch's `bias_ih` and `bias_hh`.

Two things are deliberately *not* tidied up in `app/model.py`, because they would change
predictions: the embedding has no `padding_idx` (the model was trained with `mask_zero=False`,
so row 0 is a real trained vector), and sequences are always padded to exactly 50 (the GRU
learned to expect trailing pad steps).
</details>

### 2. Throughput: batching, caching, and not blocking the event loop

The old `/predict` called `model.predict()` directly inside an `async def`, which blocks the
event loop — with two users, the second one waits for the first even though nothing is
CPU-saturated.

Now inference runs on a dedicated thread, and concurrent single-text requests are collected
into one forward pass. A GRU is dominated by its 50 sequential steps; the per-step matmul is
tiny, so 32 sentences cost barely more than one:

| batch | ms/batch | ms/item | items/s |
|------:|---------:|--------:|--------:|
| 1  | 4.0  | 4.01 | 249 |
| 8  | 6.6  | 0.83 | 1 211 |
| 32 | 14.2 | 0.44 | 2 252 |
| 64 | 24.0 | 0.37 | 2 671 |

*(2 threads, CPU only, `python scripts/benchmark.py`)*

End to end over HTTP, 400 requests at concurrency 24, same machine and same model
(`python scripts/benchmark.py --url … -n 400 -c 24`):

| | throughput | p50 | p90 | mean batch |
|---|---:|---:|---:|---:|
| batching off | 137 req/s | 170 ms | 210 ms | 1.0 |
| **batching on** | **278 req/s** | **76 ms** | **139 ms** | 7.7 |

Batching is **opportunistic**: the worker never idles waiting for company. It takes the first
request, sweeps up whatever is already queued behind it, and dispatches — so a lone user on an
idle server sees no added latency, while a loaded server naturally forms batches. Repeated
text is served from an LRU cache keyed on the *normalised* string, so `"I'm happy!"` and
`"im happy"` share an entry (~0.03 ms).

### 3. The API tells you when it is guessing

`dair-ai/emotion` is heavily templated ("i feel …"), and the model inherited that. It is
confident and right on sentences shaped like its training data, and shaky elsewhere:

```
i feel happy and excited              → joy       0.79   ✓
i can't believe how happy i am        → anger     0.29   ✗  (low_confidence: true)
i feel so much affection for him      → joy       0.99   ✗  (love is the weak class)
i feel grateful                       → sadness   0.99   ✗  and confident about it
```

That last one is the worst case found so far, and it is worth being precise about: it is
**inherited, not introduced**. "grateful" is in vocabulary (index 735), and the original Keras
graph returns the same probabilities to 1e-7. Converting the model did not break it; the model
was always like that, and nothing in the old UI would have told you.

Responses now carry `vocab_coverage`, `tokens`, and a `low_confidence` flag, and the UI shows
a caveat instead of presenting a 29% guess with the same flourish as a 99% one. Those cases
are pinned in `tests/test_engine.py::TestKnownWeakness` so a retrained model has to visibly
improve on them. Retraining is what `training/train.py` is for — see below.

### 4. Other fixes

- **`all_probabilites` → `probabilities`.** The misspelled key is still emitted as a
  deprecated alias, so nothing that reads it breaks.
- **`Depends`-injected engine** replaces the module-level `dl_model` dict, and `/ready`
  returns 503 while loading (`/health` stays 200, so liveness and readiness are separable).
- **Configuration** moved from module constants to `app/config.py` (`MOODLINE_*` env vars,
  `.env` supported). Nothing needs editing to deploy.
- **CORS** no longer sends `allow_credentials=True` alongside `allow_origins=["*"]` — a
  combination browsers reject outright.
- **Input validation**: whitespace-only text is rejected, over-long text is truncated rather
  than silently reshaped, and errors come back as `{detail, request_id}` with a request id on
  every response.
- **Optional API key and per-IP rate limiting**, both off by default.
- **Front-end**: aborts superseded requests, backs off health polling (1 s → 30 s, and pauses
  on a hidden tab) instead of polling every 3 s forever, builds the DOM with `textContent`
  rather than `innerHTML`, and adds labels, focus rings and `aria-live` regions.

---

## API

| method | path | purpose |
|---|---|---|
| `POST` | `/predict` | one sentence |
| `POST` | `/predict/batch` | up to 64 sentences in one forward pass |
| `GET` | `/health` | liveness — answers while the model loads |
| `GET` | `/ready` | readiness — 503 until the model can serve |
| `GET` | `/labels` | label names, emoji, indices |
| `GET` | `/model` | version, device, parameter count, thread count |
| `GET` | `/metrics` | latency percentiles, cache hit rate, mean batch size |
| `GET` | `/docs` | OpenAPI UI |

```bash
curl -s localhost:8000/predict -H 'content-type: application/json' \
     -d '{"text":"i feel so alone and hopeless today"}' | jq
```

```json
{
  "text": "i feel so alone and hopeless today",
  "predicted_emotion": "sadness",
  "emoji": "😢",
  "confidence": 0.998,
  "probabilities": { "sadness": 0.998, "joy": 0.001, "love": 0.0, "anger": 0.0, "fear": 0.001, "surprise": 0.0 },
  "all_probabilites": { "…": "deprecated alias of probabilities" },
  "tokens": 7,
  "vocab_coverage": 1.0,
  "low_confidence": false,
  "cached": false,
  "model_version": "bigru-20260928-b0e24325",
  "took_ms": 4.1
}
```

---

## Training

`training/train.py` replaces the notebook. Four things differ from what the notebook did:

- **Masking.** Sequences are packed, so the recurrence stops at the real last token instead of
  running 50 steps over padding. Faster, and better behaved on lengths the training set
  under-represents.
- **Honest validation.** The notebook early-stopped on the *test* set, which turns the reported
  test accuracy into a training metric. Here the dataset's own validation split drives early
  stopping and test is touched once, at the end.
- **Attention pooling** over the top GRU's outputs, so a cue in the middle of a long sentence
  is not diluted by everything after it.
- **AdamW + one-cycle LR + label smoothing + gradient clipping**, with the balanced class
  weights kept from the original.

```bash
pip install -r requirements-train.txt
python training/train.py                       # dair-ai/emotion from the hub
python training/train.py --data-dir data/      # or local train/validation/test.csv
python training/train.py --legacy-arch         # reproduce the old network for comparison
```

It writes the **same bundle format** the API loads, so a retrained model is a drop-in swap:

```bash
MOODLINE_ARTIFACT_PATH=artifacts/emotion_bigru_v2.pt uvicorn app.main:app
```

Compare the two on identical rows before switching:

```bash
python scripts/evaluate.py --artifact artifacts/emotion_bigru.pt
python scripts/evaluate.py --artifact artifacts/emotion_bigru_v2.pt
```

The original notebook is kept at `notebooks/01_model_exploration.ipynb` for the RNN/LSTM/GRU
comparison that motivated the architecture.

---

## Configuration

All settings take a `MOODLINE_` prefix; see `.env.example` for the full list.

| variable | default | notes |
|---|---|---|
| `MOODLINE_ARTIFACT_PATH` | `artifacts/emotion_bigru.pt` | swap in a retrained model |
| `MOODLINE_TORCH_THREADS` | auto (`min(4, cores)`) | set to 2 in a container with a CPU quota |
| `MOODLINE_QUANTIZE` | `false` | int8 dynamic: ~18 % faster, max \|Δp\| < 0.01, same argmax |
| `MOODLINE_CACHE_SIZE` | `2048` | 0 disables |
| `MOODLINE_BATCH_MAX_SIZE` | `32` | |
| `MOODLINE_CORS_ORIGINS` | `*` | comma-separated list in production |
| `MOODLINE_API_KEY` | unset | when set, `/predict*` require `X-API-Key` |
| `MOODLINE_RATE_LIMIT_PER_MINUTE` | `240` | per client IP, per process; 0 disables |

---

## Deployment

```bash
docker build -t moodline .
docker run -p 8000:8000 moodline
```

`render.yaml` and `Procfile` are included for buildpack hosts. **Run one worker per instance**
— the model is loaded per process, so a second worker is another ~250 MB of RSS for no extra
throughput on a small box. Scale with instances.

---

## Layout

```
app/                 the service
  config.py          MOODLINE_* settings
  tokenizer.py       Keras-compatible encoder, pure Python + NumPy
  model.py           the PyTorch network and the bundle format
  engine.py          loading, inference, LRU cache
  batching.py        opportunistic micro-batcher
  metrics.py         counters, latency percentiles, rate limiter
  main.py            FastAPI wiring
training/            data loading, vocabulary, the training loop
scripts/             convert_keras_to_pytorch.py · evaluate.py · benchmark.py
static/              the UI
tests/               72 tests: tokenizer, engine, batching, API, conversion parity
artifacts/           emotion_bigru.pt  (+ the legacy .keras / .pkl inputs)
```

## Tests

```bash
make install-dev
make test
```

Tests that need the model artifact skip themselves when it is absent, so a fresh clone without
it still exercises the tokenizer, schemas and error handling. `tests/test_conversion.py` needs
TensorFlow and skips without it.
