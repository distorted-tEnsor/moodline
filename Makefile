.DEFAULT_GOAL := help
PY ?= python3
PORT ?= 8000

.PHONY: help install install-dev install-train convert train evaluate run dev test lint format bench bench-http docker docker-run clean

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-16s\033[0m %s\n", $$1, $$2}'

install: ## Runtime dependencies only
	$(PY) -m pip install -r requirements.txt

install-dev: ## Runtime + test/lint tooling
	$(PY) -m pip install -r requirements-dev.txt

install-train: ## Runtime + training/evaluation tooling
	$(PY) -m pip install -r requirements-train.txt

convert: ## Rebuild artifacts/emotion_bigru.pt from the Keras model
	$(PY) scripts/convert_keras_to_pytorch.py

train: ## Train a new model (see training/train.py --help)
	$(PY) training/train.py

evaluate: ## Score the current artifact on the test split
	$(PY) scripts/evaluate.py

run: ## Serve on $(PORT)
	uvicorn app.main:app --host 0.0.0.0 --port $(PORT)

dev: ## Serve with autoreload
	uvicorn app.main:app --reload --port $(PORT)

test: ## Run the test suite
	TF_CPP_MIN_LOG_LEVEL=3 $(PY) -m pytest

lint: ## Lint and type-check
	ruff check .
	ruff format --check .
	mypy app

format: ## Apply formatting and import sorting
	ruff format .
	ruff check --fix .

bench: ## Benchmark the engine directly
	$(PY) scripts/benchmark.py

bench-http: ## Benchmark a running server (make bench-http PORT=8000)
	$(PY) scripts/benchmark.py --url http://127.0.0.1:$(PORT) -n 300 -c 16

docker: ## Build the image
	docker build -t moodline .

docker-run: ## Run the image on $(PORT)
	docker run --rm -p $(PORT):8000 moodline

clean: ## Remove caches
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
	rm -rf .pytest_cache .ruff_cache .mypy_cache
