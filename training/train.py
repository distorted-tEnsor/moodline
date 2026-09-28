#!/usr/bin/env python3
"""Train the Moodline emotion classifier in PyTorch.

This replaces the notebook's Keras training. What changed, and why:

* **Masking.** The old model ran the GRU over 50 steps no matter how long the
  sentence was, and learned to read padding as signal. Here sequences are packed,
  so the recurrence stops at the real last token — faster *and* better behaved on
  sentence lengths the training set under-represents.
* **Honest validation.** The notebook early-stopped on the test set. Here the
  dataset's own validation split drives early stopping and the test split is
  touched exactly once, at the end.
* **Attention pooling** over the top GRU's outputs instead of taking only the
  final hidden state, so a strong emotional cue in the middle of a long sentence
  is not diluted by everything that follows it.
* **AdamW + one-cycle LR + label smoothing + gradient clipping**, and class
  weights kept from the original (the corpus is heavily skewed towards joy and
  sadness).
* Seeded, resumable, and it exports the *same bundle format* the API loads, so a
  retrained model is a drop-in replacement for the converted one.

Examples
--------
    python training/train.py                          # HF hub, default recipe
    python training/train.py --data-dir data/ --epochs 15
    python training/train.py --legacy-arch            # reproduce the old network
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.utils.class_weight import compute_class_weight

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.model import BiGRUEmotionClassifier, Bundle, ModelConfig
from training.data import (
    EmotionDataset,
    build_vocab,
    ensure_validation,
    load_splits,
    make_loader,
)

ROOT = Path(__file__).resolve().parents[1]
logger = logging.getLogger("train")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def pick_device(preference: str) -> torch.device:
    if preference != "auto":
        return torch.device(preference)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


@dataclass
class EpochResult:
    loss: float
    accuracy: float
    macro_f1: float

    def __str__(self) -> str:
        return f"loss {self.loss:.4f}  acc {self.accuracy:.4f}  macroF1 {self.macro_f1:.4f}"


class EarlyStopping:
    """Stops when the monitored score has not improved for ``patience`` epochs."""

    def __init__(self, patience: int = 4, min_delta: float = 1e-4) -> None:
        self.patience = patience
        self.min_delta = min_delta
        self.best = -float("inf")
        self.best_state: dict[str, torch.Tensor] | None = None
        self.best_epoch = 0
        self._waited = 0

    def step(self, score: float, model: nn.Module, epoch: int) -> bool:
        if score > self.best + self.min_delta:
            self.best = score
            self.best_epoch = epoch
            self.best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            self._waited = 0
            return False
        self._waited += 1
        return self._waited >= self.patience

    def restore(self, model: nn.Module) -> None:
        if self.best_state is not None:
            model.load_state_dict(self.best_state)


# ---------------------------------------------------------------------------
# Train / evaluate
# ---------------------------------------------------------------------------


def run_epoch(
    model: nn.Module,
    loader,
    criterion: nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None = None,
    scaler: torch.amp.GradScaler | None = None,
    clip: float = 1.0,
) -> tuple[EpochResult, np.ndarray, np.ndarray]:
    training = optimizer is not None
    model.train(training)

    total_loss = 0.0
    seen = 0
    predictions: list[np.ndarray] = []
    targets: list[np.ndarray] = []

    for tokens, lengths, labels in loader:
        tokens = tokens.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        with torch.set_grad_enabled(training):
            use_amp = scaler is not None and device.type == "cuda"
            with torch.autocast(device_type=device.type, enabled=use_amp):
                logits = model(tokens, lengths)
                loss = criterion(logits, labels)

            if training:
                optimizer.zero_grad(set_to_none=True)
                if use_amp:
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    nn.utils.clip_grad_norm_(model.parameters(), clip)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    nn.utils.clip_grad_norm_(model.parameters(), clip)
                    optimizer.step()
                if scheduler is not None:
                    scheduler.step()

        batch = labels.size(0)
        total_loss += float(loss.detach()) * batch
        seen += batch
        predictions.append(logits.detach().argmax(dim=-1).cpu().numpy())
        targets.append(labels.detach().cpu().numpy())

    y_pred = np.concatenate(predictions)
    y_true = np.concatenate(targets)
    result = EpochResult(
        loss=total_loss / max(seen, 1),
        accuracy=float((y_pred == y_true).mean()),
        macro_f1=float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
    )
    return result, y_true, y_pred


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    data = parser.add_argument_group("data")
    data.add_argument("--dataset", default="dair-ai/emotion")
    data.add_argument("--data-dir", type=Path, default=None, help="CSV splits instead of the hub")
    data.add_argument("--max-words", type=int, default=10_000)
    data.add_argument(
        "--min-freq", type=int, default=2, help="drop hapax words; they only add noise"
    )
    data.add_argument("--max-length", type=int, default=50)

    arch = parser.add_argument_group("architecture")
    arch.add_argument("--embedding-dim", type=int, default=300)
    arch.add_argument("--hidden", type=int, nargs=2, default=[128, 64])
    arch.add_argument("--dropout", type=float, default=0.4)
    arch.add_argument("--pooling", choices=["attention", "last"], default="attention")
    arch.add_argument(
        "--legacy-arch",
        action="store_true",
        help="unmasked + last-state pooling, as in the notebook",
    )

    optim = parser.add_argument_group("optimisation")
    optim.add_argument("--epochs", type=int, default=15)
    optim.add_argument("--batch-size", type=int, default=64)
    optim.add_argument("--lr", type=float, default=2e-3)
    optim.add_argument("--weight-decay", type=float, default=1e-2)
    optim.add_argument("--label-smoothing", type=float, default=0.05)
    optim.add_argument("--patience", type=int, default=4)
    optim.add_argument("--clip", type=float, default=1.0)
    optim.add_argument("--no-class-weights", action="store_true")

    misc = parser.add_argument_group("misc")
    misc.add_argument("--device", default="auto")
    misc.add_argument("--workers", type=int, default=0)
    misc.add_argument("--seed", type=int, default=42)
    misc.add_argument("--out", type=Path, default=ROOT / "artifacts" / "emotion_bigru_v2.pt")
    misc.add_argument("--report", type=Path, default=ROOT / "artifacts" / "training_report.json")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s  %(levelname)-7s %(message)s", datefmt="%H:%M:%S"
    )
    args = parse_args(argv)
    set_seed(args.seed)
    device = pick_device(args.device)
    logger.info("device: %s", device)

    splits, label_names = load_splits(args.data_dir, args.dataset)
    ensure_validation(splits, seed=args.seed)
    for name, split in splits.items():
        logger.info("%-11s %6d rows", name, len(split))

    vocab = build_vocab(splits["train"].texts, args.max_words, args.min_freq)

    config = ModelConfig(
        vocab_size=args.max_words,
        embedding_dim=args.embedding_dim,
        hidden_sizes=tuple(args.hidden),
        num_classes=len(label_names),
        dropout=args.dropout,
        max_sequence_length=args.max_length,
        masked=not args.legacy_arch,
        pooling="last" if args.legacy_arch else args.pooling,
    )
    model = BiGRUEmotionClassifier(config).to(device)
    logger.info("model: %s", config.to_dict())
    logger.info("parameters: %s", f"{model.num_parameters():,}")

    loaders = {
        name: make_loader(
            EmotionDataset(split, vocab, args.max_length),
            args.batch_size,
            shuffle=(name == "train"),
            workers=args.workers,
        )
        for name, split in splits.items()
    }

    weights = None
    if not args.no_class_weights:
        classes = np.arange(len(label_names))
        weights = torch.tensor(
            compute_class_weight("balanced", classes=classes, y=np.array(splits["train"].labels)),
            dtype=torch.float32,
            device=device,
        )
        logger.info("class weights: %s", np.round(weights.cpu().numpy(), 3).tolist())

    criterion = nn.CrossEntropyLoss(weight=weights, label_smoothing=args.label_smoothing)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=args.lr,
        epochs=args.epochs,
        steps_per_epoch=max(1, len(loaders["train"])),
        pct_start=0.25,
    )
    scaler = torch.amp.GradScaler(enabled=device.type == "cuda")
    stopper = EarlyStopping(patience=args.patience)

    history: list[dict[str, float]] = []
    started = time.perf_counter()

    for epoch in range(1, args.epochs + 1):
        epoch_start = time.perf_counter()
        train_result, _, _ = run_epoch(
            model, loaders["train"], criterion, device, optimizer, scheduler, scaler, args.clip
        )
        val_result, _, _ = run_epoch(model, loaders["validation"], criterion, device)
        elapsed = time.perf_counter() - epoch_start

        logger.info(
            "epoch %2d/%d  %.0fs | train %s | val %s",
            epoch,
            args.epochs,
            elapsed,
            train_result,
            val_result,
        )
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_result.loss,
                "train_accuracy": train_result.accuracy,
                "val_loss": val_result.loss,
                "val_accuracy": val_result.accuracy,
                "val_macro_f1": val_result.macro_f1,
                "lr": scheduler.get_last_lr()[0],
                "seconds": elapsed,
            }
        )

        if stopper.step(val_result.macro_f1, model, epoch):
            logger.info("early stop: no improvement for %d epochs", args.patience)
            break

    stopper.restore(model)
    logger.info(
        "restored best weights from epoch %d (val macroF1 %.4f) after %.0fs",
        stopper.best_epoch,
        stopper.best,
        time.perf_counter() - started,
    )

    report: dict[str, object] = {"history": history, "best_epoch": stopper.best_epoch}

    if "test" in loaders:
        test_result, y_true, y_pred = run_epoch(model, loaders["test"], criterion, device)
        logger.info("test: %s", test_result)
        text_report = classification_report(
            y_true, y_pred, target_names=label_names, digits=4, zero_division=0
        )
        print("\n" + text_report)
        report["test"] = {
            "loss": test_result.loss,
            "accuracy": test_result.accuracy,
            "macro_f1": test_result.macro_f1,
            "per_class": classification_report(
                y_true, y_pred, target_names=label_names, output_dict=True, zero_division=0
            ),
            "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
        }

    bundle = Bundle(
        config=config,
        labels=label_names,
        vocab=vocab,
        tokenizer={
            "num_words": args.max_words,
            "oov_token": "<unk>",
            "lower": True,
            "max_length": args.max_length,
            "padding": "post",
            "truncating": "post",
            "strip_apostrophes": True,
        },
        state_dict=model.state_dict(),
        meta={
            "source": "training/train.py",
            "dataset": str(args.data_dir or args.dataset),
            "trained_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "torch_version": str(torch.__version__),
            "parameters": model.num_parameters(),
            "best_epoch": stopper.best_epoch,
            "val_macro_f1": stopper.best,
            "seed": args.seed,
        },
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(bundle.to_payload(), args.out)
    logger.info("saved %s (%.1f MB)", args.out, args.out.stat().st_size / 1e6)

    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2), encoding="utf-8")
    logger.info("wrote %s", args.report)
    logger.info("serve it with: MOODLINE_ARTIFACT_PATH=%s uvicorn app.main:app", args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
