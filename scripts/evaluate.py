#!/usr/bin/env python3
"""Score a saved bundle against a dataset split and write a readable report.

Works with either artifact — the converted Keras model or a retrained one — so
you can compare them on the same rows before swapping one for the other.

    python scripts/evaluate.py --artifact artifacts/emotion_bigru.pt
    python scripts/evaluate.py --artifact artifacts/emotion_bigru_v2.pt --data-dir data/ --split test
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.config import get_settings
from app.engine import EmotionEngine
from training.data import load_splits

ROOT = Path(__file__).resolve().parents[1]


def render_confusion(matrix: np.ndarray, labels: list[str]) -> str:
    width = max(len(label) for label in labels) + 2
    header = " " * width + "".join(f"{label[:6]:>8}" for label in labels)
    rows = [header]
    for index, label in enumerate(labels):
        counts = "".join(f"{value:>8}" for value in matrix[index])
        rows.append(f"{label:<{width}}{counts}")
    return "\n".join(rows)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--artifact", type=Path, default=ROOT / "artifacts" / "emotion_bigru.pt")
    parser.add_argument("--dataset", default="dair-ai/emotion")
    parser.add_argument("--data-dir", type=Path, default=None)
    parser.add_argument("--split", default="test")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--out", type=Path, default=None, help="write the report as JSON")
    args = parser.parse_args()

    from sklearn.metrics import classification_report, confusion_matrix, f1_score

    settings = get_settings().model_copy(update={"artifact_path": args.artifact, "cache_size": 0})
    engine = EmotionEngine.load(settings)
    print(f"model      : {engine.version} ({args.artifact.name})")

    splits, label_names = load_splits(args.data_dir, args.dataset)
    if args.split not in splits:
        raise SystemExit(f"split '{args.split}' not in {sorted(splits)}")
    split = splits[args.split]
    print(f"evaluating : {len(split)} rows from '{args.split}'\n")

    predictions: list[int] = []
    confidences: list[float] = []
    label_index = {name: index for index, name in enumerate(engine.labels)}

    for start in range(0, len(split), args.batch_size):
        chunk = split.texts[start : start + args.batch_size]
        for prediction in engine.predict_many(chunk, use_cache=False):
            predictions.append(label_index[prediction.label])
            confidences.append(prediction.probabilities[prediction.label])

    y_true = np.array(split.labels)
    y_pred = np.array(predictions)
    confidence = np.array(confidences)
    correct = y_pred == y_true

    print(
        classification_report(y_true, y_pred, target_names=label_names, digits=4, zero_division=0)
    )
    matrix = confusion_matrix(y_true, y_pred, labels=list(range(len(label_names))))
    print("confusion matrix (rows = truth)\n")
    print(render_confusion(matrix, label_names))

    print(
        f"\naccuracy {correct.mean():.4f}   macro-F1 "
        f"{f1_score(y_true, y_pred, average='macro', zero_division=0):.4f}"
    )
    print(
        f"mean confidence: {confidence.mean():.3f} overall, "
        f"{confidence[correct].mean():.3f} when right, "
        f"{confidence[~correct].mean() if (~correct).any() else float('nan'):.3f} when wrong"
    )
    low = confidence < 0.5
    print(
        f"predictions below 0.5 confidence: {low.mean():.1%} (accuracy there: {correct[low].mean() if low.any() else float('nan'):.3f})"
    )

    if args.out:
        args.out.write_text(
            json.dumps(
                {
                    "artifact": str(args.artifact),
                    "model_version": engine.version,
                    "split": args.split,
                    "accuracy": float(correct.mean()),
                    "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
                    "per_class": classification_report(
                        y_true, y_pred, target_names=label_names, output_dict=True, zero_division=0
                    ),
                    "confusion_matrix": matrix.tolist(),
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
