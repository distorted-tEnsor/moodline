"""Dataset loading and vocabulary building for training.

Kept separate from the training loop so both ``train.py`` and ``evaluate.py`` see
exactly the same text normalisation the *serving* path uses (``app.tokenizer``).
A mismatch there is the classic way a model that scored 92% offline scores 60%
in production.
"""

from __future__ import annotations

import csv
import logging
import sys
from collections import Counter
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.model import DEFAULT_LABELS
from app.tokenizer import normalise

logger = logging.getLogger(__name__)

PAD_INDEX = 0
OOV_INDEX = 1
OOV_TOKEN = "<unk>"


@dataclass
class Split:
    texts: list[str]
    labels: list[int]

    def __len__(self) -> int:
        return len(self.texts)


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------


def load_from_hub(name: str = "dair-ai/emotion") -> tuple[dict[str, Split], list[str]]:
    from datasets import load_dataset  # imported lazily: heavy, and optional

    raw = load_dataset(name)
    label_names = list(raw["train"].features["label"].names)
    splits = {key: Split(list(value["text"]), list(value["label"])) for key, value in raw.items()}
    return splits, label_names


def load_from_csv(directory: Path) -> tuple[dict[str, Split], list[str]]:
    """Expects ``train.csv`` / ``validation.csv`` / ``test.csv`` with ``text,label``.

    ``label`` may be an index or a name; names are mapped through DEFAULT_LABELS.
    """
    splits: dict[str, Split] = {}
    label_names = list(DEFAULT_LABELS)
    lookup = {name: index for index, name in enumerate(label_names)}

    for split in ("train", "validation", "test"):
        path = directory / f"{split}.csv"
        if not path.exists():
            continue
        texts: list[str] = []
        labels: list[int] = []
        with path.open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                raw_label = str(row["label"]).strip()
                labels.append(int(raw_label) if raw_label.isdigit() else lookup[raw_label])
                texts.append(row["text"])
        splits[split] = Split(texts, labels)

    if "train" not in splits:
        raise FileNotFoundError(f"no train.csv in {directory}")
    return splits, label_names


def load_splits(
    data_dir: Path | None = None, dataset: str = "dair-ai/emotion"
) -> tuple[dict[str, Split], list[str]]:
    if data_dir is not None:
        logger.info("loading CSV splits from %s", data_dir)
        return load_from_csv(data_dir)
    try:
        logger.info("loading %s from the Hugging Face hub", dataset)
        return load_from_hub(dataset)
    except Exception as exc:
        raise SystemExit(
            f"could not load '{dataset}' ({exc}).\n"
            "Either `pip install datasets` with network access, or pass "
            "--data-dir pointing at train.csv / validation.csv / test.csv."
        ) from exc


def ensure_validation(splits: dict[str, Split], fraction: float = 0.1, seed: int = 42) -> None:
    """Carve a validation split out of train if the dataset has none.

    The original notebook early-stopped on the *test* set, which quietly turns the
    reported test accuracy into a training metric.
    """
    if "validation" in splits:
        return
    train = splits["train"]
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(train))
    cut = int(len(train) * (1 - fraction))
    keep, hold = order[:cut], order[cut:]
    splits["train"] = Split([train.texts[i] for i in keep], [train.labels[i] for i in keep])
    splits["validation"] = Split([train.texts[i] for i in hold], [train.labels[i] for i in hold])
    logger.info("no validation split found; held out %d rows from train", len(hold))


# ---------------------------------------------------------------------------
# Vocabulary
# ---------------------------------------------------------------------------


def build_vocab(texts: Iterable[str], max_words: int = 10_000, min_freq: int = 1) -> dict[str, int]:
    """Frequency-ranked vocabulary. Index 0 is padding, index 1 is the OOV token."""
    counter: Counter[str] = Counter()
    for text in texts:
        cleaned = normalise(text)
        if cleaned:
            counter.update(cleaned.split(" "))

    vocab = {OOV_TOKEN: OOV_INDEX}
    for word, count in counter.most_common():
        if count < min_freq or len(vocab) + 1 >= max_words:
            break
        vocab[word] = len(vocab) + 1
    logger.info("vocabulary: %d distinct words seen, %d kept", len(counter), len(vocab))
    return vocab


# ---------------------------------------------------------------------------
# Torch plumbing
# ---------------------------------------------------------------------------


class EmotionDataset(Dataset):
    def __init__(self, split: Split, vocab: dict[str, int], max_length: int = 50) -> None:
        self.labels = split.labels
        self.max_length = max_length
        get = vocab.get
        self.sequences = [
            [get(word, OOV_INDEX) for word in normalise(text).split(" ") if word][:max_length]
            or [OOV_INDEX]
            for text in split.texts
        ]

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, index: int) -> tuple[list[int], int]:
        return self.sequences[index], self.labels[index]


def collate(
    batch: Sequence[tuple[list[int], int]],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Pads to the longest member of *this* batch, not to a fixed 50.

    With masking on, short sentences no longer pay for 50 GRU steps; on the
    emotion corpus (median ~17 tokens) that alone roughly halves training time.
    """
    lengths = torch.tensor([len(sequence) for sequence, _ in batch], dtype=torch.int64)
    width = int(lengths.max())
    tokens = torch.zeros((len(batch), width), dtype=torch.int64)
    for row, (sequence, _) in enumerate(batch):
        tokens[row, : len(sequence)] = torch.tensor(sequence, dtype=torch.int64)
    labels = torch.tensor([label for _, label in batch], dtype=torch.int64)
    return tokens, lengths, labels


def make_loader(
    dataset: EmotionDataset, batch_size: int, *, shuffle: bool, workers: int = 0
) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collate,
        num_workers=workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
        persistent_workers=workers > 0,
    )
