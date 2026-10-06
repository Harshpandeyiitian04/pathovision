#!/usr/bin/env python3
"""Discover BreakHis, validate it, and create leakage-safe patient splits."""
from __future__ import annotations

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse
from pathlib import Path
import json

import pandas as pd
import matplotlib.pyplot as plt

from src.config import load_config
from src.data.parse_breakhis import build_metadata, validate_images
from src.data.patient_split import assert_no_patient_leakage, save_splits, split_by_patient
from src.seed import seed_everything


def _limit_split(df: pd.DataFrame, max_images: int | None, seed: int) -> pd.DataFrame:
    if max_images is None or len(df) <= max_images:
        return df.copy()
    if max_images < 2:
        raise ValueError("A split image limit must be at least 2")
    # Apply image limits after splitting so a patient's images stay in one set.
    pieces = []
    for _, group in df.groupby("label"):
        n = max(1, round(max_images * len(group) / len(df)))
        pieces.append(group.sample(n=min(n, len(group)), random_state=seed))
    sampled = pd.concat(pieces, ignore_index=True)
    if len(sampled) > max_images:
        sampled = sampled.sample(max_images, random_state=seed)
    return sampled.reset_index(drop=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--validate-images", action="store_true")
    args = parser.parse_args()
    cfg = load_config(args.config)
    seed_everything(int(cfg["seed"]), bool(cfg["reproducibility"].get("deterministic", True)))
    root = cfg["dataset"]["root"]
    metadata = build_metadata(root)
    if args.validate_images:
        validate_images(metadata)
    metadata_path = Path(cfg["paths"]["metadata_csv"])
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    metadata.to_csv(metadata_path, index=False)

    train, val, test = split_by_patient(metadata, float(cfg["split"]["test_size"]), float(cfg["split"]["val_size"]), int(cfg["split"]["random_state"]))
    limits = cfg["dataset"]
    train = _limit_split(train, limits.get("max_train_images"), int(cfg["seed"]))
    val = _limit_split(val, limits.get("max_val_images"), int(cfg["seed"]) + 1)
    test = _limit_split(test, limits.get("max_test_images"), int(cfg["seed"]) + 2)
    assert_no_patient_leakage(set(train.patient_id), set(val.patient_id), set(test.patient_id))
    save_splits(train, val, test, cfg["paths"]["splits_dir"])

    stats = {
        "total_images": int(len(metadata)), "benign": int((metadata.label == 0).sum()), "malignant": int((metadata.label == 1).sum()),
        "patients": int(metadata.patient_id.nunique()),
        "split_patients": {"train": int(train.patient_id.nunique()), "val": int(val.patient_id.nunique()), "test": int(test.patient_id.nunique())},
        "split_images": {"train": int(len(train)), "val": int(len(val)), "test": int(len(test))},
        "magnifications": {str(k): int(v) for k, v in metadata.magnification.value_counts().sort_index().items()},
        "class_by_split": {name: {"benign": int((df.label == 0).sum()), "malignant": int((df.label == 1).sum())} for name, df in (("train", train), ("val", val), ("test", test))},
    }
    results = Path(cfg["paths"]["results_dir"])
    (results / "metrics").mkdir(parents=True, exist_ok=True)
    (results / "metrics" / "dataset_sanity.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
    class_counts = metadata.assign(class_name=metadata.label.map({0: "Benign", 1: "Malignant"}))["class_name"].value_counts()
    fig, ax = plt.subplots(figsize=(5, 4))
    class_counts.plot(kind="bar", ax=ax)
    ax.set_ylabel("Image count")
    ax.set_title("BreakHis Class Distribution")
    fig.tight_layout()
    fig.savefig(results / "figures" / "class_distribution.png", dpi=180)
    plt.close(fig)
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
