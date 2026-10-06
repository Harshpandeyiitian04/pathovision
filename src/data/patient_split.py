"""Deterministic patient-level splitting with leakage checks."""
from __future__ import annotations

from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split


def _patient_table(metadata: pd.DataFrame) -> pd.DataFrame:
    labels_per_patient = metadata.groupby("patient_id")["label"].agg(lambda s: int(round(s.mean())))
    consistency = metadata.groupby("patient_id")["label"].nunique()
    if (consistency > 1).any():
        bad = consistency[consistency > 1].index.tolist()[:10]
        raise ValueError(f"A patient has multiple labels; refusing to split: {bad}")
    return labels_per_patient.rename("label").reset_index()


def split_by_patient(
    metadata: pd.DataFrame,
    test_size: float = 0.20,
    val_size: float = 0.15,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split images by patient; val_size is fraction of the entire dataset."""
    patients = _patient_table(metadata)
    if len(patients) < 5:
        raise ValueError("At least 5 unique patients are required for a meaningful split.")
    temp_size = test_size + val_size
    temp_fraction = temp_size
    stratify = patients["label"] if patients["label"].nunique() == 2 else None
    try:
        train_p, temp_p = train_test_split(
            patients, test_size=temp_fraction, random_state=random_state, stratify=stratify
        )
        relative_val = val_size / temp_fraction
        temp_stratify = temp_p["label"] if temp_p["label"].nunique() == 2 else None
        val_p, test_p = train_test_split(
            temp_p, test_size=test_size / temp_fraction, random_state=random_state, stratify=temp_stratify
        )
    except ValueError as exc:
        # Small cohorts may not support stratification; keep the patient split deterministic.
        print(f"Warning: stratified patient split unavailable ({exc}); using deterministic unstratified split.")
        train_p, temp_p = train_test_split(patients, test_size=temp_fraction, random_state=random_state, stratify=None)
        val_p, test_p = train_test_split(temp_p, test_size=test_size / temp_fraction, random_state=random_state, stratify=None)

    assignments = {
        "train": set(train_p["patient_id"]),
        "val": set(val_p["patient_id"]),
        "test": set(test_p["patient_id"]),
    }
    assert_no_patient_leakage(*assignments.values())
    split_frames = tuple(metadata[metadata["patient_id"].isin(assignments[name])].copy() for name in ("train", "val", "test"))
    for frame, name in zip(split_frames, ("train", "val", "test")):
        frame["split"] = name
    return split_frames


def assert_no_patient_leakage(train_patients, val_patients, test_patients) -> None:
    train_patients, val_patients, test_patients = map(set, (train_patients, val_patients, test_patients))
    if train_patients & val_patients:
        raise AssertionError("Patient leakage detected between train and validation")
    if train_patients & test_patients:
        raise AssertionError("Patient leakage detected between train and test")
    if val_patients & test_patients:
        raise AssertionError("Patient leakage detected between validation and test")


def save_splits(train: pd.DataFrame, val: pd.DataFrame, test: pd.DataFrame, out_dir: str | Path) -> None:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    train.to_csv(out / "train.csv", index=False)
    val.to_csv(out / "val.csv", index=False)
    test.to_csv(out / "test.csv", index=False)
