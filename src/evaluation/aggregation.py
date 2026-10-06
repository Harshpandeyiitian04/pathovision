"""Tile-to-image aggregation strategies."""
from __future__ import annotations

import math
from typing import Iterable

import numpy as np
import pandas as pd


def aggregate_probabilities(probabilities: Iterable[float], method: str = "mean", top_k_fraction: float = 0.2) -> float:
    values = np.asarray(list(probabilities), dtype=float)
    if values.size == 0:
        raise ValueError("Cannot aggregate an empty probability list")
    if np.any(~np.isfinite(values)):
        raise ValueError("Probabilities contain non-finite values")
    if method == "mean":
        return float(values.mean())
    if method == "max":
        return float(values.max())
    if method == "top_k":
        if not 0 < top_k_fraction <= 1:
            raise ValueError("top_k_fraction must be in (0,1]")
        k = max(1, math.ceil(len(values) * top_k_fraction))
        return float(np.sort(values)[-k:].mean())
    raise ValueError(f"Unknown aggregation method: {method}")


def aggregate_tile_dataframe(tile_df: pd.DataFrame, method: str, top_k_fraction: float = 0.2) -> pd.DataFrame:
    required = {"image_path", "patient_id", "label", "probability_malignant"}
    missing = required - set(tile_df.columns)
    if missing:
        raise ValueError(f"Missing columns for aggregation: {sorted(missing)}")
    rows = []
    for image_path, group in tile_df.groupby("image_path", sort=False):
        prob = aggregate_probabilities(group["probability_malignant"], method, top_k_fraction)
        first = group.iloc[0]
        row = {
            "image_path": image_path,
            "patient_id": first["patient_id"],
            "label": int(first["label"]),
            "probability_malignant": prob,
            "predicted_label": int(prob >= 0.5),
            "tile_count": int(len(group)),
        }
        for col in ("magnification", "subtype", "class_name"):
            if col in group.columns:
                row[col] = first[col]
        rows.append(row)
    return pd.DataFrame(rows)
