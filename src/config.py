"""Configuration loading and validation."""
from __future__ import annotations

from pathlib import Path
from typing import Any
import copy

import yaml


def load_config(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Configuration file not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    if cfg.get("training", {}).get("mode") == "fast":
        fast = cfg.get("fast", {})
        cfg = copy.deepcopy(cfg)
        for key in ("max_train_images", "max_val_images", "max_test_images"):
            if key in fast:
                cfg["dataset"][key] = fast[key]
        for key in ("epochs_phase1", "epochs_phase2", "max_train_tiles_per_image"):
            if key in fast:
                cfg["training"][key] = fast[key]
    validate_config(cfg)
    return cfg


def validate_config(cfg: dict[str, Any]) -> None:
    required = ["seed", "dataset", "split", "model", "training", "evaluation"]
    missing = [k for k in required if k not in cfg]
    if missing:
        raise ValueError(f"Missing configuration sections: {missing}")
    ds = cfg["dataset"]
    for key in ("tile_size", "stride"):
        if int(ds[key]) <= 0:
            raise ValueError(f"dataset.{key} must be > 0")
    if int(ds["stride"]) > int(ds["tile_size"]):
        raise ValueError("dataset.stride must be <= dataset.tile_size to provide overlap")
    if not 0 <= float(ds["tissue_threshold"]) <= 1:
        raise ValueError("dataset.tissue_threshold must be in [0, 1]")
    if not 0 < float(cfg["split"]["test_size"]) < 1:
        raise ValueError("split.test_size must be in (0,1)")
    if not 0 < float(cfg["split"]["val_size"]) < 1:
        raise ValueError("split.val_size must be in (0,1)")
    if float(cfg["split"]["test_size"]) + float(cfg["split"]["val_size"]) >= 1:
        raise ValueError("test_size + val_size must be < 1")
    if float(cfg["evaluation"]["threshold"]) < 0 or float(cfg["evaluation"]["threshold"]) > 1:
        raise ValueError("evaluation.threshold must be in [0,1]")
    if cfg["evaluation"].get("aggregation") not in {"mean", "max", "top_k"}:
        raise ValueError("evaluation.aggregation must be mean, max, or top_k")
