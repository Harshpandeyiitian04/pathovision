#!/usr/bin/env python3
"""Evaluate the frozen best checkpoint on the untouched test split."""
from __future__ import annotations

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse
from pathlib import Path
import torch
import pandas as pd

from src.config import load_config
from src.evaluation.evaluate import evaluate_all
from src.models.resnet import build_model
from src.seed import seed_everything


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/default.yaml")
    args = parser.parse_args()
    cfg = load_config(args.config)
    seed_everything(int(cfg["seed"]), bool(cfg["reproducibility"].get("deterministic", True)))
    ckpt_path = Path(cfg["paths"]["checkpoint_dir"]) / "best_model.pth"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}. Train first.")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
    model = build_model(cfg).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    test_df = pd.read_csv(Path(cfg["paths"]["splits_dir"]) / "test.csv")
    report = evaluate_all(model, test_df, cfg, device)
    print("Final test metrics:")
    print(report["test_metrics"])


if __name__ == "__main__":
    main()
