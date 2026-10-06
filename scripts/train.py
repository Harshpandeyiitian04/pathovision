#!/usr/bin/env python3
"""Train PathoVision from prepared patient-level splits."""
from __future__ import annotations

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse
from pathlib import Path
import torch
import pandas as pd

from src.config import load_config
from src.data.dataset import TileDataset, build_tile_index
from src.models.resnet import build_model
from src.seed import seed_everything
from src.training.train import train_model


def load_split(path: str) -> pd.DataFrame:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Missing split file {p}. Run scripts/prepare_data.py first.")
    return pd.read_csv(p)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/default.yaml")
    args = parser.parse_args()
    cfg = load_config(args.config)
    seed_everything(int(cfg["seed"]), bool(cfg["reproducibility"].get("deterministic", True)))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {torch.cuda.get_device_name(0) if device.type == 'cuda' else 'CPU'}")
    split_dir = Path(cfg["paths"]["splits_dir"])
    train_df, val_df = load_split(split_dir / "train.csv"), load_split(split_dir / "val.csv")
    train_records = build_tile_index(train_df, cfg, cfg["training"].get("max_train_tiles_per_image"))
    val_records = build_tile_index(val_df, cfg, cfg["training"].get("max_eval_tiles_per_image"))
    print(f"Train tiles: {len(train_records)} | Validation tiles: {len(val_records)}")
    train_ds = TileDataset(train_records, cfg, train=True)
    val_ds = TileDataset(val_records, cfg, train=False)
    model = build_model(cfg).to(device)
    train_model(model, train_ds, val_ds, cfg, train_df, device)
    print(f"Saved best checkpoint to {cfg['paths']['checkpoint_dir']}/best_model.pth")


if __name__ == "__main__":
    main()
