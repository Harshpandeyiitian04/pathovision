"""Validation loop using image-level aggregation for checkpoint selection."""
from __future__ import annotations

import pandas as pd
import torch
from torch import nn
from tqdm import tqdm

from src.evaluation.aggregation import aggregate_tile_dataframe
from src.evaluation.metrics import compute_metrics


def run_tile_inference(model, loader, device: torch.device, criterion: nn.Module | None = None, desc: str = "Validation"):
    model.eval()
    rows = []
    losses = []
    with torch.no_grad():
        for batch in tqdm(loader, desc=desc, leave=False):
            images = batch["image"].to(device, non_blocking=True)
            labels = batch["label"].to(device, non_blocking=True)
            logits = model(images).reshape(-1)
            if criterion is not None:
                losses.append(float(criterion(logits, labels).item()) * len(labels))
            probs = torch.sigmoid(logits).detach().cpu().numpy()
            for i, prob in enumerate(probs):
                row = {
                    "image_path": batch["image_path"][i],
                    "patient_id": batch["patient_id"][i],
                    "tile_id": int(batch["tile_id"][i]),
                    "x": int(batch["x"][i]),
                    "y": int(batch["y"][i]),
                    "label": int(round(float(batch["label"][i]))),
                    "probability_malignant": float(prob),
                }
                rows.append(row)
    tile_df = pd.DataFrame(rows)
    avg_loss = sum(losses) / len(tile_df) if len(tile_df) else float("nan")
    return tile_df, avg_loss


def image_level_validation(tile_df: pd.DataFrame, threshold: float, aggregation: str, top_k_fraction: float) -> dict:
    image_df = aggregate_tile_dataframe(tile_df, aggregation, top_k_fraction)
    metrics = compute_metrics(image_df["label"], image_df["probability_malignant"], threshold)
    return metrics | {"image_predictions": image_df}
