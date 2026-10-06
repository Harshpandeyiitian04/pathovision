"""Final test evaluation and reporting."""
from __future__ import annotations

from pathlib import Path
import json

import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.data.dataset import TileDataset, build_tile_index, make_transform
from src.evaluation.aggregation import aggregate_tile_dataframe
from src.evaluation.metrics import compute_metrics, metrics_by_group, plot_confusion_matrix, plot_roc, save_json
from src.seed import worker_init_fn


def predict_tiles(model, dataset, cfg, device):
    loader = DataLoader(
        dataset,
        batch_size=int(cfg["training"]["batch_size"]),
        shuffle=False,
        num_workers=int(cfg["training"].get("num_workers", 0)),
        pin_memory=device.type == "cuda",
        worker_init_fn=worker_init_fn,
    )
    model.eval()
    rows = []
    with torch.no_grad():
        for batch in tqdm(loader, desc="Test inference"):
            images = batch["image"].to(device, non_blocking=True)
            probs = torch.sigmoid(model(images).reshape(-1)).cpu().numpy()
            for i, prob in enumerate(probs):
                rows.append({
                    "image_path": batch["image_path"][i], "patient_id": batch["patient_id"][i],
                    "tile_id": int(batch["tile_id"][i]), "x": int(batch["x"][i]), "y": int(batch["y"][i]),
                    "label": int(round(float(batch["label"][i]))), "probability_malignant": float(prob),
                })
    return pd.DataFrame(rows)


def evaluate_all(model, test_metadata, cfg, device):
    max_tiles = cfg["training"].get("max_eval_tiles_per_image")
    records = build_tile_index(test_metadata, cfg, max_tiles_per_image=max_tiles)
    dataset = TileDataset(records, cfg, train=False)
    tile_df = predict_tiles(model, dataset, cfg, device)
    # Keep metadata for the magnification and subtype summaries.
    meta_cols = ["image_path", "magnification", "subtype", "class_name"]
    tile_df = tile_df.merge(test_metadata[meta_cols], on="image_path", how="left", validate="many_to_one")
    results_dir = Path(cfg["paths"]["results_dir"])
    (results_dir / "predictions").mkdir(parents=True, exist_ok=True)
    tile_df.to_csv(results_dir / "predictions" / "test_tile_predictions.csv", index=False)

    threshold = float(cfg["evaluation"]["threshold"])
    aggregation_metrics = {}
    for method in ("mean", "max", "top_k"):
        image_df = aggregate_tile_dataframe(tile_df, method, float(cfg["evaluation"]["top_k_fraction"]))
        metrics = compute_metrics(image_df["label"], image_df["probability_malignant"], threshold)
        aggregation_metrics[method] = metrics
        if method == cfg["evaluation"]["aggregation"]:
            image_df.to_csv(results_dir / "predictions" / "test_predictions.csv", index=False)
            final_metrics = metrics
            final_image_df = image_df
    save_json(final_metrics, results_dir / "metrics" / "test_metrics.json")
    pd.DataFrame([
        {"aggregation": method, **{k: aggregation_metrics[method][k] for k in ("accuracy", "precision", "recall", "f1", "roc_auc")}}
        for method in aggregation_metrics
    ]).to_csv(results_dir / "metrics" / "aggregation_comparison.csv", index=False)
    by_mag = metrics_by_group(final_image_df, "magnification", threshold)
    by_mag.to_csv(results_dir / "metrics" / "by_magnification.csv", index=False)
    plot_confusion_matrix(final_metrics, results_dir / "figures" / "confusion_matrix.png")
    plot_roc(final_image_df["label"], final_image_df["probability_malignant"], results_dir / "figures" / "roc_curve.png")

    report = {
        "dataset_statistics": {
            "test_images": int(len(test_metadata)),
            "test_patients": int(test_metadata["patient_id"].nunique()),
            "test_tiles": int(len(tile_df)),
        },
        "test_metrics": final_metrics,
        "aggregation_metrics": aggregation_metrics,
        "magnification_metrics": by_mag.to_dict(orient="records"),
        "aggregation_selected": cfg["evaluation"]["aggregation"],
    }
    save_json(report, results_dir / "metrics" / "final_report.json")
    return report
