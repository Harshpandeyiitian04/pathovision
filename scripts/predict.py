#!/usr/bin/env python3
"""Single-image tile inference with optional tile CSV, visualization, and Grad-CAM."""
from __future__ import annotations

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse
from pathlib import Path
import pandas as pd
import torch
from PIL import Image, ImageDraw

from src.config import load_config
from src.data.tiling import extract_tile, select_tile_coordinates
from src.evaluation.aggregation import aggregate_probabilities
from src.explainability.gradcam import GradCAM, overlay_heatmap
from src.models.resnet import build_model
from src.seed import seed_everything


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--image", required=True)
    parser.add_argument("--save-dir", default="results/inference")
    parser.add_argument("--gradcam", action="store_true")
    args = parser.parse_args()
    cfg = load_config(args.config)
    seed_everything(int(cfg["seed"]))
    image_path = Path(args.image)
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint_path = Path(cfg["paths"]["checkpoint_dir"]) / "best_model.pth"
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model = build_model(cfg).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    from src.data.dataset import make_transform
    transform = make_transform(cfg, train=False)
    with Image.open(image_path) as img:
        image = img.convert("RGB")
        coords = select_tile_coordinates(image, int(cfg["dataset"]["tile_size"]), int(cfg["dataset"]["stride"]), float(cfg["dataset"]["tissue_threshold"]), int(cfg["dataset"].get("background_gray_threshold", 245)), seed=int(cfg["seed"]))
        if not coords:
            raise ValueError("No tissue-containing tiles were found. Lower dataset.tissue_threshold and retry.")
        rows = []
        tensors = []
        for coord in coords:
            tile = extract_tile(image, coord.x, coord.y, int(cfg["dataset"]["tile_size"]))
            tensors.append(transform(tile))
        with torch.no_grad():
            probs = torch.sigmoid(model(torch.stack(tensors).to(device)).reshape(-1)).cpu().numpy()
        for coord, prob in zip(coords, probs):
            rows.append({"image_path": str(image_path), "tile_id": coord.tile_id, "x": coord.x, "y": coord.y, "probability_malignant": float(prob), "predicted_label": int(prob >= float(cfg["evaluation"]["threshold"]))})
        tile_df = pd.DataFrame(rows)
        method = cfg["evaluation"]["aggregation"]
        image_prob = aggregate_probabilities(tile_df.probability_malignant, method, float(cfg["evaluation"]["top_k_fraction"]))
        label = "Malignant" if image_prob >= float(cfg["evaluation"]["threshold"]) else "Benign"
        out = Path(args.save_dir); out.mkdir(parents=True, exist_ok=True)
        tile_df.to_csv(out / "tile_predictions.csv", index=False)
        print(f"Image: {image_path}")
        print(f"Prediction: {label}")
        print(f"Probability: {image_prob:.4f}")
        print(f"Aggregation: {method}")
        print(f"Number of valid tiles: {len(tile_df)}")

        canvas = image.copy()
        draw = ImageDraw.Draw(canvas)
        for row in rows:
            x, y = row["x"], row["y"]
            draw.rectangle((x, y, x + int(cfg["dataset"]["tile_size"]) - 1, y + int(cfg["dataset"]["tile_size"]) - 1), width=2)
        canvas.save(out / "tile_grid.png")

        if args.gradcam:
            target_layer = model.layer4[-1].conv2
            cam = GradCAM(model, target_layer)
            top_idx = int(tile_df["probability_malignant"].idxmax())
            tile = extract_tile(image, rows[top_idx]["x"], rows[top_idx]["y"], int(cfg["dataset"]["tile_size"]))
            cam_tensor = transform(tile).unsqueeze(0).to(device)
            heatmap = cam(cam_tensor)
            overlay_heatmap(tile, heatmap, out / "gradcam_top_tile.png")
            cam.remove()


if __name__ == "__main__":
    main()
