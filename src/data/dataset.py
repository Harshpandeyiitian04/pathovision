"""PyTorch datasets for image-level metadata and on-the-fly tissue-filtered tiles."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset

from .tiling import TileCoordinate, select_tile_coordinates


@dataclass(frozen=True)
class TileRecord:
    image_path: str
    patient_id: str
    label: int
    tile_id: int
    x: int
    y: int


def make_transform(cfg: dict, train: bool) -> Callable:
    from torchvision import transforms

    aug = cfg["augmentation"]
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    if train:
        return transforms.Compose([
            transforms.Resize((int(aug["resize_size"]), int(aug["resize_size"]))),
            transforms.RandomHorizontalFlip() if aug.get("horizontal_flip", True) else transforms.Lambda(lambda x: x),
            transforms.RandomVerticalFlip() if aug.get("vertical_flip", True) else transforms.Lambda(lambda x: x),
            transforms.RandomRotation(float(aug.get("rotation_degrees", 20))),
            transforms.ColorJitter(
                brightness=float(aug.get("color_jitter_brightness", 0.15)),
                contrast=float(aug.get("color_jitter_contrast", 0.15)),
                saturation=float(aug.get("color_jitter_saturation", 0.15)),
                hue=float(aug.get("color_jitter_hue", 0.03)),
            ),
            transforms.ToTensor(),
            normalize,
        ])
    return transforms.Compose([
        transforms.Resize((int(aug["resize_size"]), int(aug["resize_size"]))),
        transforms.ToTensor(),
        normalize,
    ])


def build_tile_index(
    metadata: pd.DataFrame,
    cfg: dict,
    max_tiles_per_image: int | None = None,
) -> list[TileRecord]:
    ds = cfg["dataset"]
    records: list[TileRecord] = []
    for row_idx, row in metadata.reset_index(drop=True).iterrows():
        path = Path(row["image_path"])
        try:
            with Image.open(path) as image:
                image = image.convert("RGB")
                seed = int(cfg["seed"]) + row_idx
                coords = select_tile_coordinates(
                    image=image,
                    tile_size=int(ds["tile_size"]),
                    stride=int(ds["stride"]),
                    tissue_threshold=float(ds["tissue_threshold"]),
                    background_gray_threshold=int(ds.get("background_gray_threshold", 245)),
                    max_tiles=max_tiles_per_image,
                    seed=seed,
                )
        except Exception as exc:
            raise RuntimeError(f"Failed to index image {path}: {exc}") from exc
        for coord in coords:
            records.append(TileRecord(
                image_path=str(path), patient_id=str(row["patient_id"]), label=int(row["label"]),
                tile_id=int(coord.tile_id), x=int(coord.x), y=int(coord.y),
            ))
    if not records:
        raise ValueError("Tiling produced zero valid tiles. Lower tissue_threshold or inspect the dataset.")
    return records


class TileDataset(Dataset):
    def __init__(self, records: list[TileRecord], cfg: dict, train: bool = False):
        self.records = records
        self.cfg = cfg
        self.transform = make_transform(cfg, train=train)
        self.tile_size = int(cfg["dataset"]["tile_size"])

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int):
        record = self.records[index]
        from .tiling import extract_tile
        try:
            with Image.open(record.image_path) as image:
                tile = extract_tile(image, record.x, record.y, self.tile_size)
        except Exception as exc:
            raise RuntimeError(f"Could not read tile from {record.image_path}: {exc}") from exc
        return {
            "image": self.transform(tile),
            "label": torch.tensor(float(record.label), dtype=torch.float32),
            "image_path": record.image_path,
            "patient_id": record.patient_id,
            "tile_id": record.tile_id,
            "x": record.x,
            "y": record.y,
        }
