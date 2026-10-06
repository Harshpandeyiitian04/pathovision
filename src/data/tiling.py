"""Spatial tiling and explainable tissue filtering for histopathology images."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw


@dataclass(frozen=True)
class TileCoordinate:
    tile_id: int
    x: int
    y: int


def _positions(length: int, tile_size: int, stride: int) -> list[int]:
    if length <= tile_size:
        return [0]
    positions = list(range(0, length - tile_size + 1, stride))
    last = length - tile_size
    if positions[-1] != last:
        positions.append(last)
    return positions


def tile_coordinates(width: int, height: int, tile_size: int = 224, stride: int = 168) -> list[TileCoordinate]:
    if width <= 0 or height <= 0 or tile_size <= 0 or stride <= 0:
        raise ValueError("width, height, tile_size and stride must be positive")
    if stride > tile_size:
        raise ValueError("stride must be <= tile_size")
    coords = []
    tile_id = 0
    for y in _positions(height, tile_size, stride):
        for x in _positions(width, tile_size, stride):
            coords.append(TileCoordinate(tile_id=tile_id, x=x, y=y))
            tile_id += 1
    return coords


def extract_tile(image: Image.Image, x: int, y: int, tile_size: int = 224) -> Image.Image:
    """Extract a tile and pad right/bottom edges to exactly tile_size."""
    if tile_size <= 0:
        raise ValueError("tile_size must be positive")
    rgb = image.convert("RGB")
    arr = np.asarray(rgb)
    h, w = arr.shape[:2]
    if x < 0 or y < 0 or x >= max(w, 1) or y >= max(h, 1):
        raise ValueError(f"Tile origin ({x},{y}) is outside image dimensions {w}x{h}")
    crop = arr[y : min(y + tile_size, h), x : min(x + tile_size, w)]
    pad_h = tile_size - crop.shape[0]
    pad_w = tile_size - crop.shape[1]
    if pad_h or pad_w:
        crop = np.pad(crop, ((0, pad_h), (0, pad_w), (0, 0)), mode="edge")
    return Image.fromarray(crop.astype(np.uint8), mode="RGB")


def tissue_ratio(
    tile: Image.Image,
    background_gray_threshold: int = 245,
    saturation_threshold: int = 8,
) -> float:
    """Estimate tissue content as pixels that are dark OR sufficiently saturated."""
    rgb = np.asarray(tile.convert("RGB"), dtype=np.uint8)
    gray = (0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2])
    mx = rgb.max(axis=-1).astype(np.int16)
    mn = rgb.min(axis=-1).astype(np.int16)
    saturation = mx - mn
    tissue = (gray < background_gray_threshold) | (saturation >= saturation_threshold)
    return float(tissue.mean())


def select_tile_coordinates(
    image: Image.Image,
    tile_size: int,
    stride: int,
    tissue_threshold: float,
    background_gray_threshold: int = 245,
    max_tiles: int | None = None,
    seed: int = 42,
) -> list[TileCoordinate]:
    """Return coordinates passing the tissue filter, optionally capped deterministically."""
    all_coords = tile_coordinates(image.width, image.height, tile_size, stride)
    valid = []
    for coord in all_coords:
        tile = extract_tile(image, coord.x, coord.y, tile_size)
        if tissue_ratio(tile, background_gray_threshold) >= tissue_threshold:
            valid.append(coord)
    if max_tiles is not None and max_tiles > 0 and len(valid) > max_tiles:
        rng = np.random.default_rng(seed)
        indices = np.sort(rng.choice(len(valid), size=max_tiles, replace=False))
        valid = [valid[i] for i in indices]
    return valid


def draw_tile_grid(image: Image.Image, coords: list[TileCoordinate], tile_size: int = 224, line_width: int = 2) -> Image.Image:
    """Draw selected tile rectangles on a copy of the original image."""
    out = image.convert("RGB").copy()
    draw = ImageDraw.Draw(out)
    for coord in coords:
        draw.rectangle((coord.x, coord.y, coord.x + tile_size - 1, coord.y + tile_size - 1), width=line_width)
    return out
