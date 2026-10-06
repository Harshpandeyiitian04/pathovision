from PIL import Image
import numpy as np

from src.data.tiling import extract_tile, select_tile_coordinates, tile_coordinates, tissue_ratio


def test_tile_dimensions_and_edge_padding():
    image = Image.fromarray(np.full((300, 250, 3), 120, dtype=np.uint8))
    coords = tile_coordinates(250, 300, 224, 168)
    assert len(coords) == 4
    for c in coords:
        tile = extract_tile(image, c.x, c.y, 224)
        assert tile.size == (224, 224)


def test_tissue_filter_rejects_white_background():
    white = Image.fromarray(np.full((224, 224, 3), 255, dtype=np.uint8))
    tissue = Image.fromarray(np.full((224, 224, 3), 80, dtype=np.uint8))
    assert tissue_ratio(white) == 0.0
    assert tissue_ratio(tissue) == 1.0
    assert select_tile_coordinates(white, 224, 168, 0.2) == []
    assert len(select_tile_coordinates(tissue, 224, 168, 0.2)) == 1
