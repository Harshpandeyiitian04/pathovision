"""Robust BreakHis metadata discovery and validation."""
from __future__ import annotations

import re
from pathlib import Path
from typing import Iterable

import pandas as pd
from PIL import Image

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}
MAGNIFICATIONS = {"40X": 40, "100X": 100, "200X": 200, "400X": 400}

_FILENAME_RE = re.compile(
    r"^(?P<patient>.+?)-(?P<magnification>40|100|200|400)-(?P<index>\d+)\.(?P<ext>png|jpg|jpeg|tif|tiff|bmp)$",
    re.IGNORECASE,
)


def _class_from_parts(parts: Iterable[str]) -> tuple[str, int]:
    lower = {p.lower() for p in parts}
    if "benign" in lower:
        return "Benign", 0
    if "malignant" in lower:
        return "Malignant", 1
    raise ValueError("Could not infer Benign/Malignant class from path")


def _subtype_from_parts(path: Path, root: Path, class_name: str) -> str:
    rel_parts = path.relative_to(root).parts[:-1]
    lowered = [p.lower() for p in rel_parts]
    try:
        idx = lowered.index(class_name.lower())
        candidates = rel_parts[idx + 1 :]
        for candidate in candidates:
            if candidate.lower() not in {"40x", "100x", "200x", "400x"}:
                return candidate
    except ValueError:
        pass
    return "unknown"


def parse_image_record(path: Path, root: Path) -> dict:
    class_name, label = _class_from_parts(path.parts)
    match = _FILENAME_RE.match(path.name)
    if match:
        patient_id = match.group("patient")
        magnification = int(match.group("magnification"))
    else:
        # For non-standard filenames, read magnification and patient ID from the path.
        magnification = None
        for part in reversed(path.relative_to(root).parts[:-1]):
            m = re.fullmatch(r"(40|100|200|400)X?", part, flags=re.I)
            if m:
                magnification = int(m.group(1))
                break
        if magnification is None:
            raise ValueError(f"Cannot infer magnification from filename/path: {path}")
        # Use the filename portion before the magnification when available.
        stem_parts = re.split(r"[-_]", path.stem)
        mag_idx = next((i for i, x in enumerate(stem_parts) if x.lower().rstrip("x") in {"40", "100", "200", "400"}), None)
        patient_id = "-".join(stem_parts[:mag_idx]) if mag_idx is not None and mag_idx > 0 else path.parent.name
    subtype = _subtype_from_parts(path, root, class_name)
    return {
        "image_path": str(path.resolve()),
        "patient_id": str(patient_id),
        "label": int(label),
        "class_name": class_name,
        "subtype": subtype,
        "magnification": int(magnification),
    }


def discover_images(root: str | Path) -> list[Path]:
    root = Path(root)
    if not root.exists():
        raise FileNotFoundError(
            f"BreakHis dataset root does not exist: {root}. Place the downloaded dataset under data/raw/."
        )
    paths = sorted(p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS)
    if not paths:
        raise FileNotFoundError(f"No image files found under {root}.")
    return paths


def build_metadata(root: str | Path) -> pd.DataFrame:
    root = Path(root).resolve()
    records = []
    errors = []
    for path in discover_images(root):
        try:
            records.append(parse_image_record(path, root))
        except (ValueError, OSError) as exc:
            errors.append(f"{path}: {exc}")
    if errors:
        preview = "\n".join(errors[:10])
        raise ValueError(f"Failed to parse {len(errors)} images. First errors:\n{preview}")
    df = pd.DataFrame(records).drop_duplicates(subset=["image_path"]).sort_values("image_path").reset_index(drop=True)
    validate_metadata(df)
    return df


def validate_metadata(df: pd.DataFrame) -> None:
    required = {"image_path", "patient_id", "label", "class_name", "subtype", "magnification"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Metadata missing columns: {sorted(missing)}")
    if df.empty:
        raise ValueError("Metadata is empty")
    if df["patient_id"].isna().any() or df["label"].isna().any():
        raise ValueError("Metadata contains missing patient IDs or labels")
    if not set(df["label"].unique()).issubset({0, 1}):
        raise ValueError("Labels must be 0/1")
    missing_paths = [p for p in df["image_path"] if not Path(p).exists()]
    if missing_paths:
        raise FileNotFoundError(f"Metadata references {len(missing_paths)} missing image paths")


def validate_images(df: pd.DataFrame, sample_limit: int | None = None) -> None:
    rows = df if sample_limit is None else df.head(sample_limit)
    bad = []
    for path in rows["image_path"]:
        try:
            with Image.open(path) as img:
                img.verify()
        except Exception as exc:  # Pillow can raise different errors for damaged images.
            bad.append(f"{path}: {exc}")
    if bad:
        raise ValueError("Invalid image files detected:\n" + "\n".join(bad[:10]))
