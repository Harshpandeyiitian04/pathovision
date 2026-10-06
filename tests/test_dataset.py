from pathlib import Path

import pandas as pd
import numpy as np
from PIL import Image
import pytest

from src.data.parse_breakhis import parse_image_record
from src.data.patient_split import assert_no_patient_leakage, split_by_patient


def test_breakhis_label_patient_and_magnification(tmp_path):
    root = tmp_path / "breakhis"
    path = root / "SOB" / "benign" / "adenosis" / "SOB_B_A" / "40X" / "SOB_B_A-14-22549AB-40-001.png"
    path.parent.mkdir(parents=True)
    Image.fromarray(np.zeros((20, 20, 3), dtype=np.uint8)).save(path)
    record = parse_image_record(path, root)
    assert record["label"] == 0
    assert record["class_name"] == "Benign"
    assert record["magnification"] == 40
    assert record["patient_id"] == "SOB_B_A-14-22549AB"


def test_patient_split_has_no_leakage():
    rows = []
    for i in range(20):
        label = i % 2
        for j in range(2):
            rows.append({"image_path": f"p{i}_{j}.png", "patient_id": f"p{i}", "label": label, "class_name": "Benign" if label == 0 else "Malignant", "subtype": "x", "magnification": 40})
    df = pd.DataFrame(rows)
    train, val, test = split_by_patient(df, test_size=0.2, val_size=0.2, random_state=42)
    assert not set(train.patient_id) & set(val.patient_id)
    assert not set(train.patient_id) & set(test.patient_id)
    assert not set(val.patient_id) & set(test.patient_id)


def test_leakage_assertion_raises():
    with pytest.raises(AssertionError):
        assert_no_patient_leakage({"a"}, {"a"}, {"b"})
