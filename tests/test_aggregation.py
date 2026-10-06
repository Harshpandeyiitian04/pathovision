import pandas as pd
import pytest

from src.evaluation.aggregation import aggregate_probabilities, aggregate_tile_dataframe


def test_aggregation_values():
    values = [0.1, 0.2, 0.9, 0.8]
    assert aggregate_probabilities(values, "mean") == pytest.approx(0.5)
    assert aggregate_probabilities(values, "max") == pytest.approx(0.9)
    assert aggregate_probabilities(values, "top_k", 0.5) == pytest.approx(0.85)


def test_empty_aggregation_raises():
    with pytest.raises(ValueError):
        aggregate_probabilities([], "mean")


def test_dataframe_aggregation():
    df = pd.DataFrame({"image_path": ["a"] * 3 + ["b"] * 2, "patient_id": ["p1"] * 3 + ["p2"] * 2, "label": [0, 0, 0, 1, 1], "probability_malignant": [0.1, 0.2, 0.3, 0.7, 0.9]})
    out = aggregate_tile_dataframe(df, "max")
    assert len(out) == 2
    assert out.loc[out.image_path == "a", "probability_malignant"].iloc[0] == pytest.approx(0.3)
    assert out.loc[out.image_path == "b", "probability_malignant"].iloc[0] == pytest.approx(0.9)
