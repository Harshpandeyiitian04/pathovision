"""Classification metrics and plotting helpers."""
from __future__ import annotations

from pathlib import Path

import json
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score,
    roc_curve,
)


def compute_metrics(y_true, probabilities, threshold: float = 0.5) -> dict:
    y_true = np.asarray(y_true).astype(int)
    probabilities = np.asarray(probabilities, dtype=float)
    if len(y_true) != len(probabilities):
        raise ValueError("y_true and probabilities must have the same length")
    if len(y_true) == 0:
        raise ValueError("Cannot compute metrics on an empty dataset")
    y_pred = (probabilities >= threshold).astype(int)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    specificity = float(tn / (tn + fp)) if (tn + fp) else float("nan")
    auc = float(roc_auc_score(y_true, probabilities)) if len(np.unique(y_true)) == 2 else float("nan")
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": auc,
        "specificity": specificity,
        "sensitivity": float(recall_score(y_true, y_pred, zero_division=0)),
        "support": int(len(y_true)),
        "positive_support": int(y_true.sum()),
        "negative_support": int((y_true == 0).sum()),
        "threshold": float(threshold),
        "confusion_matrix": [[int(v) for v in row] for row in cm.tolist()],
    }


def metrics_by_group(df: pd.DataFrame, group_col: str, threshold: float = 0.5) -> pd.DataFrame:
    rows = []
    for value, group in df.groupby(group_col, dropna=False):
        metrics = compute_metrics(group["label"], group["probability_malignant"], threshold)
        rows.append({group_col: value, **{k: metrics[k] for k in ("accuracy", "precision", "recall", "f1", "roc_auc")}, "sample_count": len(group)})
    return pd.DataFrame(rows)


def save_json(data: dict, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, allow_nan=True), encoding="utf-8")


def plot_confusion_matrix(metrics: dict, path: str | Path) -> None:
    import matplotlib.pyplot as plt
    cm = np.asarray(metrics["confusion_matrix"])
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm)
    ax.set_xticks([0, 1], labels=["Benign", "Malignant"])
    ax.set_yticks([0, 1], labels=["Benign", "Malignant"])
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("Confusion Matrix")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, int(cm[i, j]), ha="center", va="center")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_roc(y_true, probabilities, path: str | Path) -> None:
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(5, 4))
    if len(np.unique(y_true)) == 2:
        fpr, tpr, _ = roc_curve(y_true, probabilities)
        auc = roc_auc_score(y_true, probabilities)
        ax.plot(fpr, tpr, label=f"ROC-AUC = {auc:.3f}")
        ax.legend(loc="lower right")
    else:
        ax.text(0.5, 0.5, "ROC curve unavailable: only one class present", ha="center", va="center")
    ax.plot([0, 1], [0, 1], linestyle="--")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve")
    fig.tight_layout()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_training_history(history: dict, path: str | Path, metric: str) -> None:
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(history.get("epoch", []), history.get(f"train_{metric}", []), label=f"Train {metric}")
    ax.plot(history.get("epoch", []), history.get(f"val_{metric}", []), label=f"Val {metric}")
    ax.set_xlabel("Epoch")
    ax.set_ylabel(metric.upper())
    ax.set_title(f"Training vs Validation {metric.upper()}")
    ax.legend()
    fig.tight_layout()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
