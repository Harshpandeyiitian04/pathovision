"""Training engine for PathoVision."""
from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import pandas as pd
import yaml
import torch
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.evaluation.metrics import compute_metrics, plot_training_history
from src.seed import worker_init_fn
from .validate import image_level_validation, run_tile_inference


def make_loader(dataset, cfg: dict, shuffle: bool) -> DataLoader:
    generator = torch.Generator()
    generator.manual_seed(int(cfg["seed"]))
    return DataLoader(
        dataset,
        batch_size=int(cfg["training"]["batch_size"]),
        shuffle=shuffle,
        num_workers=int(cfg["training"].get("num_workers", 0)),
        pin_memory=torch.cuda.is_available(),
        worker_init_fn=worker_init_fn,
        generator=generator,
        persistent_workers=False,
    )


def _pos_weight(train_labels) -> torch.Tensor:
    labels = pd.Series(train_labels).astype(int)
    positives = int((labels == 1).sum())
    negatives = int((labels == 0).sum())
    if positives == 0:
        raise ValueError("Training split contains no malignant images; cannot calculate pos_weight.")
    return torch.tensor([negatives / positives], dtype=torch.float32)


def _train_epoch(model, loader, optimizer, criterion, device, scaler, use_amp: bool):
    model.train()
    running_loss = 0.0
    seen = 0
    all_labels = []
    all_probs = []
    for batch in tqdm(loader, desc="Train", leave=False):
        images = batch["image"].to(device, non_blocking=True)
        labels = batch["label"].to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        if use_amp:
            with torch.amp.autocast("cuda"):
                logits = model(images).reshape(-1)
                loss = criterion(logits, labels)
        else:
            logits = model(images).reshape(-1)
            loss = criterion(logits, labels)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        n = len(labels)
        running_loss += float(loss.item()) * n
        seen += n
        all_labels.append(labels.detach().cpu())
        all_probs.append(torch.sigmoid(logits.detach()).cpu())
    train_metrics = compute_metrics(torch.cat(all_labels).numpy(), torch.cat(all_probs).numpy(), 0.5)
    return running_loss / max(seen, 1), train_metrics


def _set_phase(model, phase: int):
    if phase == 1:
        for name, param in model.named_parameters():
            param.requires_grad = name.startswith("fc.")
    else:
        for param in model.parameters():
            param.requires_grad = True


def train_model(model, train_dataset, val_dataset, cfg: dict, train_metadata: pd.DataFrame, device: torch.device) -> dict[str, Any]:
    train_loader = make_loader(train_dataset, cfg, shuffle=True)
    val_loader = make_loader(val_dataset, cfg, shuffle=False)
    use_amp = bool(cfg["training"].get("use_amp", True)) and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    pos_weight = _pos_weight(train_metadata["label"]) if cfg.get("loss", {}).get("use_pos_weight", True) else None
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight.to(device) if pos_weight is not None else None)

    history = {"epoch": [], "train_loss": [], "train_accuracy": [], "train_precision": [], "train_recall": [], "train_f1": [], "train_roc_auc": [], "val_loss": [], "val_accuracy": [], "val_precision": [], "val_recall": [], "val_f1": [], "val_roc_auc": []}
    best_score = float("-inf")
    best_state = None
    epochs = [(1, int(cfg["training"]["epochs_phase1"])), (2, int(cfg["training"]["epochs_phase2"]))]
    global_epoch = 0
    patience_count = 0

    checkpoint_dir = Path(cfg["paths"]["checkpoint_dir"])
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    for phase, phase_epochs in epochs:
        if phase_epochs <= 0:
            continue
        _set_phase(model, phase)
        lr = float(cfg["training"]["learning_rate"] if phase == 1 else cfg["training"]["backbone_learning_rate"])
        optimizer = AdamW([p for p in model.parameters() if p.requires_grad], lr=lr, weight_decay=float(cfg["training"]["weight_decay"]))
        scheduler = ReduceLROnPlateau(
            optimizer, mode="max", factor=float(cfg["training"]["scheduler_factor"]),
            patience=int(cfg["training"]["scheduler_patience"]), min_lr=float(cfg["training"]["min_lr"]),
        )
        for _ in range(phase_epochs):
            global_epoch += 1
            train_loss, train_metrics = _train_epoch(model, train_loader, optimizer, criterion, device, scaler, use_amp)
            val_tiles, val_loss = run_tile_inference(model, val_loader, device, criterion, desc=f"Validation P{phase}")
            val_result = image_level_validation(
                val_tiles, float(cfg["evaluation"]["threshold"]), cfg["evaluation"]["aggregation"], float(cfg["evaluation"]["top_k_fraction"])
            )
            metrics = {k: val_result[k] for k in ("accuracy", "precision", "recall", "f1", "roc_auc")}
            history["epoch"].append(global_epoch)
            history["train_loss"].append(train_loss)
            for k in ("accuracy", "precision", "recall", "f1", "roc_auc"):
                history[f"train_{k}"].append(train_metrics[k])
            history["val_loss"].append(val_loss)
            for k in ("accuracy", "precision", "recall", "f1", "roc_auc"):
                history[f"val_{k}"].append(metrics[k])
            score = metrics["roc_auc"] if metrics["roc_auc"] == metrics["roc_auc"] else metrics["f1"]
            scheduler.step(score)
            print(f"Epoch {global_epoch}: train_loss={train_loss:.4f} val_loss={val_loss:.4f} val_f1={metrics['f1']:.4f} val_auc={metrics['roc_auc']:.4f}")
            if score > best_score:
                best_score = score
                best_state = copy.deepcopy(model.state_dict())
                checkpoint = {
                    "model_state_dict": best_state,
                    "architecture": cfg["model"]["architecture"],
                    "threshold": float(cfg["evaluation"]["threshold"]),
                    "normalization": {"mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]},
                    "aggregation": cfg["evaluation"]["aggregation"],
                    "top_k_fraction": float(cfg["evaluation"]["top_k_fraction"]),
                    "config": cfg,
                    "best_validation_score": float(best_score),
                    "epoch": global_epoch,
                }
                torch.save(checkpoint, checkpoint_dir / "best_model.pth")
                (checkpoint_dir / "best_config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
                patience_count = 0
            else:
                patience_count += 1
                if patience_count >= int(cfg["training"]["early_stopping_patience"]):
                    print("Early stopping triggered.")
                    break
        if patience_count >= int(cfg["training"]["early_stopping_patience"]):
            break

    if best_state is None:
        raise RuntimeError("Training finished without a valid checkpoint")
    model.load_state_dict(best_state)
    (checkpoint_dir / "training_history.json").write_text(json.dumps(history, indent=2, allow_nan=True), encoding="utf-8")
    results_dir = Path(cfg["paths"]["results_dir"])
    plot_training_history(history, results_dir / "figures" / "loss_curves.png", "loss")
    plot_training_history(history, results_dir / "figures" / "f1_curves.png", "f1")
    plot_training_history(history, results_dir / "figures" / "roc_auc_curves.png", "roc_auc")
    return history
