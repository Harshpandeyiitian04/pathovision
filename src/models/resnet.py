"""ResNet18 transfer-learning classifier."""
from __future__ import annotations

import torch
from torch import nn
from torchvision.models import ResNet18_Weights, resnet18


def build_model(cfg: dict) -> nn.Module:
    model_cfg = cfg["model"]
    pretrained = bool(model_cfg.get("pretrained", True))
    try:
        weights = ResNet18_Weights.DEFAULT if pretrained else None
        model = resnet18(weights=weights)
    except Exception as exc:
        if pretrained:
            raise RuntimeError(
                "Could not load ImageNet ResNet18 weights. If internet is unavailable, set model.pretrained=false."
            ) from exc
        raise
    in_features = model.fc.in_features
    model.fc = nn.Sequential(nn.Dropout(float(model_cfg.get("dropout", 0.3))), nn.Linear(in_features, 1))
    if bool(model_cfg.get("freeze_backbone", True)):
        for name, param in model.named_parameters():
            if not name.startswith("fc."):
                param.requires_grad = False
    return model


def unfreeze_backbone(model: nn.Module) -> None:
    for param in model.parameters():
        param.requires_grad = True


def model_output_shape(model: nn.Module, device: torch.device) -> tuple[int, ...]:
    x = torch.randn(2, 3, 224, 224, device=device)
    with torch.no_grad():
        return tuple(model(x).shape)
