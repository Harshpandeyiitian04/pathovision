import torch

from src.models.resnet import build_model


def config():
    return {"model": {"architecture": "resnet18", "pretrained": False, "freeze_backbone": True, "dropout": 0.3}}


def test_model_output_shape():
    model = build_model(config())
    x = torch.randn(2, 3, 224, 224)
    y = model(x)
    assert y.shape == (2, 1)


def test_bce_logits_compatible():
    model = build_model(config())
    x = torch.randn(2, 3, 224, 224)
    logits = model(x).reshape(-1)
    labels = torch.tensor([0.0, 1.0])
    loss = torch.nn.BCEWithLogitsLoss()(logits, labels)
    assert torch.isfinite(loss)


def test_fast_mode_overrides(tmp_path):
    from src.config import load_config
    p = tmp_path / "cfg.yaml"
    p.write_text("""seed: 1
dataset:
  tile_size: 224
  stride: 168
  tissue_threshold: 0.2
split:
  test_size: 0.2
  val_size: 0.2
model:
  architecture: resnet18
training:
  mode: fast
  epochs_phase1: 10
  epochs_phase2: 10
  max_train_tiles_per_image: 20
evaluation:
  threshold: 0.5
  aggregation: top_k
fast:
  max_train_images: 10
  max_val_images: 5
  max_test_images: 5
  epochs_phase1: 1
  epochs_phase2: 2
  max_train_tiles_per_image: 3
""")
    cfg = load_config(p)
    assert cfg["dataset"]["max_train_images"] == 10
    assert cfg["training"]["epochs_phase1"] == 1
    assert cfg["training"]["max_train_tiles_per_image"] == 3
