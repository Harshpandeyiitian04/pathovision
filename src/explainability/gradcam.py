"""Dependency-free Grad-CAM implementation for ResNet18."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from PIL import Image


class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.activations = None
        self.gradients = None
        self._hooks = [
            target_layer.register_forward_hook(self._save_activation),
        ]

    def _save_activation(self, module, inputs, output):
        self.activations = output.detach()
        if output.requires_grad:
            output.register_hook(self._save_gradient_tensor)

    def _save_gradient_tensor(self, gradient):
        self.gradients = gradient.detach()

    def remove(self):
        for hook in self._hooks:
            hook.remove()

    def __call__(self, input_tensor: torch.Tensor) -> np.ndarray:
        self.model.zero_grad(set_to_none=True)
        self.model.eval()
        input_tensor = input_tensor.clone().detach().requires_grad_(True)
        logits = self.model(input_tensor)
        score = logits.reshape(-1)[0]
        score.backward()
        if self.activations is None or self.gradients is None:
            raise RuntimeError("Grad-CAM hooks did not capture activations/gradients")
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = (weights * self.activations).sum(dim=1).squeeze(0)
        cam = torch.relu(cam)
        cam -= cam.min()
        max_value = cam.max()
        if float(max_value) > 0:
            cam /= max_value
        return cam.cpu().numpy()


def overlay_heatmap(image: Image.Image, cam: np.ndarray, output_path: str | Path, alpha: float = 0.45) -> None:
    import matplotlib.pyplot as plt
    image_arr = np.asarray(image.convert("RGB"))
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.imshow(image_arr)
    ax.imshow(cam, cmap="jet", alpha=alpha, extent=(0, image_arr.shape[1], image_arr.shape[0], 0))
    ax.axis("off")
    fig.tight_layout(pad=0)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180, bbox_inches="tight", pad_inches=0)
    plt.close(fig)
