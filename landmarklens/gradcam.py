"""Grad-CAM heatmap generation.

Wraps `pytorch-grad-cam` so both the CLI example generator and the Flask
service produce identical overlays. Grad-CAM hooks the last residual block
(layer4[-1]), whose 7x7 activation map is the finest spatial resolution
available before global pooling collapses it.

The overlay is produced at the *model's* view of the image -- the 224x224
centre crop -- so what you see is exactly what the network scored, not a
resampled approximation of it.
"""
from __future__ import annotations

import io
from typing import Optional, Tuple

import numpy as np
import torch
from PIL import Image
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

from .data import IMAGENET_MEAN, IMAGENET_STD, IMAGE_SIZE, eval_transforms
from .model import target_layer


def _display_crop(image: Image.Image, size: int = IMAGE_SIZE) -> np.ndarray:
    """The same resize+centre-crop the eval transform applies, as float RGB."""
    from torchvision import transforms

    pipeline = transforms.Compose([
        transforms.Resize(int(size * 1.14)),
        transforms.CenterCrop(size),
    ])
    return np.asarray(pipeline(image.convert("RGB")), dtype=np.float32) / 255.0


class HeatmapGenerator:
    """Reusable Grad-CAM generator bound to one model."""

    def __init__(self, model: torch.nn.Module, size: int = IMAGE_SIZE):
        self.model = model
        self.size = size
        self.transform = eval_transforms(size)
        self.cam = GradCAM(model=model, target_layers=[target_layer(model)])

    def overlay(self, image: Image.Image, class_index: Optional[int] = None,
                alpha: float = 0.5) -> Tuple[Image.Image, int]:
        """Return (overlay image, class index the heatmap explains).

        `class_index=None` explains the model's own top prediction.
        """
        tensor = self.transform(image.convert("RGB")).unsqueeze(0)

        if class_index is None:
            with torch.no_grad():
                class_index = int(self.model(tensor).argmax(dim=1).item())

        grayscale = self.cam(
            input_tensor=tensor,
            targets=[ClassifierOutputTarget(class_index)],
        )[0]

        rgb = _display_crop(image, self.size)
        blended = show_cam_on_image(rgb, grayscale, use_rgb=True,
                                    image_weight=1.0 - alpha)
        return Image.fromarray(blended), class_index

    def overlay_bytes(self, image: Image.Image,
                      class_index: Optional[int] = None,
                      alpha: float = 0.5) -> Tuple[bytes, int]:
        img, idx = self.overlay(image, class_index, alpha)
        buf = io.BytesIO()
        img.save(buf, format="PNG", optimize=True)
        return buf.getvalue(), idx


__all__ = ["HeatmapGenerator", "IMAGENET_MEAN", "IMAGENET_STD"]
