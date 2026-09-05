"""Model construction and checkpoint I/O for LandmarkLens.

Transfer learning from an ImageNet-pretrained backbone. With ~100 images per
class the whole network cannot be trained from scratch, so training runs in two
phases (see scripts/train.py):

  1. warmup  -- backbone frozen, only the new classifier head trains
  2. finetune -- the last residual stage (layer4) is unfrozen at a lower LR

Freezing layer1-layer3 keeps the generic edge/texture filters intact, which is
what makes a dataset this small workable at all, and it roughly halves CPU
training time since no gradients flow through the early (highest-resolution)
convolutions.
"""
from __future__ import annotations

import json
import os
from typing import Dict, List

import torch
import torch.nn as nn
from torchvision import models

BACKBONES = {
    "resnet50": (models.resnet50, models.ResNet50_Weights.IMAGENET1K_V2),
    "resnet34": (models.resnet34, models.ResNet34_Weights.IMAGENET1K_V1),
    "resnet18": (models.resnet18, models.ResNet18_Weights.IMAGENET1K_V1),
}


def build_model(num_classes: int, backbone: str = "resnet50",
                pretrained: bool = True, dropout: float = 0.2) -> nn.Module:
    if backbone not in BACKBONES:
        raise ValueError(f"unknown backbone {backbone!r}; "
                         f"choose from {sorted(BACKBONES)}")
    ctor, weights = BACKBONES[backbone]
    model = ctor(weights=weights if pretrained else None)
    in_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(dropout),
        nn.Linear(in_features, num_classes),
    )
    return model


def set_trainable(model: nn.Module, mode: str) -> List[nn.Parameter]:
    """Freeze/unfreeze according to the training phase.

    mode "head"     -> only model.fc trains
    mode "layer4"   -> model.fc + the final residual stage train
    mode "all"      -> everything trains
    Returns the list of parameters that require grad.
    """
    for p in model.parameters():
        p.requires_grad = False

    if mode == "all":
        for p in model.parameters():
            p.requires_grad = True
    elif mode == "layer4":
        for p in model.layer4.parameters():
            p.requires_grad = True
        for p in model.fc.parameters():
            p.requires_grad = True
    elif mode == "head":
        for p in model.fc.parameters():
            p.requires_grad = True
    else:
        raise ValueError(f"unknown trainable mode {mode!r}")

    return [p for p in model.parameters() if p.requires_grad]


def target_layer(model: nn.Module):
    """Layer Grad-CAM hooks: the last conv block before global pooling."""
    return model.layer4[-1]


def save_checkpoint(path: str, model: nn.Module, classes: List[str],
                    backbone: str, meta: Dict | None = None) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    torch.save({
        "state_dict": model.state_dict(),
        "classes": classes,
        "backbone": backbone,
        "meta": meta or {},
    }, path)
    # Sidecar JSON so the serving container can read labels without torch.load
    with open(os.path.splitext(path)[0] + "_classes.json", "w",
              encoding="utf-8") as fh:
        json.dump({"classes": classes, "backbone": backbone,
                   "meta": meta or {}}, fh, indent=2)


def load_checkpoint(path: str, map_location: str = "cpu"):
    """Returns (model_in_eval_mode, classes, backbone, meta)."""
    ckpt = torch.load(path, map_location=map_location, weights_only=False)
    classes = ckpt["classes"]
    backbone = ckpt.get("backbone", "resnet50")
    model = build_model(len(classes), backbone=backbone, pretrained=False)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model, classes, backbone, ckpt.get("meta", {})
