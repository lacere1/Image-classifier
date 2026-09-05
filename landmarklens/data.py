"""Datasets and augmentation for LandmarkLens.

The augmentation pipeline targets the deployment condition: a phone photo taken
on a London street. That means arbitrary framing and distance (RandomResizedCrop
with an aggressive scale range), varying weather and time of day (ColorJitter),
handheld tilt (small rotation), and occasional occlusion by traffic, people or
scaffolding (RandomErasing).

Horizontal flip is safe for buildings and street furniture here -- none of the
16 classes is distinguished by chirality. Text on signage does flip, which is
mildly unrealistic, but the model is learning shape and colour layout rather
than reading the text, and flipping measurably helped the roundel/bus-stop
classes in early runs.
"""
from __future__ import annotations

from typing import Tuple

import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
IMAGE_SIZE = 224


def train_transforms(size: int = IMAGE_SIZE, strength: str = "standard"):
    """Augmentation for training.

    strength:
        "light"    - resize + flip only (baseline / ablation run)
        "standard" - random resized crop, flip, colour jitter
        "strong"   - standard + rotation, grayscale, random erasing
    """
    if strength == "light":
        return transforms.Compose([
            transforms.Resize((size, size)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])

    ops = [
        transforms.RandomResizedCrop(size, scale=(0.6, 1.0), ratio=(0.75, 1.33)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.3, contrast=0.3,
                               saturation=0.3, hue=0.05),
    ]
    if strength == "strong":
        ops.insert(1, transforms.RandomRotation(12))
        ops.append(transforms.RandomGrayscale(p=0.05))

    ops += [transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)]

    if strength == "strong":
        ops.append(transforms.RandomErasing(p=0.25, scale=(0.02, 0.15)))

    return transforms.Compose(ops)


def eval_transforms(size: int = IMAGE_SIZE):
    """Deterministic pipeline used for val, test, export and serving."""
    return transforms.Compose([
        transforms.Resize(int(size * 1.14)),
        transforms.CenterCrop(size),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


def build_loaders(splits_dir: str, batch_size: int = 32, workers: int = 4,
                  aug: str = "standard", size: int = IMAGE_SIZE
                  ) -> Tuple[DataLoader, DataLoader, DataLoader, list[str]]:
    import os

    train_ds = ImageFolder(os.path.join(splits_dir, "train"),
                           transform=train_transforms(size, aug))
    val_ds = ImageFolder(os.path.join(splits_dir, "val"),
                         transform=eval_transforms(size))
    test_ds = ImageFolder(os.path.join(splits_dir, "test"),
                          transform=eval_transforms(size))

    common = dict(num_workers=workers, pin_memory=False,
                  persistent_workers=workers > 0)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              drop_last=False, **common)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, **common)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, **common)
    return train_loader, val_loader, test_loader, train_ds.classes


def class_weights(splits_dir: str, classes: list[str]) -> torch.Tensor:
    """Inverse-frequency weights, normalised to mean 1.

    Commons yields uneven counts per class (a bus stop pole has far fewer good
    photos than Tower Bridge), so the loss is reweighted rather than the data
    resampled.
    """
    import os

    counts = []
    for c in classes:
        d = os.path.join(splits_dir, "train", c)
        counts.append(len(os.listdir(d)) if os.path.isdir(d) else 0)
    t = torch.tensor(counts, dtype=torch.float32).clamp(min=1)
    w = t.sum() / (len(t) * t)
    return w / w.mean()
