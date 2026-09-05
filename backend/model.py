"""Inference wrapper used by the LandmarkLens service.

Two prediction backends are supported, selected with LANDMARKLENS_BACKEND:

  "torch" (default)  eager PyTorch, float32
  "onnx"             ONNX Runtime session over an exported .onnx file, which is
                     what the benchmark measured as the faster CPU path

The PyTorch model is loaded either way, because Grad-CAM needs gradients and so
cannot run against the ONNX graph. When the ONNX backend is active, the torch
model is used only for heatmaps -- the two share the same weights, so the
explanation still describes the deployed classifier.
"""
from __future__ import annotations

import os
import threading
from typing import List, Tuple

import numpy as np
import torch
from PIL import Image

from landmarklens.classes import BY_SLUG
from landmarklens.data import eval_transforms
from landmarklens.gradcam import HeatmapGenerator
from landmarklens.model import load_checkpoint

DEFAULT_CHECKPOINT = os.environ.get(
    "LANDMARKLENS_CHECKPOINT",
    os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                 "artifacts", "best_model.pt"),
)
DEFAULT_ONNX = os.environ.get("LANDMARKLENS_ONNX", "")


class Prediction:
    __slots__ = ("slug", "display", "location_name", "resolvable", "confidence")

    def __init__(self, slug: str, confidence: float):
        meta = BY_SLUG.get(slug)
        self.slug = slug
        self.display = meta.display if meta else slug
        self.location_name = meta.location_name if meta else slug
        self.resolvable = bool(meta.resolvable) if meta else False
        self.confidence = confidence

    def as_dict(self) -> dict:
        return {
            "class": self.slug,
            "label": self.display,
            "location_name": self.location_name,
            "resolvable": self.resolvable,
            "confidence": round(self.confidence, 4),
            "score": round(self.confidence, 4),  # legacy key for the React UI
        }


class ModelWrapper:
    def __init__(self, checkpoint: str = DEFAULT_CHECKPOINT,
                 onnx_path: str = DEFAULT_ONNX,
                 backend: str = os.environ.get("LANDMARKLENS_BACKEND", "torch")):
        if not os.path.exists(checkpoint):
            raise FileNotFoundError(
                f"checkpoint not found: {checkpoint}\n"
                "Train one first (scripts/train.py) or set "
                "LANDMARKLENS_CHECKPOINT."
            )
        self.model, self.classes, self.backbone, self.meta = \
            load_checkpoint(checkpoint)
        self.image_size = int(self.meta.get("image_size", 224))
        self.transform = eval_transforms(self.image_size)
        self.checkpoint = checkpoint

        self.backend = "torch"
        self.session = None
        if backend == "onnx" and onnx_path and os.path.exists(onnx_path):
            import onnxruntime as ort
            opts = ort.SessionOptions()
            opts.intra_op_num_threads = int(
                os.environ.get("LANDMARKLENS_THREADS", "4"))
            self.session = ort.InferenceSession(
                onnx_path, opts, providers=["CPUExecutionProvider"])
            self.input_name = self.session.get_inputs()[0].name
            self.backend = "onnx"
            self.onnx_path = onnx_path

        # Grad-CAM registers backward hooks on the shared model, and the
        # library is not re-entrant, so serialise heatmap generation.
        self._cam = None
        self._cam_lock = threading.Lock()

    # ------------------------------------------------------------------ #

    def _logits(self, tensor: torch.Tensor) -> np.ndarray:
        if self.session is not None:
            return self.session.run(
                None, {self.input_name: tensor.numpy()})[0]
        with torch.no_grad():
            return self.model(tensor).numpy()

    def predict_topk(self, image: Image.Image, k: int = 3
                     ) -> Tuple[List[Prediction], int]:
        """Returns (top-k predictions, index of the top class)."""
        tensor = self.transform(image.convert("RGB")).unsqueeze(0)
        logits = self._logits(tensor)[0]
        exp = np.exp(logits - logits.max())
        probs = exp / exp.sum()
        k = min(k, len(probs))
        order = np.argsort(-probs)[:k]
        preds = [Prediction(self.classes[i], float(probs[i])) for i in order]
        return preds, int(order[0])

    def heatmap_png(self, image: Image.Image, class_index: int) -> bytes:
        with self._cam_lock:
            if self._cam is None:
                self._cam = HeatmapGenerator(self.model, self.image_size)
            blob, _ = self._cam.overlay_bytes(image, class_index)
        return blob

    def info(self) -> dict:
        return {
            "backbone": self.backbone,
            "backend": self.backend,
            "onnx_path": getattr(self, "onnx_path", None),
            "checkpoint": os.path.basename(self.checkpoint),
            "image_size": self.image_size,
            "num_classes": len(self.classes),
            "classes": self.classes,
            "trained": {
                k: self.meta.get(k)
                for k in ("run_name", "best_epoch", "val_top1", "val_top3")
                if k in self.meta
            },
        }
