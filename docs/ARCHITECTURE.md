# LandmarkLens architecture

## Pipeline

```
Wikimedia Commons API
        │  scripts/harvest_dataset.py   (license filter + attribution record)
        ▼
   data/raw/<class>/                    16 classes
        │  scripts/split_dataset.py     (stratified, seed 42, hard links)
        ▼
   data/splits/{train,val,test}/<class>/
        │  scripts/train.py             (two-phase transfer learning)
        │      ├─ phase 1: backbone frozen, new head trains
        │      └─ phase 2: layer4 unfrozen, cosine-annealed LR
        │  → MLflow (./mlruns): params, per-epoch metrics, checkpoint artifact
        ▼
   artifacts/<run>_best.pt              best epoch by val top-1
        │
        ├─ scripts/evaluate.py          held-out test top-1/top-3, per-class F1,
        │                               confusion pairs
        ├─ scripts/make_gradcam_examples.py
        │                               → docs/gradcam/*.png
        ├─ scripts/export_onnx.py       → FP32 ONNX
        │                                 INT8 dynamic ONNX  (MatMul/Gemm)
        │                                 INT8 static QDQ ONNX (calibrated)
        │                                 INT8 dynamic PyTorch
        └─ scripts/benchmark.py         latency + accuracy for every variant,
                                        same process, same thread count
        ▼
   backend/app.py                       Flask REST API (/predict, /heatmap/…)
        ├─ frontend/                    React SPA behind nginx
        └─ TravelAssistant /chat_photo  photo → location entity → journey
```

## Module responsibilities

| Path | Responsibility |
|---|---|
| `landmarklens/classes.py` | The 16-class registry. Each class carries the `location_name` it maps to and whether that name is *resolvable* (a specific place) or generic (a kind of place). |
| `landmarklens/commons.py` | Throttled Wikimedia Commons client. Enforces the license allow-list and records attribution. |
| `landmarklens/data.py` | Transforms and loaders. Three augmentation strengths so the sweep can ablate augmentation. Inverse-frequency class weights for the uneven Commons yield. |
| `landmarklens/model.py` | Backbone construction, phase freezing (`head` / `layer4` / `all`), checkpoint I/O with a JSON sidecar so serving can read labels without `torch.load`. |
| `landmarklens/gradcam.py` | One `HeatmapGenerator` shared by the CLI example generator and the API, so the README figures and the served overlays are produced by identical code. |
| `backend/model.py` | Serving wrapper. Prediction runs on PyTorch or ONNX Runtime (`LANDMARKLENS_BACKEND`); Grad-CAM always runs on the PyTorch weights because it needs gradients. |
| `backend/app.py` | REST layer. Heatmaps are held in a bounded in-memory LRU and fetched once by the client. |

## Two design decisions worth explaining

**Why freeze most of the backbone.** With roughly 100 images per class there is
not enough signal to move 25 million parameters without overfitting. Phase 1
trains only the new head so its random initialisation stops producing large,
destructive gradients; phase 2 unfreezes just `layer4`, the most
task-specific stage. Layers 1-3 keep their ImageNet edge and texture filters.
This also roughly halves CPU training time, because no gradients flow through
the early high-resolution convolutions.

**Why `resolvable` exists.** Six of the sixteen classes — a roundel, a bus stop
flag, a double-decker bus, a black cab, a red telephone box — identify a *kind*
of place. A photo of a bus stop cannot tell you which bus stop. Returning
"London bus stop" as a routable destination would be a confidently wrong
answer, so the class registry marks these `resolvable: false` and the consumer
is expected to ask the user rather than plan a journey. Getting this wrong is
the difference between a demo and something usable.
