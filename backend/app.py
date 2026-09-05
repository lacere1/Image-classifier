"""LandmarkLens REST API.

Endpoints
    GET  /healthz          liveness + which model is loaded
    GET  /classes          the 16 classes and the location each maps to
    POST /predict          image upload -> location_name / confidence / heatmap
    GET  /heatmap/<id>.png the Grad-CAM overlay produced by a /predict call

/predict accepts a multipart upload under the field name "image" or "file"
(the bundled React frontend historically posted "file"), or a raw image body.

Response shape -- the three documented keys plus detail that consumers can
ignore:

    {
      "location_name": "Tower Bridge",
      "confidence": 0.94,
      "heatmap_url": "/heatmap/3f2c....png",
      "class": "tower_bridge",
      "label": "Tower Bridge",
      "resolvable": true,
      "low_confidence": false,
      "predictions": [ {...}, {...}, {...} ]
    }

`resolvable` is the field integrators must respect: classes like a bus stop
flag or a tube roundel identify a *kind* of place, not a specific one, so a
downstream journey planner has to ask the user which one rather than routing to
a guess.
"""
from __future__ import annotations

import io
import os
import sys
import time
import uuid
from collections import OrderedDict
from threading import Lock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from flask import Flask, Response, jsonify, request
from PIL import Image, ImageOps

from model import ModelWrapper  # noqa: E402  (same directory)

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024  # 16 MB uploads

try:
    from flask_cors import CORS
    CORS(app)
except Exception:  # CORS is unnecessary behind the bundled nginx proxy
    pass

CONFIDENCE_FLOOR = float(os.environ.get("LANDMARKLENS_MIN_CONFIDENCE", "0.45"))

model = ModelWrapper()
print(f"[landmarklens] loaded {model.info()}", flush=True)

# Heatmaps are held in a small bounded in-memory cache rather than written to
# disk: they are single-use artifacts fetched once by the client immediately
# after the prediction that produced them.
_HEATMAP_CACHE: "OrderedDict[str, bytes]" = OrderedDict()
_HEATMAP_MAX = int(os.environ.get("LANDMARKLENS_HEATMAP_CACHE", "64"))
_cache_lock = Lock()


def _store_heatmap(blob: bytes) -> str:
    key = uuid.uuid4().hex
    with _cache_lock:
        _HEATMAP_CACHE[key] = blob
        while len(_HEATMAP_CACHE) > _HEATMAP_MAX:
            _HEATMAP_CACHE.popitem(last=False)
    return key


def _read_upload() -> Image.Image:
    """Pull an image out of the request, whatever field name was used."""
    fs = request.files.get("image") or request.files.get("file")
    if fs is not None and fs.filename:
        raw = fs.read()
    elif request.data:
        raw = request.data
    else:
        raise ValueError("no image supplied (expected multipart field "
                         "'image' or 'file', or a raw image body)")
    image = Image.open(io.BytesIO(raw))
    image = ImageOps.exif_transpose(image)  # honour phone camera orientation
    return image.convert("RGB")


@app.get("/healthz")
def healthz():
    return jsonify({"status": "ok", **model.info()})


@app.get("/classes")
def classes():
    from landmarklens.classes import CLASSES
    return jsonify({
        "count": len(CLASSES),
        "classes": [{
            "class": c.slug, "label": c.display,
            "location_name": c.location_name, "resolvable": c.resolvable,
        } for c in CLASSES],
    })


@app.post("/predict")
def predict():
    t0 = time.perf_counter()
    try:
        image = _read_upload()
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    except Exception as exc:
        return jsonify({"error": f"invalid image: {exc}"}), 400

    want_heatmap = request.args.get("heatmap", "true").lower() not in (
        "0", "false", "no")

    try:
        preds, top_index = model.predict_topk(image, k=3)
    except Exception as exc:
        return jsonify({"error": f"inference failed: {exc}"}), 500

    infer_ms = (time.perf_counter() - t0) * 1000.0

    heatmap_url = None
    if want_heatmap:
        try:
            key = _store_heatmap(model.heatmap_png(image, top_index))
            heatmap_url = f"/heatmap/{key}.png"
        except Exception as exc:  # a failed overlay must not fail the request
            print(f"[landmarklens] heatmap failed: {exc}", flush=True)

    top = preds[0]
    return jsonify({
        "location_name": top.location_name,
        "confidence": round(top.confidence, 4),
        "heatmap_url": heatmap_url,
        "class": top.slug,
        "label": top.display,
        "resolvable": top.resolvable,
        "low_confidence": top.confidence < CONFIDENCE_FLOOR,
        "confidence_floor": CONFIDENCE_FLOOR,
        "predictions": [p.as_dict() for p in preds],
        "timing_ms": {"inference": round(infer_ms, 2),
                      "total": round((time.perf_counter() - t0) * 1000.0, 2)},
        "model": {"backbone": model.backbone, "backend": model.backend},
    })


@app.get("/heatmap/<key>.png")
def heatmap(key: str):
    with _cache_lock:
        blob = _HEATMAP_CACHE.get(key)
    if blob is None:
        return jsonify({"error": "heatmap expired or not found"}), 404
    return Response(blob, mimetype="image/png",
                    headers={"Cache-Control": "private, max-age=300"})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", "8000")))
