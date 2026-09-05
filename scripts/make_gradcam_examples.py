"""Generate Grad-CAM example figures from the test split, for the README.

Saves side-by-side panels (original | heatmap) for:
  * the most confident *correct* predictions, one per class where available
  * every *misclassification* on the test split, ranked by how confident the
    model was while being wrong -- these are the interesting ones, because the
    heatmap usually shows exactly which shared visual feature fooled it

    python scripts/make_gradcam_examples.py --checkpoint artifacts/best_model.pt
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys

import torch
from PIL import Image, ImageDraw

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from landmarklens.data import eval_transforms
from landmarklens.gradcam import HeatmapGenerator, _display_crop
from landmarklens.model import load_checkpoint

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SPLITS = os.path.join(ROOT, "data", "splits")
OUT_DIR = os.path.join(ROOT, "docs", "gradcam")

BAR_H = 26


def panel(image: Image.Image, overlay: Image.Image, caption: str) -> Image.Image:
    """Original beside its heatmap, with a caption bar underneath."""
    import numpy as np

    left = Image.fromarray((_display_crop(image) * 255).astype("uint8"))
    w, h = left.size
    out = Image.new("RGB", (w * 2 + 8, h + BAR_H), "white")
    out.paste(left, (0, 0))
    out.paste(overlay.resize((w, h)), (w + 8, 0))
    draw = ImageDraw.Draw(out)
    draw.text((4, h + 6), caption[:110], fill=(20, 20, 20))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--correct-per-class", type=int, default=1)
    ap.add_argument("--max-errors", type=int, default=6)
    ap.add_argument("--splits", default=SPLITS)
    args = ap.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)
    model, classes, backbone, meta = load_checkpoint(args.checkpoint)
    size = int(meta.get("image_size", 224))
    tf = eval_transforms(size)
    cam = HeatmapGenerator(model, size)

    # Score the whole test split first, then pick what to visualise.
    rows = []
    for ci, cname in enumerate(classes):
        for path in sorted(glob.glob(os.path.join(args.splits, "test", cname, "*"))):
            try:
                img = Image.open(path).convert("RGB")
            except Exception:
                continue
            with torch.no_grad():
                probs = torch.softmax(model(tf(img).unsqueeze(0)), dim=1)[0]
            pi = int(probs.argmax())
            rows.append({"path": path, "true": ci, "pred": pi,
                         "conf": float(probs[pi])})

    correct = [r for r in rows if r["true"] == r["pred"]]
    wrong = [r for r in rows if r["true"] != r["pred"]]
    print(f"test images: {len(rows)}   correct: {len(correct)}   "
          f"misclassified: {len(wrong)}")

    manifest = {"checkpoint": args.checkpoint, "correct": [], "errors": []}

    # --- confident correct predictions, one per class ---------------------
    for ci, cname in enumerate(classes):
        picks = sorted([r for r in correct if r["true"] == ci],
                       key=lambda r: -r["conf"])[:args.correct_per_class]
        for n, r in enumerate(picks):
            img = Image.open(r["path"]).convert("RGB")
            overlay, _ = cam.overlay(img, r["pred"])
            cap = f"CORRECT  {classes[r['true']]}  ({r['conf'] * 100:.1f}%)"
            out = os.path.join(OUT_DIR, f"correct_{cname}{'' if n == 0 else n}.png")
            panel(img, overlay, cap).save(out)
            manifest["correct"].append({
                "file": os.path.relpath(out, ROOT).replace("\\", "/"),
                "class": cname, "confidence": r["conf"],
                "source": os.path.relpath(r["path"], ROOT).replace("\\", "/"),
            })

    # --- confident mistakes ------------------------------------------------
    for n, r in enumerate(sorted(wrong, key=lambda r: -r["conf"])[:args.max_errors]):
        img = Image.open(r["path"]).convert("RGB")
        overlay, _ = cam.overlay(img, r["pred"])
        cap = (f"WRONG  true={classes[r['true']]}  "
               f"pred={classes[r['pred']]}  ({r['conf'] * 100:.1f}%)")
        out = os.path.join(
            OUT_DIR, f"error_{n + 1}_{classes[r['true']]}_as_{classes[r['pred']]}.png")
        panel(img, overlay, cap).save(out)
        manifest["errors"].append({
            "file": os.path.relpath(out, ROOT).replace("\\", "/"),
            "true": classes[r["true"]], "pred": classes[r["pred"]],
            "confidence": r["conf"],
            "source": os.path.relpath(r["path"], ROOT).replace("\\", "/"),
        })

    with open(os.path.join(OUT_DIR, "manifest.json"), "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2)
    print(f"wrote {len(manifest['correct'])} correct and "
          f"{len(manifest['errors'])} error panels to {OUT_DIR}")


if __name__ == "__main__":
    main()
