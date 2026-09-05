"""Evaluate a checkpoint on the held-out test split.

Reports overall top-1 / top-3, per-class precision / recall / F1, and the
confusion pairs the model actually gets wrong -- the per-class table is the part
worth reading, because a small Commons-sourced dataset almost always has a few
classes that carry the average.

    python scripts/evaluate.py --checkpoint artifacts/baseline_best.pt
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import torch
from sklearn.metrics import classification_report, confusion_matrix
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from landmarklens.data import eval_transforms
from landmarklens.model import load_checkpoint

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SPLITS = os.path.join(ROOT, "data", "splits")


@torch.no_grad()
def collect(model, loader, device):
    probs_all, y_all = [], []
    for x, y in loader:
        out = model(x.to(device))
        probs_all.append(torch.softmax(out, dim=1).cpu())
        y_all.append(y)
    return torch.cat(probs_all), torch.cat(y_all)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--split", default="test", choices=["test", "val", "train"])
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--splits", default=SPLITS)
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, classes, backbone, meta = load_checkpoint(args.checkpoint)
    model.to(device)

    ds = ImageFolder(os.path.join(args.splits, args.split),
                     transform=eval_transforms(meta.get("image_size", 224)))
    assert ds.classes == classes, (
        f"class order mismatch between checkpoint and {args.split} folder")
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                        num_workers=0)

    probs, y = collect(model, loader, device)
    pred = probs.argmax(dim=1)
    top3 = probs.topk(min(3, probs.shape[1]), dim=1).indices

    top1 = (pred == y).float().mean().item()
    top3_acc = (top3 == y.unsqueeze(1)).any(dim=1).float().mean().item()

    print(f"\ncheckpoint : {args.checkpoint}")
    print(f"backbone   : {backbone}")
    print(f"split      : {args.split}  (n={len(ds)})")
    print(f"top-1      : {top1:.4f}")
    print(f"top-3      : {top3_acc:.4f}\n")

    report = classification_report(y.numpy(), pred.numpy(),
                                   labels=list(range(len(classes))),
                                   target_names=classes, zero_division=0,
                                   output_dict=True)
    print(classification_report(y.numpy(), pred.numpy(),
                                labels=list(range(len(classes))),
                                target_names=classes, zero_division=0,
                                digits=3))

    cm = confusion_matrix(y.numpy(), pred.numpy(),
                          labels=list(range(len(classes))))
    print("Most frequent confusions (true -> predicted):")
    pairs = []
    for i in range(len(classes)):
        for j in range(len(classes)):
            if i != j and cm[i, j] > 0:
                pairs.append((int(cm[i, j]), classes[i], classes[j]))
    for n, a, b in sorted(pairs, reverse=True)[:10]:
        print(f"  {n:>3}x  {a} -> {b}")
    if not pairs:
        print("  (none)")

    out_path = args.out or os.path.join(
        ROOT, "artifacts",
        os.path.splitext(os.path.basename(args.checkpoint))[0]
        + f"_{args.split}_eval.json")
    payload = {
        "checkpoint": args.checkpoint, "split": args.split, "n": len(ds),
        "top1": top1, "top3": top3_acc, "classes": classes,
        "per_class": {c: report[c] for c in classes},
        "macro_f1": report["macro avg"]["f1-score"],
        "confusion_matrix": cm.tolist(),
        "top_confusions": [{"count": n, "true": a, "pred": b}
                           for n, a, b in sorted(pairs, reverse=True)[:10]],
    }
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
