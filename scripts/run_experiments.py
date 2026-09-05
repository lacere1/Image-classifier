"""Run the LandmarkLens experiment sweep and summarise it from MLflow.

Three runs that differ in one meaningful dimension each, so the MLflow history
answers a question rather than just recording a number:

  baseline        standard augmentation, lr 1e-3 / 1e-4   -- the reference
  strong_aug      strong augmentation (rotation, grayscale, random erasing)
                  -- does more aggressive augmentation help on ~100 img/class?
  high_lr         standard augmentation, 10x the fine-tuning LR
                  -- does the backbone tolerate a larger fine-tune step?

The best run by validation top-1 is copied to artifacts/best_model.pt, which is
what the serving container loads.

    python scripts/run_experiments.py
    python scripts/run_experiments.py --only baseline
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ARTIFACTS = os.path.join(ROOT, "artifacts")
PY = sys.executable

EXPERIMENTS = {
    "baseline": ["--aug", "standard", "--lr-head", "1e-3", "--lr-ft", "1e-4"],
    "strong_aug": ["--aug", "strong", "--lr-head", "1e-3", "--lr-ft", "1e-4"],
    "high_lr": ["--aug", "standard", "--lr-head", "1e-3", "--lr-ft", "1e-3"],
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--backbone", default="resnet50")
    ap.add_argument("--epochs-head", type=int, default=3)
    ap.add_argument("--epochs-ft", type=int, default=8)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--only", default="", help="comma-separated run names")
    args = ap.parse_args()

    only = {s for s in args.only.split(",") if s}
    todo = [k for k in EXPERIMENTS if not only or k in only]

    summaries = []
    for name in todo:
        cmd = [PY, "-u", os.path.join(ROOT, "scripts", "train.py"),
               "--run-name", name, "--backbone", args.backbone,
               "--epochs-head", str(args.epochs_head),
               "--epochs-ft", str(args.epochs_ft),
               "--batch-size", str(args.batch_size),
               "--workers", str(args.workers)] + EXPERIMENTS[name]
        print(f"\n{'=' * 70}\n RUN: {name}\n {' '.join(cmd[2:])}\n{'=' * 70}",
              flush=True)
        rc = subprocess.call(cmd, cwd=ROOT)
        if rc != 0:
            print(f"  run {name} exited {rc}")
            continue
    # Selecting the best model must consider every run that has ever
    # completed, not just the ones this invocation happened to execute --
    # otherwise `--only` on a resumed sweep would discard an earlier, better
    # run when it copies artifacts/best_model.pt.
    for name in EXPERIMENTS:
        path = os.path.join(ARTIFACTS, f"{name}_train.json")
        ckpt = os.path.join(ARTIFACTS, f"{name}_best.pt")
        if os.path.exists(path) and os.path.exists(ckpt):
            with open(path, encoding="utf-8") as fh:
                summaries.append(json.load(fh))

    if not summaries:
        print("no runs completed")
        return

    summaries.sort(key=lambda s: -s["val_top1"])
    print(f"\n{'=' * 70}\n EXPERIMENT SUMMARY (by best val top-1)\n{'=' * 70}")
    print(f"{'run':<14}{'val top1':>10}{'val top3':>10}{'epoch':>7}{'minutes':>9}")
    for s in summaries:
        print(f"{s['run_name']:<14}{s['val_top1']:>10.4f}{s['val_top3']:>10.4f}"
              f"{s['epoch']:>7}{s['train_minutes']:>9.1f}")

    best = summaries[0]
    for suffix in ("_best.pt", "_best_classes.json"):
        src = os.path.join(ARTIFACTS, best["run_name"] + suffix)
        dst = os.path.join(ARTIFACTS, "best_model" + suffix.replace("_best", ""))
        if os.path.exists(src):
            shutil.copy2(src, dst)
    print(f"\nbest run: {best['run_name']} -> artifacts/best_model.pt")

    with open(os.path.join(ARTIFACTS, "experiments.json"), "w",
              encoding="utf-8") as fh:
        json.dump({"runs": summaries, "best": best["run_name"]}, fh, indent=2)


if __name__ == "__main__":
    main()
