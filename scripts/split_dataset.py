"""Build a stratified train/val/test split from data/raw.

Images are hard-linked (falling back to copy) into data/splits/{train,val,test}
/<class>/ so that torchvision's ImageFolder can read them directly without
duplicating hundreds of megabytes on disk.

Corrupt or unreadable files found during the pass are dropped and reported --
Commons occasionally serves a truncated thumbnail.

    python scripts/split_dataset.py --val 0.15 --test 0.15 --seed 42
"""
from __future__ import annotations

import argparse
import json
import os
import random
import shutil
import sys

from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from landmarklens.classes import CLASSES

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW = os.path.join(ROOT, "data", "raw")
SPLITS = os.path.join(ROOT, "data", "splits")


def usable(path: str) -> bool:
    """True when PIL can fully decode the file as an RGB-convertible image."""
    try:
        with Image.open(path) as im:
            im.verify()
        with Image.open(path) as im:  # verify() leaves the file unusable
            im.convert("RGB").load()
        return True
    except Exception:
        return False


def link_or_copy(src: str, dst: str) -> None:
    if os.path.exists(dst):
        return
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--val", type=float, default=0.15)
    ap.add_argument("--test", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    if os.path.exists(SPLITS):
        shutil.rmtree(SPLITS)

    rng = random.Random(args.seed)
    stats: dict[str, dict[str, int]] = {}
    corrupt_total = 0

    for cls in CLASSES:
        src_dir = os.path.join(RAW, cls.slug)
        if not os.path.isdir(src_dir):
            print(f"  {cls.slug}: NO DATA")
            stats[cls.slug] = {"train": 0, "val": 0, "test": 0, "total": 0}
            continue

        files = sorted(f for f in os.listdir(src_dir) if not f.startswith("."))
        good, corrupt = [], 0
        for f in files:
            if usable(os.path.join(src_dir, f)):
                good.append(f)
            else:
                corrupt += 1
        corrupt_total += corrupt

        rng.shuffle(good)
        n = len(good)
        n_test = max(1, round(n * args.test)) if n else 0
        n_val = max(1, round(n * args.val)) if n else 0
        test_files = good[:n_test]
        val_files = good[n_test:n_test + n_val]
        train_files = good[n_test + n_val:]

        for split, group in (("train", train_files), ("val", val_files),
                             ("test", test_files)):
            out_dir = os.path.join(SPLITS, split, cls.slug)
            os.makedirs(out_dir, exist_ok=True)
            for f in group:
                link_or_copy(os.path.join(src_dir, f), os.path.join(out_dir, f))

        stats[cls.slug] = {
            "train": len(train_files), "val": len(val_files),
            "test": len(test_files), "total": n, "corrupt_dropped": corrupt,
        }
        print(f"  {cls.slug:<22} train={len(train_files):>3} "
              f"val={len(val_files):>3} test={len(test_files):>3} "
              f"(corrupt dropped: {corrupt})")

    totals = {k: sum(v.get(k, 0) for v in stats.values())
              for k in ("train", "val", "test", "total")}
    print(f"\n  {'TOTAL':<22} train={totals['train']} val={totals['val']} "
          f"test={totals['test']}  (all={totals['total']}, "
          f"corrupt dropped={corrupt_total})")

    out = {"seed": args.seed, "val_frac": args.val, "test_frac": args.test,
           "per_class": stats, "totals": totals}
    with open(os.path.join(ROOT, "data", "split_stats.json"), "w",
              encoding="utf-8") as fh:
        json.dump(out, fh, indent=2)
    print(f"\n  wrote data/split_stats.json")


if __name__ == "__main__":
    main()
