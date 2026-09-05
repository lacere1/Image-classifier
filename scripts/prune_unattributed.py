"""Guarantee that every image in data/raw has a recorded licence.

The licensing claim in DATA_SOURCES.md is only meaningful if *every* file on
disk appears in `data/attribution/<class>.json`. Files can fall out of sync
when a class is re-harvested from different Commons categories: the new run
rewrites the manifest, but images downloaded by the earlier run are still on
disk with no attribution record.

This reports the discrepancy and, with --delete, removes the orphans. It is
safer to lose a few images than to publish a dataset whose provenance cannot
be reproduced.

    python scripts/prune_unattributed.py            # report only
    python scripts/prune_unattributed.py --delete
"""
from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from landmarklens.classes import CLASSES

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW = os.path.join(ROOT, "data", "raw")
ATTRIB = os.path.join(ROOT, "data", "attribution")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--delete", action="store_true",
                    help="actually remove unattributed files")
    args = ap.parse_args()

    total_orphans = 0
    total_files = 0
    for cls in CLASSES:
        raw_dir = os.path.join(RAW, cls.slug)
        man_path = os.path.join(ATTRIB, f"{cls.slug}.json")
        if not os.path.isdir(raw_dir):
            continue
        on_disk = {f for f in os.listdir(raw_dir) if not f.startswith(".")}
        total_files += len(on_disk)

        attributed: set[str] = set()
        if os.path.exists(man_path):
            with open(man_path, encoding="utf-8") as fh:
                man = json.load(fh)
            attributed = {e["file"] for e in man.get("images", [])
                          if e.get("file")}

        orphans = sorted(on_disk - attributed)
        missing = sorted(attributed - on_disk)
        if orphans or missing:
            print(f"  {cls.slug:<22} on_disk={len(on_disk):>3} "
                  f"attributed={len(attributed):>3} "
                  f"orphans={len(orphans):>3} missing_file={len(missing):>3}")
        total_orphans += len(orphans)

        if args.delete:
            for f in orphans:
                os.remove(os.path.join(raw_dir, f))

    print(f"\n  {total_files} files, {total_orphans} without an attribution "
          f"record")
    if total_orphans and not args.delete:
        print("  re-run with --delete to remove them")
    elif args.delete and total_orphans:
        print(f"  deleted {total_orphans} unattributed files")
    elif not total_orphans:
        print("  OK: every image has a recorded licence")


if __name__ == "__main__":
    main()
