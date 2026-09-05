"""Harvest the LandmarkLens dataset from Wikimedia Commons.

For each class in the registry: gather candidate file titles from its Commons
categories (and, if short, from full-text search), resolve them to 480px-wide
thumbnails with license metadata, filter to freely-licensed raster photos, then
download up to --per-class of them.

Every kept image is recorded with its source page, author and license so that
DATA_SOURCES.md can be regenerated from data/attribution/*.json.

Re-running is safe: already-downloaded files are skipped, so an interrupted run
resumes where it stopped.

    python scripts/harvest_dataset.py --per-class 110
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from landmarklens import commons
from landmarklens.classes import CLASSES

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DIR = os.path.join(ROOT, "data", "raw")
ATTRIB_DIR = os.path.join(ROOT, "data", "attribution")


def safe_name(title: str) -> str:
    """Stable, filesystem-safe filename derived from the Commons title."""
    stem = title.replace("File:", "")
    ext = os.path.splitext(stem)[1].lower()
    if ext not in (".jpg", ".jpeg", ".png"):
        ext = ".jpg"
    digest = hashlib.sha1(title.encode("utf-8")).hexdigest()[:16]
    return f"{digest}{ext}"


def harvest_class(cls, per_class: int, oversample: float) -> dict:
    out_dir = os.path.join(RAW_DIR, cls.slug)
    os.makedirs(out_dir, exist_ok=True)

    existing = {f for f in os.listdir(out_dir) if not f.startswith(".")}
    print(f"\n=== {cls.slug} ===  (already have {len(existing)})")
    if len(existing) >= per_class:
        print("  already complete, skipping fetch")
        attrib_path = os.path.join(ATTRIB_DIR, f"{cls.slug}.json")
        if os.path.exists(attrib_path):
            with open(attrib_path, encoding="utf-8") as fh:
                return json.load(fh)

    # 1. Candidate titles ---------------------------------------------------
    titles: list[str] = []
    seen = set()
    for cat in cls.categories:
        try:
            got = commons.category_files(cat, depth=1, cap=600)
        except Exception as exc:  # a missing/renamed category must not abort
            print(f"  category {cat!r} failed: {exc}")
            continue
        new = [t for t in got if t not in seen]
        seen.update(new)
        titles.extend(new)
        print(f"  category {cat!r}: {len(got)} files ({len(titles)} candidates)")

    want = int(per_class * oversample)
    if len(titles) < want:
        for query in cls.searches:
            try:
                got = commons.search_files(query, limit=300)
            except Exception as exc:
                print(f"  search {query!r} failed: {exc}")
                continue
            new = [t for t in got if t not in seen]
            seen.update(new)
            titles.extend(new)
            print(f"  search {query!r}: +{len(new)} ({len(titles)} candidates)")
            if len(titles) >= want:
                break

    if not titles:
        print("  NO CANDIDATES FOUND")
        return {"class": cls.slug, "images": [], "sources": []}

    # Shuffle deterministically so we do not take only alphabetically-early
    # files (which on Commons often means one photographer's upload batch).
    random.Random(1234).shuffle(titles)

    # 2. Metadata + license filter -----------------------------------------
    # Resolving metadata costs one throttled API call per 40 titles, so only
    # inspect a modest surplus over the target rather than the whole category:
    # in practice ~95% of Commons candidates clear the license/size filter.
    probe = max(per_class + 40, int(per_class * 1.5))
    records = commons.file_info(titles[:probe], thumb_width=480, min_width=400)
    # Top up if the filter was harsher than expected for this class.
    while len(records) < per_class and probe < len(titles):
        nxt = min(probe + per_class, len(titles))
        records += commons.file_info(titles[probe:nxt], thumb_width=480,
                                     min_width=400)
        probe = nxt
    print(f"  {len(records)} files pass license/size filter "
          f"(from {probe} candidates inspected)", flush=True)

    # 3. Download ------------------------------------------------------------
    # Files already on disk count towards the target (so an interrupted run
    # resumes); the rest are fetched concurrently in one batch.
    kept: list[dict] = []
    pending: list[tuple] = []
    by_dest: dict[str, dict] = {}
    for rec in records:
        if len(kept) + len(pending) >= per_class:
            break
        fname = safe_name(rec.title)
        dest = os.path.join(out_dir, fname)
        entry = rec.as_dict()
        entry["file"] = fname
        if os.path.exists(dest):
            kept.append(entry)
        else:
            pending.append((rec.thumb_url, dest))
            by_dest[dest] = entry

    if pending:
        print(f"    fetching {len(pending)} images "
              f"({len(kept)} already on disk)...", flush=True)
        for _, dest in commons.download_many(pending, workers=6):
            kept.append(by_dest[dest])

    print(f"  -> {len(kept)} images in {out_dir}", flush=True)

    licenses: dict[str, int] = {}
    for e in kept:
        licenses[e["license_short"]] = licenses.get(e["license_short"], 0) + 1

    manifest = {
        "class": cls.slug,
        "display": cls.display,
        "location_name": cls.location_name,
        "resolvable": cls.resolvable,
        "commons_categories": cls.categories,
        "commons_searches": cls.searches,
        "count": len(kept),
        "license_breakdown": licenses,
        "images": kept,
    }
    os.makedirs(ATTRIB_DIR, exist_ok=True)
    with open(os.path.join(ATTRIB_DIR, f"{cls.slug}.json"), "w",
              encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=1, ensure_ascii=False)
    return manifest


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-class", type=int, default=110)
    ap.add_argument("--oversample", type=float, default=1.3,
                    help="candidate multiplier before license filtering")
    ap.add_argument("--only", default="", help="comma-separated class slugs")
    args = ap.parse_args()

    os.makedirs(RAW_DIR, exist_ok=True)
    os.makedirs(ATTRIB_DIR, exist_ok=True)

    only = {s for s in args.only.split(",") if s}
    targets = [c for c in CLASSES if not only or c.slug in only]

    summary = []
    for cls in targets:
        try:
            manifest = harvest_class(cls, args.per_class, args.oversample)
        except Exception as exc:
            print(f"  !! {cls.slug} failed: {exc}")
            manifest = {"class": cls.slug, "count": 0}
        summary.append((cls.slug, manifest.get("count", 0)))

    print("\n================ HARVEST SUMMARY ================")
    total = 0
    for slug, n in summary:
        total += n
        flag = "" if n >= args.per_class * 0.7 else "   <-- SHORT"
        print(f"  {slug:<22} {n:>4}{flag}")
    print(f"  {'TOTAL':<22} {total:>4}")


if __name__ == "__main__":
    main()
