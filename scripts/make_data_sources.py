"""Regenerate DATA_SOURCES.md from the harvest attribution manifests.

Every image in the dataset came from Wikimedia Commons under a free license.
This writes the per-class summary (counts, categories used, license breakdown)
plus a machine-readable per-image attribution index, so the licensing claim is
reproducible rather than asserted.

    python scripts/make_data_sources.py
"""
from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from landmarklens.classes import CLASSES

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ATTRIB = os.path.join(ROOT, "data", "attribution")


def main() -> None:
    lines: list[str] = []
    lines.append("# Data sources and attribution\n")
    lines.append(
        "Every training image in LandmarkLens was downloaded from "
        "[Wikimedia Commons](https://commons.wikimedia.org) via the MediaWiki "
        "API by `scripts/harvest_dataset.py`.\n"
    )
    lines.append(
        "The harvester keeps a file **only** if its `LicenseShortName` matches "
        "one of the free licenses allow-listed in `landmarklens/commons.py` "
        "(CC BY, CC BY-SA, CC0/CC Zero, Public Domain / PD-*, or Attribution). "
        "Non-free files -- including the London Underground roundel *as a "
        "trademark* -- are rejected at harvest time.\n"
    )
    lines.append(
        "Images are stored as 480px-wide thumbnails served by Commons, not as "
        "the full-resolution originals.\n"
    )
    lines.append(
        "**Per-image attribution** (Commons title, author, license, and source "
        "page URL for every single file) lives in `data/attribution/<class>.json`. "
        "That directory is the authoritative record; the tables below are a "
        "summary of it.\n"
    )
    lines.append(
        "> Trademark note: several classes depict trademarks belonging to "
        "Transport for London. The images are used here under their "
        "photographic copyright licenses for non-commercial research and "
        "portfolio purposes. This project is not affiliated with or endorsed "
        "by TfL.\n"
    )

    total = 0
    license_totals: dict[str, int] = {}
    lines.append("\n## Summary by class\n")
    lines.append("| Class | Images | Commons categories | Licenses |")
    lines.append("|---|---:|---|---|")

    per_class_rows = []
    for cls in CLASSES:
        path = os.path.join(ATTRIB, f"{cls.slug}.json")
        if not os.path.exists(path):
            per_class_rows.append((cls, 0, {}, "*(not harvested)*"))
            continue
        with open(path, encoding="utf-8") as fh:
            man = json.load(fh)
        n = man.get("count", 0)
        total += n
        lic = man.get("license_breakdown", {})
        for k, v in lic.items():
            license_totals[k] = license_totals.get(k, 0) + v
        cats = ", ".join(f"`{c}`" for c in man.get("commons_categories", []))
        lic_str = ", ".join(f"{k} ({v})" for k, v in
                            sorted(lic.items(), key=lambda kv: -kv[1]))
        lines.append(f"| `{cls.slug}` | {n} | {cats} | {lic_str or '-'} |")
        per_class_rows.append((cls, n, lic, cats))

    lines.append(f"| **Total** | **{total}** | | |")

    lines.append("\n## License breakdown across the whole dataset\n")
    lines.append("| License | Images | Share |")
    lines.append("|---|---:|---:|")
    for k, v in sorted(license_totals.items(), key=lambda kv: -kv[1]):
        lines.append(f"| {k} | {v} | {v / max(total, 1) * 100:.1f}% |")

    lines.append("\n## Class to location mapping\n")
    lines.append(
        "`resolvable = false` means the class identifies a *kind* of place "
        "rather than a specific one, so TravelAssistant asks the user which "
        "one instead of planning a journey to a guess.\n"
    )
    lines.append("| Class | Label | `location_name` | Resolvable |")
    lines.append("|---|---|---|---|")
    for cls in CLASSES:
        lines.append(f"| `{cls.slug}` | {cls.display} | {cls.location_name} | "
                     f"{'yes' if cls.resolvable else 'no'} |")

    lines.append("\n## Reproducing\n")
    lines.append("```bash")
    lines.append("python scripts/harvest_dataset.py --per-class 110")
    lines.append("python scripts/make_data_sources.py")
    lines.append("```\n")
    lines.append(
        "Commons categories change over time, so a later run will not return "
        "a byte-identical dataset. The attribution JSON records exactly which "
        "files this particular dataset was built from.\n"
    )

    out = os.path.join(ROOT, "DATA_SOURCES.md")
    with open(out, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines))
    print(f"wrote {out}  ({total} images across "
          f"{sum(1 for _, n, _, _ in per_class_rows if n)} classes)")


if __name__ == "__main__":
    main()
