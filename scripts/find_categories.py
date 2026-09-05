"""Find Wikimedia Commons categories that actually exist and hold photos.

Several plausible-looking category names in the class registry turned out to be
empty or non-existent (Commons renames and merges categories constantly), which
silently pushed those classes onto the weaker full-text-search fallback. This
searches the Category namespace for a term, then reports how many files each
candidate category actually contains, so the registry can be corrected against
reality instead of guesses.

    python scripts/find_categories.py "black cab taxi london" "bus stop flag"
"""
from __future__ import annotations

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from landmarklens import commons


def category_search(term: str, limit: int = 12) -> list[str]:
    data = commons.api(action="query", list="search", srsearch=term,
                       srnamespace="14", srlimit=str(limit))
    return [h["title"][len("Category:"):]
            for h in data.get("query", {}).get("search", [])]


def direct_count(category: str) -> int:
    """Files directly in a category (no subcategory descent) -- one API call."""
    data = commons.api(action="query", list="categorymembers",
                       cmtitle=f"Category:{category}", cmtype="file",
                       cmlimit="500")
    return len(data.get("query", {}).get("categorymembers", []))


def main() -> None:
    terms = sys.argv[1:]
    if not terms:
        print(__doc__)
        return
    for term in terms:
        print(f"\n=== {term!r} ===", flush=True)
        try:
            cats = category_search(term)
        except Exception as exc:
            print(f"  search failed: {exc}")
            continue
        for cat in cats:
            try:
                n = direct_count(cat)
            except Exception as exc:
                n = -1
                print(f"  {'ERR':>5}  {cat}  ({exc})")
                continue
            marker = "  <-- usable" if n >= 40 else ""
            print(f"  {n:>5}  {cat}{marker}", flush=True)


if __name__ == "__main__":
    main()
