"""Rate-limited Wikimedia Commons API client.

Commons enforces per-IP rate limits and *requires* a descriptive User-Agent.
This client serialises requests behind a minimum interval and honours
429/Retry-After with exponential backoff, so a long harvest run does not get
the caller blocked.

Only freely-licensed files are kept: anything whose license shortname is not on
the allow-list below is skipped, and the license/author of every file that IS
kept is recorded so it can be reproduced in DATA_SOURCES.md.
"""
from __future__ import annotations

import json
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import asdict, dataclass
from typing import Any, Dict, Iterable, List, Optional

API = "https://commons.wikimedia.org/w/api.php"
USER_AGENT = (
    "LandmarkLens/0.1 (educational portfolio project; "
    "London landmark image classifier)"
)

# License shortnames accepted for the dataset: public domain and the CC
# variants that permit reuse with attribution. Anything else (non-free logos,
# fair-use rationales) is rejected outright.
ALLOWED_LICENSE_PATTERNS = [
    re.compile(r"^cc[ -]?by([ -]sa)?([ -]\d(\.\d)?)?", re.I),
    re.compile(r"^cc[ -]?zero", re.I),
    re.compile(r"^cc0", re.I),
    re.compile(r"^public domain", re.I),
    re.compile(r"^pd[- ]", re.I),
    re.compile(r"^attribution$", re.I),
]

_MIN_INTERVAL = 2.0  # seconds between API calls; Commons is strict
_last_call = [0.0]


def _throttle() -> None:
    delta = time.time() - _last_call[0]
    if delta < _MIN_INTERVAL:
        time.sleep(_MIN_INTERVAL - delta)
    _last_call[0] = time.time()


def api(**params: Any) -> Dict[str, Any]:
    """One MediaWiki API GET with throttling and 429/5xx backoff."""
    params.setdefault("format", "json")
    params.setdefault("formatversion", "2")
    url = API + "?" + urllib.parse.urlencode(params)
    backoff = 5.0
    for attempt in range(6):
        _throttle()
        req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
        try:
            with urllib.request.urlopen(req, timeout=60) as resp:
                return json.load(resp)
        except urllib.error.HTTPError as exc:
            if exc.code in (429, 500, 502, 503, 504):
                wait = float(exc.headers.get("Retry-After") or backoff)
                print(f"    [commons] HTTP {exc.code}; sleeping {wait:.0f}s "
                      f"(attempt {attempt + 1}/6)")
                time.sleep(wait)
                backoff = min(backoff * 2, 120)
                continue
            raise
        except (urllib.error.URLError, TimeoutError) as exc:
            print(f"    [commons] network error {exc}; retry in {backoff:.0f}s")
            time.sleep(backoff)
            backoff = min(backoff * 2, 120)
    raise RuntimeError(f"Commons API failed after retries: {url}")


def category_files(category: str, depth: int = 1, cap: int = 1200) -> List[str]:
    """File titles in a category, descending `depth` levels of subcategories."""
    found: List[str] = []
    seen = set()
    frontier = [category]
    for level in range(depth + 1):
        next_frontier: List[str] = []
        for cat in frontier:
            cont: Dict[str, str] = {}
            while len(found) < cap:
                data = api(
                    action="query", list="categorymembers",
                    cmtitle=f"Category:{cat}", cmtype="file|subcat",
                    cmlimit="500", **cont,
                )
                members = data.get("query", {}).get("categorymembers", [])
                for m in members:
                    title = m["title"]
                    if title.startswith("Category:"):
                        if level < depth:
                            next_frontier.append(title[len("Category:"):])
                    elif title not in seen:
                        seen.add(title)
                        found.append(title)
                cont = data.get("continue") or {}
                if not cont:
                    break
        frontier = next_frontier
        if not frontier or len(found) >= cap:
            break
    return found[:cap]


def search_files(query: str, limit: int = 300) -> List[str]:
    """File titles from Commons full-text search (File: namespace only)."""
    titles: List[str] = []
    offset = 0
    while len(titles) < limit:
        data = api(
            action="query", list="search", srsearch=query,
            srnamespace="6", srlimit="50", sroffset=str(offset),
        )
        hits = data.get("query", {}).get("search", [])
        if not hits:
            break
        titles.extend(h["title"] for h in hits)
        offset += len(hits)
        if "continue" not in data:
            break
    return titles[:limit]


@dataclass
class FileRecord:
    title: str
    thumb_url: str
    descriptionurl: str
    license_short: str
    license_url: str
    artist: str
    width: int
    height: int
    mime: str

    def as_dict(self) -> Dict[str, Any]:
        return asdict(self)


_TAG_RE = re.compile(r"<[^>]+>")


def _clean(html: Optional[str]) -> str:
    if not html:
        return ""
    text = _TAG_RE.sub(" ", html)
    text = re.sub(r"\s+", " ", text).strip()
    return text[:300]


def _license_ok(short: str) -> bool:
    return any(p.match(short.strip()) for p in ALLOWED_LICENSE_PATTERNS)


def file_info(titles: Iterable[str], thumb_width: int = 480,
              min_width: int = 400) -> List[FileRecord]:
    """Resolve file titles to thumbnail URLs + license metadata, in batches.

    Files that are not raster photos, are too small, or carry a license outside
    the allow-list are dropped.
    """
    titles = list(titles)
    out: List[FileRecord] = []
    for i in range(0, len(titles), 40):
        batch = titles[i:i + 40]
        data = api(
            action="query", titles="|".join(batch), prop="imageinfo",
            iiprop="url|size|mime|extmetadata", iiurlwidth=str(thumb_width),
        )
        for page in data.get("query", {}).get("pages", []):
            infos = page.get("imageinfo") or []
            if not infos:
                continue
            info = infos[0]
            if info.get("mime", "") not in ("image/jpeg", "image/png"):
                continue
            w, h = int(info.get("width", 0)), int(info.get("height", 0))
            if w < min_width or h < 200:
                continue
            # Reject extreme panoramas / banner crops: after a square-ish
            # resize they carry almost no usable subject detail.
            if w and h and (max(w, h) / min(w, h)) > 2.6:
                continue
            meta = info.get("extmetadata", {}) or {}
            short = _clean(meta.get("LicenseShortName", {}).get("value"))
            if not _license_ok(short):
                continue
            thumb = info.get("thumburl") or info.get("url")
            if not thumb:
                continue
            out.append(FileRecord(
                title=page.get("title", ""),
                thumb_url=thumb,
                descriptionurl=info.get("descriptionurl", ""),
                license_short=short or "unknown",
                license_url=_clean(meta.get("LicenseUrl", {}).get("value")),
                artist=_clean(meta.get("Artist", {}).get("value")) or "unknown",
                width=int(info.get("width", 0)),
                height=int(info.get("height", 0)),
                mime=info.get("mime", ""),
            ))
    return out


def download(url: str, dest: str, timeout: int = 45) -> bool:
    """Fetch one thumbnail. Returns True on success."""
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    for attempt in range(3):
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                blob = resp.read()
            if len(blob) < 2048:  # too small to be a usable photo
                return False
            with open(dest, "wb") as fh:
                fh.write(blob)
            return True
        except urllib.error.HTTPError as exc:
            if exc.code == 429:
                time.sleep(10 * (attempt + 1))
                continue
            return False
        except Exception:
            time.sleep(2 * (attempt + 1))
    return False


def download_many(jobs: List[tuple], workers: int = 6) -> List[tuple]:
    """Download (url, dest) pairs concurrently.

    Thumbnails come from upload.wikimedia.org, a CDN, so a handful of parallel
    connections is fine there -- unlike the API, which stays strictly serialised
    behind `_throttle`. Returns the subset of jobs that succeeded, in the order
    they were requested.
    """
    from concurrent.futures import ThreadPoolExecutor

    if not jobs:
        return []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        ok = list(pool.map(lambda j: download(j[0], j[1]), jobs))
    return [j for j, good in zip(jobs, ok) if good]
