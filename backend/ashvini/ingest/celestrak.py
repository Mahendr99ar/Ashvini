"""CelesTrak general perturbations data.

CelesTrak updates most groups a few times a day and asks users not to download the
same data more often than it changes. Every group is cached on disk and re-used for
two hours, and a request that fails falls back to the cached copy.
"""

import json
import logging
import time
from collections.abc import Iterable
from pathlib import Path

import httpx

from ..orbits.elements import OrbitRecord, from_omm

log = logging.getLogger(__name__)

GP_URL = "https://celestrak.org/NORAD/elements/gp.php"
USER_AGENT = "Ashvini/0.1 (+https://github.com/Mahendr99ar/ashvini)"
CACHE_SECONDS = 2 * 3600


def cache_path(cache_dir: Path, group: str) -> Path:
    return cache_dir / f"celestrak-{group}.json"


def read_cache(path: Path, max_age_s: float) -> list[dict] | None:
    if not path.exists() or time.time() - path.stat().st_mtime > max_age_s:
        return None
    return json.loads(path.read_text())


def download_group(client: httpx.Client, group: str) -> list[dict]:
    resp = client.get(GP_URL, params={"GROUP": group, "FORMAT": "json"})
    resp.raise_for_status()
    if not resp.text.lstrip().startswith("["):
        raise ValueError(f"CelesTrak returned no data for group {group!r}: {resp.text[:80]}")
    return resp.json()


def fetch_group(client: httpx.Client, group: str, cache_dir: Path) -> list[dict]:
    path = cache_path(cache_dir, group)
    cached = read_cache(path, CACHE_SECONDS)
    if cached is not None:
        return cached
    try:
        rows = download_group(client, group)
    except (httpx.HTTPError, ValueError) as exc:
        stale = read_cache(path, float("inf"))
        if stale is None:
            raise
        log.warning("CelesTrak %s failed (%s); using cached copy", group, exc)
        return stale
    path.write_text(json.dumps(rows))
    return rows


def records_from_rows(rows: Iterable[dict], source: str = "celestrak") -> list[OrbitRecord]:
    records = []
    for row in rows:
        try:
            records.append(from_omm(row, source))
        except (KeyError, ValueError) as exc:
            log.warning("skipping malformed GP record %s: %s", row.get("NORAD_CAT_ID"), exc)
    return records


def newest_per_object(records: Iterable[OrbitRecord]) -> list[OrbitRecord]:
    best: dict[int, OrbitRecord] = {}
    for rec in records:
        old = best.get(rec.norad_id)
        if old is None or rec.satrec.jdsatepoch + rec.satrec.jdsatepochF > old.satrec.jdsatepoch + old.satrec.jdsatepochF:
            best[rec.norad_id] = rec
    return list(best.values())


def fetch_catalogue(groups: list[str], cache_dir: Path, client: httpx.Client | None = None) -> list[OrbitRecord]:
    cache_dir.mkdir(parents=True, exist_ok=True)
    own_client = client is None
    client = client or httpx.Client(timeout=60, headers={"User-Agent": USER_AGENT}, follow_redirects=True)
    try:
        records = []
        for group in groups:
            records.extend(records_from_rows(fetch_group(client, group, cache_dir)))
        return newest_per_object(records)
    finally:
        if own_client:
            client.close()
