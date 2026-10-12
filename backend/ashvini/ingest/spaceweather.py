"""Planetary K index from NOAA SWPC, observed and forecast.

SWPC has published these products both as a header row followed by lists and as
lists of objects, so the parser accepts either shape.
"""

import math
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta

import httpx
import numpy as np

OBSERVED_URL = "https://services.swpc.noaa.gov/products/noaa-planetary-k-index.json"
FORECAST_URL = "https://services.swpc.noaa.gov/products/noaa-planetary-k-index-forecast.json"


@dataclass
class KpReading:
    time: datetime
    kp: float
    kind: str  # observed, estimated or predicted


def storm_level(kp: float) -> str:
    """NOAA G-scale: Kp 5 is G1, up to Kp 9 for G5."""
    if kp >= 5:
        return f"G{min(5, int(math.floor(kp)) - 4)}"
    return "quiet" if kp < 4 else "active"


def _parse_time(text: str) -> datetime:
    return datetime.fromisoformat(text.replace("Z", "").replace(" ", "T")).replace(tzinfo=UTC)


def rows_as_dicts(payload: list) -> list[dict]:
    if not payload:
        return []
    if isinstance(payload[0], dict):
        return [{k.lower(): v for k, v in row.items()} for row in payload]
    header = [str(h).lower() for h in payload[0]]
    return [dict(zip(header, row, strict=False)) for row in payload[1:]]


def parse_kp(payload: list, default_kind: str) -> list[KpReading]:
    readings = []
    for row in rows_as_dicts(payload):
        kp_value = row.get("kp", row.get("kp_index", row.get("estimated_kp")))
        if kp_value in (None, "") or "time_tag" not in row:
            continue
        kind = str(row.get("observed", default_kind) or default_kind).lower()
        readings.append(KpReading(_parse_time(str(row["time_tag"])), float(kp_value), kind))
    return readings


def fetch_kp(client: httpx.Client | None = None) -> list[KpReading]:
    own = client is None
    client = client or httpx.Client(timeout=30)
    try:
        observed = parse_kp(client.get(OBSERVED_URL).json(), "observed")
        forecast = parse_kp(client.get(FORECAST_URL).json(), "predicted")
    finally:
        if own:
            client.close()
    merged = {r.time: r for r in observed}
    for r in forecast:
        if r.time not in merged or merged[r.time].kind != "observed":
            merged[r.time] = r
    return [merged[t] for t in sorted(merged)]


def synthetic_kp(now: datetime, seed: int = 7) -> list[KpReading]:
    """Two quiet days, then a storm building to Kp 7 about a day ahead, at 3-hour cadence."""
    rng = np.random.default_rng(seed)
    start = now.replace(minute=0, second=0, microsecond=0) - timedelta(hours=now.hour % 3 + 48)
    readings = []
    for k in range(32):
        t = start + timedelta(hours=3 * k)
        hours_ahead = (t - now).total_seconds() / 3600
        storm = 5.2 * math.exp(-(((hours_ahead - 24) / 9) ** 2))
        kp = round(float(np.clip(round((1.7 + storm + rng.normal(0, 0.35)) * 3) / 3, 0, 9)), 2)
        readings.append(KpReading(t, kp, "observed" if t <= now else "predicted"))
    return readings
