"""Time conversions between Python datetimes and the Julian dates SGP4 expects."""

from datetime import UTC, datetime, timedelta

import numpy as np
from sgp4.api import jday

from .constants import JD_J2000, SECONDS_PER_DAY


def utc_now() -> datetime:
    return datetime.now(UTC).replace(microsecond=0)


def as_utc(when: datetime) -> datetime:
    return when.replace(tzinfo=UTC) if when.tzinfo is None else when.astimezone(UTC)


def to_jd(when: datetime) -> tuple[float, float]:
    """Split Julian date (whole, fraction), the form SGP4 wants for precision."""
    w = as_utc(when)
    return jday(w.year, w.month, w.day, w.hour, w.minute, w.second + w.microsecond / 1e6)


def from_jd(jd: float, fr: float = 0.0) -> datetime:
    days = (jd - JD_J2000) + fr
    return datetime(2000, 1, 1, 12, tzinfo=UTC) + timedelta(days=days)


def time_grid(start: datetime, seconds: float, step_s: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Evenly spaced times as (offset seconds, jd, fraction) arrays."""
    offsets = np.arange(0.0, seconds + step_s, step_s)
    jd0, fr0 = to_jd(start)
    fr = fr0 + offsets / SECONDS_PER_DAY
    whole = np.floor(fr)
    return offsets, jd0 + whole, fr - whole


def centuries_since_j2000(when: datetime) -> float:
    jd, fr = to_jd(when)
    return ((jd - JD_J2000) + fr) / 36525.0


def parse_ccsds_time(text: str) -> datetime:
    """CCSDS times look like 2026-10-12T04:31:09.123 or 2026-285T04:31:09 (day of year)."""
    text = text.strip().rstrip("Z")
    if len(text) >= 9 and text[4] == "-" and text[8] in "T " and text[5:8].isdigit():
        year, doy = int(text[:4]), int(text[5:8])
        rest = text[9:] or "00:00:00"
        base = datetime(year, 1, 1, tzinfo=UTC) + timedelta(days=doy - 1)
        t = datetime.strptime(rest.split(".")[0], "%H:%M:%S")
        frac = float("0." + rest.split(".")[1]) if "." in rest else 0.0
        return base + timedelta(hours=t.hour, minutes=t.minute, seconds=t.second + frac)
    whole, _, frac = text.partition(".")
    base = datetime.strptime(whole.replace(" ", "T"), "%Y-%m-%dT%H:%M:%S").replace(tzinfo=UTC)
    return base + timedelta(seconds=float("0." + frac) if frac else 0.0)


def format_ccsds_time(when: datetime) -> str:
    return as_utc(when).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3]
