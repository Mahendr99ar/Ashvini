"""Propagation: many objects, many times, one call into the SGP4 C extension."""

from collections.abc import Sequence
from datetime import datetime, timedelta

import numpy as np
from sgp4.api import SatrecArray

from ..constants import SECONDS_PER_DAY
from ..timeutil import time_grid, to_jd
from .elements import OrbitRecord


def propagate_many(records: Sequence[OrbitRecord], jd: np.ndarray, fr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Positions and velocities in TEME, shaped (objects, times, 3). Failed points become NaN."""
    if not records:
        empty = np.empty((0, len(jd), 3))
        return empty, empty.copy()
    errors, r, v = SatrecArray([rec.satrec for rec in records]).sgp4(jd, fr)
    bad = errors != 0
    r[bad] = np.nan
    v[bad] = np.nan
    return r, v


def state_at(record: OrbitRecord, when: datetime) -> tuple[np.ndarray, np.ndarray]:
    jd, fr = to_jd(when)
    err, r, v = record.satrec.sgp4(jd, fr)
    if err:
        return np.full(3, np.nan), np.full(3, np.nan)
    return np.array(r), np.array(v)


def state_at_offset(record: OrbitRecord, jd0: float, fr0: float, seconds: float) -> tuple[np.ndarray, np.ndarray]:
    fr = fr0 + seconds / SECONDS_PER_DAY
    whole = np.floor(fr)
    err, r, v = record.satrec.sgp4(jd0 + whole, fr - whole)
    if err:
        return np.full(3, np.nan), np.full(3, np.nan)
    return np.array(r), np.array(v)


def ground_track(record: OrbitRecord, start: datetime, minutes: float, step_s: float = 60.0) -> list[tuple[str, float, float]]:
    """(time, lat, lon) points for a map line."""
    from .frames import ecef_to_latlon, teme_to_ecef

    offsets, jd, fr = time_grid(start, minutes * 60, step_s)
    r, _ = propagate_many([record], jd, fr)
    track = []
    for k, sec in enumerate(offsets):
        if np.isnan(r[0, k, 0]):
            continue
        when = start + timedelta(seconds=float(sec))
        lat, lon, _ = ecef_to_latlon(teme_to_ecef(r[0, k], when))
        track.append((when.isoformat(), round(lat, 3), round(lon, 3)))
    return track
