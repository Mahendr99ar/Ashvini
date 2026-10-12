"""Find close approaches between a set of primaries and the whole catalogue.

Three passes, cheapest first:

1. Apogee/perigee filter (Hoots, Crawford and Roehrich, 1984). Two orbits whose
   altitude bands do not overlap can never meet, which removes most pairs.
2. Coarse sieve. Positions on a fixed time grid go into a k-d tree. Anything within
   `report_km` plus the distance two objects can close in half a step is a candidate.
3. Refinement. For each pass of each candidate pair, the miss distance is minimised
   with Brent's method in the two steps around the closest grid sample.
"""

from collections import defaultdict
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timedelta

import numpy as np
from scipy.optimize import minimize_scalar
from scipy.spatial import cKDTree

from ..orbits.elements import OrbitRecord
from ..orbits.ephemeris import EphemerisTrack
from ..orbits.frames import ric_basis
from ..orbits.propagate import propagate_many, state_at_offset
from ..timeutil import time_grid, to_jd

MAX_CLOSING_SPEED_KMS = 15.8  # two circular LEO orbits head-on
CHUNK_STEPS = 360

StateFn = Callable[[float], tuple[np.ndarray, np.ndarray]]


@dataclass
class Encounter:
    primary_id: int
    secondary_id: int
    tca: datetime
    miss_km: float
    rel_speed_kms: float
    r1: np.ndarray = field(repr=False)
    v1: np.ndarray = field(repr=False)
    r2: np.ndarray = field(repr=False)
    v2: np.ndarray = field(repr=False)

    @property
    def miss_ric_km(self) -> np.ndarray:
        """Miss vector in the primary's radial / in-track / cross-track frame."""
        return ric_basis(self.r1, self.v1) @ (self.r2 - self.r1)


def altitude_band(records: Sequence[OrbitRecord]) -> tuple[np.ndarray, np.ndarray]:
    return np.array([r.perigee_km for r in records]), np.array([r.apogee_km for r in records])


def candidate_secondaries(primaries: Sequence[OrbitRecord], catalogue: Sequence[OrbitRecord], pad_km: float) -> list[OrbitRecord]:
    """Objects whose altitude band overlaps at least one primary's band."""
    if not primaries:
        return []
    p_lo, p_hi = altitude_band(primaries)
    s_lo, s_hi = altitude_band(catalogue)
    gap = np.maximum(p_lo[:, None], s_lo[None, :]) - np.minimum(p_hi[:, None], s_hi[None, :])
    keep = (gap <= pad_km).any(axis=0)
    return [rec for rec, k in zip(catalogue, keep, strict=True) if k]


def coarse_radius(report_km: float, step_s: float) -> float:
    return report_km + MAX_CLOSING_SPEED_KMS * step_s / 2


def sieve(p_pos: np.ndarray, s_pos: np.ndarray, radius: float, step0: int, hits: dict) -> None:
    """For every time step, record (primary, secondary) index pairs closer than `radius`."""
    for t in range(p_pos.shape[1]):
        s_ok = ~np.isnan(s_pos[:, t, 0])
        p_ok = ~np.isnan(p_pos[:, t, 0])
        if not s_ok.any() or not p_ok.any():
            continue
        s_index = np.flatnonzero(s_ok)
        p_index = np.flatnonzero(p_ok)
        tree = cKDTree(s_pos[s_ok, t])
        for pi, near in zip(p_index, tree.query_ball_point(p_pos[p_ok, t], radius), strict=True):
            for j in near:
                si = int(s_index[j])
                d = float(np.linalg.norm(s_pos[si, t] - p_pos[pi, t]))
                hits[(int(pi), si)].append((step0 + t, d))


def split_passes(samples: list[tuple[int, float]]) -> list[list[tuple[int, float]]]:
    samples.sort()
    passes, current = [], [samples[0]]
    for s in samples[1:]:
        if s[0] == current[-1][0] + 1:
            current.append(s)
        else:
            passes.append(current)
            current = [s]
    passes.append(current)
    return passes


def refine(f1: StateFn, f2: StateFn, center_s: float, step_s: float) -> tuple[float, float]:
    """Time (seconds from start) and distance of closest approach near `center_s`."""
    def distance(t: float) -> float:
        d = np.linalg.norm(f2(t)[0] - f1(t)[0])
        return float(d) if np.isfinite(d) else 1e9

    best = minimize_scalar(distance, bounds=(center_s - step_s, center_s + step_s), method="bounded",
                           options={"xatol": 1e-3})
    return float(best.x), float(best.fun)


def sgp4_state_fn(rec: OrbitRecord, start: datetime) -> StateFn:
    jd0, fr0 = to_jd(start)
    return lambda s: state_at_offset(rec, jd0, fr0, s)


def ephemeris_state_fn(track: EphemerisTrack, start: datetime) -> StateFn:
    return lambda s: track.state(start + timedelta(seconds=s))


def primary_positions(primaries: Sequence[OrbitRecord], ephem: dict[int, EphemerisTrack], start: datetime,
                      offsets: np.ndarray, jd: np.ndarray, fr: np.ndarray) -> np.ndarray:
    pos, _ = propagate_many(primaries, jd, fr)
    t0 = start + timedelta(seconds=float(offsets[0]))
    for i, rec in enumerate(primaries):
        track = ephem.get(rec.norad_id)
        if track is not None and track.covers(t0):
            pos[i] = track.positions(start, offsets)
    return pos


def screen(primaries: Sequence[OrbitRecord], catalogue: Sequence[OrbitRecord], start: datetime, days: float,
           step_s: float = 30.0, report_km: float = 5.0,
           ephemerides: dict[int, EphemerisTrack] | None = None) -> list[Encounter]:
    ephem = ephemerides or {}
    radius = coarse_radius(report_km, step_s)
    secondaries = candidate_secondaries(primaries, catalogue, pad_km=report_km + 10)
    offsets, jd, fr = time_grid(start, days * 86400, step_s)

    hits: dict[tuple[int, int], list[tuple[int, float]]] = defaultdict(list)
    for c0 in range(0, len(offsets), CHUNK_STEPS):
        sl = slice(c0, c0 + CHUNK_STEPS)
        p_pos = primary_positions(primaries, ephem, start, offsets[sl], jd[sl], fr[sl])
        s_pos, _ = propagate_many(secondaries, jd[sl], fr[sl])
        sieve(p_pos, s_pos, radius, c0, hits)

    found: dict[tuple[int, int, int], Encounter] = {}
    for (pi, si), samples in hits.items():
        p, s = primaries[pi], secondaries[si]
        if p.norad_id == s.norad_id:
            continue
        f1 = ephemeris_state_fn(ephem[p.norad_id], start) if p.norad_id in ephem else sgp4_state_fn(p, start)
        f2 = sgp4_state_fn(s, start)
        for one_pass in split_passes(samples):
            k = min(one_pass, key=lambda x: x[1])[0]
            t_s, miss = refine(f1, f2, float(offsets[k]), step_s)
            if miss > report_km:
                continue
            r1, v1 = f1(t_s)
            r2, v2 = f2(t_s)
            enc = Encounter(p.norad_id, s.norad_id, start + timedelta(seconds=t_s), miss,
                            float(np.linalg.norm(v2 - v1)), r1, v1, r2, v2)
            key = (*sorted((p.norad_id, s.norad_id)), round(t_s / 60))
            if key not in found or found[key].miss_km > miss:
                found[key] = enc
    return sorted(found.values(), key=lambda e: e.tca)
