"""Operator ephemerides: a table of states that replaces public TLEs for the operator's own objects."""

from dataclasses import dataclass
from datetime import datetime

import numpy as np

from ..timeutil import as_utc


@dataclass
class EphemerisTrack:
    norad_id: int
    name: str
    epochs: np.ndarray  # POSIX seconds, increasing
    states: np.ndarray  # (n, 6) TEME km and km/s

    @property
    def start(self) -> float:
        return float(self.epochs[0])

    @property
    def stop(self) -> float:
        return float(self.epochs[-1])

    def covers(self, when: datetime) -> bool:
        t = as_utc(when).timestamp()
        return self.start <= t <= self.stop

    def state(self, when: datetime) -> tuple[np.ndarray, np.ndarray]:
        return hermite_state(self.epochs, self.states, as_utc(when).timestamp())

    def positions(self, start: datetime, offsets: np.ndarray) -> np.ndarray:
        t0 = as_utc(start).timestamp()
        return np.array([hermite_state(self.epochs, self.states, t0 + s)[0] for s in offsets])


def hermite_state(epochs: np.ndarray, states: np.ndarray, t: float) -> tuple[np.ndarray, np.ndarray]:
    """Cubic Hermite between the two neighbouring points, using both positions and velocities."""
    if t < epochs[0] or t > epochs[-1]:
        nan = np.full(3, np.nan)
        return nan, nan.copy()
    k = int(np.clip(np.searchsorted(epochs, t) - 1, 0, len(epochs) - 2))
    t0, t1 = epochs[k], epochs[k + 1]
    h = t1 - t0
    s = (t - t0) / h
    p0, v0, p1, v1 = states[k, :3], states[k, 3:], states[k + 1, :3], states[k + 1, 3:]
    h00, h10, h01, h11 = 2 * s**3 - 3 * s**2 + 1, s**3 - 2 * s**2 + s, -2 * s**3 + 3 * s**2, s**3 - s**2
    pos = h00 * p0 + h10 * h * v0 + h01 * p1 + h11 * h * v1
    d00, d10, d01, d11 = 6 * s**2 - 6 * s, 3 * s**2 - 4 * s + 1, -6 * s**2 + 6 * s, 3 * s**2 - 2 * s
    vel = (d00 * p0 + d10 * h * v0 + d01 * p1 + d11 * h * v1) / h
    return pos, vel
