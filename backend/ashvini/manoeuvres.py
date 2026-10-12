"""Spot orbit changes in an object's element history.

Drag lowers an orbit slowly and smoothly. A thruster firing moves the semi-major
axis in one step. This baseline compares each step with the typical step size for
that object (median absolute deviation) and flags the outliers. The learned
detector in `ml/manoeuvre` is measured against this.
"""

from dataclasses import dataclass
from datetime import datetime

import numpy as np

from .orbits.elements import semi_major_axis_km


@dataclass
class ManoeuvreEvent:
    norad_id: int
    after: datetime
    before: datetime
    delta_a_km: float
    score: float  # step size in units of typical step

    @property
    def direction(self) -> str:
        return "raised" if self.delta_a_km > 0 else "lowered"


def semi_major_axes(history: list[dict]) -> tuple[list[datetime], np.ndarray]:
    rows = sorted(history, key=lambda r: r["epoch"])
    return [r["epoch"] for r in rows], np.array([semi_major_axis_km(r["mean_motion"]) for r in rows])


def robust_scale(values: np.ndarray) -> float:
    if len(values) == 0:
        return 0.0
    mad = float(np.median(np.abs(values - np.median(values))))
    return 1.4826 * mad


def detect(history: list[dict], min_km: float = 0.08, k: float = 6.0) -> list[ManoeuvreEvent]:
    """Flag steps in semi-major axis that are both large in km and large for this object."""
    if len(history) < 4:
        return []
    epochs, a = semi_major_axes(history)
    steps = np.diff(a)
    typical = np.median(steps)
    scale = max(robust_scale(steps), 0.005)
    events = []
    for i, step in enumerate(steps):
        score = abs(step - typical) / scale
        if abs(step - typical) >= min_km and score >= k:
            events.append(ManoeuvreEvent(history[0]["norad_id"], epochs[i], epochs[i + 1], float(step - typical), float(score)))
    return events
