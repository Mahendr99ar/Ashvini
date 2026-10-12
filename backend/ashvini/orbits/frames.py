"""Reference frames.

SGP4 returns positions in TEME (true equator, mean equinox). Operators often send
ephemerides in EME2000 (J2000). Converting between them needs precession (IAU 1976),
nutation (the four largest IAU 1980 terms, good to about half an arcsecond) and the
equation of the equinoxes. Half an arcsecond is roughly 17 m at low-orbit radius,
well inside public-data uncertainty.
"""

import math
from datetime import datetime

import numpy as np

from ..timeutil import centuries_since_j2000, to_jd

ARCSEC = math.pi / (180 * 3600)


def rot_x(a: float) -> np.ndarray:
    c, s = math.cos(a), math.sin(a)
    return np.array([[1, 0, 0], [0, c, s], [0, -s, c]])


def rot_z(a: float) -> np.ndarray:
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, s, 0], [-s, c, 0], [0, 0, 1]])


def gmst_rad(when: datetime) -> float:
    """Greenwich mean sidereal time (IAU 1982), the angle SGP4's TEME frame is spun by."""
    jd, fr = to_jd(when)
    t = ((jd - 2451545.0) + fr) / 36525.0
    seconds = 67310.54841 + (876600 * 3600 + 8640184.812866) * t + 0.093104 * t**2 - 6.2e-6 * t**3
    return math.radians((seconds % 86400) / 240.0)


def precession_matrix(t: float) -> np.ndarray:
    """J2000 mean equator and equinox to mean of date (IAU 1976)."""
    zeta = (2306.2181 * t + 0.30188 * t**2 + 0.017998 * t**3) * ARCSEC
    z = (2306.2181 * t + 1.09468 * t**2 + 0.018203 * t**3) * ARCSEC
    theta = (2004.3109 * t - 0.42665 * t**2 - 0.041833 * t**3) * ARCSEC
    return rot_z(-z) @ rot_y(theta) @ rot_z(-zeta)


def rot_y(a: float) -> np.ndarray:
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, 0, -s], [0, 1, 0], [s, 0, c]])


def nutation_angles(t: float) -> tuple[float, float, float]:
    """Mean obliquity, nutation in longitude and in obliquity (radians), largest terms only."""
    omega = math.radians(125.04452 - 1934.136261 * t)
    sun = math.radians(280.4665 + 36000.7698 * t)
    moon = math.radians(218.3165 + 481267.8813 * t)
    eps0 = math.radians(23.439291 - 0.0130042 * t)
    dpsi = (-17.20 * math.sin(omega) - 1.32 * math.sin(2 * sun) - 0.23 * math.sin(2 * moon) + 0.21 * math.sin(2 * omega)) * ARCSEC
    deps = (9.20 * math.cos(omega) + 0.57 * math.cos(2 * sun) + 0.10 * math.cos(2 * moon) - 0.09 * math.cos(2 * omega)) * ARCSEC
    return eps0, dpsi, deps


def j2000_to_teme_matrix(when: datetime) -> np.ndarray:
    t = centuries_since_j2000(when)
    eps0, dpsi, deps = nutation_angles(t)
    nutation = rot_x(-(eps0 + deps)) @ rot_z(-dpsi) @ rot_x(eps0)
    equation_of_equinoxes = dpsi * math.cos(eps0)
    return rot_z(equation_of_equinoxes) @ nutation @ precession_matrix(t)


def teme_to_ecef(r_teme: np.ndarray, when: datetime) -> np.ndarray:
    """Polar motion is ignored; it moves things by metres."""
    return rot_z(gmst_rad(when)) @ np.asarray(r_teme)


def ecef_to_latlon(r_ecef: np.ndarray) -> tuple[float, float, float]:
    """Spherical latitude, longitude (degrees) and altitude (km). Good enough for a map label."""
    x, y, z = r_ecef
    radius = math.sqrt(x * x + y * y + z * z)
    lat = math.degrees(math.asin(z / radius))
    lon = math.degrees(math.atan2(y, x))
    return lat, lon, radius - 6378.135


def ric_basis(r: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Rows are radial, in-track and cross-track unit vectors of the orbit at (r, v)."""
    radial = r / np.linalg.norm(r)
    cross = np.cross(r, v)
    cross = cross / np.linalg.norm(cross)
    intrack = np.cross(cross, radial)
    return np.vstack([radial, intrack, cross])
