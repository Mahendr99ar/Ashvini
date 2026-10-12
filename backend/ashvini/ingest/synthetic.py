"""A synthetic catalogue for tests, offline demos and CI.

The shells copy the shape of the real population (broadband constellations near
550 km, sun-synchronous imagers, debris clouds near 800 km, navigation and
geostationary belts). On top of that, a few close approaches are planted at known
times with known miss distances, and one satellite gets a known manoeuvre in its
element history, so the tests can check that the pipeline finds exactly those.
"""

import math
from dataclasses import dataclass
from datetime import datetime, timedelta

import numpy as np
from scipy.optimize import minimize
from sgp4.api import WGS72, Satrec

from ..constants import EARTH_RADIUS_KM, MU_EARTH_KM3_S2
from ..orbits.elements import DEBRIS, PAYLOAD, ROCKET_BODY, OrbitRecord
from ..orbits.propagate import state_at
from ..timeutil import to_jd

SGP4_EPOCH_JD = 2433281.5  # 1949-12-31 00:00 UT, the zero point sgp4init counts from


@dataclass
class PlantedEncounter:
    primary_id: int
    secondary_id: int
    tca: datetime
    miss_km: float


@dataclass
class PlantedManoeuvre:
    norad_id: int
    when: datetime
    delta_a_km: float


@dataclass
class SyntheticWorld:
    records: list[OrbitRecord]
    encounters: list[PlantedEncounter]
    manoeuvres: list[PlantedManoeuvre]
    history: list[dict]


def mean_motion_rad_min(alt_km: float) -> float:
    a = EARTH_RADIUS_KM + alt_km
    return math.sqrt(MU_EARTH_KM3_S2 / a**3) * 60.0


def make_satrec(norad: int, epoch: datetime, alt_km: float, inc_deg: float, raan_deg: float,
                ma_deg: float, ecc: float = 0.0005, bstar: float = 2e-5, argp_deg: float = 0.0) -> Satrec:
    jd, fr = to_jd(epoch)
    sat = Satrec()
    sat.sgp4init(WGS72, "i", norad, (jd - SGP4_EPOCH_JD) + fr, bstar, 0.0, 0.0, ecc,
                 math.radians(argp_deg % 360), math.radians(inc_deg), math.radians(ma_deg % 360),
                 mean_motion_rad_min(alt_km), math.radians(raan_deg % 360))
    sat.classification = "U"
    return sat


def make_record(norad: int, name: str, kind: str, launch: str, sat: Satrec) -> OrbitRecord:
    sat.intldesg = launch.replace("-", "")[2:]
    return OrbitRecord(norad, name, kind, launch, sat, source="synthetic")


class Builder:
    def __init__(self, seed: int, now: datetime):
        self.rng = np.random.default_rng(seed)
        self.now = now
        self.next_id = 70001
        self.records: list[OrbitRecord] = []

    def epoch(self) -> datetime:
        return self.now - timedelta(hours=float(self.rng.uniform(2, 40)))

    def add(self, name: str, kind: str, launch: str, alt: float, inc: float, raan: float, ma: float, **kw) -> OrbitRecord:
        norad = self.next_id
        self.next_id += 1
        rec = make_record(norad, name, kind, launch, make_satrec(norad, self.epoch(), alt, inc, raan, ma, **kw))
        self.records.append(rec)
        return rec

    def shell(self, prefix: str, launch: str, alt: float, inc: float, planes: int, per: int, jitter: float = 1.5) -> None:
        for p in range(planes):
            for k in range(per):
                ma = k * 360 / per + p * 360 / per * 0.37
                self.add(f"{prefix}-{len(self.records):04d}", PAYLOAD, launch, alt + self.rng.uniform(-jitter, jitter), inc, p * 360 / planes, ma)

    def scatter(self, count: int, kind: str, name: str, launch: str, alt: tuple[float, float], inc: tuple[float, float]) -> None:
        for _ in range(count):
            self.add(f"{name} {self.rng.integers(10000, 99999)}", kind, launch, self.rng.uniform(*alt), self.rng.uniform(*inc),
                     self.rng.uniform(0, 360), self.rng.uniform(0, 360), ecc=float(self.rng.uniform(0.0003, 0.004)))


def unit_position(sat: Satrec, when: datetime) -> np.ndarray:
    jd, fr = to_jd(when)
    _, r, _ = sat.sgp4(jd, fr)
    r = np.array(r)
    return r / np.linalg.norm(r)


def plane_through(p: np.ndarray, inc_deg: float) -> tuple[float, float]:
    """RAAN and argument of latitude (degrees) of a circular orbit with this inclination passing through p."""
    i = math.radians(inc_deg)
    u = math.asin(max(-1.0, min(1.0, p[2] / math.sin(i))))
    raan = math.atan2(p[1], p[0]) - math.atan2(math.sin(u) * math.cos(i), math.cos(u))
    return math.degrees(raan), math.degrees(u)


def plant_encounter(b: Builder, primary: OrbitRecord, tca: datetime, miss_km: float, inc_deg: float, kind: str, name: str) -> PlantedEncounter:
    """Add a secondary whose SGP4 track passes `miss_km` from the primary at `tca`.

    The secondary is slightly eccentric, with perigee at the crossing point and a
    mean altitude 40 km higher, so the two meet once rather than every orbit.
    """
    r_primary, _ = state_at(primary, tca)
    r_target = float(np.linalg.norm(r_primary))
    a = r_target + 40.0
    ecc = 1 - r_target / a
    epoch = b.epoch()
    raan0, u0 = plane_through(r_primary / r_target, inc_deg)
    n_deg_s = math.degrees(mean_motion_rad_min(a - EARTH_RADIUS_KM)) / 60.0
    ma0 = -n_deg_s * (tca - epoch).total_seconds()
    norad = b.next_id
    b.next_id += 1
    jd, fr = to_jd(tca)

    def build(params: np.ndarray) -> Satrec:
        return make_satrec(norad, epoch, a - EARTH_RADIUS_KM, inc_deg, params[0], params[1],
                           ecc=ecc + params[3] * 1e-4, argp_deg=params[2])

    # aim the secondary so that, at TCA, it sits `miss_km` straight above the primary
    goal = r_primary + miss_km * r_primary / r_target
    x = np.array([raan0, ma0, u0, 0.0])
    for _ in range(4):
        x = minimize(lambda p: float(np.sum((_position(build(p), jd, fr) - goal) ** 2)), x0=x, method="Nelder-Mead",
                     options={"xatol": 1e-10, "fatol": 1e-14, "maxiter": 8000}).x
    x_target = x
    b.records.append(make_record(norad, name, kind, "1999-025", build(x_target)))
    return PlantedEncounter(primary.norad_id, norad, tca, miss_km)


def _position(sat: Satrec, jd: float, fr: float) -> np.ndarray:
    _, r, _ = sat.sgp4(jd, fr)
    return np.array(r)


def element_history(rec: OrbitRecord, now: datetime, rng: np.random.Generator, jump: PlantedManoeuvre | None) -> list[dict]:
    """Ten element sets, twelve hours apart, with slow drag decay and, optionally, one orbit raise."""
    a_now = EARTH_RADIUS_KM + (rec.apogee_km + rec.perigee_km) / 2
    rows = []
    for k in range(10, 0, -1):
        epoch = now - timedelta(hours=12 * k)
        a = a_now + 0.004 * k + rng.normal(0, 0.002)
        if jump and epoch < jump.when:
            a -= jump.delta_a_km
        n_rev_day = math.sqrt(MU_EARTH_KM3_S2 / a**3) * 86400 / (2 * math.pi)
        rows.append({"norad_id": rec.norad_id, "epoch": epoch, "mean_motion": n_rev_day,
                     "eccentricity": rec.satrec.ecco, "inclination": rec.inclination_deg})
    return rows


def build_world(seed: int = 7, size: int = 2500, now: datetime | None = None) -> SyntheticWorld:
    from ..timeutil import utc_now

    now = now or utc_now()
    b = Builder(seed, now)
    scale = max(size, 400) / 2500

    b.add("ISS (ZARYA)", PAYLOAD, "1998-067A", 418, 51.64, 212, 40)
    b.add("CSS (TIANHE)", PAYLOAD, "2021-035A", 385, 41.47, 95, 210)
    b.shell("LUMEN", "2024-101", 550, 53.0, max(4, int(36 * scale)), 20)
    b.shell("ARC", "2023-045", 1200, 87.9, max(3, int(12 * scale)), 18)
    for k in range(int(220 * scale)):
        alt = float(b.rng.uniform(480, 790))
        b.add(f"TERRA-OBS {k + 1}", PAYLOAD, "2022-010", alt, 96.6 + (alt - 400) * 0.0042, b.rng.uniform(0, 360), b.rng.uniform(0, 360))
    for q in range(6):
        for k in range(4):
            b.add(f"NAV-G{q * 4 + k + 1:02d}", PAYLOAD, "2018-109", 20180, 55.0, q * 60, k * 90 + q * 15)
    for k in range(int(60 * scale)):
        b.add(f"GEO-COM {k + 1}", PAYLOAD, "2019-001", 35786, float(b.rng.uniform(0, 0.1)), b.rng.uniform(0, 360), b.rng.uniform(0, 360))
    b.scatter(int(420 * scale), DEBRIS, "FY-1C DEB", "1999-025", (700, 1050), (98.4, 99.2))
    b.scatter(int(240 * scale), DEBRIS, "COSMOS 2251 DEB", "1993-036", (760, 900), (74.0, 74.1))
    b.scatter(int(160 * scale), DEBRIS, "DEB", "2021-000", (350, 1500), (20, 100))
    b.scatter(int(140 * scale), ROCKET_BODY, "R/B", "2010-000", (350, 1700), (28, 100))

    eo = next(r for r in b.records if r.name == "TERRA-OBS 7")
    lumen = next(r for r in b.records if r.name.startswith("LUMEN-0030"))
    iss = b.records[0]
    encounters = [
        plant_encounter(b, eo, now + timedelta(hours=20), 0.21, 60.0, DEBRIS, "FY-1C DEB 48211"),
        plant_encounter(b, lumen, now + timedelta(hours=41), 1.40, 82.0, ROCKET_BODY, "SL-8 R/B 23119"),
        plant_encounter(b, iss, now + timedelta(hours=9), 3.20, 74.0, DEBRIS, "COSMOS 2251 DEB 34427"),
    ]
    manoeuvre = PlantedManoeuvre(eo.norad_id, now - timedelta(hours=30), 0.35)
    history = []
    for rec in [eo, lumen, iss]:
        history.extend(element_history(rec, now, b.rng, manoeuvre if rec is eo else None))
    return SyntheticWorld(b.records, encounters, [manoeuvre], history)
