"""Turn a raw encounter into numbers an operator can act on: covariance, Pc, and the features the risk model reads."""

from dataclasses import dataclass
from datetime import datetime

import numpy as np

from ..orbits.elements import OrbitRecord
from ..screening.covariance import Sigma, inertial_covariance, sigma_for
from ..screening.pc import collision_probability, log10_or_floor, max_pc
from ..screening.screen import Encounter


@dataclass
class Assessment:
    pc: float
    log_pc: float
    max_pc: float
    mahalanobis: float
    sigma1: Sigma
    sigma2: Sigma
    age1_days: float
    age2_days: float
    miss_ric_km: np.ndarray

    @property
    def data_age_days(self) -> float:
        return max(self.age1_days, self.age2_days)


def element_age_days(record: OrbitRecord, at: datetime) -> float:
    return (at - record.epoch).total_seconds() / 86400.0


def assess(enc: Encounter, primary: OrbitRecord, secondary: OrbitRecord, hard_body_radius_m: float,
           primary_sigma: Sigma | None = None) -> Assessment:
    """`primary_sigma` lets an operator's own ephemeris (usually far better than a TLE) set the primary's uncertainty."""
    age1 = element_age_days(primary, enc.tca)
    age2 = element_age_days(secondary, enc.tca)
    s1 = primary_sigma or sigma_for(primary, age1)
    s2 = sigma_for(secondary, age2)
    cov1 = inertial_covariance(s1, enc.r1, enc.v1)
    cov2 = inertial_covariance(s2, enc.r2, enc.v2)
    hbr_km = hard_body_radius_m / 1000.0
    pc, mahal = collision_probability(enc.r2 - enc.r1, enc.v2 - enc.v1, cov1, cov2, hbr_km)
    return Assessment(pc, log10_or_floor(pc), max_pc(enc.miss_km, hbr_km), mahal, s1, s2, age1, age2, enc.miss_ric_km)


def model_features(enc: Encounter, a: Assessment, now: datetime, secondary_type: str) -> dict[str, float]:
    """The subset of ESA challenge features that can also be computed from public data.

    Names follow the Kelvins dataset so the same model can be trained there and served here.
    """
    ric = a.miss_ric_km * 1000.0
    return {
        "time_to_tca": (enc.tca - now).total_seconds() / 86400.0,
        "risk": a.log_pc,
        "max_risk_estimate": log10_or_floor(a.max_pc),
        "miss_distance": enc.miss_km * 1000.0,
        "relative_speed": enc.rel_speed_kms * 1000.0,
        "relative_position_r": float(ric[0]),
        "relative_position_t": float(ric[1]),
        "relative_position_n": float(ric[2]),
        "mahalanobis_distance": a.mahalanobis,
        "t_sigma_r": a.sigma1.r_km * 1000, "t_sigma_t": a.sigma1.t_km * 1000, "t_sigma_n": a.sigma1.n_km * 1000,
        "c_sigma_r": a.sigma2.r_km * 1000, "c_sigma_t": a.sigma2.t_km * 1000, "c_sigma_n": a.sigma2.n_km * 1000,
        "c_object_type_debris": 1.0 if secondary_type == "DEBRIS" else 0.0,
        "c_object_type_rocket_body": 1.0 if secondary_type == "ROCKET BODY" else 0.0,
    }
