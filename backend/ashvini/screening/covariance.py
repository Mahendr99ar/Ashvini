"""Position uncertainty for public element sets.

TLEs carry no covariance, so Ashvini uses a simple growth model: an uncertainty at
epoch that grows linearly with the age of the elements, largest along-track. The
numbers sit in the range reported by Flohrer et al. (2008), "Assessment and
categorization of TLE orbit errors for the US SSN catalogue", and are deliberately
rough. When an operator uploads a CDM, its real covariance replaces this model.
"""

from dataclasses import dataclass

import numpy as np

from ..orbits.elements import DEBRIS, OrbitRecord
from ..orbits.frames import ric_basis


@dataclass(frozen=True)
class Sigma:
    r_km: float
    t_km: float
    n_km: float

    def scaled(self, k: float) -> "Sigma":
        return Sigma(self.r_km * k, self.t_km * k, self.n_km * k)


# (sigma at epoch, growth per day), radial / in-track / cross-track, km
REGIMES = {
    "LEO": (Sigma(0.10, 0.40, 0.10), Sigma(0.08, 1.20, 0.08)),
    "MEO": (Sigma(0.50, 2.00, 0.50), Sigma(0.20, 3.00, 0.20)),
    "GEO": (Sigma(1.00, 5.00, 1.00), Sigma(0.30, 4.00, 0.30)),
}
DEBRIS_FACTOR = 1.5


def regime(record: OrbitRecord) -> str:
    if record.perigee_km < 2000:
        return "LEO"
    if record.apogee_km > 30000:
        return "GEO"
    return "MEO"


def sigma_for(record: OrbitRecord, age_days: float) -> Sigma:
    base, growth = REGIMES[regime(record)]
    age = max(0.0, age_days)
    sigma = Sigma(base.r_km + growth.r_km * age, base.t_km + growth.t_km * age, base.n_km + growth.n_km * age)
    return sigma.scaled(DEBRIS_FACTOR) if record.object_type == DEBRIS else sigma


def ric_covariance(sigma: Sigma) -> np.ndarray:
    return np.diag([sigma.r_km**2, sigma.t_km**2, sigma.n_km**2])


def inertial_covariance(sigma: Sigma, r: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Rotate a diagonal RIC covariance into the inertial frame of r and v."""
    basis = ric_basis(r, v)
    return basis.T @ ric_covariance(sigma) @ basis
