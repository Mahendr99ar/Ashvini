"""Probability of collision for a short encounter (Foster and Estes, 1992).

At high relative speed the two objects pass so quickly that their relative motion is
a straight line. Project the combined position covariance onto the plane
perpendicular to the relative velocity, and Pc is the probability mass of that 2-D
Gaussian inside a circle of the combined hard-body radius, centred on the miss vector.
"""

import math

import numpy as np

GL_NODES, GL_WEIGHTS = np.polynomial.legendre.leggauss(48)


def encounter_plane(rel_pos: np.ndarray, rel_vel: np.ndarray) -> np.ndarray:
    """2x3 matrix whose rows span the plane perpendicular to the relative velocity."""
    z = rel_vel / np.linalg.norm(rel_vel)
    x = rel_pos - np.dot(rel_pos, z) * z
    if np.linalg.norm(x) < 1e-12:
        x = np.cross(z, [1.0, 0.0, 0.0])
        if np.linalg.norm(x) < 1e-12:
            x = np.cross(z, [0.0, 1.0, 0.0])
    x = x / np.linalg.norm(x)
    y = np.cross(z, x)
    return np.vstack([x, y])


def gaussian_mass_in_circle(center: np.ndarray, cov: np.ndarray, radius: float, n_theta: int = 96) -> float:
    """Integrate a 2-D Gaussian over a disc using polar Gauss-Legendre quadrature."""
    inv = np.linalg.inv(cov)
    norm = 1.0 / (2 * math.pi * math.sqrt(np.linalg.det(cov)))
    rho = (GL_NODES + 1) * radius / 2
    w_rho = GL_WEIGHTS * radius / 2
    theta = np.linspace(0, 2 * math.pi, n_theta, endpoint=False)
    pts_x = rho[:, None] * np.cos(theta)[None, :] - center[0]
    pts_y = rho[:, None] * np.sin(theta)[None, :] - center[1]
    q = inv[0, 0] * pts_x**2 + 2 * inv[0, 1] * pts_x * pts_y + inv[1, 1] * pts_y**2
    integrand = np.exp(-0.5 * q) * rho[:, None]
    return float(norm * (w_rho[:, None] * integrand).sum() * (2 * math.pi / n_theta))


def collision_probability(rel_pos_km: np.ndarray, rel_vel_kms: np.ndarray, cov1_km2: np.ndarray,
                          cov2_km2: np.ndarray, hard_body_radius_km: float) -> tuple[float, float]:
    """Pc and the Mahalanobis distance of the miss in the encounter plane."""
    plane = encounter_plane(rel_pos_km, rel_vel_kms)
    miss_2d = plane @ rel_pos_km
    cov_2d = plane @ (cov1_km2 + cov2_km2) @ plane.T
    pc = gaussian_mass_in_circle(miss_2d, cov_2d, hard_body_radius_km)
    mahalanobis = float(math.sqrt(miss_2d @ np.linalg.inv(cov_2d) @ miss_2d))
    return min(1.0, max(0.0, pc)), mahalanobis


def max_pc(miss_km: float, hard_body_radius_km: float) -> float:
    """Worst case over covariance size for a known miss distance (Alfano, 2005), circular covariance."""
    if miss_km <= 0:
        return 1.0
    return min(1.0, hard_body_radius_km**2 / (math.e * miss_km**2))


def log10_or_floor(p: float, floor: float = -30.0) -> float:
    return math.log10(p) if p > 10**floor else floor
