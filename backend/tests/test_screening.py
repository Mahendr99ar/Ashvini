import math
from datetime import timedelta

import numpy as np

from ashvini.screening.covariance import Sigma, inertial_covariance, sigma_for
from ashvini.screening.pc import collision_probability, gaussian_mass_in_circle, max_pc
from ashvini.screening.screen import candidate_secondaries, screen

from .conftest import NOW


def test_pc_matches_the_closed_form_for_a_centred_circular_gaussian():
    sigma, radius = 0.1, 0.02
    pc = gaussian_mass_in_circle(np.zeros(2), np.eye(2) * sigma**2, radius)
    assert abs(pc - (1 - math.exp(-(radius**2) / (2 * sigma**2)))) < 1e-9


def test_pc_falls_as_the_miss_grows():
    cov = np.diag([0.1, 0.5, 0.1]) ** 2
    v = np.array([0.0, 0.0, 14.0])
    near, _ = collision_probability(np.array([0.05, 0, 0]), v, cov, cov, 0.02)
    far, _ = collision_probability(np.array([1.0, 0, 0]), v, cov, cov, 0.02)
    assert near > far > 0


def test_max_pc_bounds_any_covariance():
    for sigma in (0.05, 0.2, 1.0, 3.0):
        pc = gaussian_mass_in_circle(np.array([0.5, 0.0]), np.eye(2) * sigma**2, 0.02)
        assert pc <= max_pc(0.5, 0.02) * 1.01


def test_uncertainty_grows_with_age_and_is_largest_along_track(world):
    rec = world.records[0]
    young, old = sigma_for(rec, 0.1), sigma_for(rec, 3.0)
    assert old.t_km > young.t_km and old.t_km > old.r_km
    cov = inertial_covariance(Sigma(0.1, 1.0, 0.1), np.array([7000.0, 0, 0]), np.array([0, 7.5, 0]))
    assert np.allclose(np.linalg.eigvalsh(cov), sorted([0.01, 1.0, 0.01]))


def test_apogee_perigee_filter_drops_objects_that_can_never_meet(world):
    leo = [r for r in world.records if r.name == "ISS (ZARYA)"]
    kept = candidate_secondaries(leo, world.records, pad_km=15)
    assert all(r.perigee_km < 450 for r in kept)
    assert not any(r.name.startswith("GEO-COM") for r in kept)


def test_screening_finds_every_planted_encounter_at_the_right_time_and_distance(world):
    ids = {e.primary_id for e in world.encounters}
    primaries = [r for r in world.records if r.norad_id in ids]
    found = screen(primaries, world.records, NOW, 2.0, 30, 5)
    for planted in world.encounters:
        match = [e for e in found if {e.primary_id, e.secondary_id} == {planted.primary_id, planted.secondary_id}]
        assert len(match) == 1, f"expected exactly one pass for {planted}"
        assert abs(match[0].tca - planted.tca) < timedelta(seconds=2)
        assert abs(match[0].miss_km - planted.miss_km) < 0.01
