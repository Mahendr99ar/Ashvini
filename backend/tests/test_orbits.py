import math
from datetime import UTC, datetime

import numpy as np

from ashvini.orbits.elements import (
    cospar_from_tle_field,
    object_type_from_name,
    parse_tle_text,
    semi_major_axis_km,
)
from ashvini.orbits.ephemeris import EphemerisTrack
from ashvini.orbits.frames import ecef_to_latlon, gmst_rad, j2000_to_teme_matrix, precession_matrix, ric_basis
from ashvini.orbits.propagate import state_at

ISS_TLE = """ISS (ZARYA)
1 25544U 98067A   08264.51782528 -.00002182  00000-0 -11606-4 0  2927
2 25544  51.6416 247.4627 0006703 130.5360 325.0288 15.72125391563537
"""


def test_reads_a_three_line_element_set():
    [iss] = parse_tle_text(ISS_TLE)
    assert iss.norad_id == 25544
    assert iss.name == "ISS (ZARYA)"
    assert iss.intl_designator == "1998-067A"
    assert 330 < iss.perigee_km < 360 and 340 < iss.apogee_km < 370
    assert abs(iss.inclination_deg - 51.6416) < 1e-4


def test_names_tell_debris_and_rocket_bodies_apart():
    assert object_type_from_name("FENGYUN 1C DEB") == "DEBRIS"
    assert object_type_from_name("SL-16 R/B") == "ROCKET BODY"
    assert object_type_from_name("SENTINEL-1A") == "PAYLOAD"


def test_cospar_conversion_handles_both_centuries():
    assert cospar_from_tle_field("98067A  ") == "1998-067A"
    assert cospar_from_tle_field("21035A") == "2021-035A"


def test_semi_major_axis_of_a_geostationary_orbit():
    assert abs(semi_major_axis_km(1.00273791) - 42164) < 2


def test_precession_is_identity_at_j2000_and_rotations_stay_orthonormal():
    assert np.allclose(precession_matrix(0.0), np.eye(3))
    m = j2000_to_teme_matrix(datetime(2026, 10, 10, tzinfo=UTC))
    assert np.allclose(m @ m.T, np.eye(3), atol=1e-12)
    # about 26.8 years of precession is roughly 0.37 degrees
    angle = math.degrees(math.acos((np.trace(precession_matrix(0.268)) - 1) / 2))
    assert 0.3 < angle < 0.45


def test_ric_basis_is_right_handed():
    r, v = np.array([7000.0, 0, 0]), np.array([0, 7.5, 0.5])
    b = ric_basis(r, v)
    assert np.allclose(b @ b.T, np.eye(3))
    assert np.allclose(np.cross(b[0], b[1]), b[2])


def test_gmst_and_ground_position_are_sane():
    g = gmst_rad(datetime(2000, 1, 1, 12, tzinfo=UTC))
    assert abs(math.degrees(g) - 280.46) < 0.01
    lat, lon, alt = ecef_to_latlon(np.array([0.0, 6378.135 + 500, 0.0]))
    assert abs(lat) < 1e-9 and abs(lon - 90) < 1e-9 and abs(alt - 500) < 1e-9


def test_hermite_interpolation_tracks_sgp4_closely():
    [iss] = parse_tle_text(ISS_TLE)
    t0 = datetime(2008, 9, 20, 12, tzinfo=UTC)
    stamps = np.arange(0, 3600 * 3, 60.0) + t0.timestamp()
    states = []
    for s in stamps:
        r, v = state_at(iss, datetime.fromtimestamp(s, UTC))
        states.append(np.concatenate([r, v]))
    track = EphemerisTrack(25544, "ISS", stamps, np.array(states))
    when = datetime.fromtimestamp(stamps[10] + 31, UTC)
    truth, _ = state_at(iss, when)
    interp, _ = track.state(when)
    assert np.linalg.norm(truth - interp) < 0.01  # under 10 m between 60 s samples
