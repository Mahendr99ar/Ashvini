from datetime import UTC, datetime, timedelta

import numpy as np
import pytest

from ashvini.ingest.ccsds import CcsdsError, format_cdm, format_oem, parse_cdm, parse_oem
from ashvini.orbits.ephemeris import EphemerisTrack
from ashvini.orbits.frames import j2000_to_teme_matrix
from ashvini.timeutil import parse_ccsds_time

SAMPLE_CDM = """CCSDS_CDM_VERS = 1.0
CREATION_DATE = 2026-10-10T06:00:00.000
ORIGINATOR = EXAMPLE OPS
MESSAGE_ID = EX-0001
TCA = 2026-10-11T02:00:00.000
MISS_DISTANCE = 210 [m]
RELATIVE_SPEED = 4980 [m/s]
COLLISION_PROBABILITY = 3.1e-04
OBJECT = OBJECT1
OBJECT_DESIGNATOR = 70439
OBJECT_NAME = TERRA-OBS 7
INTERNATIONAL_DESIGNATOR = 2022-010A
OBJECT_TYPE = PAYLOAD
CR_R = 1.0e+02 [m**2]
CT_T = 2.5e+05 [m**2]
CN_N = 4.0e+02 [m**2]
OBJECT = OBJECT2
OBJECT_DESIGNATOR = 71049
OBJECT_NAME = FY-1C DEB 48211
OBJECT_TYPE = DEBRIS
CR_R = 9.0e+04 [m**2]
CT_T = 4.0e+06 [m**2]
CN_N = 9.0e+04 [m**2]
"""


def test_reads_a_cdm():
    cdm = parse_cdm(SAMPLE_CDM)
    assert cdm.norad_ids == (70439, 71049)
    assert cdm.miss_distance_m == 210
    assert cdm.collision_probability == pytest.approx(3.1e-4)
    assert cdm.objects[0].sigma_t_m == pytest.approx(500)
    assert cdm.objects[1].object_type == "DEBRIS"


def test_cdm_round_trip_keeps_the_numbers():
    text = format_cdm(message_id="X", created=datetime(2026, 10, 10, tzinfo=UTC), tca=datetime(2026, 10, 11, 2, tzinfo=UTC),
                      miss_m=210, rel_speed_ms=4980, pc=3.1e-4,
                      objects=[{"designator": 1, "name": "A", "intl": "", "type": "PAYLOAD", "sigma_r_m": 10, "sigma_t_m": 500, "sigma_n_m": 20},
                               {"designator": 2, "name": "B", "intl": "", "type": "DEBRIS", "sigma_r_m": 300, "sigma_t_m": 2000, "sigma_n_m": 300}])
    back = parse_cdm(text)
    assert back.miss_distance_m == pytest.approx(210) and back.objects[1].sigma_t_m == pytest.approx(2000)


def test_a_cdm_without_two_objects_is_refused():
    with pytest.raises(CcsdsError):
        parse_cdm("TCA = 2026-10-11T02:00:00\nMISS_DISTANCE = 10\nOBJECT = OBJECT1\n")


def test_day_of_year_times():
    assert parse_ccsds_time("2026-283T04:30:00.5") == datetime(2026, 10, 10, 4, 30, 0, 500000, tzinfo=UTC)


def test_oem_in_eme2000_is_rotated_into_teme():
    t0 = datetime(2026, 10, 10, tzinfo=UTC)
    lines = ["CCSDS_OEM_VERS = 2.0", "META_START", "OBJECT_NAME = SAT", "OBJECT_ID = 12345", "CENTER_NAME = EARTH",
             "REF_FRAME = EME2000", "TIME_SYSTEM = UTC", "META_STOP"]
    for k in range(3):
        t = t0 + timedelta(minutes=k)
        lines.append(f"{t:%Y-%m-%dT%H:%M:%S}.000 7000 0 0 0 7.5 0")
    [track] = parse_oem("\n".join(lines))
    expected = j2000_to_teme_matrix(t0) @ np.array([7000.0, 0, 0])
    assert np.allclose(track.states[0, :3], expected)
    assert track.norad_id == 12345


def test_oem_in_an_unknown_frame_is_refused():
    with pytest.raises(CcsdsError, match="ITRF"):
        parse_oem("META_START\nREF_FRAME = ITRF\nOBJECT_ID = 1\nMETA_STOP\n2026-10-10T00:00:00 1 2 3 4 5 6\n")


def test_oem_round_trip():
    track = EphemerisTrack(5, "X", np.array([1.79e9, 1.79e9 + 60]), np.array([[7000, 0, 0, 0, 7.5, 0], [7000, 450, 0, -0.5, 7.4, 0]], float))
    [back] = parse_oem(format_oem(track))
    assert np.allclose(back.states, track.states) and np.allclose(back.epochs, track.epochs)
