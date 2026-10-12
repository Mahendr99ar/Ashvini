"""Orbit records: one object, its latest mean elements, and the numbers people ask about first."""

import math
from dataclasses import dataclass, field
from datetime import datetime

from sgp4 import exporter, omm
from sgp4.api import WGS72, Satrec

from ..constants import EARTH_RADIUS_KM, MINUTES_PER_DAY, MU_EARTH_KM3_S2
from ..timeutil import from_jd

PAYLOAD, DEBRIS, ROCKET_BODY, UNKNOWN = "PAYLOAD", "DEBRIS", "ROCKET BODY", "UNKNOWN"


@dataclass
class OrbitRecord:
    norad_id: int
    name: str
    object_type: str
    intl_designator: str
    satrec: Satrec = field(repr=False)
    source: str = "celestrak"

    @property
    def epoch(self) -> datetime:
        return from_jd(self.satrec.jdsatepoch, self.satrec.jdsatepochF)

    @property
    def apogee_km(self) -> float:
        return self.satrec.alta * EARTH_RADIUS_KM

    @property
    def perigee_km(self) -> float:
        return self.satrec.altp * EARTH_RADIUS_KM

    @property
    def inclination_deg(self) -> float:
        return math.degrees(self.satrec.inclo)

    @property
    def mean_motion_rev_day(self) -> float:
        return self.satrec.no_kozai * MINUTES_PER_DAY / (2 * math.pi)

    @property
    def period_min(self) -> float:
        return MINUTES_PER_DAY / self.mean_motion_rev_day

    def tle_lines(self) -> tuple[str, str]:
        return exporter.export_tle(self.satrec)


def object_type_from_name(name: str) -> str:
    """CelesTrak names debris 'X DEB' and upper stages 'X R/B'; everything else is a payload."""
    upper = name.upper()
    if " DEB" in upper or upper.endswith("DEB"):
        return DEBRIS
    if "R/B" in upper:
        return ROCKET_BODY
    return PAYLOAD


def semi_major_axis_km(mean_motion_rev_day: float) -> float:
    n_rad_s = mean_motion_rev_day * 2 * math.pi / 86400.0
    return (MU_EARTH_KM3_S2 / n_rad_s**2) ** (1 / 3)


def from_tle(name: str, line1: str, line2: str, source: str = "tle") -> OrbitRecord:
    sat = Satrec.twoline2rv(line1, line2, WGS72)
    intl = cospar_from_tle_field(line1[9:17])
    clean = name.strip().removeprefix("0 ").strip()
    return OrbitRecord(sat.satnum, clean, object_type_from_name(clean), intl, sat, source)


def from_omm(fields: dict, source: str = "celestrak") -> OrbitRecord:
    """CelesTrak's JSON GP records use the CCSDS OMM keyword names, which sgp4 reads directly."""
    sat = Satrec()
    omm.initialize(sat, {k: str(v) for k, v in fields.items()})
    name = str(fields.get("OBJECT_NAME", "")).strip()
    return OrbitRecord(
        norad_id=int(fields["NORAD_CAT_ID"]),
        name=name,
        object_type=str(fields.get("OBJECT_TYPE") or object_type_from_name(name)),
        intl_designator=str(fields.get("OBJECT_ID", "")),
        satrec=sat,
        source=source,
    )


def parse_tle_text(text: str, source: str = "tle") -> list[OrbitRecord]:
    """Read two-line or three-line element text; a name line is optional."""
    lines = [ln.rstrip() for ln in text.splitlines() if ln.strip()]
    records, name, i = [], "", 0
    while i < len(lines):
        line = lines[i]
        if line.startswith("1 ") and i + 1 < len(lines) and lines[i + 1].startswith("2 "):
            records.append(from_tle(name or f"NORAD {line[2:7].strip()}", line, lines[i + 1], source))
            name, i = "", i + 2
        else:
            name, i = line, i + 1
    return records


def cospar_from_tle_field(field_text: str) -> str:
    """TLE line 1 stores '98067A'; everyone else writes '1998-067A'."""
    raw = field_text.strip()
    if len(raw) < 5 or not raw[:5].isdigit():
        return raw
    yy = int(raw[:2])
    year = 1900 + yy if yy >= 57 else 2000 + yy
    return f"{year}-{raw[2:5]}{raw[5:]}"


def launch_key(intl_designator: str) -> str:
    """'1999-025ABC' -> '1999-025': the launch every piece from that event shares."""
    return intl_designator[:8] if len(intl_designator) >= 8 else intl_designator
