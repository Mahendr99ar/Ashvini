"""CCSDS messages in KVN (keyword = value) form: CDM for conjunctions, OEM for ephemerides.

Operators already exchange these, so Ashvini reads what they have and writes what
their tools can open. Only the keywords Ashvini uses are interpreted; everything
else is kept in `extra` so nothing is silently dropped.
"""

import math
import re
from dataclasses import dataclass, field
from datetime import UTC, datetime

import numpy as np

from ..orbits.ephemeris import EphemerisTrack
from ..orbits.frames import j2000_to_teme_matrix
from ..timeutil import format_ccsds_time, parse_ccsds_time

KVN_LINE = re.compile(r"^\s*([A-Z0-9_]+)\s*=\s*(.*?)\s*(\[[^\]]*\])?\s*$")


class CcsdsError(ValueError):
    pass


@dataclass
class CdmObject:
    designator: str = ""
    name: str = ""
    intl_designator: str = ""
    object_type: str = ""
    sigma_r_m: float | None = None
    sigma_t_m: float | None = None
    sigma_n_m: float | None = None
    extra: dict = field(default_factory=dict)


@dataclass
class Cdm:
    message_id: str
    creation_date: datetime
    originator: str
    tca: datetime
    miss_distance_m: float
    relative_speed_ms: float | None
    collision_probability: float | None
    objects: tuple[CdmObject, CdmObject]
    extra: dict = field(default_factory=dict)

    @property
    def norad_ids(self) -> tuple[int | None, int | None]:
        return tuple(_as_int(o.designator) for o in self.objects)  # type: ignore[return-value]


def _as_int(text: str) -> int | None:
    digits = re.sub(r"\D", "", text or "")
    return int(digits) if digits else None


def _as_float(text: str | None) -> float | None:
    if text in (None, ""):
        return None
    return float(text)


def kvn_pairs(text: str) -> list[tuple[str, str]]:
    pairs = []
    for raw in text.splitlines():
        if not raw.strip() or raw.strip().startswith("COMMENT"):
            continue
        m = KVN_LINE.match(raw)
        if m:
            pairs.append((m.group(1), m.group(2)))
    return pairs


def parse_cdm(text: str) -> Cdm:
    header: dict[str, str] = {}
    objects: list[dict[str, str]] = []
    for key, value in kvn_pairs(text):
        if key == "OBJECT":
            objects.append({})
            continue
        (objects[-1] if objects else header)[key] = value
    if len(objects) != 2:
        raise CcsdsError("a CDM needs exactly two OBJECT sections")
    for required in ("TCA", "MISS_DISTANCE"):
        if required not in header:
            raise CcsdsError(f"CDM is missing {required}")

    def to_object(d: dict[str, str]) -> CdmObject:
        cr, ct, cn = (_as_float(d.get(k)) for k in ("CR_R", "CT_T", "CN_N"))
        known = {"OBJECT_DESIGNATOR", "OBJECT_NAME", "INTERNATIONAL_DESIGNATOR", "OBJECT_TYPE", "CR_R", "CT_T", "CN_N"}
        return CdmObject(
            designator=d.get("OBJECT_DESIGNATOR", ""), name=d.get("OBJECT_NAME", ""),
            intl_designator=d.get("INTERNATIONAL_DESIGNATOR", ""), object_type=d.get("OBJECT_TYPE", ""),
            sigma_r_m=math.sqrt(cr) if cr else None, sigma_t_m=math.sqrt(ct) if ct else None,
            sigma_n_m=math.sqrt(cn) if cn else None, extra={k: v for k, v in d.items() if k not in known},
        )

    known = {"MESSAGE_ID", "CREATION_DATE", "ORIGINATOR", "TCA", "MISS_DISTANCE", "RELATIVE_SPEED", "COLLISION_PROBABILITY"}
    return Cdm(
        message_id=header.get("MESSAGE_ID", ""),
        creation_date=parse_ccsds_time(header.get("CREATION_DATE", header["TCA"])),
        originator=header.get("ORIGINATOR", ""),
        tca=parse_ccsds_time(header["TCA"]),
        miss_distance_m=float(header["MISS_DISTANCE"]),
        relative_speed_ms=_as_float(header.get("RELATIVE_SPEED")),
        collision_probability=_as_float(header.get("COLLISION_PROBABILITY")),
        objects=(to_object(objects[0]), to_object(objects[1])),
        extra={k: v for k, v in header.items() if k not in known},
    )


def format_cdm(*, message_id: str, created: datetime, tca: datetime, miss_m: float, rel_speed_ms: float,
               pc: float, objects: list[dict], originator: str = "ASHVINI") -> str:
    """A minimal CDM 1.0 in KVN. `objects` holds designator, name, intl, type and sigmas in metres."""
    lines = [
        "CCSDS_CDM_VERS = 1.0",
        f"CREATION_DATE = {format_ccsds_time(created)}",
        f"ORIGINATOR = {originator}",
        f"MESSAGE_ID = {message_id}",
        f"TCA = {format_ccsds_time(tca)}",
        f"MISS_DISTANCE = {miss_m:.1f} [m]",
        f"RELATIVE_SPEED = {rel_speed_ms:.1f} [m/s]",
        f"COLLISION_PROBABILITY = {pc:.4e}",
        "COLLISION_PROBABILITY_METHOD = FOSTER-1992",
        "COMMENT Screening on public GP data. Not an official conjunction assessment.",
    ]
    for label, obj in zip(("OBJECT1", "OBJECT2"), objects, strict=True):
        lines += [
            f"OBJECT = {label}",
            f"OBJECT_DESIGNATOR = {obj['designator']}",
            "CATALOG_NAME = SATCAT",
            f"OBJECT_NAME = {obj['name']}",
            f"INTERNATIONAL_DESIGNATOR = {obj['intl']}",
            f"OBJECT_TYPE = {obj['type']}",
            "REF_FRAME = RTN",
            f"CR_R = {obj['sigma_r_m'] ** 2:.4e} [m**2]",
            f"CT_T = {obj['sigma_t_m'] ** 2:.4e} [m**2]",
            f"CN_N = {obj['sigma_n_m'] ** 2:.4e} [m**2]",
        ]
    return "\n".join(lines) + "\n"


def parse_oem(text: str) -> list[EphemerisTrack]:
    """Read every segment of an OEM. EME2000 states are rotated into TEME so SGP4 tracks and these agree."""
    tracks: list[EphemerisTrack] = []
    meta: dict[str, str] = {}
    rows: list[list[float]] = []
    epochs: list[datetime] = []
    in_meta = in_cov = False

    def close_segment() -> None:
        if not rows:
            return
        frame = meta.get("REF_FRAME", "").upper()
        states = np.array(rows, dtype=float)
        if frame in ("EME2000", "J2000", "GCRF", "ICRF"):
            states = np.array([rotate_state(s, j2000_to_teme_matrix(t)) for s, t in zip(states, epochs, strict=True)])
        elif frame != "TEME":
            raise CcsdsError(f"REF_FRAME {frame or '(missing)'} is not supported; send TEME or EME2000")
        norad = _as_int(meta.get("OBJECT_ID", "")) or 0
        stamps = np.array([t.timestamp() for t in epochs])
        tracks.append(EphemerisTrack(norad, meta.get("OBJECT_NAME", f"OBJECT {norad}"), stamps, states))

    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("COMMENT"):
            continue
        if line == "META_START":
            close_segment()
            meta, rows, epochs, in_meta = {}, [], [], True
            continue
        if line == "META_STOP":
            in_meta = False
            continue
        if line in ("COVARIANCE_START", "COVARIANCE_STOP"):
            in_cov = line == "COVARIANCE_START"
            continue
        if in_meta:
            m = KVN_LINE.match(line)
            if m:
                meta[m.group(1)] = m.group(2)
            continue
        if in_cov or "=" in line:
            continue
        parts = line.split()
        if len(parts) >= 7:
            epochs.append(parse_ccsds_time(parts[0]))
            rows.append([float(x) for x in parts[1:7]])
    close_segment()
    if not tracks:
        raise CcsdsError("no ephemeris data lines found")
    return tracks


def rotate_state(state: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    return np.concatenate([matrix @ state[:3], matrix @ state[3:]])


def format_oem(track: EphemerisTrack, originator: str = "ASHVINI") -> str:
    lines = ["CCSDS_OEM_VERS = 2.0", f"CREATION_DATE = {format_ccsds_time(datetime.now(UTC))}",
             f"ORIGINATOR = {originator}", "META_START", f"OBJECT_NAME = {track.name}", f"OBJECT_ID = {track.norad_id}",
             "CENTER_NAME = EARTH", "REF_FRAME = TEME", "TIME_SYSTEM = UTC", "META_STOP"]
    for t, s in zip(track.epochs, track.states, strict=True):
        stamp = format_ccsds_time(datetime.fromtimestamp(float(t), UTC))
        lines.append(stamp + " " + " ".join(f"{x:.6f}" for x in s))
    return "\n".join(lines) + "\n"
