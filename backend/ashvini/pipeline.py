"""One full run: load orbits, screen, score, update events, look for manoeuvres, refresh space weather, alert.

Safe to run twice in a row. Events are matched to the ones already stored, so a
rerun updates numbers and appends to each event's history, and never touches a
person's decision.

    python -m ashvini.pipeline
"""

import argparse
import hashlib
import logging
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta
from pathlib import Path

import httpx
import numpy as np
from sqlalchemy import delete, func, or_, select
from sqlalchemy.orm import Session, sessionmaker

from . import manoeuvres
from .alerts import dispatch
from .config import Settings, get_settings
from .db import (
    Conjunction,
    ConjunctionUpdate,
    ElementHistory,
    Ephemeris,
    Event,
    KpValue,
    ManoeuvreRow,
    Run,
    SpaceObject,
    Watch,
    make_sessionmaker,
)
from .ingest import celestrak, spaceweather, synthetic
from .orbits.elements import OrbitRecord
from .orbits.ephemeris import EphemerisTrack
from .risk.assess import assess, model_features
from .risk.decision import Thresholds, recommend
from .risk.model import fallback_prediction, load_model
from .screening.covariance import Sigma
from .screening.screen import Encounter, screen
from .timeutil import as_utc, utc_now

log = logging.getLogger("ashvini.pipeline")
SAME_EVENT_WINDOW = timedelta(minutes=30)
OPERATOR_SIGMA = Sigma(0.02, 0.06, 0.02)  # a typical operator ephemeris, km


@dataclass
class RunSummary:
    run_id: int
    source: str
    objects: int
    primaries: int
    encounters: int
    events_created: int
    events_updated: int
    events_passed: int
    manoeuvres: int
    alerts_sent: int
    seconds: float


@dataclass
class Catalogue:
    records: list[OrbitRecord]
    history: list[dict]
    primary_ids: set[int]
    source: str


def load_catalogue(settings: Settings, cache_dir: Path, now: datetime) -> Catalogue:
    if settings.data_source == "synthetic":
        world = synthetic.build_world(settings.synthetic_seed, settings.synthetic_size, now)
        primaries = {e.primary_id for e in world.encounters} | {world.records[0].norad_id, world.records[1].norad_id}
        return Catalogue(world.records, world.history, primaries, "synthetic")
    groups = settings.group_list(settings.celestrak_groups)
    records = celestrak.fetch_catalogue(groups, cache_dir)
    primary_ids = set(settings.id_list(settings.primary_norad_ids))
    for group in settings.group_list(settings.primary_groups):
        primary_ids |= {r.norad_id for r in celestrak.fetch_catalogue([group], cache_dir)}
    history = [history_row(r) for r in records if r.object_type == "PAYLOAD"]
    return Catalogue(records, history, primary_ids, "celestrak")


def history_row(rec: OrbitRecord) -> dict:
    return {"norad_id": rec.norad_id, "epoch": rec.epoch, "mean_motion": rec.mean_motion_rev_day,
            "eccentricity": rec.satrec.ecco, "inclination": rec.inclination_deg}


def watched_ids(session: Session) -> set[int]:
    return set(session.scalars(select(Watch.norad_id)).all())


def load_ephemerides(session: Session, now: datetime) -> dict[int, EphemerisTrack]:
    tracks: dict[int, EphemerisTrack] = {}
    rows = session.scalars(select(Ephemeris).order_by(Ephemeris.uploaded_at)).all()
    for row in rows:
        if as_utc(row.stop) < now:
            continue
        tracks[row.norad_id] = EphemerisTrack(row.norad_id, row.name, np.array(row.epochs), np.array(row.states))
    return tracks


def upsert_objects(session: Session, records: list[OrbitRecord], run_id: int) -> None:
    existing = {o.norad_id: o for o in session.scalars(select(SpaceObject)).all()}
    for rec in records:
        line1, line2 = rec.tle_lines()
        values = dict(name=rec.name, object_type=rec.object_type, intl_designator=rec.intl_designator, epoch=rec.epoch,
                      line1=line1, line2=line2, perigee_km=rec.perigee_km, apogee_km=rec.apogee_km,
                      inclination_deg=rec.inclination_deg, period_min=rec.period_min, source=rec.source, run_id=run_id)
        obj = existing.get(rec.norad_id)
        if obj is None:
            session.add(SpaceObject(norad_id=rec.norad_id, **values))
        else:
            for k, v in values.items():
                setattr(obj, k, v)


HISTORY_DAYS = 21


def store_history(session: Session, rows: list[dict], now: datetime, only_new_objects: bool = False) -> set[int]:
    """Append element sets newer than what is stored, drop sets older than three weeks, and return the
    objects that got a new set. Debris and rocket bodies cannot manoeuvre, so callers pass payloads only.
    Synthetic history is written once per object, since every synthetic run invents fresh epochs."""
    latest = dict(session.execute(select(ElementHistory.norad_id, func.max(ElementHistory.epoch))
                                  .group_by(ElementHistory.norad_id)).all())
    changed: set[int] = set()
    for row in rows:
        known = latest.get(row["norad_id"])
        epoch = as_utc(row["epoch"]).replace(microsecond=0)
        if known is not None and (only_new_objects or epoch <= as_utc(known)):
            continue
        session.add(ElementHistory(**{**row, "epoch": epoch}))
        changed.add(row["norad_id"])
    session.execute(delete(ElementHistory).where(ElementHistory.epoch < now - timedelta(days=HISTORY_DAYS)))
    return changed


def new_event_id(a: int, b: int, tca: datetime, taken=lambda _id: False) -> str:
    """Stable id from the pair and the minute of closest approach. Neighbours in one shell can meet twice in
    an hour, so the minute matters; `taken` guards the rare clash."""
    lo, hi = sorted((a, b))
    salt = 0
    while True:
        candidate = hashlib.sha1(f"{lo}-{hi}-{tca:%Y%m%d%H%M}-{salt}".encode()).hexdigest()[:16]
        if not taken(candidate):
            return candidate
        salt += 1


def find_existing(session: Session, a: int, b: int, tca: datetime) -> Conjunction | None:
    pair = or_((Conjunction.primary_id == a) & (Conjunction.secondary_id == b),
               (Conjunction.primary_id == b) & (Conjunction.secondary_id == a))
    for c in session.scalars(select(Conjunction).where(pair)).all():
        if abs(as_utc(c.tca) - tca) <= SAME_EVENT_WINDOW:
            return c
    return None


def score(enc: Encounter, by_id: dict[int, OrbitRecord], ephem: dict[int, EphemerisTrack], settings: Settings,
          model, now: datetime) -> dict:
    primary, secondary = by_id[enc.primary_id], by_id[enc.secondary_id]
    sigma1 = OPERATOR_SIGMA if enc.primary_id in ephem else None
    a = assess(enc, primary, secondary, settings.hard_body_radius_m, sigma1)
    features = model_features(enc, a, now, secondary.object_type)
    prediction = model.predict(features) if model else fallback_prediction(a.log_pc)
    rec = recommend(a.log_pc, prediction.log_risk, prediction.high, prediction.source, enc.miss_km,
                    (enc.tca - now).total_seconds() / 3600, a.data_age_days,
                    Thresholds(settings.action_pc, settings.monitor_pc))
    return dict(
        primary_id=enc.primary_id, secondary_id=enc.secondary_id, tca=enc.tca, miss_km=enc.miss_km,
        rel_speed_kms=enc.rel_speed_kms, miss_ric_km=[round(float(x), 4) for x in a.miss_ric_km],
        pc=a.pc, log_pc=a.log_pc, max_pc=a.max_pc, mahalanobis=a.mahalanobis,
        sigmas={"primary": asdict(a.sigma1), "secondary": asdict(a.sigma2), "primary_source": "operator ephemeris" if sigma1 else "TLE age model"},
        data_age_days=a.data_age_days, predicted_log_risk=prediction.log_risk, predicted_low=prediction.low,
        predicted_high=prediction.high, prediction_source=prediction.source, recommendation=rec.level,
        reasons=rec.reasons, features=features,
    )


def save_event(session: Session, fields: dict, run_id: int, now: datetime) -> bool:
    """Insert or update one event and append its history row. Returns True when the event is new."""
    conj = find_existing(session, fields["primary_id"], fields["secondary_id"], fields["tca"])
    created = conj is None
    if created:
        event_id = new_event_id(fields["primary_id"], fields["secondary_id"], fields["tca"],
                                lambda i: session.get(Conjunction, i) is not None)
        conj = Conjunction(id=event_id,
                           first_run_id=run_id, created_at=now, **fields)
        session.add(conj)
        session.add(Event(kind="conjunction.created", payload={"id": conj.id, "recommendation": conj.recommendation}))
    else:
        escalated = fields["recommendation"] != conj.recommendation
        for k, v in fields.items():
            setattr(conj, k, v)
        if escalated:
            session.add(Event(kind="conjunction.changed", payload={"id": conj.id, "recommendation": conj.recommendation}))
    conj.last_run_id, conj.updated_at, conj.status = run_id, now, "open"
    session.flush()
    session.add(ConjunctionUpdate(conjunction_id=conj.id, run_id=run_id, seen_at=now, tca=fields["tca"],
                                  miss_km=fields["miss_km"], log_pc=fields["log_pc"],
                                  predicted_log_risk=fields["predicted_log_risk"]))
    return created


def close_passed(session: Session, now: datetime) -> int:
    passed = 0
    for conj in session.scalars(select(Conjunction).where(Conjunction.status == "open")).all():
        if as_utc(conj.tca) < now:
            conj.status = "passed"
            passed += 1
    return passed


def record_manoeuvres(session: Session, norad_ids: set[int]) -> int:
    """Run the detector on objects that received a new element set in this run."""
    added = 0
    known = {(m.norad_id, as_utc(m.after)) for m in session.scalars(select(ManoeuvreRow)).all()}
    ids = sorted(norad_ids)
    for start in range(0, len(ids), 500):
        chunk = ids[start:start + 500]
        rows = session.scalars(select(ElementHistory).where(ElementHistory.norad_id.in_(chunk))
                               .order_by(ElementHistory.norad_id, ElementHistory.epoch)).all()
        by_object: dict[int, list[dict]] = {}
        for h in rows:
            by_object.setdefault(h.norad_id, []).append({"norad_id": h.norad_id, "epoch": as_utc(h.epoch), "mean_motion": h.mean_motion})
        for history in by_object.values():
            for ev in manoeuvres.detect(history):
                if (ev.norad_id, ev.after) in known:
                    continue
                known.add((ev.norad_id, ev.after))
                session.add(ManoeuvreRow(norad_id=ev.norad_id, after=ev.after, before=ev.before,
                                         delta_a_km=ev.delta_a_km, score=ev.score))
                session.add(Event(kind="manoeuvre.detected", payload={"norad_id": ev.norad_id, "delta_a_km": round(ev.delta_a_km, 3)}))
                added += 1
    return added


def refresh_space_weather(session: Session, settings: Settings, now: datetime) -> int:
    try:
        readings = spaceweather.synthetic_kp(now) if settings.spaceweather_source == "synthetic" else spaceweather.fetch_kp()
    except (httpx.HTTPError, ValueError) as exc:
        log.warning("space weather refresh failed: %s", exc)
        return 0
    for r in readings:
        session.merge(KpValue(time=r.time, kp=r.kp, kind=r.kind))
    return len(readings)


def run_pipeline(settings: Settings | None = None, session_factory: sessionmaker | None = None,
                 now: datetime | None = None, cache_dir: Path = Path(".cache")) -> RunSummary:
    settings = settings or get_settings()
    session_factory = session_factory or make_sessionmaker(settings.database_url)
    now = now or utc_now()
    t0 = time.perf_counter()
    model = load_model(settings.risk_model_path)

    with session_factory() as session:
        run = Run(source=settings.data_source, started_at=now,
                  params={"screen_days": settings.screen_days, "step_s": settings.screen_step_s,
                          "report_miss_km": settings.report_miss_km, "hbr_m": settings.hard_body_radius_m,
                          "action_pc": settings.action_pc, "monitor_pc": settings.monitor_pc,
                          "model": model.version if model else None})
        session.add(run)
        session.commit()
        try:
            cat = load_catalogue(settings, cache_dir, now)
            by_id = {r.norad_id: r for r in cat.records}
            primary_ids = (cat.primary_ids | watched_ids(session)) & by_id.keys()
            primaries = [by_id[i] for i in sorted(primary_ids)]
            ephem = load_ephemerides(session, now)
            upsert_objects(session, cat.records, run.id)
            changed = store_history(session, cat.history, now, only_new_objects=cat.source == "synthetic")
            session.flush()

            encounters = screen(primaries, cat.records, now, settings.screen_days, settings.screen_step_s,
                                settings.report_miss_km, ephem)
            created = updated = 0
            for enc in encounters:
                if save_event(session, score(enc, by_id, ephem, settings, model, now), run.id, now):
                    created += 1
                else:
                    updated += 1
            passed = close_passed(session, now)
            found = record_manoeuvres(session, changed)
            refresh_space_weather(session, settings, now)
            session.commit()
            sent = dispatch(session, settings)
            summary = RunSummary(run.id, cat.source, len(cat.records), len(primaries), len(encounters), created,
                                 updated, passed, found, sent, round(time.perf_counter() - t0, 2))
            run.status, run.finished_at, run.counts = "ok", utc_now(), asdict(summary)
            session.add(Event(kind="run.finished", payload={"run_id": run.id, "events_created": created}))
            session.commit()
            return summary
        except Exception as exc:
            session.rollback()
            run.status, run.error, run.finished_at = "failed", str(exc)[:2000], utc_now()
            session.add(run)
            session.commit()
            raise


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the Ashvini pipeline once.")
    parser.add_argument("--source", choices=["celestrak", "synthetic"], help="override ASHVINI_DATA_SOURCE")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    settings = get_settings()
    if args.source:
        settings = settings.model_copy(update={"data_source": args.source})
    summary = run_pipeline(settings)
    for k, v in asdict(summary).items():
        print(f"{k:>16}: {v}")


if __name__ == "__main__":
    main()
