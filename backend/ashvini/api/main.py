"""HTTP API.

Reads are public by default (ASHVINI_PUBLIC_READ). Writes need an API key with the
right role: analysts record decisions and upload files, admins manage channels,
watches and thresholds. Every write lands in the audit log.

    uvicorn ashvini.api.main:app --reload
"""

import asyncio
import json
import time
from collections import defaultdict, deque
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Literal

from fastapi import Depends, FastAPI, File, Header, HTTPException, Query, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import PlainTextResponse, StreamingResponse
from pydantic import BaseModel, Field
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from .. import __version__
from ..auth import PUBLIC, Caller, audit, resolve
from ..cdm_intake import apply_cdm
from ..config import get_settings
from ..copilot import ask
from ..db import (
    AuditLog,
    Channel,
    Conjunction,
    Ephemeris,
    Event,
    KpValue,
    ManoeuvreRow,
    Review,
    Run,
    SpaceObject,
    Watch,
    make_sessionmaker,
)
from ..ingest.ccsds import CcsdsError, format_cdm, parse_cdm, parse_oem
from ..ingest.spaceweather import storm_level
from ..orbits.elements import from_tle
from ..orbits.propagate import ground_track
from ..risk.decision import ACTION
from ..risk.model import load_model
from ..service import (
    conjunction_view,
    conjunctions_query,
    object_brief,
    object_names,
    recommendation_for,
    thresholds_for,
)
from ..timeutil import as_utc, utc_now

settings = get_settings()
SessionLocal = make_sessionmaker(settings.database_url)
risk_model = load_model(settings.risk_model_path)

app = FastAPI(title="Ashvini", version=__version__,
              description="Close approaches, collision risk, orbit changes and space weather for Earth orbit.")
app.add_middleware(GZipMiddleware, minimum_size=2000)
app.add_middleware(CORSMiddleware, allow_origins=[o.strip() for o in settings.cors_origins.split(",") if o.strip()],
                   allow_methods=["*"], allow_headers=["*"])


# ---------- dependencies ----------

def get_session():
    with SessionLocal() as session:
        yield session


def get_caller(session: Session = Depends(get_session), x_api_key: str | None = Header(default=None)) -> Caller:
    if x_api_key:
        caller = resolve(session, x_api_key, settings.secret)
        if caller is None:
            raise HTTPException(401, "That API key is not valid or has been revoked.")
        return caller
    if not settings.public_read:
        raise HTTPException(401, "Send an API key in the X-API-Key header.")
    return PUBLIC


def require(role: str):
    def check(caller: Caller = Depends(get_caller)) -> Caller:
        if caller is PUBLIC or not caller.can(role):
            raise HTTPException(403, f"This needs an API key with the {role} role.")
        return caller
    return check


RATE: dict[str, deque] = defaultdict(deque)


def rate_limited(request: Request) -> None:
    key = request.client.host if request.client else "unknown"
    window, now = RATE[key], time.monotonic()
    while window and now - window[0] > 60:
        window.popleft()
    if len(window) >= settings.copilot_per_minute:
        raise HTTPException(429, "Too many questions in a minute. Try again shortly.")
    window.append(now)


# ---------- health ----------

@app.get("/api/health")
def health(session: Session = Depends(get_session)):
    last = session.scalars(select(Run).order_by(Run.id.desc()).limit(1)).first()
    newest_epoch = session.scalar(select(func.max(SpaceObject.epoch)))
    newest_kp = session.scalar(select(func.max(KpValue.time)).where(KpValue.kind == "observed"))
    now = utc_now()

    def age_h(t):
        return None if t is None else round((now - as_utc(t)).total_seconds() / 3600, 1)

    return {
        "status": "ok", "version": __version__, "source": settings.data_source,
        "objects": session.scalar(select(func.count()).select_from(SpaceObject)),
        "open_events": session.scalar(select(func.count()).select_from(Conjunction).where(Conjunction.status == "open")),
        "data_age_hours": {"newest_elements": age_h(newest_epoch), "newest_kp": age_h(newest_kp),
                           "last_run": age_h(last.finished_at) if last and last.finished_at else None},
        "last_run": None if last is None else {"id": last.id, "status": last.status, "counts": last.counts,
                                               "params": last.params, "error": last.error},
        "risk_model": risk_model.version if risk_model else None,
    }


@app.get("/api/me")
def me(caller: Caller = Depends(get_caller)):
    return {"name": caller.name, "role": caller.role, "org": caller.org.name if caller.org else None,
            "thresholds": vars(thresholds_for(caller.org))}


# ---------- catalogue ----------

@app.get("/api/catalogue")
def catalogue(session: Session = Depends(get_session),
              types: str = Query("PAYLOAD,DEBRIS,ROCKET BODY", description="comma-separated object types")):
    """Compact element sets for the globe, which propagates them in the browser."""
    wanted = [t.strip() for t in types.split(",")]
    rows = session.execute(select(SpaceObject.norad_id, SpaceObject.name, SpaceObject.object_type,
                                  SpaceObject.line1, SpaceObject.line2).where(SpaceObject.object_type.in_(wanted))).all()
    return {"count": len(rows), "objects": [{"n": r[0], "name": r[1], "t": r[2], "l1": r[3], "l2": r[4]} for r in rows]}


@app.get("/api/objects/{norad_id}")
def object_detail(norad_id: int, session: Session = Depends(get_session)):
    obj = session.get(SpaceObject, norad_id)
    if obj is None:
        raise HTTPException(404, f"No object {norad_id} in the catalogue.")
    rec = from_tle(obj.name, obj.line1, obj.line2)
    return {**object_brief(obj, norad_id), "period_min": round(obj.period_min, 2),
            "ground_track": ground_track(rec, utc_now(), minutes=obj.period_min)}


# ---------- conjunctions ----------

@app.get("/api/conjunctions")
def list_conjunctions(status: Literal["open", "passed", "all"] = "open", limit: int = Query(100, le=500),
                      session: Session = Depends(get_session), caller: Caller = Depends(get_caller)):
    rows = conjunctions_query(session, caller.org, None if status == "all" else status, limit)
    return [conjunction_view(session, c, caller.org) for c in rows]


@app.get("/api/conjunctions/{event_id}")
def get_conjunction(event_id: str, session: Session = Depends(get_session), caller: Caller = Depends(get_caller)):
    conj = session.get(Conjunction, event_id)
    if conj is None:
        raise HTTPException(404, "No such event.")
    return conjunction_view(session, conj, caller.org, detail=True)


@app.get("/api/conjunctions/{event_id}/cdm", response_class=PlainTextResponse)
def export_cdm(event_id: str, session: Session = Depends(get_session)):
    conj = session.get(Conjunction, event_id)
    if conj is None:
        raise HTTPException(404, "No such event.")
    names = object_names(session, {conj.primary_id, conj.secondary_id})
    objs = []
    for norad, key in ((conj.primary_id, "primary"), (conj.secondary_id, "secondary")):
        o, sig = names.get(norad), conj.sigmas.get(key, {})
        objs.append({"designator": norad, "name": o.name if o else str(norad), "intl": o.intl_designator if o else "",
                     "type": o.object_type if o else "UNKNOWN", "sigma_r_m": sig.get("r_km", 0) * 1000,
                     "sigma_t_m": sig.get("t_km", 0) * 1000, "sigma_n_m": sig.get("n_km", 0) * 1000})
    return format_cdm(message_id=f"ASHVINI-{conj.id}", created=utc_now(), tca=as_utc(conj.tca), miss_m=conj.miss_km * 1000,
                      rel_speed_ms=conj.rel_speed_kms * 1000, pc=conj.pc, objects=objs)


class ReviewIn(BaseModel):
    decision: Literal["manoeuvre", "monitor", "dismiss"]
    note: str = Field("", max_length=2000)


@app.post("/api/conjunctions/{event_id}/reviews", status_code=201)
def add_review(event_id: str, body: ReviewIn, session: Session = Depends(get_session), caller: Caller = Depends(require("analyst"))):
    conj = session.get(Conjunction, event_id)
    if conj is None:
        raise HTTPException(404, "No such event.")
    rec = recommendation_for(conj, thresholds_for(caller.org))
    if (body.decision == ACTION or body.decision != rec.level) and not body.note.strip():
        raise HTTPException(422, "Add a note: it is required for a manoeuvre and whenever you differ from the recommendation.")
    review = Review(conjunction_id=conj.id, org_id=caller.org.id if caller.org else None, decision=body.decision,
                    note=body.note.strip(), reviewer=caller.name, recommendation_at_time=rec.level, log_pc_at_time=conj.log_pc)
    session.add(review)
    session.add(Event(kind="conjunction.reviewed", payload={"id": conj.id, "decision": body.decision}))
    audit(session, caller.org.id if caller.org else None, caller.name, "review.added", {"event": conj.id, "decision": body.decision})
    session.commit()
    return conjunction_view(session, conj, caller.org, detail=True)


# ---------- operator uploads ----------

@app.post("/api/upload/cdm", status_code=201)
async def upload_cdm(file: UploadFile = File(...), session: Session = Depends(get_session),
                     caller: Caller = Depends(require("analyst"))):
    raw = (await file.read()).decode("utf-8", errors="replace")
    try:
        cdm = parse_cdm(raw)
        conj, created = apply_cdm(session, cdm, raw, caller.org, risk_model)
    except (CcsdsError, ValueError) as exc:
        raise HTTPException(422, f"Could not read that CDM: {exc}") from exc
    audit(session, caller.org.id if caller.org else None, caller.name, "cdm.uploaded", {"event": conj.id, "created": created})
    session.commit()
    return {"created": created, "event": conjunction_view(session, conj, caller.org, detail=True)}


@app.post("/api/upload/oem", status_code=201)
async def upload_oem(file: UploadFile = File(...), session: Session = Depends(get_session),
                     caller: Caller = Depends(require("analyst"))):
    if caller.org is None:
        raise HTTPException(403, "Ephemerides belong to an organisation; use an organisation key.")
    raw = (await file.read()).decode("utf-8", errors="replace")
    try:
        tracks = parse_oem(raw)
    except (CcsdsError, ValueError) as exc:
        raise HTTPException(422, f"Could not read that OEM: {exc}") from exc
    stored = []
    for t in tracks:
        if not t.norad_id:
            raise HTTPException(422, "Each OEM segment needs OBJECT_ID set to the NORAD catalogue number.")
        session.add(Ephemeris(org_id=caller.org.id, norad_id=t.norad_id, name=t.name,
                              start=datetime.fromtimestamp(t.start, UTC), stop=datetime.fromtimestamp(t.stop, UTC),
                              epochs=t.epochs.tolist(), states=t.states.tolist()))
        if session.scalars(select(Watch).where(Watch.org_id == caller.org.id, Watch.norad_id == t.norad_id)).first() is None:
            session.add(Watch(org_id=caller.org.id, norad_id=t.norad_id))
        stored.append({"norad_id": t.norad_id, "points": len(t.epochs),
                       "hours": round((t.stop - t.start) / 3600, 1)})
    audit(session, caller.org.id, caller.name, "oem.uploaded", {"segments": stored})
    session.commit()
    return {"segments": stored, "note": "Used from the next pipeline run for screening these objects."}


# ---------- space weather and manoeuvres ----------

@app.get("/api/spaceweather")
def spaceweather(session: Session = Depends(get_session)):
    now = utc_now()
    rows = session.scalars(select(KpValue).where(KpValue.time >= now - timedelta(days=2)).order_by(KpValue.time)).all()
    series = [{"time": as_utc(r.time).isoformat(), "kp": r.kp, "kind": r.kind} for r in rows]
    past = [r for r in rows if as_utc(r.time) <= now]
    ahead = [r for r in rows if as_utc(r.time) > now]
    peak = max(ahead, key=lambda r: r.kp) if ahead else None
    return {"kp_now": past[-1].kp if past else None, "level_now": storm_level(past[-1].kp) if past else None,
            "peak": None if peak is None else {"kp": peak.kp, "time": as_utc(peak.time).isoformat(), "level": storm_level(peak.kp)},
            "series": series, "source": settings.spaceweather_source}


@app.get("/api/manoeuvres")
def manoeuvres(days: int = 14, session: Session = Depends(get_session)):
    since = utc_now() - timedelta(days=days)
    rows = [m for m in session.scalars(select(ManoeuvreRow).order_by(ManoeuvreRow.after.desc())).all() if as_utc(m.after) >= since]
    names = object_names(session, {m.norad_id for m in rows})
    return [{"norad_id": m.norad_id, "name": names[m.norad_id].name if m.norad_id in names else None,
             "before": as_utc(m.before).isoformat(), "after": as_utc(m.after).isoformat(),
             "delta_a_km": round(m.delta_a_km, 3), "score": round(m.score, 1)} for m in rows]


# ---------- copilot ----------

class Question(BaseModel):
    question: str = Field(min_length=2, max_length=500)


@app.post("/api/copilot", dependencies=[Depends(rate_limited)])
def copilot(body: Question, session: Session = Depends(get_session), caller: Caller = Depends(get_caller)):
    answer = ask(session, caller.org, body.question, settings)
    return {"answer": answer.text, "mode": answer.mode, "refused": answer.refused, "tools": answer.tools_used}


# ---------- models ----------

def settings_models_dir() -> str:
    return str(Path(settings.risk_model_path).parent) if settings.risk_model_path else "models"


@app.get("/api/models")
def models():
    """Model cards written by ml/*/train.py, if any were copied next to the API."""
    cards = []
    for path in sorted(Path(settings_models_dir()).glob("*.json")):
        if path.name.endswith(".serving.json") or path.name == "index.json":
            continue
        try:
            cards.append(json.loads(path.read_text()))
        except json.JSONDecodeError:
            continue
    return {"serving": risk_model.version if risk_model else "baseline: current Pc carried forward", "cards": cards}


# ---------- admin ----------

class ChannelIn(BaseModel):
    kind: Literal["webhook", "slack", "email"]
    target: str = Field(min_length=3, max_length=400)
    min_level: Literal["monitor", "manoeuvre"] = "manoeuvre"


@app.get("/api/admin/channels")
def list_channels(session: Session = Depends(get_session), caller: Caller = Depends(require("admin"))):
    rows = session.scalars(select(Channel).where(Channel.org_id == caller.org.id)).all()
    return [{"id": c.id, "kind": c.kind, "target": c.target[:24] + "...", "min_level": c.min_level, "active": c.active} for c in rows]


@app.post("/api/admin/channels", status_code=201)
def create_channel(body: ChannelIn, session: Session = Depends(get_session), caller: Caller = Depends(require("admin"))):
    ch = Channel(org_id=caller.org.id, kind=body.kind, target=body.target, min_level=body.min_level)
    session.add(ch)
    audit(session, caller.org.id, caller.name, "channel.added", {"kind": body.kind, "min_level": body.min_level})
    session.commit()
    return {"id": ch.id}


class ThresholdsIn(BaseModel):
    action_pc: float = Field(gt=0, lt=1)
    monitor_pc: float = Field(gt=0, lt=1)


@app.put("/api/admin/thresholds")
def set_thresholds(body: ThresholdsIn, session: Session = Depends(get_session), caller: Caller = Depends(require("admin"))):
    if body.monitor_pc >= body.action_pc:
        raise HTTPException(422, "The monitor line must be below the action line.")
    old = {"action_pc": caller.org.action_pc, "monitor_pc": caller.org.monitor_pc}
    org = session.merge(caller.org)
    org.action_pc, org.monitor_pc = body.action_pc, body.monitor_pc
    audit(session, org.id, caller.name, "thresholds.changed", {"from": old, "to": body.model_dump()})
    session.commit()
    return body.model_dump()


class WatchIn(BaseModel):
    norad_ids: list[int] = Field(min_length=1, max_length=500)


@app.post("/api/admin/watch")
def add_watch(body: WatchIn, session: Session = Depends(get_session), caller: Caller = Depends(require("admin"))):
    have = set(session.scalars(select(Watch.norad_id).where(Watch.org_id == caller.org.id)).all())
    new = [n for n in body.norad_ids if n not in have]
    for n in new:
        session.add(Watch(org_id=caller.org.id, norad_id=n))
    audit(session, caller.org.id, caller.name, "watch.added", {"norad_ids": new})
    session.commit()
    return {"watching": len(have) + len(new)}


@app.get("/api/admin/audit")
def audit_log(limit: int = Query(100, le=1000), session: Session = Depends(get_session), caller: Caller = Depends(require("admin"))):
    rows = session.scalars(select(AuditLog).where(AuditLog.org_id == caller.org.id).order_by(AuditLog.id.desc()).limit(limit)).all()
    return [{"at": as_utc(r.created_at).isoformat(), "actor": r.actor, "action": r.action, "detail": r.detail} for r in rows]


# ---------- live stream ----------

@app.get("/api/stream")
async def stream(request: Request, after: int | None = None):
    """Server-sent events from the events table, so there is no broker to host."""

    async def gen():
        with SessionLocal() as s:
            last = after if after is not None else (s.scalar(select(func.max(Event.id))) or 0)
        yield "retry: 5000\n\n"
        while not await request.is_disconnected():
            with SessionLocal() as s:
                rows = s.scalars(select(Event).where(Event.id > last).order_by(Event.id).limit(50)).all()
            for ev in rows:
                last = ev.id
                yield f"id: {ev.id}\nevent: {ev.kind}\ndata: {json.dumps(ev.payload)}\n\n"
            await asyncio.sleep(3)

    return StreamingResponse(gen(), media_type="text/event-stream", headers={"Cache-Control": "no-cache"})

