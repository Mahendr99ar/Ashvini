"""Bring an operator's CDM into Ashvini: match it to an event (or open one) and keep it in the event's history."""

import hashlib
import math
from datetime import datetime

from sqlalchemy.orm import Session

from .db import CdmMessage, Conjunction, ConjunctionUpdate, Event, Org
from .ingest.ccsds import Cdm
from .risk.decision import recommend
from .risk.model import RiskModel, fallback_prediction
from .screening.pc import log10_or_floor, max_pc
from .service import thresholds_for
from .timeutil import utc_now


def cdm_features(cdm: Cdm, now: datetime) -> dict[str, float]:
    o1, o2 = cdm.objects
    pc = cdm.collision_probability or 0.0
    feats = {
        "time_to_tca": (cdm.tca - now).total_seconds() / 86400.0,
        "risk": log10_or_floor(pc),
        "miss_distance": cdm.miss_distance_m,
        "relative_speed": cdm.relative_speed_ms or float("nan"),
        "c_object_type_debris": 1.0 if "DEBRIS" in o2.object_type.upper() else 0.0,
        "c_object_type_rocket_body": 1.0 if "ROCKET" in o2.object_type.upper() else 0.0,
    }
    for prefix, obj in (("t", o1), ("c", o2)):
        for axis in ("r", "t", "n"):
            value = getattr(obj, f"sigma_{axis}_m")
            feats[f"{prefix}_sigma_{axis}"] = value if value is not None else float("nan")
    return feats


def apply_cdm(session: Session, cdm: Cdm, raw: str, org: Org | None, model: RiskModel | None) -> tuple[Conjunction, bool]:
    from .pipeline import find_existing, new_event_id

    now = utc_now()
    a, b = cdm.norad_ids
    if a is None or b is None:
        raise ValueError("CDM objects need numeric OBJECT_DESIGNATOR values (NORAD ids)")
    pc = cdm.collision_probability or 0.0
    log_pc = log10_or_floor(pc)
    miss_km = cdm.miss_distance_m / 1000.0
    features = cdm_features(cdm, now)
    prediction = model.predict(features) if model else fallback_prediction(log_pc)
    sigmas = {"primary": {f"{ax}_km": (getattr(cdm.objects[0], f"sigma_{ax}_m") or 0) / 1000 for ax in "rtn"},
              "secondary": {f"{ax}_km": (getattr(cdm.objects[1], f"sigma_{ax}_m") or 0) / 1000 for ax in "rtn"},
              "primary_source": f"CDM {cdm.originator}"}
    rec = recommend(log_pc, prediction.log_risk, prediction.high, prediction.source, miss_km,
                    (cdm.tca - now).total_seconds() / 3600, 0.0, thresholds_for(org))
    fields = dict(tca=cdm.tca, miss_km=miss_km, rel_speed_kms=(cdm.relative_speed_ms or 0) / 1000.0, pc=pc, log_pc=log_pc,
                  max_pc=max_pc(miss_km, 0.02), mahalanobis=0.0, sigmas=sigmas,
                  data_age_days=0.0, predicted_log_risk=prediction.log_risk, predicted_low=prediction.low,
                  predicted_high=prediction.high, prediction_source=prediction.source, recommendation=rec.level,
                  reasons=rec.reasons, features={k: (None if isinstance(v, float) and math.isnan(v) else v) for k, v in features.items()},
                  origin="cdm", status="open", updated_at=now)
    conj = find_existing(session, a, b, cdm.tca)
    created = conj is None
    if created:
        conj = Conjunction(id=new_event_id(a, b, cdm.tca, lambda i: session.get(Conjunction, i) is not None), primary_id=a, secondary_id=b, miss_ric_km=[],
                           created_at=now, **fields)
        session.add(conj)
    else:
        for k, v in fields.items():
            setattr(conj, k, v)
    session.flush()
    session.add(ConjunctionUpdate(conjunction_id=conj.id, seen_at=now, tca=cdm.tca, miss_km=miss_km, log_pc=log_pc,
                                  predicted_log_risk=prediction.log_risk, origin="cdm"))
    message_id = cdm.message_id or hashlib.sha1(raw.encode()).hexdigest()[:16]
    session.add(CdmMessage(conjunction_id=conj.id, org_id=org.id if org else None, message_id=message_id,
                           originator=cdm.originator, created=cdm.creation_date, raw=raw))
    session.add(Event(kind="cdm.received", payload={"id": conj.id, "message_id": message_id}))
    return conj, created
