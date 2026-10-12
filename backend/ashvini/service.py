"""Read-side helpers shared by the API, the alert dispatcher and the copilot."""

from datetime import datetime

from sqlalchemy import or_, select
from sqlalchemy.orm import Session

from .config import get_settings
from .db import Conjunction, ConjunctionUpdate, Org, Review, SpaceObject, Watch
from .risk.decision import Recommendation, Thresholds, recommend
from .timeutil import as_utc, utc_now


def thresholds_for(org: Org | None) -> Thresholds:
    if org is None:
        s = get_settings()
        return Thresholds(s.action_pc, s.monitor_pc)
    return Thresholds(org.action_pc, org.monitor_pc)


def recommendation_for(conj: Conjunction, thresholds: Thresholds, now: datetime | None = None) -> Recommendation:
    """Recompute the recommendation with this organisation's lines; the stored one uses the defaults."""
    now = now or utc_now()
    hours = (as_utc(conj.tca) - now).total_seconds() / 3600
    return recommend(conj.log_pc, conj.predicted_log_risk, conj.predicted_high, conj.prediction_source,
                     conj.miss_km, hours, conj.data_age_days, thresholds)


def object_names(session: Session, ids: set[int]) -> dict[int, SpaceObject]:
    if not ids:
        return {}
    return {o.norad_id: o for o in session.scalars(select(SpaceObject).where(SpaceObject.norad_id.in_(ids))).all()}


def watched_by(session: Session, org: Org | None) -> set[int] | None:
    if org is None:
        return None
    return set(session.scalars(select(Watch.norad_id).where(Watch.org_id == org.id)).all())


def conjunctions_query(session: Session, org: Org | None, status: str | None = "open", limit: int = 200) -> list[Conjunction]:
    q = select(Conjunction)
    if status:
        q = q.where(Conjunction.status == status)
    ids = watched_by(session, org)
    if ids:
        q = q.where(or_(Conjunction.primary_id.in_(ids), Conjunction.secondary_id.in_(ids)))
    return list(session.scalars(q.order_by(Conjunction.log_pc.desc(), Conjunction.tca).limit(limit)).all())


def object_brief(obj: SpaceObject | None, norad_id: int) -> dict:
    if obj is None:
        return {"norad_id": norad_id, "name": f"NORAD {norad_id}", "type": "UNKNOWN"}
    return {"norad_id": obj.norad_id, "name": obj.name, "type": obj.object_type, "intl": obj.intl_designator,
            "perigee_km": round(obj.perigee_km, 1), "apogee_km": round(obj.apogee_km, 1),
            "inclination_deg": round(obj.inclination_deg, 2), "epoch": as_utc(obj.epoch).isoformat()}


def latest_review(session: Session, conj_id: str, org: Org | None) -> Review | None:
    q = select(Review).where(Review.conjunction_id == conj_id)
    if org is not None:
        q = q.where(Review.org_id == org.id)
    return session.scalars(q.order_by(Review.id.desc()).limit(1)).first()


def conjunction_view(session: Session, conj: Conjunction, org: Org | None, detail: bool = False) -> dict:
    names = object_names(session, {conj.primary_id, conj.secondary_id})
    t = thresholds_for(org)
    rec = recommendation_for(conj, t)
    review = latest_review(session, conj.id, org)
    view = {
        "id": conj.id, "status": conj.status, "tca": as_utc(conj.tca).isoformat(),
        "primary": object_brief(names.get(conj.primary_id), conj.primary_id),
        "secondary": object_brief(names.get(conj.secondary_id), conj.secondary_id),
        "miss_km": round(conj.miss_km, 4), "rel_speed_kms": round(conj.rel_speed_kms, 3),
        "pc": conj.pc, "log_pc": round(conj.log_pc, 2),
        "predicted": {"log_risk": round(conj.predicted_log_risk, 2), "low": round(conj.predicted_low, 2),
                      "high": round(conj.predicted_high, 2), "source": conj.prediction_source},
        "recommendation": rec.level, "reasons": rec.reasons,
        "thresholds": {"action_pc": t.action_pc, "monitor_pc": t.monitor_pc},
        "data_age_days": round(conj.data_age_days, 2), "origin": conj.origin,
        "decision": None if review is None else {"decision": review.decision, "note": review.note,
                                                  "reviewer": review.reviewer, "at": as_utc(review.created_at).isoformat()},
    }
    if detail:
        history = session.scalars(select(ConjunctionUpdate).where(ConjunctionUpdate.conjunction_id == conj.id)
                                  .order_by(ConjunctionUpdate.id)).all()
        reviews = session.scalars(select(Review).where(Review.conjunction_id == conj.id).order_by(Review.id)).all()
        view |= {
            "miss_ric_km": conj.miss_ric_km, "mahalanobis": round(conj.mahalanobis, 3), "max_pc": conj.max_pc,
            "sigmas": conj.sigmas, "features": conj.features,
            "history": [{"seen_at": as_utc(h.seen_at).isoformat(), "tca": as_utc(h.tca).isoformat(), "miss_km": round(h.miss_km, 4),
                         "log_pc": round(h.log_pc, 2), "predicted_log_risk": round(h.predicted_log_risk, 2), "origin": h.origin}
                        for h in history],
            "reviews": [{"decision": r.decision, "note": r.note, "reviewer": r.reviewer,
                         "recommendation_at_time": r.recommendation_at_time, "at": as_utc(r.created_at).isoformat()}
                        for r in reviews if org is None or r.org_id == org.id],
            "lineage": {"first_run": conj.first_run_id, "last_run": conj.last_run_id},
        }
    return view
