"""Send alerts where operators already work: webhooks, Slack, email.

Each (channel, event, level) is delivered at most once. An event that escalates
from monitor to manoeuvre alerts again; one that stays the same does not.
"""

import logging
import smtplib
from email.message import EmailMessage

import httpx
from sqlalchemy import or_, select
from sqlalchemy.orm import Session

from .config import Settings
from .db import Channel, Conjunction, Delivery, Org, Watch
from .risk.decision import LEVELS
from .service import conjunction_view, recommendation_for, thresholds_for

log = logging.getLogger(__name__)


def alert_text(view: dict) -> str:
    p, s = view["primary"]["name"], view["secondary"]["name"]
    miss = view["miss_km"]
    miss_text = f"{miss * 1000:.0f} m" if miss < 1 else f"{miss:.2f} km"
    return (f"[{view['recommendation'].upper()}] {p} and {s} at {view['tca'][:16].replace('T', ' ')} UTC. "
            f"Miss {miss_text}, Pc {view['pc']:.1e}. {view['reasons'][0]}")


def send_webhook(client: httpx.Client, url: str, view: dict) -> None:
    client.post(url, json={"type": "ashvini.conjunction", "event": view}).raise_for_status()


def send_slack(client: httpx.Client, url: str, view: dict) -> None:
    client.post(url, json={"text": alert_text(view)}).raise_for_status()


def send_email(settings: Settings, to: str, view: dict) -> None:
    if not settings.smtp_host:
        raise RuntimeError("email channel configured but ASHVINI_SMTP_HOST is not set")
    msg = EmailMessage()
    msg["From"], msg["To"] = settings.smtp_from, to
    msg["Subject"] = f"Ashvini: {view['recommendation']} for {view['primary']['name']}"
    msg.set_content(alert_text(view) + "\n\nReasons:\n- " + "\n- ".join(view["reasons"]))
    with smtplib.SMTP(settings.smtp_host, settings.smtp_port, timeout=20) as smtp:
        smtp.starttls()
        if settings.smtp_user:
            smtp.login(settings.smtp_user, settings.smtp_password)
        smtp.send_message(msg)


def deliver(channel: Channel, view: dict, settings: Settings, client: httpx.Client) -> None:
    if channel.kind == "webhook":
        send_webhook(client, channel.target, view)
    elif channel.kind == "slack":
        send_slack(client, channel.target, view)
    elif channel.kind == "email":
        send_email(settings, channel.target, view)
    else:
        raise ValueError(f"unknown channel kind {channel.kind}")


def events_for_org(session: Session, org: Org) -> list[Conjunction]:
    ids = set(session.scalars(select(Watch.norad_id).where(Watch.org_id == org.id)).all())
    if not ids:
        return []
    q = select(Conjunction).where(Conjunction.status == "open",
                                  or_(Conjunction.primary_id.in_(ids), Conjunction.secondary_id.in_(ids)))
    return list(session.scalars(q).all())


def already_sent(session: Session, channel_id: int, conj_id: str, level: str) -> bool:
    q = select(Delivery.id).where(Delivery.channel_id == channel_id, Delivery.conjunction_id == conj_id,
                                  Delivery.level == level, Delivery.status == "sent")
    return session.scalars(q).first() is not None


def dispatch(session: Session, settings: Settings, client: httpx.Client | None = None) -> int:
    own = client is None
    client = client or httpx.Client(timeout=15)
    sent = 0
    try:
        for org in session.scalars(select(Org)).all():
            channels = session.scalars(select(Channel).where(Channel.org_id == org.id, Channel.active)).all()
            if not channels:
                continue
            thresholds = thresholds_for(org)
            for conj in events_for_org(session, org):
                level = recommendation_for(conj, thresholds).level
                for ch in channels:
                    if LEVELS[level] < LEVELS[ch.min_level] or already_sent(session, ch.id, conj.id, level):
                        continue
                    view = conjunction_view(session, conj, org)
                    try:
                        deliver(ch, view, settings, client)
                        session.add(Delivery(channel_id=ch.id, conjunction_id=conj.id, level=level, status="sent"))
                        sent += 1
                    except Exception as exc:  # one bad channel must not stop the others
                        log.warning("alert to channel %s failed: %s", ch.id, exc)
                        session.add(Delivery(channel_id=ch.id, conjunction_id=conj.id, level=level,
                                             status="failed", detail=str(exc)[:500]))
                    session.flush()
        session.commit()
    finally:
        if own:
            client.close()
    return sent
