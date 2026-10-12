"""Database schema. SQLite for local runs and tests, Postgres in production.

Two tables are append-only by design: `reviews` (what a person decided and why) and
`audit_log` (who changed what). Pipeline reruns update events, never decisions.
"""

from datetime import datetime
from typing import Any

from sqlalchemy import (
    JSON,
    Boolean,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
    UniqueConstraint,
    create_engine,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column, sessionmaker

from .timeutil import utc_now


class Base(DeclarativeBase):
    type_annotation_map = {dict[str, Any]: JSON, list[Any]: JSON}


class Run(Base):
    __tablename__ = "runs"
    id: Mapped[int] = mapped_column(primary_key=True)
    started_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now)
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    status: Mapped[str] = mapped_column(String(16), default="running")
    source: Mapped[str] = mapped_column(String(32))
    params: Mapped[dict[str, Any]] = mapped_column(default=dict)
    counts: Mapped[dict[str, Any]] = mapped_column(default=dict)
    error: Mapped[str | None] = mapped_column(Text)


class SpaceObject(Base):
    __tablename__ = "objects"
    norad_id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=False)
    name: Mapped[str] = mapped_column(String(64))
    object_type: Mapped[str] = mapped_column(String(16))
    intl_designator: Mapped[str] = mapped_column(String(16), default="")
    epoch: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    line1: Mapped[str] = mapped_column(String(80))
    line2: Mapped[str] = mapped_column(String(80))
    perigee_km: Mapped[float] = mapped_column(Float)
    apogee_km: Mapped[float] = mapped_column(Float)
    inclination_deg: Mapped[float] = mapped_column(Float)
    period_min: Mapped[float] = mapped_column(Float)
    source: Mapped[str] = mapped_column(String(16))
    run_id: Mapped[int | None] = mapped_column(ForeignKey("runs.id"))


class ElementHistory(Base):
    __tablename__ = "element_history"
    __table_args__ = (UniqueConstraint("norad_id", "epoch"),)
    id: Mapped[int] = mapped_column(primary_key=True)
    norad_id: Mapped[int] = mapped_column(Integer, index=True)
    epoch: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    mean_motion: Mapped[float] = mapped_column(Float)
    eccentricity: Mapped[float] = mapped_column(Float)
    inclination: Mapped[float] = mapped_column(Float)


class Conjunction(Base):
    __tablename__ = "conjunctions"
    id: Mapped[str] = mapped_column(String(16), primary_key=True)
    primary_id: Mapped[int] = mapped_column(Integer, index=True)
    secondary_id: Mapped[int] = mapped_column(Integer, index=True)
    tca: Mapped[datetime] = mapped_column(DateTime(timezone=True), index=True)
    miss_km: Mapped[float] = mapped_column(Float)
    rel_speed_kms: Mapped[float] = mapped_column(Float)
    miss_ric_km: Mapped[list[Any]] = mapped_column(default=list)
    pc: Mapped[float] = mapped_column(Float)
    log_pc: Mapped[float] = mapped_column(Float)
    max_pc: Mapped[float] = mapped_column(Float)
    mahalanobis: Mapped[float] = mapped_column(Float)
    sigmas: Mapped[dict[str, Any]] = mapped_column(default=dict)
    data_age_days: Mapped[float] = mapped_column(Float)
    predicted_log_risk: Mapped[float] = mapped_column(Float)
    predicted_low: Mapped[float] = mapped_column(Float)
    predicted_high: Mapped[float] = mapped_column(Float)
    prediction_source: Mapped[str] = mapped_column(String(48))
    recommendation: Mapped[str] = mapped_column(String(16))
    reasons: Mapped[list[Any]] = mapped_column(default=list)
    features: Mapped[dict[str, Any]] = mapped_column(default=dict)
    origin: Mapped[str] = mapped_column(String(16), default="screening")
    status: Mapped[str] = mapped_column(String(16), default="open")
    first_run_id: Mapped[int | None] = mapped_column(Integer)
    last_run_id: Mapped[int | None] = mapped_column(Integer)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now)


class ConjunctionUpdate(Base):
    """One row per run that saw the event: the history a CDM sequence would carry."""

    __tablename__ = "conjunction_updates"
    id: Mapped[int] = mapped_column(primary_key=True)
    conjunction_id: Mapped[str] = mapped_column(ForeignKey("conjunctions.id"), index=True)
    run_id: Mapped[int | None] = mapped_column(Integer)
    seen_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now)
    tca: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    miss_km: Mapped[float] = mapped_column(Float)
    log_pc: Mapped[float] = mapped_column(Float)
    predicted_log_risk: Mapped[float] = mapped_column(Float)
    origin: Mapped[str] = mapped_column(String(16), default="screening")


class CdmMessage(Base):
    __tablename__ = "cdm_messages"
    id: Mapped[int] = mapped_column(primary_key=True)
    conjunction_id: Mapped[str] = mapped_column(ForeignKey("conjunctions.id"), index=True)
    org_id: Mapped[int | None] = mapped_column(ForeignKey("orgs.id"))
    message_id: Mapped[str] = mapped_column(String(64))
    originator: Mapped[str] = mapped_column(String(64))
    created: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    raw: Mapped[str] = mapped_column(Text)


class Review(Base):
    __tablename__ = "reviews"
    id: Mapped[int] = mapped_column(primary_key=True)
    conjunction_id: Mapped[str] = mapped_column(ForeignKey("conjunctions.id"), index=True)
    org_id: Mapped[int | None] = mapped_column(ForeignKey("orgs.id"))
    decision: Mapped[str] = mapped_column(String(16))
    note: Mapped[str] = mapped_column(Text, default="")
    reviewer: Mapped[str] = mapped_column(String(64))
    recommendation_at_time: Mapped[str] = mapped_column(String(16))
    log_pc_at_time: Mapped[float] = mapped_column(Float)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now)


class Org(Base):
    __tablename__ = "orgs"
    id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str] = mapped_column(String(80), unique=True)
    action_pc: Mapped[float] = mapped_column(Float, default=1e-4)
    monitor_pc: Mapped[float] = mapped_column(Float, default=1e-6)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now)


class ApiKey(Base):
    __tablename__ = "api_keys"
    id: Mapped[int] = mapped_column(primary_key=True)
    org_id: Mapped[int] = mapped_column(ForeignKey("orgs.id"))
    prefix: Mapped[str] = mapped_column(String(12), unique=True)
    digest: Mapped[str] = mapped_column(String(64))
    role: Mapped[str] = mapped_column(String(12))
    label: Mapped[str] = mapped_column(String(64), default="")
    revoked: Mapped[bool] = mapped_column(Boolean, default=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now)


class Watch(Base):
    __tablename__ = "watches"
    __table_args__ = (UniqueConstraint("org_id", "norad_id"),)
    id: Mapped[int] = mapped_column(primary_key=True)
    org_id: Mapped[int] = mapped_column(ForeignKey("orgs.id"))
    norad_id: Mapped[int] = mapped_column(Integer, index=True)


class Ephemeris(Base):
    __tablename__ = "ephemerides"
    id: Mapped[int] = mapped_column(primary_key=True)
    org_id: Mapped[int] = mapped_column(ForeignKey("orgs.id"))
    norad_id: Mapped[int] = mapped_column(Integer, index=True)
    name: Mapped[str] = mapped_column(String(64))
    start: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    stop: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    epochs: Mapped[list[Any]] = mapped_column(default=list)
    states: Mapped[list[Any]] = mapped_column(default=list)
    uploaded_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now)


class Channel(Base):
    __tablename__ = "channels"
    id: Mapped[int] = mapped_column(primary_key=True)
    org_id: Mapped[int] = mapped_column(ForeignKey("orgs.id"))
    kind: Mapped[str] = mapped_column(String(12))  # webhook, slack, email
    target: Mapped[str] = mapped_column(String(400))
    min_level: Mapped[str] = mapped_column(String(16), default="manoeuvre")
    active: Mapped[bool] = mapped_column(Boolean, default=True)


class Delivery(Base):
    __tablename__ = "deliveries"
    id: Mapped[int] = mapped_column(primary_key=True)
    channel_id: Mapped[int] = mapped_column(ForeignKey("channels.id"))
    conjunction_id: Mapped[str] = mapped_column(ForeignKey("conjunctions.id"))
    level: Mapped[str] = mapped_column(String(16))
    status: Mapped[str] = mapped_column(String(12))
    detail: Mapped[str] = mapped_column(Text, default="")
    sent_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now)


class AuditLog(Base):
    __tablename__ = "audit_log"
    id: Mapped[int] = mapped_column(primary_key=True)
    org_id: Mapped[int | None] = mapped_column(Integer)
    actor: Mapped[str] = mapped_column(String(64))
    action: Mapped[str] = mapped_column(String(48))
    detail: Mapped[dict[str, Any]] = mapped_column(default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now)


class KpValue(Base):
    __tablename__ = "kp_values"
    time: Mapped[datetime] = mapped_column(DateTime(timezone=True), primary_key=True)
    kp: Mapped[float] = mapped_column(Float)
    kind: Mapped[str] = mapped_column(String(12))


class ManoeuvreRow(Base):
    __tablename__ = "manoeuvres"
    __table_args__ = (UniqueConstraint("norad_id", "after"),)
    id: Mapped[int] = mapped_column(primary_key=True)
    norad_id: Mapped[int] = mapped_column(Integer, index=True)
    after: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    before: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    delta_a_km: Mapped[float] = mapped_column(Float)
    score: Mapped[float] = mapped_column(Float)


class Event(Base):
    """Feed for the live stream: new events, reviews, manoeuvres."""

    __tablename__ = "events"
    id: Mapped[int] = mapped_column(primary_key=True)
    kind: Mapped[str] = mapped_column(String(32))
    payload: Mapped[dict[str, Any]] = mapped_column(default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now)


def normalise_url(url: str) -> str:
    """Neon and Render hand out postgres:// URLs; SQLAlchemy wants a driver name."""
    if url.startswith("postgres://"):
        url = "postgresql://" + url[len("postgres://"):]
    if url.startswith("postgresql://"):
        url = "postgresql+psycopg://" + url[len("postgresql://"):]
    return url


def make_engine(url: str):
    url = normalise_url(url)
    kwargs = {"connect_args": {"check_same_thread": False}} if url.startswith("sqlite") else {"pool_pre_ping": True}
    return create_engine(url, **kwargs)


def make_sessionmaker(url: str) -> sessionmaker[Session]:
    engine = make_engine(url)
    Base.metadata.create_all(engine)
    return sessionmaker(engine, expire_on_commit=False)
