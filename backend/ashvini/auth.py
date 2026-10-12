"""API keys and roles.

A key looks like `ash_<prefix>_<secret>`. Only a salted SHA-256 of the secret is
stored, so a leaked database does not leak working keys.
"""

import hashlib
import hmac
import secrets
from dataclasses import dataclass

from sqlalchemy import select
from sqlalchemy.orm import Session

from .db import ApiKey, AuditLog, Org

ROLES = {"viewer": 0, "analyst": 1, "admin": 2}


@dataclass
class Caller:
    org: Org | None
    role: str
    label: str

    def can(self, role: str) -> bool:
        return ROLES[self.role] >= ROLES[role]

    @property
    def name(self) -> str:
        return self.label or (self.org.name if self.org else "public")


PUBLIC = Caller(None, "viewer", "public")


def digest(secret: str, pepper: str) -> str:
    return hashlib.sha256((pepper + ":" + secret).encode()).hexdigest()


def issue_key(session: Session, org: Org, role: str, pepper: str, label: str = "") -> str:
    if role not in ROLES:
        raise ValueError(f"role must be one of {', '.join(ROLES)}")
    prefix, secret = secrets.token_hex(4), secrets.token_urlsafe(24)
    session.add(ApiKey(org_id=org.id, prefix=prefix, digest=digest(secret, pepper), role=role, label=label))
    audit(session, org.id, "admin-cli", "key.issued", {"prefix": prefix, "role": role})
    return f"ash_{prefix}_{secret}"


def resolve(session: Session, raw_key: str | None, pepper: str) -> Caller | None:
    """Caller for a key, or None when the key is wrong or revoked."""
    if not raw_key:
        return None
    parts = raw_key.split("_", 2)
    if len(parts) != 3 or parts[0] != "ash":
        return None
    row = session.scalars(select(ApiKey).where(ApiKey.prefix == parts[1], ApiKey.revoked.is_(False))).first()
    if row is None or not hmac.compare_digest(row.digest, digest(parts[2], pepper)):
        return None
    return Caller(session.get(Org, row.org_id), row.role, row.label)


def audit(session: Session, org_id: int | None, actor: str, action: str, detail: dict) -> None:
    session.add(AuditLog(org_id=org_id, actor=actor, action=action, detail=detail))
