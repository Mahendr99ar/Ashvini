"""Set up an operator from the command line.

    python -m ashvini.admin create-org "Example Orbital" --action-pc 1e-4 --monitor-pc 1e-6
    python -m ashvini.admin create-key "Example Orbital" --role admin
    python -m ashvini.admin watch "Example Orbital" 25544 43013
    python -m ashvini.admin add-channel "Example Orbital" slack https://hooks.slack.com/...
    python -m ashvini.admin list
"""

import argparse
import sys

from sqlalchemy import select
from sqlalchemy.orm import Session

from .auth import audit, issue_key
from .config import get_settings
from .db import Channel, Org, Watch, make_sessionmaker


def find_org(session: Session, name: str) -> Org:
    org = session.scalars(select(Org).where(Org.name == name)).first()
    if org is None:
        sys.exit(f"no organisation called {name!r}; create it first with create-org")
    return org


def create_org(session: Session, args) -> str:
    org = Org(name=args.org, action_pc=args.action_pc, monitor_pc=args.monitor_pc)
    session.add(org)
    session.flush()
    audit(session, org.id, "admin-cli", "org.created", {"action_pc": args.action_pc, "monitor_pc": args.monitor_pc})
    return f"created {org.name} (id {org.id})"


def create_key(session: Session, args) -> str:
    key = issue_key(session, find_org(session, args.org), args.role, get_settings().secret, args.label)
    return f"{key}\nStore this now; it is not shown again."


def watch(session: Session, args) -> str:
    org = find_org(session, args.org)
    have = set(session.scalars(select(Watch.norad_id).where(Watch.org_id == org.id)).all())
    added = [n for n in args.norad_ids if n not in have]
    for n in added:
        session.add(Watch(org_id=org.id, norad_id=n))
    audit(session, org.id, "admin-cli", "watch.added", {"norad_ids": added})
    return f"{org.name} now watches {len(have) + len(added)} objects"


def add_channel(session: Session, args) -> str:
    org = find_org(session, args.org)
    ch = Channel(org_id=org.id, kind=args.kind, target=args.target, min_level=args.min_level)
    session.add(ch)
    session.flush()
    audit(session, org.id, "admin-cli", "channel.added", {"kind": args.kind, "min_level": args.min_level})
    return f"channel {ch.id} ({args.kind}) alerts at {args.min_level} and above"


def list_all(session: Session, args) -> str:
    lines = []
    for org in session.scalars(select(Org)).all():
        n = len(session.scalars(select(Watch).where(Watch.org_id == org.id)).all())
        c = len(session.scalars(select(Channel).where(Channel.org_id == org.id)).all())
        lines.append(f"{org.id:>3}  {org.name:<30} action {org.action_pc:.0e}  monitor {org.monitor_pc:.0e}  "
                     f"watching {n}  channels {c}")
    return "\n".join(lines) or "no organisations yet"


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="python -m ashvini.admin")
    sub = p.add_subparsers(dest="command", required=True)
    c = sub.add_parser("create-org")
    c.add_argument("org")
    c.add_argument("--action-pc", type=float, default=1e-4)
    c.add_argument("--monitor-pc", type=float, default=1e-6)
    c.set_defaults(fn=create_org)
    k = sub.add_parser("create-key")
    k.add_argument("org")
    k.add_argument("--role", choices=["viewer", "analyst", "admin"], default="analyst")
    k.add_argument("--label", default="")
    k.set_defaults(fn=create_key)
    w = sub.add_parser("watch")
    w.add_argument("org")
    w.add_argument("norad_ids", nargs="+", type=int)
    w.set_defaults(fn=watch)
    a = sub.add_parser("add-channel")
    a.add_argument("org")
    a.add_argument("kind", choices=["webhook", "slack", "email"])
    a.add_argument("target")
    a.add_argument("--min-level", choices=["monitor", "manoeuvre"], default="manoeuvre")
    a.set_defaults(fn=add_channel)
    sub.add_parser("list").set_defaults(fn=list_all)
    return p


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    factory = make_sessionmaker(get_settings().database_url)
    with factory() as session:
        print(args.fn(session, args))
        session.commit()


if __name__ == "__main__":
    main()
