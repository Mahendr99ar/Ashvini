"""A read-only copilot for operators.

It answers through four tools that only read. It turns down requests to act (record
a decision, change a threshold, send an alert), because those belong to a person.
Every number in an answer is checked against the tool results it was built from;
if a model invents one, the offline writer answers instead.

Without a configured model, the offline writer builds the answer from the same
tools, so the copilot works on a free tier with no API key.
"""

import json
import re
from dataclasses import dataclass, field
from datetime import timedelta

import httpx
from sqlalchemy import select
from sqlalchemy.orm import Session

from .config import Settings
from .db import Conjunction, KpValue, ManoeuvreRow, Org
from .ingest.spaceweather import storm_level
from .service import conjunction_view, conjunctions_query, object_names
from .timeutil import as_utc, utc_now

ACTION_PATTERNS = re.compile(
    r"\b(record|log|approve|dismiss|decide|execute|perform|plan|send|notify|alert|change|set|update|delete|remove|"
    r"burn|fire)\b.*\b(decision|manoeuvre|maneuver|threshold|alert|event|channel|burn|thruster)\b", re.I)
EVENT_ID = re.compile(r"\b[0-9a-f]{16}\b")
NUMBER = re.compile(r"(?<![\w.])-?\d+(?:\.\d+)?(?:e-?\d+)?")


# ---------- tools ----------

def tool_list_events(session: Session, org: Org | None, limit: int = 5) -> list[dict]:
    rows = conjunctions_query(session, org, "open", limit=int(limit))
    return [{k: v for k, v in conjunction_view(session, c, org).items() if k not in ("reasons", "thresholds", "decision")}
            for c in rows]


def tool_get_event(session: Session, org: Org | None, event_id: str) -> dict:
    conj = session.get(Conjunction, event_id)
    if conj is None:
        return {"error": f"no event {event_id}"}
    view = conjunction_view(session, conj, org, detail=True)
    view.pop("features", None)
    return view


def tool_space_weather(session: Session, org: Org | None) -> dict:
    now = utc_now()
    rows = session.scalars(select(KpValue).order_by(KpValue.time)).all()
    past = [r for r in rows if as_utc(r.time) <= now]
    ahead = [r for r in rows if now < as_utc(r.time) <= now + timedelta(hours=48)]
    if not rows:
        return {"error": "no space weather data yet"}
    peak = max(ahead, key=lambda r: r.kp) if ahead else None
    return {"kp_now": past[-1].kp if past else None,
            "peak_next_48h": peak.kp if peak else None,
            "peak_time": as_utc(peak.time).isoformat() if peak else None,
            "level_at_peak": storm_level(peak.kp) if peak else None}


def tool_recent_manoeuvres(session: Session, org: Org | None, days: int = 7) -> list[dict]:
    since = utc_now() - timedelta(days=int(days))
    rows = [m for m in session.scalars(select(ManoeuvreRow).order_by(ManoeuvreRow.after.desc())).all() if as_utc(m.after) >= since]
    names = object_names(session, {m.norad_id for m in rows})
    return [{"norad_id": m.norad_id, "name": names[m.norad_id].name if m.norad_id in names else str(m.norad_id),
             "between": [as_utc(m.before).isoformat(), as_utc(m.after).isoformat()],
             "delta_a_km": round(m.delta_a_km, 3)} for m in rows]


TOOLS = {
    "list_events": (tool_list_events, "Open close-approach events, highest risk first.", {"limit": {"type": "integer"}}),
    "get_event": (tool_get_event, "Full detail for one event, including its history and decisions.", {"event_id": {"type": "string"}}),
    "space_weather": (tool_space_weather, "Current Kp and the forecast peak in the next 48 hours.", {}),
    "recent_manoeuvres": (tool_recent_manoeuvres, "Orbit changes detected in recent element sets.", {"days": {"type": "integer"}}),
}


def call_tool(session: Session, org: Org | None, name: str, args: dict):
    fn = TOOLS[name][0]
    return fn(session, org, **args)


# ---------- guards ----------

def is_action_request(question: str) -> bool:
    return bool(ACTION_PATTERNS.search(question))


def numbers_in(text: str) -> set[str]:
    return set(NUMBER.findall(text))


def grounded(answer: str, evidence: list) -> bool:
    """Every number in the answer must appear in the evidence, allowing for rounding."""
    blob = json.dumps(evidence, default=str)
    known = [float(x) for x in NUMBER.findall(blob)]
    for raw in numbers_in(answer):
        value = float(raw)
        if raw in blob:
            continue
        if not any(abs(value - k) <= max(0.051 * abs(k), 0.051) for k in known):
            return False
    return True


# ---------- offline writer ----------

def miss_text(km: float) -> str:
    return f"{km * 1000:.0f} m" if km < 1 else f"{km:.2f} km"


def describe_event(e: dict) -> str:
    return (f"{e['primary']['name']} and {e['secondary']['name']} pass {miss_text(e['miss_km'])} apart at "
            f"{e['tca'][:16].replace('T', ' ')} UTC (Pc {e['pc']:.1e}, recommendation: {e['recommendation']}).")


def offline_answer(session: Session, org: Org | None, question: str) -> tuple[str, list]:
    q = question.lower()
    match = EVENT_ID.search(q)
    if match:
        e = tool_get_event(session, org, match.group(0))
        if "error" in e:
            return e["error"] + ".", [e]
        lines = [describe_event(e), *e["reasons"]]
        if e["decision"]:
            lines.append(f"Last decision: {e['decision']['decision']} by {e['decision']['reviewer']}.")
        return " ".join(lines), [e]
    if any(w in q for w in ("weather", "storm", "kp", "solar", "sun")):
        w = tool_space_weather(session, org)
        if "error" in w:
            return "No space weather data has been loaded yet.", [w]
        text = f"Kp is {w['kp_now']} now."
        if w["peak_next_48h"] is not None:
            text += f" The forecast peaks at Kp {w['peak_next_48h']} ({w['level_at_peak']}) at {w['peak_time'][:16].replace('T', ' ')} UTC."
            if w["peak_next_48h"] >= 5:
                text += " Low orbits will see more drag, so expect Pc values to move after the next element sets."
        return text, [w]
    if any(w in q for w in ("manoeuvre", "maneuver", "burn", "raised", "moved")):
        rows = tool_recent_manoeuvres(session, org)
        if not rows:
            return "No orbit changes detected in the last 7 days.", [rows]
        parts = [f"{m['name']} {'raised' if m['delta_a_km'] > 0 else 'lowered'} its orbit by {abs(m['delta_a_km'])} km "
                 f"between {m['between'][0][:16].replace('T', ' ')} and {m['between'][1][:16].replace('T', ' ')} UTC." for m in rows[:3]]
        return " ".join(parts), [rows]
    events = tool_list_events(session, org, 3)
    if not events:
        return "No open close approaches right now.", [events]
    return "Highest-risk open events: " + " ".join(describe_event(e) for e in events), [events]


# ---------- model loop ----------

@dataclass
class Answer:
    text: str
    mode: str
    refused: bool = False
    tools_used: list[str] = field(default_factory=list)


def tool_specs() -> list[dict]:
    return [{"type": "function", "function": {"name": n, "description": d,
                                              "parameters": {"type": "object", "properties": p}}}
            for n, (_, d, p) in TOOLS.items()]


SYSTEM = ("You help satellite operators understand close approaches, orbit changes and space weather. "
          "Use the tools for every fact. Quote numbers exactly as the tools give them. You cannot record decisions, "
          "change thresholds or send alerts; say so if asked. Keep answers under 120 words.")


def model_answer(session: Session, org: Org | None, question: str, settings: Settings,
                 client: httpx.Client) -> tuple[str, list, list[str]]:
    messages = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": question}]
    evidence, used = [], []
    for _ in range(4):
        resp = client.post(f"{settings.copilot_base_url.rstrip('/')}/chat/completions",
                           headers={"Authorization": f"Bearer {settings.copilot_api_key}"},
                           json={"model": settings.copilot_model, "messages": messages, "tools": tool_specs()})
        resp.raise_for_status()
        msg = resp.json()["choices"][0]["message"]
        calls = msg.get("tool_calls") or []
        if not calls:
            return msg.get("content") or "", evidence, used
        messages.append(msg)
        for call in calls:
            name = call["function"]["name"]
            args = json.loads(call["function"].get("arguments") or "{}")
            result = call_tool(session, org, name, args) if name in TOOLS else {"error": "unknown tool"}
            evidence.append(result)
            used.append(name)
            messages.append({"role": "tool", "tool_call_id": call["id"], "content": json.dumps(result, default=str)})
    return "", evidence, used


def ask(session: Session, org: Org | None, question: str, settings: Settings, client: httpx.Client | None = None) -> Answer:
    if is_action_request(question):
        return Answer("I can't do that. Decisions, thresholds and alerts are set by a person in the console. "
                      "I can explain an event or the space weather if that helps.", "guard", refused=True)
    if settings.copilot_base_url and settings.copilot_model:
        own = client is None
        client = client or httpx.Client(timeout=30)
        try:
            text, evidence, used = model_answer(session, org, question, settings, client)
            if text and grounded(text, evidence):
                return Answer(text, "model", tools_used=used)
        except (httpx.HTTPError, KeyError, ValueError):
            pass
        finally:
            if own:
                client.close()
    text, _ = offline_answer(session, org, question)
    return Answer(text, "offline")
