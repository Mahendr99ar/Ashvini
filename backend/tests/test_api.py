import importlib

import httpx
import pytest
from fastapi.testclient import TestClient
from sqlalchemy import select

from ashvini import config
from ashvini.alerts import dispatch
from ashvini.auth import issue_key
from ashvini.db import Channel, Conjunction, Delivery, Org, Watch
from ashvini.pipeline import run_pipeline

from .conftest import NOW, make_settings
from .test_ccsds import SAMPLE_CDM


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("ASHVINI_DATABASE_URL", f"sqlite:///{tmp_path}/test.db")
    monkeypatch.setenv("ASHVINI_SECRET", "test-pepper")
    config.get_settings.cache_clear()
    settings = make_settings(tmp_path, secret="test-pepper")
    from ashvini.db import make_sessionmaker

    factory = make_sessionmaker(settings.database_url)
    run_pipeline(settings, factory, NOW, cache_dir=tmp_path / "cache")
    with factory() as s:
        org = Org(name="Example Orbital")
        s.add(org)
        s.flush()
        keys = {role: issue_key(s, org, role, "test-pepper") for role in ("viewer", "analyst", "admin")}
        closest = s.scalars(select(Conjunction).order_by(Conjunction.miss_km)).first()
        s.add(Watch(org_id=org.id, norad_id=closest.primary_id))
        s.commit()
    import ashvini.api.main as main

    main = importlib.reload(main)
    c = TestClient(main.app)
    c.keys, c.factory, c.closest_id = keys, factory, closest.id
    yield c
    config.get_settings.cache_clear()


def test_health_reports_data_age_and_counts(client):
    body = client.get("/api/health").json()
    assert body["status"] == "ok" and body["objects"] > 1000 and body["open_events"] >= 3
    assert body["last_run"]["counts"]["events_created"] >= 3


def test_catalogue_is_compact_and_complete(client):
    body = client.get("/api/catalogue").json()
    assert body["count"] == len(body["objects"]) > 1000
    assert {"n", "name", "t", "l1", "l2"} <= body["objects"][0].keys()


def test_an_org_key_sees_only_its_watched_events_with_its_thresholds(client):
    everyone = client.get("/api/conjunctions").json()
    mine = client.get("/api/conjunctions", headers={"X-API-Key": client.keys["viewer"]}).json()
    assert 0 < len(mine) < len(everyone)
    assert mine[0]["thresholds"]["action_pc"] == 1e-4


def test_reviews_need_the_right_role_and_a_note_when_it_matters(client):
    url = f"/api/conjunctions/{client.closest_id}/reviews"
    assert client.post(url, json={"decision": "dismiss"}).status_code == 403
    assert client.post(url, json={"decision": "dismiss"}, headers={"X-API-Key": client.keys["viewer"]}).status_code == 403
    no_note = client.post(url, json={"decision": "dismiss"}, headers={"X-API-Key": client.keys["analyst"]})
    assert no_note.status_code == 422
    ok = client.post(url, json={"decision": "manoeuvre", "note": "burn planned at T-6h"}, headers={"X-API-Key": client.keys["analyst"]})
    assert ok.status_code == 201
    assert ok.json()["reviews"][-1]["note"] == "burn planned at T-6h"


def test_a_bad_key_is_refused(client):
    assert client.get("/api/me", headers={"X-API-Key": "ash_nope_nope"}).status_code == 401


def test_cdm_upload_attaches_to_the_matching_event(client):
    files = {"file": ("event.cdm", SAMPLE_CDM, "text/plain")}
    r = client.post("/api/upload/cdm", files=files, headers={"X-API-Key": client.keys["analyst"]})
    assert r.status_code == 201
    body = r.json()
    assert body["created"] is False
    assert body["event"]["origin"] == "cdm"
    assert any(h["origin"] == "cdm" for h in body["event"]["history"])


def test_cdm_export_is_readable_kvn(client):
    text = client.get(f"/api/conjunctions/{client.closest_id}/cdm").text
    assert "CCSDS_CDM_VERS = 1.0" in text and "OBJECT = OBJECT2" in text


def test_admin_can_change_thresholds_and_it_is_audited(client):
    h = {"X-API-Key": client.keys["admin"]}
    assert client.put("/api/admin/thresholds", json={"action_pc": 1e-6, "monitor_pc": 1e-3}, headers=h).status_code == 422
    assert client.put("/api/admin/thresholds", json={"action_pc": 1e-5, "monitor_pc": 1e-7}, headers=h).status_code == 200
    log = client.get("/api/admin/audit", headers=h).json()
    assert log[0]["action"] == "thresholds.changed"


def test_copilot_answers_from_data_and_refuses_to_act(client):
    a = client.post("/api/copilot", json={"question": "what is most urgent?"}).json()
    assert a["mode"] == "offline" and "Pc" in a["answer"]
    r = client.post("/api/copilot", json={"question": "record a dismiss decision for the top event"}).json()
    assert r["refused"] is True
    w = client.post("/api/copilot", json={"question": "is a storm coming?"}).json()
    assert "Kp" in w["answer"]


def test_alerts_go_out_once_per_level(client):
    seen = []
    transport = httpx.MockTransport(lambda req: seen.append(req) or httpx.Response(200))
    with client.factory() as s:
        org = s.scalars(select(Org)).one()
        s.add(Channel(org_id=org.id, kind="slack", target="https://hooks.example/x", min_level="manoeuvre"))
        s.commit()
        settings = make_settings("/tmp")
        with httpx.Client(transport=transport) as http:
            first = dispatch(s, settings, http)
            again = dispatch(s, settings, http)
        assert first == 1 and again == 0
        assert len(s.scalars(select(Delivery)).all()) == 1
    assert b"MANOEUVRE" in seen[0].content


def test_bare_address_points_to_the_docs(client):
    r = client.get("/", follow_redirects=False)
    assert r.status_code in (302, 307) and r.headers["location"] == "/docs"
