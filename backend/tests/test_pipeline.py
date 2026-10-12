from sqlalchemy import select

from ashvini.db import Conjunction, ConjunctionUpdate, KpValue, ManoeuvreRow, Review
from ashvini.manoeuvres import detect
from ashvini.pipeline import run_pipeline
from ashvini.risk.decision import Thresholds, recommend

from .conftest import NOW


def test_pipeline_creates_one_event_per_planted_encounter(ran, factory, world):
    assert ran.events_created >= len(world.encounters)
    with factory() as s:
        for planted in world.encounters:
            rows = s.scalars(select(Conjunction).where(Conjunction.secondary_id == planted.secondary_id)).all()
            assert len(rows) == 1
            assert abs(rows[0].miss_km - planted.miss_km) < 0.01


def test_the_closest_planted_event_is_recommended_for_a_manoeuvre(ran, factory, world):
    closest = min(world.encounters, key=lambda e: e.miss_km)
    with factory() as s:
        conj = s.scalars(select(Conjunction).where(Conjunction.secondary_id == closest.secondary_id)).one()
        assert conj.pc > 1e-4
        assert conj.recommendation == "manoeuvre"
        assert any("action line" in r for r in conj.reasons)


def test_rerun_updates_events_but_never_creates_duplicates_or_touches_decisions(ran, settings, factory, tmp_path):
    with factory() as s:
        conj = s.scalars(select(Conjunction)).first()
        s.add(Review(conjunction_id=conj.id, decision="monitor", note="waiting for the next CDM", reviewer="test",
                     recommendation_at_time=conj.recommendation, log_pc_at_time=conj.log_pc))
        s.commit()
        before = s.scalars(select(Conjunction)).all()
    second = run_pipeline(settings, factory, NOW, cache_dir=tmp_path / "cache")
    assert second.events_created == 0
    with factory() as s:
        assert len(s.scalars(select(Conjunction)).all()) == len(before)
        assert len(s.scalars(select(Review)).all()) == 1
        history = s.scalars(select(ConjunctionUpdate).where(ConjunctionUpdate.conjunction_id == conj.id)).all()
        assert len(history) == 2


def test_the_planted_manoeuvre_is_found_and_nothing_else(ran, factory, world):
    with factory() as s:
        rows = s.scalars(select(ManoeuvreRow)).all()
    assert [r.norad_id for r in rows] == [world.manoeuvres[0].norad_id]
    assert abs(rows[0].delta_a_km - world.manoeuvres[0].delta_a_km) < 0.05


def test_space_weather_is_stored(ran, factory):
    with factory() as s:
        assert len(s.scalars(select(KpValue)).all()) == 32


def test_manoeuvre_detector_ignores_steady_decay():
    from datetime import timedelta

    history = [{"norad_id": 1, "epoch": NOW + timedelta(hours=12 * k), "mean_motion": 15.2 + 1e-5 * k} for k in range(10)]
    assert detect(history) == []


def test_recommendation_respects_the_operator_lines():
    strict = Thresholds(action_pc=1e-5, monitor_pc=1e-7)
    loose = Thresholds(action_pc=1e-3, monitor_pc=1e-5)
    args = (-4.5, -4.5, -4.5, "current-pc", 0.8, 30, 0.5)
    assert recommend(*args, strict).level == "manoeuvre"
    assert recommend(*args, loose).level == "monitor"


def test_old_elements_never_produce_a_dismissal():
    rec = recommend(-12, -12, -12, "current-pc", 3.0, 30, 4.5, Thresholds())
    assert rec.level == "monitor" and any("days old" in r for r in rec.reasons)


def test_two_passes_of_one_pair_in_the_same_hour_get_different_ids():
    from datetime import timedelta

    from ashvini.pipeline import new_event_id

    first = new_event_id(1, 2, NOW)
    assert new_event_id(2, 1, NOW) == first
    assert new_event_id(1, 2, NOW + timedelta(minutes=48)) != first
    assert new_event_id(1, 2, NOW, taken=lambda i: i == first) != first
