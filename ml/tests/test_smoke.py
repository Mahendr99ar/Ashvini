import json

from ashvini_ml.train_all import synthetic_jobs


def test_every_module_trains_end_to_end_and_writes_a_card(tmp_path):
    jobs = synthetic_jobs(tmp_path, quick=True)
    for name, job in jobs.items():
        results = job()
        assert results, name
    cards = {p.stem for p in tmp_path.glob("*.json") if not p.name.endswith(".serving.json")}
    assert cards == {"collision-risk", "kp-forecast", "density", "telemetry-anomalies", "orbit-error", "manoeuvre-detection",
                     "debris-families", "avoidance-policy", "spacecraft-pose", "flare-forecast"}
    for path in tmp_path.glob("*.json"):
        if path.name.endswith(".serving.json"):
            continue
        card = json.loads(path.read_text())
        assert {"baseline", "results", "beats_baseline", "limitations"} <= card.keys()
    serving = json.loads((tmp_path / "collision-risk.serving.json").read_text())
    assert len(serving["features"]) == 17 and serving["conformal_half_width"] > 0
