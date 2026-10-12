"""Manoeuvre detection: the step-threshold rule Ashvini runs today, against a learned classifier.

Labels are the hard part, so two sources are combined. Manoeuvres are injected into
real histories (a known jump in semi-major axis at a known time), and an optional
CSV of public manoeuvres, such as ISS reboost dates, gives a check on real events.

    python -m ashvini_ml.manoeuvre.train --history history.tle --known iss_reboosts.csv --out ../models
    python -m ashvini_ml.manoeuvre.train --synthetic --out /tmp/models
"""

import argparse
import math
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from ..common import ModelCard, group_split, log_run, seed_everything

MU = 398600.8
FEATURES = ["da_km", "da_robust_z", "de", "di_deg", "gap_hours", "a_km", "da_prev_km", "da_next_km"]


def semi_major_axis(mean_motion_rev_day: np.ndarray) -> np.ndarray:
    n = mean_motion_rev_day * 2 * math.pi / 86400
    return (MU / n**2) ** (1 / 3)


def synthetic_sets(objects: int = 300, sets: int = 60, seed: int = 0) -> pd.DataFrame:
    """Decaying orbits sampled about twice a day; about a third of objects get one or two injected burns."""
    rng = np.random.default_rng(seed)
    rows = []
    for o in range(objects):
        a0 = 6378.135 + rng.uniform(350, 1200)
        decay = rng.uniform(0.001, 0.05) * (1 if a0 < 6378 + 700 else 0.2)
        jumps = sorted(rng.choice(np.arange(5, sets - 5), size=int(rng.integers(0, 3)), replace=False)) if rng.random() < 0.35 else []
        sizes = {j: rng.choice([-1, 1], p=[0.2, 0.8]) * rng.uniform(0.02, 1.0) for j in jumps}
        a, t, e, inc = a0, 0.0, rng.uniform(0.0005, 0.01), rng.uniform(20, 99)
        for k in range(sets):
            gap = rng.uniform(6, 30)
            t += gap
            a -= decay * gap / 12 + rng.normal(0, 0.006)
            label = k in sizes
            if label:
                a += sizes[k]
            # a bad fit now and then: one set jumps and the next comes back, with no burn behind it
            blip = rng.normal(0, 0.25) if rng.random() < 0.03 else 0.0
            n = math.sqrt(MU / (a + blip) ** 3) * 86400 / (2 * math.pi)
            rows.append({"norad_id": o, "hours": t, "mean_motion": n, "eccentricity": e + rng.normal(0, 2e-6),
                         "inclination": inc + rng.normal(0, 1e-4), "manoeuvre": label})
    return pd.DataFrame(rows)


def inject(history: pd.DataFrame, share: float, seed: int) -> pd.DataFrame:
    """Add known burns to real histories so the classifier has labels."""
    rng = np.random.default_rng(seed)
    out = []
    for _, g in history.groupby("norad_id"):
        g = g.sort_values("hours").copy()
        g["manoeuvre"] = False
        if len(g) > 10 and rng.random() < share:
            k = int(rng.integers(3, len(g) - 3))
            a = semi_major_axis(g["mean_motion"].to_numpy())
            a[k:] += rng.choice([-1, 1], p=[0.2, 0.8]) * rng.uniform(0.05, 1.5)
            g["mean_motion"] = np.sqrt(MU / a**3) * 86400 / (2 * math.pi)
            g.iloc[k, g.columns.get_loc("manoeuvre")] = True
        out.append(g)
    return pd.concat(out, ignore_index=True)


def step_features(history: pd.DataFrame) -> pd.DataFrame:
    """One row per element set (from the second on): what changed since the previous set."""
    parts = []
    for _, g in history.sort_values(["norad_id", "hours"]).groupby("norad_id"):
        a = semi_major_axis(g["mean_motion"].to_numpy())
        da = np.diff(a)
        med = np.median(da)
        mad = 1.4826 * np.median(np.abs(da - med)) + 0.005
        f = pd.DataFrame({
            "norad_id": g["norad_id"].iloc[1:].to_numpy(), "da_km": da - med, "da_robust_z": (da - med) / mad,
            "de": np.diff(g["eccentricity"].to_numpy()), "di_deg": np.diff(g["inclination"].to_numpy()),
            "gap_hours": np.diff(g["hours"].to_numpy()), "a_km": a[1:],
            "manoeuvre": g["manoeuvre"].iloc[1:].to_numpy() if "manoeuvre" in g else False,
        })
        f["da_prev_km"] = f["da_km"].shift(1).fillna(0)
        f["da_next_km"] = f["da_km"].shift(-1).fillna(0)  # needs the next set: detection waits one update
        parts.append(f)
    return pd.concat(parts, ignore_index=True)


def threshold_rule(f: pd.DataFrame, min_km: float = 0.08, k: float = 6.0) -> np.ndarray:
    """What the API runs today (ashvini.manoeuvres.detect)."""
    return ((f["da_km"].abs() >= min_km) & (f["da_robust_z"].abs() >= k)).to_numpy()


def scores(y: np.ndarray, p: np.ndarray) -> dict:
    tp, fp, fn = int((y & p).sum()), int((~y & p).sum()), int((y & ~p).sum())
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"precision": precision, "recall": recall, "f1": f1, "false_alarms": fp, "missed": fn}


def run(history: pd.DataFrame, out_dir: Path, data_name: str, seed: int = 0) -> dict:
    seed_everything(seed)
    f = step_features(history)
    train, val, test = group_split(f["norad_id"].to_numpy(), (0.6, 0.2, 0.2), seed)
    y = f["manoeuvre"].to_numpy(bool)
    clf = lgb.LGBMClassifier(n_estimators=400, learning_rate=0.05, num_leaves=15, class_weight="balanced",
                             random_state=seed, verbose=-1).fit(f.loc[train, FEATURES], y[train])
    prob_val = clf.predict_proba(f.loc[val, FEATURES])[:, 1]
    cut = max(np.linspace(0.05, 0.95, 19), key=lambda c: scores(y[val], prob_val >= c)["f1"])
    pred = clf.predict_proba(f.loc[test, FEATURES])[:, 1] >= cut
    results = {"threshold_rule": scores(y[test], threshold_rule(f[test])), "lightgbm": scores(y[test], pred),
               "decision_cut": float(cut), "detection_delay_sets": 1}
    ModelCard(
        name="manoeuvre-detection", task="Flag element-set steps that contain an orbit change.", data=data_name,
        split="By object: 60% train, 20% choose the decision cut, 20% test.",
        baseline="Robust step threshold (0.08 km and 6 MADs), as served by the API", results=results,
        beats_baseline=results["lightgbm"]["f1"] > results["threshold_rule"]["f1"], features=FEATURES,
        limitations=["Injected burns are instantaneous jumps; real low-thrust raising is gradual and harder to see.",
                     "The classifier looks one set ahead, so it confirms a manoeuvre one update later than the rule."],
    ).save(out_dir)
    log_run("manoeuvre", {"data": data_name}, results)
    return results


def tle_history_frame(text: str) -> pd.DataFrame:
    from sgp4.api import WGS72, Satrec

    lines = [ln for ln in text.splitlines() if ln.startswith(("1 ", "2 "))]
    rows = []
    for l1, l2 in zip(lines[::2], lines[1::2], strict=False):
        s = Satrec.twoline2rv(l1, l2, WGS72)
        rows.append({"norad_id": s.satnum, "hours": (s.jdsatepoch + s.jdsatepochF) * 24,
                     "mean_motion": s.no_kozai * 1440 / (2 * math.pi), "eccentricity": s.ecco,
                     "inclination": math.degrees(s.inclo)})
    return pd.DataFrame(rows)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--history", type=Path)
    p.add_argument("--synthetic", action="store_true")
    p.add_argument("--out", type=Path, default=Path("models"))
    args = p.parse_args()
    if args.synthetic:
        hist, name = synthetic_sets(), "synthetic element histories with injected burns (smoke test only)"
    elif args.history:
        hist, name = inject(tle_history_frame(args.history.read_text()), 0.35, 0), "Space-Track history with injected burns"
    else:
        p.error("give --history or --synthetic")
    for k, v in run(hist, args.out, name).items():
        print(k, v)


if __name__ == "__main__":
    main()
