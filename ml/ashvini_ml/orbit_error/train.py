"""SGP4 error correction.

Propagate an element set forward and compare it with a later element set for the
same object, which is the best public stand-in for truth. The difference, in the
later set's radial / in-track / cross-track frame, is what a model learns to predict
from the older set's own elements and the horizon. The corrected prediction is
SGP4 plus the predicted error. Split by object, so the model is tested on
satellites it never saw.

Input: TLE history text (Space-Track "gp_history" in 3LE or TLE form). Space-Track
data may not be redistributed, so only this script and the resulting numbers go in
the repository.

    python -m ashvini_ml.orbit_error.train --history history.tle --out ../models
    python -m ashvini_ml.orbit_error.train --synthetic --out /tmp/models
"""

import argparse
import math
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sgp4.api import WGS72, Satrec

from ..common import ModelCard, group_split, log_run, seed_everything

HORIZONS = (1.0, 3.0, 7.0)
TOLERANCE_DAYS = 0.5
FEATURES = ["horizon_days", "bstar", "mean_motion", "ecc", "inc_deg", "ndot", "perigee_km", "age_gap_days"]
TARGETS = ["err_r", "err_t", "err_n"]


def read_history(text: str) -> dict[int, list[Satrec]]:
    lines = [ln for ln in text.splitlines() if ln.startswith(("1 ", "2 "))]
    by_object: dict[int, list[Satrec]] = {}
    for l1, l2 in zip(lines[::2], lines[1::2], strict=False):
        sat = Satrec.twoline2rv(l1, l2, WGS72)
        by_object.setdefault(sat.satnum, []).append(sat)
    for sats in by_object.values():
        sats.sort(key=lambda s: s.jdsatepoch + s.jdsatepochF)
    return by_object


def epoch(sat: Satrec) -> float:
    return sat.jdsatepoch + sat.jdsatepochF


def ric(r: np.ndarray, v: np.ndarray) -> np.ndarray:
    radial = r / np.linalg.norm(r)
    cross = np.cross(r, v)
    cross /= np.linalg.norm(cross)
    return np.vstack([radial, np.cross(cross, radial), cross])


def pair_rows(norad: int, sats: list[Satrec]) -> list[dict]:
    rows, epochs = [], np.array([epoch(s) for s in sats])
    for k, old in enumerate(sats):
        for h in HORIZONS:
            j = int(np.argmin(np.abs(epochs - (epochs[k] + h))))
            if j <= k or abs(epochs[j] - epochs[k] - h) > TOLERANCE_DAYS:
                continue
            new = sats[j]
            e1, r1, _ = old.sgp4(new.jdsatepoch, new.jdsatepochF)
            e2, r2, v2 = new.sgp4(new.jdsatepoch, new.jdsatepochF)
            if e1 or e2:
                continue
            err = ric(np.array(r2), np.array(v2)) @ (np.array(r1) - np.array(r2))
            rows.append({"norad_id": norad, "horizon_days": epochs[j] - epochs[k], "bstar": old.bstar,
                         "mean_motion": old.no_kozai * 1440 / (2 * math.pi), "ecc": old.ecco, "inc_deg": math.degrees(old.inclo),
                         "ndot": old.ndot, "perigee_km": old.altp * 6378.135, "age_gap_days": epochs[j] - epochs[k] - h,
                         "err_r": err[0], "err_t": err[1], "err_n": err[2]})
    return rows


def _state(sat: Satrec, days_since_1950: float) -> tuple[np.ndarray, np.ndarray]:
    jd = 2433281.5 + days_since_1950
    _, r, v = sat.sgp4(math.floor(jd), jd - math.floor(jd))
    return np.array(r), np.array(v)


def republish(truth: Satrec, epoch_days: float, bstar: float, mean_anomaly_noise: float) -> Satrec:
    """New elements at `epoch_days` that follow `truth`: start from its mean elements, then nudge the mean
    motion until the two agree along track one day later (SGP4's mean-motion conventions differ internally)."""
    _state(truth, epoch_days)
    e, w, i, m, node, n = truth.em, truth.om % (2 * math.pi), truth.im, truth.mm, truth.Om % (2 * math.pi), truth.nm
    r_true, v_true = _state(truth, epoch_days + 1.0)
    along = v_true / np.linalg.norm(v_true)

    def build(n_try: float) -> Satrec:
        sat = Satrec()
        sat.sgp4init(WGS72, "i", truth.satnum, epoch_days, bstar, 0.0, 0.0, e, w, i, (m + mean_anomaly_noise) % (2 * math.pi), n_try, node)
        return sat

    for _ in range(5):
        r, _ = _state(build(n), epoch_days + 1.0)
        lead_km = float(np.dot(r - r_true, along))
        n -= lead_km / np.linalg.norm(r_true) / 1440.0
    return build(n)


def synthetic_history(objects: int = 120, sets_per_object: int = 40, seed: int = 0) -> dict[int, list[Satrec]]:
    """A 'true' orbit propagated by SGP4 with its real drag, re-published every 12 hours from its mean elements
    at that moment, but with a noisy B* (as real fits are). The noisy B* is the error the model has to learn."""
    rng = np.random.default_rng(seed)
    out: dict[int, list[Satrec]] = {}
    for o in range(objects):
        alt = rng.uniform(350, 800)
        n0 = math.sqrt(398600.8 / (6378.135 + alt) ** 3) * 60
        bstar_true = rng.uniform(0.2, 2.0) * 1e-4 * math.exp(-(alt - 350) / 120)
        truth = Satrec()
        truth.sgp4init(WGS72, "i", 10000 + o, 27678.0, bstar_true, 0.0, 0.0, 0.001, math.radians(rng.uniform(0, 360)),
                       math.radians(rng.uniform(40, 99)), math.radians(rng.uniform(0, 360)), n0, math.radians(rng.uniform(0, 360)))
        sats = []
        for k in range(sets_per_object):
            t_days = 27678.0 + k * 0.5 + rng.uniform(-0.04, 0.04)
            jd = 2433281.5 + t_days
            err, _, _ = truth.sgp4(math.floor(jd), jd - math.floor(jd))
            if err:
                break
            fit_noise = rng.normal(0, 1.0 / (6378.135 + alt))  # about a kilometre along track
            # fits under-estimate drag for low orbits here, a systematic error the model can learn
            fitted_bstar = bstar_true * (0.6 if alt < 500 else 1.0) * rng.lognormal(0, 0.15)
            pub = republish(truth, t_days, fitted_bstar, fit_noise)
            sats.append(pub)
        out[10000 + o] = sats
    return out


def train_component(x: pd.DataFrame, y: pd.Series, seed: int) -> lgb.LGBMRegressor:
    return lgb.LGBMRegressor(n_estimators=600, learning_rate=0.03, num_leaves=31, min_child_samples=20,
                             random_state=seed, verbose=-1).fit(x, y)


def summarise(errors: np.ndarray) -> dict:
    norm = np.linalg.norm(errors, axis=1)
    return {"median_km": float(np.median(norm)), "mean_km": float(norm.mean()), "p90_km": float(np.quantile(norm, 0.9))}


def run(history: dict[int, list[Satrec]], out_dir: Path, data_name: str, seed: int = 0) -> dict:
    seed_everything(seed)
    table = pd.DataFrame([row for norad, sats in history.items() for row in pair_rows(norad, sats)])
    train, val, test = group_split(table["norad_id"].to_numpy(), (0.7, 0.0, 0.3), seed)
    train |= val
    models = {t: train_component(table.loc[train, FEATURES], table.loc[train, t], seed) for t in TARGETS}
    raw = table.loc[test, TARGETS].to_numpy()
    predicted = np.column_stack([models[t].predict(table.loc[test, FEATURES]) for t in TARGETS])
    corrected = raw - predicted
    results = {}
    for h in HORIZONS:
        near = (np.abs(table.loc[test, "horizon_days"].to_numpy() - h) <= TOLERANCE_DAYS)
        if not near.any():
            continue
        results[f"{h:g}d"] = {"sgp4": summarise(raw[near]), "corrected": summarise(corrected[near]),
                              "objects_improved_share": float(np.mean(np.linalg.norm(corrected[near], axis=1) < np.linalg.norm(raw[near], axis=1)))}
    beats = all(r["corrected"]["median_km"] < r["sgp4"]["median_km"] for r in results.values())
    ModelCard(
        name="orbit-error", task="Predict and remove SGP4's position error from the element set's own features.",
        data=data_name, split="By object: 70% of objects train, 30% test.", baseline="Plain SGP4", results=results,
        beats_baseline=beats, features=FEATURES,
        limitations=["Truth is a later element set, which has its own error of order a kilometre.",
                     "Objects that manoeuvre between sets are not removed yet and inflate both errors."],
    ).save(out_dir)
    log_run("orbit-error", {"data": data_name}, results)
    return results


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--history", type=Path)
    p.add_argument("--synthetic", action="store_true")
    p.add_argument("--out", type=Path, default=Path("models"))
    args = p.parse_args()
    if args.synthetic:
        hist, name = synthetic_history(), "synthetic element histories (smoke test only)"
    elif args.history:
        hist, name = read_history(args.history.read_text()), "Space-Track element set history"
    else:
        p.error("give --history or --synthetic")
    for k, v in run(hist, args.out, name).items():
        print(k, v)


if __name__ == "__main__":
    main()
