"""Debris families from orbital elements.

Every catalogued fragment carries its parent's launch in its international
designator ('1999-025' for Fengyun-1C pieces), which gives free labels. Two tasks:

1. Unsupervised: do HDBSCAN clusters on the elements recover the families?
   Scored with the adjusted Rand index against the designator labels.
2. Attribution: for newly catalogued pieces, which family do they belong to?
   A classifier against the nearest-orbit-plane baseline.

RAAN drifts at a rate that depends on altitude and inclination, so old clouds
spread around the Earth; the features include both RAAN and the drift-free
elements so the models can lean on whichever still holds the family together.

    python -m ashvini_ml.debris.train --catalogue gp.csv --out ../models     # CelesTrak GP CSV
    python -m ashvini_ml.debris.train --synthetic --out /tmp/models
"""

import argparse
import math
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.cluster import HDBSCAN
from sklearn.metrics import adjusted_rand_score
from sklearn.preprocessing import StandardScaler

from ..common import ModelCard, log_run, seed_everything

MU = 398600.8
FEATURES = ["a_km", "ecc", "inc_deg", "raan_sin", "raan_cos", "perigee_km", "apogee_km"]


def from_gp_csv(path: Path, min_family: int = 30) -> pd.DataFrame:
    gp = pd.read_csv(path)
    gp = gp[gp["OBJECT_NAME"].str.contains(" DEB", na=False)]
    n = gp["MEAN_MOTION"] * 2 * math.pi / 86400
    df = pd.DataFrame({"norad_id": gp["NORAD_CAT_ID"], "family": gp["OBJECT_ID"].str[:8],
                       "a_km": (MU / n**2) ** (1 / 3), "ecc": gp["ECCENTRICITY"], "inc_deg": gp["INCLINATION"],
                       "raan_deg": gp["RA_OF_ASC_NODE"]})
    big = df["family"].value_counts()
    return df[df["family"].isin(big[big >= min_family].index)].reset_index(drop=True)


def synthetic_breakups(families: int = 8, seed: int = 0) -> pd.DataFrame:
    """Each parent breaks up with velocity kicks; pieces spread in a, e, i, and RAAN drifts apart over the years."""
    rng = np.random.default_rng(seed)
    rows = []
    for f in range(families):
        a0, i0, raan0 = 6378 + rng.uniform(600, 1100), rng.uniform(65, 100), rng.uniform(0, 360)
        age_years = rng.uniform(1, 15)
        for k in range(int(rng.integers(60, 300))):
            dv = rng.normal(0, 0.025, 3)  # km/s
            v = math.sqrt(MU / a0)
            a = 1 / (2 / a0 - (v + dv[0]) ** 2 / MU)
            ecc = abs(np.linalg.norm(dv[:2]) / v * 2) + rng.uniform(0, 0.002)
            inc = i0 + math.degrees(dv[2] / v)
            drift = -2.06474e14 * a ** -3.5 * math.cos(math.radians(inc)) * 365.25 * age_years  # deg, J2
            drift0 = -2.06474e14 * a0 ** -3.5 * math.cos(math.radians(i0)) * 365.25 * age_years
            rows.append({"norad_id": f * 1000 + k, "family": f"FAM-{f}", "a_km": a, "ecc": ecc, "inc_deg": inc,
                         "raan_deg": (raan0 + drift - drift0 + rng.normal(0, 1)) % 360})
    return pd.DataFrame(rows)


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["raan_sin"], df["raan_cos"] = np.sin(np.radians(df["raan_deg"])), np.cos(np.radians(df["raan_deg"]))
    df["perigee_km"] = df["a_km"] * (1 - df["ecc"]) - 6378.135
    df["apogee_km"] = df["a_km"] * (1 + df["ecc"]) - 6378.135
    return df


def nearest_plane(train: pd.DataFrame, test: pd.DataFrame) -> np.ndarray:
    """Baseline: the family whose mean orbit plane (inclination, RAAN) is closest."""
    centres = train.groupby("family")[["inc_deg", "raan_sin", "raan_cos"]].mean()
    d_inc = (test["inc_deg"].to_numpy()[:, None] - centres["inc_deg"].to_numpy()[None]) / 1.0
    d_raan = np.hypot(test["raan_sin"].to_numpy()[:, None] - centres["raan_sin"].to_numpy()[None],
                      test["raan_cos"].to_numpy()[:, None] - centres["raan_cos"].to_numpy()[None]) * 30
    return centres.index.to_numpy()[np.argmin(np.hypot(d_inc, d_raan), axis=1)]


def run(df: pd.DataFrame, out_dir: Path, data_name: str, seed: int = 0) -> dict:
    seed_everything(seed)
    df = add_features(df)
    scaled = StandardScaler().fit_transform(df[FEATURES])
    clusters = HDBSCAN(min_cluster_size=15, copy=True).fit_predict(scaled)
    rng = np.random.default_rng(seed)
    test = rng.random(len(df)) < 0.25  # pieces catalogued later
    train_df, test_df = df[~test], df[test]
    clf = lgb.LGBMClassifier(n_estimators=300, learning_rate=0.05, num_leaves=15, random_state=seed, verbose=-1)
    clf.fit(train_df[FEATURES], train_df["family"])
    results = {
        "clustering": {"adjusted_rand_index": float(adjusted_rand_score(df["family"], clusters)),
                       "clusters_found": int(len(set(clusters)) - (1 if -1 in clusters else 0)),
                       "families": int(df["family"].nunique()), "noise_share": float(np.mean(clusters == -1))},
        "attribution": {"nearest_plane_top1": float(np.mean(nearest_plane(train_df, test_df) == test_df["family"].to_numpy())),
                        "lightgbm_top1": float(np.mean(clf.predict(test_df[FEATURES]) == test_df["family"].to_numpy()))},
    }
    ModelCard(
        name="debris-families", task="Group fragments into break-up families and attribute new pieces.", data=data_name,
        split="Clustering on all pieces; attribution trains on 75% of pieces and tests on the rest.",
        baseline="Nearest mean orbit plane", results=results,
        beats_baseline=results["attribution"]["lightgbm_top1"] > results["attribution"]["nearest_plane_top1"], features=FEATURES,
        limitations=["Families that share an orbit (two break-ups in one plane) cannot be separated by elements alone.",
                     "The designator labels the launch, which merges pieces from separate events on the same launch."],
    ).save(out_dir)
    log_run("debris", {"data": data_name}, results)
    return results


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--catalogue", type=Path, help="CelesTrak GP data as CSV")
    p.add_argument("--synthetic", action="store_true")
    p.add_argument("--out", type=Path, default=Path("models"))
    args = p.parse_args()
    if args.synthetic:
        df, name = synthetic_breakups(), "synthetic break-ups (smoke test only)"
    elif args.catalogue:
        df, name = from_gp_csv(args.catalogue), "CelesTrak GP catalogue, debris only"
    else:
        p.error("give --catalogue or --synthetic")
    for k, v in run(df, args.out, name).items():
        print(k, v)


if __name__ == "__main__":
    main()
