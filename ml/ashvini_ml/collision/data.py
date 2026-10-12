"""ESA Collision Avoidance Challenge data (Zenodo 4463683) and a synthetic stand-in with the same columns.

Each row is one CDM. Rows sharing an `event_id` are the warnings for one close
approach, issued as tracking improves and `time_to_tca` (days) falls. Following the
challenge, a model may only see CDMs issued at least two days before closest
approach, and must predict the `risk` (log10 Pc) of the last CDM of the event.
"""

from pathlib import Path

import numpy as np
import pandas as pd

CUTOFF_DAYS = 2.0
HIGH_RISK = -6.0
FLOOR = -6.001  # the challenge clips low risks here so that "very safe" values do not dominate the error

SERVING_FEATURES = [
    "time_to_tca", "risk", "max_risk_estimate", "miss_distance", "relative_speed",
    "relative_position_r", "relative_position_t", "relative_position_n", "mahalanobis_distance",
    "t_sigma_r", "t_sigma_t", "t_sigma_n", "c_sigma_r", "c_sigma_t", "c_sigma_n",
    "c_object_type_debris", "c_object_type_rocket_body",
]


def load_kelvins(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    return add_object_type_flags(df)


def add_object_type_flags(df: pd.DataFrame) -> pd.DataFrame:
    kind = df.get("c_object_type", pd.Series("", index=df.index)).astype(str).str.upper()
    df = df.copy()
    df["c_object_type_debris"] = (kind == "DEBRIS").astype(float)
    df["c_object_type_rocket_body"] = (kind == "ROCKET BODY").astype(float)
    return df


def synthetic_cdms(n_events: int = 600, seed: int = 0) -> pd.DataFrame:
    """Events whose CDMs drift toward a hidden final risk as time to TCA shrinks.

    About one event in eight ends high risk, roughly the imbalance of the real set.
    """
    rng = np.random.default_rng(seed)
    rows = []
    for event in range(n_events):
        high = rng.random() < 0.125
        final = rng.uniform(-5.8, -3.0) if high else rng.uniform(-25, -7)
        n_cdm = int(rng.integers(4, 15))
        times = np.sort(rng.uniform(0.05, 7.0, n_cdm))[::-1]
        times[-1] = rng.uniform(0.0, 0.9)
        obj_type = rng.choice(["DEBRIS", "PAYLOAD", "ROCKET BODY"], p=[0.6, 0.25, 0.15])
        trend = rng.normal(0, 0.6)
        for t in times:
            noise = rng.normal(0, 0.25 + 0.35 * t)
            risk = final + trend * t + noise
            sigma_scale = 1 + 0.5 * t
            miss = float(np.exp(rng.normal(np.log(80 + 60 * max(-risk - 3, 0)), 0.4)))
            rows.append({
                "event_id": event, "mission_id": int(rng.integers(1, 20)), "time_to_tca": float(t),
                "risk": float(min(risk, -1.0)), "max_risk_estimate": float(min(risk + abs(rng.normal(1.5, 0.5)), -0.5)),
                "miss_distance": miss, "relative_speed": float(rng.uniform(1000, 15000)),
                "relative_position_r": float(rng.normal(0, miss / 4)), "relative_position_t": float(rng.normal(0, miss / 2)),
                "relative_position_n": float(rng.normal(0, miss / 3)), "mahalanobis_distance": float(abs(rng.normal(1.5 - risk / 10, 0.5))),
                "t_sigma_r": 15 * sigma_scale, "t_sigma_t": 300 * sigma_scale, "t_sigma_n": 20 * sigma_scale,
                "c_sigma_r": 80 * sigma_scale, "c_sigma_t": 1500 * sigma_scale, "c_sigma_n": 90 * sigma_scale,
                "c_object_type": obj_type,
            })
    return add_object_type_flags(pd.DataFrame(rows))


def prepare_events(cdms: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    """Per event: the CDMs a model may see (>= 2 days out), and the target from the final CDM.

    Returns (visible CDMs, target per event, latest visible CDM per event).
    """
    cdms = cdms.sort_values(["event_id", "time_to_tca"], ascending=[True, False])
    final = cdms.groupby("event_id").tail(1).set_index("event_id")
    usable = final[final["time_to_tca"] < 1.0].index
    visible = cdms[(cdms["time_to_tca"] >= CUTOFF_DAYS) & cdms["event_id"].isin(usable)]
    keep = visible["event_id"].unique()
    target = final.loc[keep, "risk"].clip(lower=FLOOR)
    latest = visible.groupby("event_id").tail(1).set_index("event_id").loc[keep]
    return visible, target, latest


def trend_features(visible: pd.DataFrame) -> pd.DataFrame:
    """How the event has been moving: count, first risk, slope and last change."""
    def per_event(g: pd.DataFrame) -> pd.Series:
        t, r = g["time_to_tca"].to_numpy(), g["risk"].to_numpy()
        slope = float(np.polyfit(t, r, 1)[0]) if len(g) >= 2 and np.ptp(t) > 0 else 0.0
        return pd.Series({
            "n_cdms": len(g), "risk_first": r[0], "risk_slope": slope,
            "risk_delta_last": r[-1] - r[-2] if len(g) >= 2 else 0.0,
            "miss_delta_last": g["miss_distance"].iloc[-1] - g["miss_distance"].iloc[-2] if len(g) >= 2 else 0.0,
            "span_days": float(np.ptp(t)),
        })

    return visible.groupby("event_id").apply(per_event, include_groups=False)


def feature_table(visible: pd.DataFrame, latest: pd.DataFrame, columns: list[str] | None = None) -> pd.DataFrame:
    numeric = latest.select_dtypes("number").drop(columns=["mission_id"], errors="ignore")
    if columns is not None:
        numeric = numeric.reindex(columns=columns)
    return numeric.join(trend_features(visible))
