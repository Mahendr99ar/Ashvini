"""OMNI hourly data and the forecasting table built from it.

OMNI2 low-resolution files (omni2_YYYY.dat, or omni2_all_years.dat) hold 55
whitespace-separated words per hour. The words used here, counted from 1, are:
year (1), day of year (2), hour (3), Bz GSM in nT (17), proton density (24),
flow speed (25), flow pressure (29), Kp x 10 (39), Dst (41), ap (50) and F10.7 (51).
Fill values are all-nines of each field's width.
"""

from pathlib import Path

import numpy as np
import pandas as pd

WORDS = {"bz": 17, "density": 24, "speed": 25, "pressure": 29, "kp10": 39, "dst": 41, "ap": 50, "f107": 51}
FILL = {"bz": 999.9, "density": 999.9, "speed": 9999.0, "pressure": 99.99, "kp10": 99, "dst": 99999, "ap": 999, "f107": 999.9}
HORIZONS = (3, 6, 12, 24)


def load_omni(paths: list[Path]) -> pd.DataFrame:
    frames = [pd.read_csv(p, sep=r"\s+", header=None) for p in paths]
    raw = pd.concat(frames, ignore_index=True)
    time = pd.to_datetime(raw[0].astype(str), format="%Y") + pd.to_timedelta(raw[1] - 1, unit="D") + pd.to_timedelta(raw[2], unit="h")
    df = pd.DataFrame({name: raw[word - 1].astype(float) for name, word in WORDS.items()})
    for name, fill in FILL.items():
        df.loc[df[name] >= fill, name] = np.nan
    df["kp"] = df.pop("kp10") / 10.0
    df.index = pd.DatetimeIndex(time, name="time")
    return df.sort_index()


def synthetic_omni(hours: int = 24 * 365 * 3, seed: int = 0) -> pd.DataFrame:
    """Solar wind with occasional southward-Bz storms; Kp responds a few hours later."""
    rng = np.random.default_rng(seed)
    t = pd.date_range("2015-01-01", periods=hours, freq="h")
    speed = 400 + 80 * np.sin(2 * np.pi * np.arange(hours) / (27 * 24)) + rng.normal(0, 30, hours)
    bz = rng.normal(0, 2.5, hours)
    for start in rng.choice(hours - 48, size=hours // 400, replace=False):
        length = int(rng.integers(6, 30))
        depth = rng.uniform(5, 25)
        bz[start:start + length] -= depth * np.hanning(length)
        speed[start:start + length + 12] += depth * 12
    density = np.clip(rng.lognormal(1.6, 0.4, hours), 0.5, None)
    coupling = np.clip(-bz, 0, None) * speed / 1000.0
    driver = np.convolve(coupling, np.exp(-np.arange(12) / 4.0)[::-1], mode="full")[:hours]
    kp_hourly = np.clip(1.0 + 0.55 * driver + 0.0015 * (speed - 400) + rng.normal(0, 0.3, hours), 0, 9)
    kp = pd.Series(kp_hourly, index=t).resample("3h").mean().round(1).reindex(t, method="ffill")
    return pd.DataFrame({"bz": bz, "density": density, "speed": speed, "pressure": 1.67e-6 * density * speed**2,
                         "kp": kp.values, "dst": -10 * driver, "ap": np.nan, "f107": 120.0}, index=pd.DatetimeIndex(t, name="time"))


def kp_at(df: pd.DataFrame, hours_ahead: int) -> pd.Series:
    return df["kp"].shift(-hours_ahead)


def build_table(df: pd.DataFrame) -> pd.DataFrame:
    """One row per hour: what is known now, and the Kp values to predict."""
    out = pd.DataFrame(index=df.index)
    out["kp_now"] = df["kp"]
    for lag in (3, 6, 12, 24):
        out[f"kp_lag{lag}"] = df["kp"].shift(lag)
    for window in (3, 6, 12):
        out[f"bz_mean{window}"] = df["bz"].rolling(window, min_periods=1).mean()
        out[f"bz_min{window}"] = df["bz"].rolling(window, min_periods=1).min()
        out[f"speed_mean{window}"] = df["speed"].rolling(window, min_periods=1).mean()
    out["ey"] = -df["speed"] * df["bz"] / 1000.0  # dawn-dusk electric field, mV/m
    out["ey_mean3"] = out["ey"].rolling(3, min_periods=1).mean()
    out["density"] = df["density"]
    out["pressure"] = df["pressure"]
    hours = df.index.hour + df.index.minute / 60
    doy = df.index.dayofyear
    out["hour_sin"], out["hour_cos"] = np.sin(2 * np.pi * hours / 24), np.cos(2 * np.pi * hours / 24)
    out["doy_sin"], out["doy_cos"] = np.sin(2 * np.pi * doy / 365.25), np.cos(2 * np.pi * doy / 365.25)
    for h in HORIZONS:
        out[f"kp_recur27_h{h}"] = df["kp"].shift(27 * 24 - h)  # one solar rotation ago, at the target time
        out[f"target_h{h}"] = kp_at(df, h)
    return out


FEATURES = [c for c in build_table(synthetic_omni(24 * 40)).columns if not c.startswith("target")]
