"""Thermosphere density: an empirical-model baseline against an MLP with Monte Carlo dropout.

Density at 400 km can change several-fold during a storm, and drag scales with it,
so this model is the link between space weather and orbit prediction. Following
Licata and Mehta (2022), the network predicts log10 density and keeps dropout on at
inference: the spread of many forward passes is the uncertainty, and a conformal
correction on held-out data makes the 90% interval honest.

Input: a CSV with time, alt_km, lat, lst (local solar time, hours) and density
(kg/m^3), as derived from CHAMP, GRACE or Swarm accelerometers (TU Delft), plus
OMNI for F10.7 and ap.

    python -m ashvini_ml.density.train --densities champ.csv --omni omni/omni2_*.dat --out ../models
    python -m ashvini_ml.density.train --synthetic --out /tmp/models
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn

from ..common import ModelCard, conformal_half_width, interval_coverage, log_run, mae, seed_everything
from ..kp.data import load_omni

FEATURES = ["alt_km", "lat_sin", "lat_cos", "lst_sin", "lst_cos", "doy_sin", "doy_cos", "f107", "f107a", "ap", "ap_mean24"]


def add_drivers(dens: pd.DataFrame, omni: pd.DataFrame) -> pd.DataFrame:
    drivers = omni[["f107", "ap"]].copy()
    drivers["f107"] = drivers["f107"].ffill()
    drivers["f107a"] = drivers["f107"].rolling(81 * 24, min_periods=24).mean()
    drivers["ap"] = drivers["ap"].ffill()
    drivers["ap_mean24"] = drivers["ap"].rolling(24, min_periods=1).mean()
    merged = pd.merge_asof(dens.sort_values("time"), drivers.reset_index(), on="time", direction="backward")
    return merged


def add_geometry(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    lat, lst, doy = np.radians(df["lat"]), 2 * np.pi * df["lst"] / 24, 2 * np.pi * df["time"].dt.dayofyear / 365.25
    df["lat_sin"], df["lat_cos"] = np.sin(lat), np.cos(lat)
    df["lst_sin"], df["lst_cos"] = np.sin(lst), np.cos(lst)
    df["doy_sin"], df["doy_cos"] = np.sin(doy), np.cos(doy)
    df["log_density"] = np.log10(df["density"])
    if "lon" not in df:
        hours = df["time"].dt.hour + df["time"].dt.minute / 60
        df["lon"] = ((df["lst"] - hours) * 15 + 540) % 360 - 180
    return df


def synthetic_density(n: int = 40000, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    t = pd.Timestamp("2015-01-01") + pd.to_timedelta(np.sort(rng.uniform(0, 3 * 365, n)), unit="D")
    alt, lat, lst = rng.uniform(300, 520, n), rng.uniform(-87, 87, n), rng.uniform(0, 24, n)
    f107 = 120 + 40 * np.sin(2 * np.pi * np.arange(n) / n) + rng.normal(0, 10, n)
    ap = np.clip(rng.lognormal(1.8, 0.9, n), 0, 300)
    scale_h = 50 + 0.2 * (f107 - 70)
    rho = 3e-12 * np.exp(-(alt - 400) / scale_h) * (1 + 0.004 * (f107 - 120)) * (1 + 0.008 * ap) \
        * (1 + 0.3 * np.cos(2 * np.pi * (lst - 14) / 24)) * np.exp(rng.normal(0, 0.08, n))
    return pd.DataFrame({"time": t, "alt_km": alt, "lat": lat, "lst": lst, "density": rho,
                         "f107": f107, "f107a": f107, "ap": ap, "ap_mean24": ap})


def empirical_baseline(train: pd.DataFrame, test: pd.DataFrame) -> tuple[np.ndarray, str]:
    """NRLMSISE-00 through pymsis when it is installed; otherwise a fitted exponential climatology."""
    try:
        import pymsis

        out = pymsis.calculate(test["time"].to_numpy(), test["lon"].to_numpy(), test["lat"].to_numpy(), test["alt_km"].to_numpy(),
                               f107s=test["f107"].to_numpy(), f107as=test["f107a"].to_numpy(),
                               aps=np.repeat(test["ap"].to_numpy()[:, None], 7, axis=1), version=0)
        return np.log10(out[..., 0].ravel()), "NRLMSISE-00 (pymsis)"
    except (ImportError, AttributeError, TypeError):
        cols = ["alt_km", "f107", "ap"]
        a = np.c_[np.ones(len(train)), train[cols]]
        coef, *_ = np.linalg.lstsq(a, train["log_density"], rcond=None)
        return np.c_[np.ones(len(test)), test[cols]] @ coef, "Exponential climatology fitted on training data (pymsis not installed)"


class DropoutMlp(nn.Module):
    def __init__(self, n: int, p: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(n, 128), nn.SiLU(), nn.Dropout(p), nn.Linear(128, 128), nn.SiLU(),
                                 nn.Dropout(p), nn.Linear(128, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


def train_mlp(x: np.ndarray, y: np.ndarray, epochs: int, seed: int) -> tuple[DropoutMlp, np.ndarray, np.ndarray, float, float]:
    torch.manual_seed(seed)
    xm, xs = x.mean(0), x.std(0) + 1e-6
    ym, ys = float(y.mean()), float(y.std() + 1e-6)
    xt, yt = torch.tensor((x - xm) / xs, dtype=torch.float32), torch.tensor((y - ym) / ys, dtype=torch.float32)
    model = DropoutMlp(x.shape[1])
    opt = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
    for _ in range(epochs):
        order = torch.randperm(len(yt))
        for i in range(0, len(yt), 512):
            b = order[i:i + 512]
            opt.zero_grad()
            nn.functional.mse_loss(model(xt[b]), yt[b]).backward()
            opt.step()
    return model, xm, xs, ym, ys


@torch.no_grad()
def mc_predict(model: DropoutMlp, x: np.ndarray, xm, xs, ym, ys, passes: int = 50) -> tuple[np.ndarray, np.ndarray]:
    model.train()  # dropout stays on: that is the point
    xt = torch.tensor((x - xm) / xs, dtype=torch.float32)
    draws = torch.stack([model(xt) for _ in range(passes)]).numpy() * ys + ym
    model.eval()
    return draws.mean(0), draws.std(0)


def run(df: pd.DataFrame, out_dir: Path, data_name: str, epochs: int = 30, seed: int = 0) -> dict:
    seed_everything(seed)
    df = add_geometry(df).dropna(subset=FEATURES + ["log_density"]).sort_values("time").reset_index(drop=True)
    n = len(df)
    train, cal, test = df.iloc[: int(0.7 * n)], df.iloc[int(0.7 * n): int(0.85 * n)], df.iloc[int(0.85 * n):]
    base, base_name = empirical_baseline(train, test)
    model, *norm = train_mlp(train[FEATURES].to_numpy(np.float32), train["log_density"].to_numpy(), epochs, seed)
    cal_mean, cal_std = mc_predict(model, cal[FEATURES].to_numpy(np.float32), *norm)
    z = conformal_half_width((cal["log_density"].to_numpy() - cal_mean) / np.maximum(cal_std, 1e-6), 0.9)
    mean, std = mc_predict(model, test[FEATURES].to_numpy(np.float32), *norm)
    y = test["log_density"].to_numpy()
    raw_cov = interval_coverage(y, mean - 1.645 * std, mean + 1.645 * std)
    results = {
        "baseline": {"mae_log10": mae(y, base), "median_percent_error": float(np.median(np.abs(10 ** (base - y) - 1)) * 100)},
        "mc_dropout_mlp": {"mae_log10": mae(y, mean), "median_percent_error": float(np.median(np.abs(10 ** (mean - y) - 1)) * 100),
                           "coverage_90_raw": raw_cov, "coverage_90_conformal": interval_coverage(y, mean - z * std, mean + z * std),
                           "conformal_z": z},
    }
    ModelCard(
        name="density", task="Predict log10 thermosphere density with a calibrated 90% interval.", data=data_name,
        split="By time: first 70% train, next 15% conformal calibration, last 15% test.", baseline=base_name,
        results=results, beats_baseline=results["mc_dropout_mlp"]["mae_log10"] < results["baseline"]["mae_log10"],
        features=FEATURES,
        limitations=["Trained on one or a few satellites' altitude bands; extrapolation outside them is unreliable.",
                     "Accelerometer-derived densities carry their own calibration error, which the model inherits."],
    ).save(out_dir)
    log_run("density", {"data": data_name}, results)
    return results


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--densities", type=Path)
    p.add_argument("--omni", nargs="*", type=Path)
    p.add_argument("--synthetic", action="store_true")
    p.add_argument("--out", type=Path, default=Path("models"))
    p.add_argument("--epochs", type=int, default=30)
    args = p.parse_args()
    if args.synthetic:
        df, name = synthetic_density(), "synthetic densities (smoke test only)"
    elif args.densities and args.omni:
        dens = pd.read_csv(args.densities, parse_dates=["time"])
        df, name = add_drivers(dens, load_omni(args.omni)), f"{args.densities.name} with NASA OMNI drivers"
    else:
        p.error("give --densities and --omni, or --synthetic")
    for k, v in run(df, args.out, name, args.epochs).items():
        print(k, v)


if __name__ == "__main__":
    main()
