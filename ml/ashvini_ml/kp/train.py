"""Kp forecasts: persistence, 27-day recurrence, gradient boosting, a temporal CNN and (optionally) Chronos.

    python -m ashvini_ml.kp.train --omni omni/omni2_*.dat --out ../models
    python -m ashvini_ml.kp.train --synthetic --out /tmp/models
"""

import argparse
import warnings
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
import torch
from torch import nn

from ..common import ModelCard, log_run, rmse, seed_everything, time_split, true_skill_statistic
from .data import FEATURES, HORIZONS, build_table, load_omni, synthetic_omni

STORM_KP = 5.0
WINDOW = 48
DRIVERS = ["bz", "speed", "density", "kp"]


def score(y: np.ndarray, p: np.ndarray, cut: float | None = None) -> dict:
    """RMSE, storm skill when a storm is called at Kp 5, and, if a tuned cut is given, storm skill at that cut."""
    ok = ~np.isnan(y) & ~np.isnan(p)
    out = {"rmse": rmse(y[ok], p[ok]), "storm_tss": true_skill_statistic(y[ok] >= STORM_KP, p[ok] >= STORM_KP)}
    if cut is not None:
        out["storm_cut"] = round(float(cut), 2)
        out["storm_tss_tuned"] = true_skill_statistic(y[ok] >= STORM_KP, p[ok] >= cut)
    return out


def storm_cut(y_val: np.ndarray, p_val: np.ndarray) -> float:
    """A regressor pulls forecasts toward the mean, so it rarely reaches Kp 5 even before a storm.
    Pick, on validation data only, the forecast level at which calling a storm gives the best skill."""
    ok = ~np.isnan(y_val) & ~np.isnan(p_val)
    cuts = np.arange(2.5, 7.01, 0.1)
    return float(max(cuts, key=lambda c: true_skill_statistic(y_val[ok] >= STORM_KP, p_val[ok] >= c)))


def train_gbm(x: pd.DataFrame, y: pd.Series, xv: pd.DataFrame, yv: pd.Series, seed: int) -> lgb.LGBMRegressor:
    warnings.filterwarnings("ignore", message=".*eval_set.*", category=Warning)
    keep, keep_v = y.notna(), yv.notna()
    weights = np.where(y[keep] >= STORM_KP, 3.0, 1.0)  # storms are rare and the costly ones to miss
    model = lgb.LGBMRegressor(n_estimators=1500, learning_rate=0.03, num_leaves=63, min_child_samples=40,
                              subsample=0.8, subsample_freq=1, colsample_bytree=0.8, random_state=seed, verbose=-1)
    model.fit(x[keep], y[keep], sample_weight=weights, eval_set=[(xv[keep_v], yv[keep_v])], callbacks=[lgb.early_stopping(100, verbose=False)])
    return model


class Tcn(nn.Module):
    """Dilated causal convolutions over the last two days of drivers; one output per horizon."""

    def __init__(self, n_in: int, n_out: int, width: int = 32):
        super().__init__()
        layers, c = [], n_in
        for d in (1, 2, 4, 8, 16):
            layers += [nn.ConstantPad1d((2 * d, 0), 0.0), nn.Conv1d(c, width, 3, dilation=d), nn.GELU()]
            c = width
        self.body = nn.Sequential(*layers)
        self.head = nn.Linear(width, n_out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # x: (batch, time, features)
        return self.head(self.body(x.transpose(1, 2))[:, :, -1])


def windows(raw: pd.DataFrame, table: pd.DataFrame, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    drivers = raw[DRIVERS].interpolate(limit=6).fillna(raw[DRIVERS].median()).to_numpy(np.float32)
    targets = table[[f"target_h{h}" for h in HORIZONS]].to_numpy(np.float32)
    idx = np.flatnonzero(mask)
    idx = idx[(idx >= WINDOW) & ~np.isnan(targets[idx]).any(axis=1)]
    x = np.stack([drivers[i - WINDOW + 1:i + 1] for i in idx])
    return x, targets[idx]


def train_tcn(x: np.ndarray, y: np.ndarray, epochs: int, seed: int) -> tuple[Tcn, np.ndarray, np.ndarray]:
    torch.manual_seed(seed)
    mean, std = x.reshape(-1, x.shape[-1]).mean(0), x.reshape(-1, x.shape[-1]).std(0) + 1e-6
    xs, ys = torch.tensor((x - mean) / std), torch.tensor(y)
    model = Tcn(x.shape[-1], y.shape[1])
    opt = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
    for _ in range(epochs):
        order = torch.randperm(len(ys))
        for i in range(0, len(ys), 256):
            b = order[i:i + 256]
            opt.zero_grad()
            loss = nn.functional.mse_loss(model(xs[b]), ys[b])
            loss.backward()
            opt.step()
    return model.eval(), mean, std


def chronos_forecast(history: pd.Series, origins: np.ndarray, horizon: int) -> np.ndarray | None:
    """Zero-shot Chronos median forecast, if the package and weights are available. Otherwise None."""
    try:
        from chronos import ChronosPipeline
    except ImportError:
        return None
    pipe = ChronosPipeline.from_pretrained("amazon/chronos-t5-small", device_map="cpu", torch_dtype=torch.float32)
    values = history.to_numpy(np.float32)
    contexts = [torch.tensor(values[max(0, i - 512):i + 1]) for i in origins]
    out = pipe.predict(contexts, prediction_length=horizon)
    return out.median(dim=1).values[:, -1].numpy()


def run(raw: pd.DataFrame, out_dir: Path, data_name: str, train_end, val_end, epochs: int = 20, seed: int = 0,
        chronos_samples: int = 0) -> dict:
    seed_everything(seed)
    table = build_table(raw)
    train, val, test = time_split(table.index.to_numpy(), np.datetime64(train_end), np.datetime64(val_end))
    results: dict = {}
    for h in HORIZONS:
        y = table[f"target_h{h}"]
        r = {"persistence": score(y[test].to_numpy(), table["kp_now"][test].to_numpy()),
             "recurrence_27d": score(y[test].to_numpy(), table[f"kp_recur27_h{h}"][test].to_numpy())}
        gbm = train_gbm(table.loc[train, FEATURES], y[train], table.loc[val, FEATURES], y[val], seed)
        cut = storm_cut(y[val].to_numpy(), gbm.predict(table.loc[val, FEATURES]))
        r["lightgbm"] = score(y[test].to_numpy(), gbm.predict(table.loc[test, FEATURES]), cut)
        results[f"h{h}"] = r

    x_tr, y_tr = windows(raw, table, train)
    x_va, y_va = windows(raw, table, val)
    x_te, y_te = windows(raw, table, test)
    tcn, mean, std = train_tcn(x_tr, y_tr, epochs, seed)
    with torch.no_grad():
        p_va = tcn(torch.tensor((x_va - mean) / std)).numpy()
        p_te = tcn(torch.tensor((x_te - mean) / std)).numpy()
    for k, h in enumerate(HORIZONS):
        results[f"h{h}"]["tcn"] = score(y_te[:, k], p_te[:, k], storm_cut(y_va[:, k], p_va[:, k]))

    if chronos_samples:
        origins = np.flatnonzero(test)[::max(1, test.sum() // chronos_samples)][:chronos_samples]
        for h in HORIZONS:
            pred = chronos_forecast(raw["kp"].ffill(), origins, h)
            if pred is not None:
                results[f"h{h}"]["chronos_zero_shot"] = score(table[f"target_h{h}"].to_numpy()[origins], pred)

    best_h3 = min((k for k in results["h3"] if k not in ("persistence", "recurrence_27d")), key=lambda k: results["h3"][k]["rmse"])
    lower_error = results["h3"][best_h3]["rmse"] < results["h3"]["persistence"]["rmse"] and \
        results["h24"]["lightgbm"]["rmse"] < results["h24"]["persistence"]["rmse"]
    # guardrail from product/eval-plan.md: storm skill must not fall below persistence
    storms_ok = results["h3"]["lightgbm"].get("storm_tss_tuned", 0) >= results["h3"]["persistence"]["storm_tss"]
    beats = lower_error and storms_ok
    ModelCard(
        name="kp-forecast", task="Forecast the planetary K index 3, 6, 12 and 24 hours ahead from upstream solar wind.",
        data=data_name, split=f"By time: train before {train_end}, validate to {val_end}, test after.",
        baseline="Persistence (Kp now) and 27-day recurrence", results=results, beats_baseline=beats, features=FEATURES,
        limitations=["OMNI is propagated to the bow shock, so lead time beyond about an hour comes from the model, not the data.",
                     "Storms are rare; storm TSS on a few years of test data has wide error bars.",
                     "Chronos runs zero-shot on Kp alone and does not see the solar wind.",
                     "Regression pulls forecasts toward the mean; storm calls use a cut tuned on validation, reported separately."],
        extra={"storm_threshold_kp": STORM_KP, "lower_error_than_persistence": lower_error,
               "storm_guardrail_passed": storms_ok},
    ).save(out_dir)
    log_run("kp-forecast", {"data": data_name, "epochs": epochs}, results)
    return results


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--omni", nargs="*", type=Path)
    p.add_argument("--synthetic", action="store_true")
    p.add_argument("--out", type=Path, default=Path("models"))
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--chronos-samples", type=int, default=0, help="test origins for the zero-shot Chronos comparison")
    args = p.parse_args()
    if args.synthetic:
        raw, name, tr, va = synthetic_omni(), "synthetic solar wind (smoke test only)", "2016-10-01", "2017-04-01"
    elif args.omni:
        raw, name, tr, va = load_omni(args.omni), "NASA OMNI hourly (SPDF)", "2017-01-01", "2020-01-01"
    else:
        p.error("give --omni files or --synthetic")
    for k, v in run(raw, args.out, name, tr, va, args.epochs, chronos_samples=args.chronos_samples).items():
        print(k, v)


if __name__ == "__main__":
    main()
