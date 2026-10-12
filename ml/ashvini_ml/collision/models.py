"""Baseline, gradient boosting, a sequence Transformer and a deep ensemble for final-risk prediction."""

import warnings
from dataclasses import dataclass

import lightgbm as lgb
import numpy as np
import pandas as pd
import torch
from torch import nn

from ..common import fbeta, torch_device
from .data import FLOOR, HIGH_RISK

# ---------- the challenge metric ----------

def challenge_score(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """ESA's score: MSE on truly high-risk events divided by F2 of the high-risk call. Lower is better."""
    y = np.maximum(np.asarray(y_true), FLOOR)
    p = np.maximum(np.asarray(y_pred), FLOOR)
    true_hr, pred_hr = y >= HIGH_RISK, p >= HIGH_RISK
    tp = int((true_hr & pred_hr).sum())
    fp = int((~true_hr & pred_hr).sum())
    fn = int((true_hr & ~pred_hr).sum())
    f2 = fbeta(tp, fp, fn, 2.0)
    mse_hr = float(np.mean((y[true_hr] - p[true_hr]) ** 2)) if true_hr.any() else 0.0
    return {"score": mse_hr / f2 if f2 > 0 else float("inf"), "mse_hr": mse_hr, "f2": f2,
            "missed_high_risk": fn, "false_alarms": fp}


def baseline_predict(latest: pd.DataFrame) -> np.ndarray:
    """The latest visible risk, carried forward. Hard to beat in the original competition."""
    return latest["risk"].clip(lower=FLOOR).to_numpy()


# ---------- residual target and the decision step ----------

def residual_target(y: np.ndarray, base: np.ndarray) -> np.ndarray:
    """Learn the change from the latest visible risk, not the risk itself. Most events barely move,
    so the model starts from the strong baseline instead of having to rediscover it."""
    return np.asarray(y) - np.asarray(base)


def adjust(base: np.ndarray, delta: np.ndarray, shrink: float, cut: float) -> np.ndarray:
    """Turn a predicted change into a final forecast.

    `shrink` scales how far we trust the model's change (0 means the baseline exactly).
    `cut` is the level above which an event is called high risk; anything below it is sent
    to the floor, anything above it is kept at least at -6 so the call is made.
    """
    p = np.asarray(base) + shrink * np.asarray(delta)
    return np.where(p >= cut, np.maximum(p, HIGH_RISK), FLOOR)


def tune_decision(y_val: np.ndarray, base_val: np.ndarray, delta_val: np.ndarray) -> dict:
    """Pick shrink and cut on validation events only. shrink=0, cut=-6 is the baseline itself,
    so the search can only choose something else if it scored better on validation."""
    best = {"shrink": 0.0, "cut": HIGH_RISK, "val_score": challenge_score(y_val, adjust(base_val, delta_val, 0.0, HIGH_RISK))["score"]}
    for shrink in np.round(np.arange(0.0, 1.01, 0.1), 2):
        for cut in np.round(np.arange(-7.5, -4.99, 0.1), 2):
            sc = challenge_score(y_val, adjust(base_val, delta_val, shrink, cut))["score"]
            if sc < best["val_score"] - 1e-9:
                best = {"shrink": float(shrink), "cut": float(cut), "val_score": sc}
    return best


# ---------- gradient boosting ----------

def lgbm_params(seed: int, n_estimators: int = 2000) -> dict:
    return dict(n_estimators=n_estimators, learning_rate=0.02, num_leaves=31, min_child_samples=10,
                subsample=0.8, subsample_freq=1, colsample_bytree=0.8, random_state=seed, verbose=-1)


def fit_lgbm_fixed(x: pd.DataFrame, y, n_estimators: int, seed: int = 0) -> lgb.LGBMRegressor:
    """Refit with a known number of trees, no early stopping, so no held-out data is touched."""
    return lgb.LGBMRegressor(**lgbm_params(seed, max(n_estimators, 50))).fit(x, y)


def train_lgbm(x_train: pd.DataFrame, y_train, x_val: pd.DataFrame, y_val, seed: int = 0,
               weights: np.ndarray | None = None) -> lgb.LGBMRegressor:
    """Plain regression. Run 3 showed that up-weighting high-risk events pulls every forecast toward -6
    and floods the high-risk call with false alarms, so weights are now opt-in."""
    warnings.filterwarnings("ignore", message=".*eval_set.*", category=Warning)
    model = lgb.LGBMRegressor(**lgbm_params(seed))
    model.fit(x_train, y_train, sample_weight=weights, eval_set=[(x_val, y_val)],
              callbacks=[lgb.early_stopping(100, verbose=False)])
    return model


# ---------- sequence model ----------

SEQ_COLUMNS = ["time_to_tca", "risk", "max_risk_estimate", "miss_distance", "mahalanobis_distance",
               "t_sigma_t", "c_sigma_t", "relative_speed"]
MAX_LEN = 16


@dataclass
class Scaler:
    mean: np.ndarray
    std: np.ndarray

    @classmethod
    def fit(cls, x: np.ndarray) -> "Scaler":
        x = np.where(np.isfinite(x), x, np.nan).astype(np.float64)  # ignore inf when learning the scale
        return cls(np.nan_to_num(np.nanmean(x, 0)), np.nan_to_num(np.nanstd(x, 0)) + 1e-6)

    def __call__(self, x: np.ndarray) -> np.ndarray:
        z = (np.asarray(x, np.float64) - self.mean) / self.std
        return np.clip(np.nan_to_num(z, nan=0.0, posinf=10.0, neginf=-10.0), -10, 10).astype(np.float32)


LOG_COLUMNS = ("miss_distance", "mahalanobis_distance", "t_sigma_t", "c_sigma_t", "relative_speed")


def log_columns(frame: pd.DataFrame) -> pd.DataFrame:
    """Some ESA columns span dozens of orders of magnitude (Mahalanobis distance reaches values that do not
    fit in float32), so they go in as log10, and anything left is clipped before the float32 cast."""
    out = frame.copy()
    for c in LOG_COLUMNS:
        out[c] = np.log10(out[c].astype(float).abs().clip(lower=1e-3))
    num = [c for c in out.columns if c != "event_id"]
    out[num] = out[num].astype(float).replace([np.inf, -np.inf], np.nan).clip(-1e6, 1e6)
    return out


def sequences(visible: pd.DataFrame, event_ids: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(events, MAX_LEN, features) arrays plus a padding mask, latest CDM last."""
    data = log_columns(visible[["event_id", *SEQ_COLUMNS]])
    groups = {k: g[SEQ_COLUMNS].to_numpy(dtype=np.float32) for k, g in data.groupby("event_id")}
    x = np.zeros((len(event_ids), MAX_LEN, len(SEQ_COLUMNS)), np.float32)
    pad = np.ones((len(event_ids), MAX_LEN), bool)
    for i, e in enumerate(event_ids):
        g = groups[e][-MAX_LEN:]
        x[i, -len(g):] = g
        pad[i, -len(g):] = False
    return x, pad


class CdmTransformer(nn.Module):
    """Reads the CDM sequence of one event and predicts its final log-risk."""

    def __init__(self, n_features: int, d_model: int = 64, layers: int = 2, dropout: float = 0.1):
        super().__init__()
        self.embed = nn.Linear(n_features, d_model)
        block = nn.TransformerEncoderLayer(d_model, nhead=4, dim_feedforward=128, dropout=dropout, batch_first=True)
        self.encoder = nn.TransformerEncoder(block, layers)
        self.head = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, 1))

    def forward(self, x: torch.Tensor, pad: torch.Tensor) -> torch.Tensor:
        h = self.encoder(self.embed(x), src_key_padding_mask=pad)
        last = h[:, -1]  # the newest CDM, which has seen the whole history through attention
        return self.head(last).squeeze(-1)


def fit_torch(model: nn.Module, batches, epochs: int, lr: float = 2e-3) -> nn.Module:
    """Train on the GPU when there is one; the fitted model comes back on the CPU for prediction."""
    dev = torch_device()
    model.to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    model.train()
    for _ in range(epochs):
        for inputs, target, weight in batches():
            inputs, target, weight = [t.to(dev) for t in inputs], target.to(dev), weight.to(dev)
            opt.zero_grad()
            loss = (weight * (model(*inputs) - target) ** 2).mean()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
    model.eval()
    return model.cpu()


def train_transformer(x: np.ndarray, pad: np.ndarray, y: np.ndarray, scaler: Scaler, epochs: int = 40, seed: int = 0,
                      weights: np.ndarray | None = None) -> CdmTransformer:
    torch.manual_seed(seed)
    xs = torch.tensor(scaler(x.reshape(-1, x.shape[-1])).reshape(x.shape), dtype=torch.float32)
    ps, ys = torch.tensor(pad), torch.tensor(y, dtype=torch.float32)
    ws = torch.ones_like(ys) if weights is None else torch.tensor(weights, dtype=torch.float32)
    model = CdmTransformer(x.shape[-1])

    def batches():
        order = torch.randperm(len(ys))
        for i in range(0, len(ys), 64):
            idx = order[i:i + 64]
            yield (xs[idx], ps[idx]), ys[idx], ws[idx]

    return fit_torch(model, batches, epochs)


@torch.no_grad()
def predict_transformer(model: CdmTransformer, x: np.ndarray, pad: np.ndarray, scaler: Scaler) -> np.ndarray:
    xs = torch.tensor(scaler(x.reshape(-1, x.shape[-1])).reshape(x.shape), dtype=torch.float32)
    return model(xs, torch.tensor(pad)).numpy()


# ---------- deep ensemble on tabular features ----------

class Mlp(nn.Module):
    def __init__(self, n: int):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(n, 128), nn.GELU(), nn.Dropout(0.1), nn.Linear(128, 64), nn.GELU(), nn.Linear(64, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


def train_ensemble(x: np.ndarray, y: np.ndarray, members: int = 5, epochs: int = 60,
                   weights: np.ndarray | None = None) -> tuple[list[Mlp], Scaler]:
    scaler = Scaler.fit(x)
    xs, ys = torch.tensor(scaler(x), dtype=torch.float32), torch.tensor(y, dtype=torch.float32)
    ws = torch.ones_like(ys) if weights is None else torch.tensor(weights, dtype=torch.float32)
    models = []
    for m in range(members):
        torch.manual_seed(100 + m)
        model = Mlp(x.shape[1])

        def batches():
            order = torch.randperm(len(ys))
            for i in range(0, len(ys), 64):
                idx = order[i:i + 64]
                yield (xs[idx],), ys[idx], ws[idx]

        models.append(fit_torch(model, batches, epochs))
    return models, scaler


@torch.no_grad()
def predict_ensemble(models: list[Mlp], scaler: Scaler, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    xs = torch.tensor(scaler(x), dtype=torch.float32)
    preds = np.stack([m(xs).numpy() for m in models])
    return preds.mean(0), preds.std(0)
