"""Telemetry anomalies: Isolation Forest on window statistics against a convolutional autoencoder.

ESA-ADB (Kotowski et al., 2024) releases real telemetry from ESA missions with
labelled anomalies. Its own preprocessing resamples every channel onto one time
grid; export that to a table with a `time` column plus one column per channel, and
the labels to a CSV with `StartTime` and `EndTime`.

Scoring is event-wise: a predicted event counts if it overlaps a labelled anomaly,
and a labelled anomaly counts as found if any predicted event overlaps it. The
benchmark also defines a stricter corrected score; this simple version is the one
reported here, and the card says so.

    python -m ashvini_ml.telemetry.train --channels mission1.parquet --labels labels.csv --out ../models
    python -m ashvini_ml.telemetry.train --synthetic --out /tmp/models
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.ensemble import IsolationForest
from torch import nn

from ..common import ModelCard, log_run, seed_everything

WINDOW, STRIDE = 64, 16


def synthetic_telemetry(n: int = 40000, channels: int = 6, seed: int = 0) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    t = np.arange(n)
    data = {f"ch{c}": np.sin(2 * np.pi * t / rng.uniform(200, 900) + rng.uniform(0, 6)) * rng.uniform(0.5, 2)
            + 0.3 * np.sin(2 * np.pi * t / 5800) + rng.normal(0, 0.08, n) for c in range(channels)}
    labels = []
    for start in sorted(rng.choice(np.arange(int(0.3 * n), n - 400), size=24, replace=False)):
        length = int(rng.integers(40, 300))
        c = f"ch{rng.integers(channels)}"
        kind = rng.integers(3)
        if kind == 0:
            data[c][start:start + length] += rng.choice([-1, 1]) * rng.uniform(1.2, 3)
        elif kind == 1:
            data[c][start:start + length] = data[c][start]
        else:
            data[c][start:start + length] += rng.normal(0, 0.8, length)
        labels.append((start, start + length))
    times = pd.date_range("2020-01-01", periods=n, freq="30s")
    frame = pd.DataFrame(data, index=pd.DatetimeIndex(times, name="time"))
    lab = pd.DataFrame({"StartTime": [times[a] for a, _ in labels], "EndTime": [times[min(b, n - 1)] for _, b in labels]})
    return frame, lab


def make_windows(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    starts = np.arange(0, len(x) - WINDOW + 1, STRIDE)
    return np.stack([x[s:s + WINDOW] for s in starts]).astype(np.float32), starts


def window_stats(w: np.ndarray) -> np.ndarray:
    diffs = np.diff(w, axis=1)
    return np.concatenate([w.mean(1), w.std(1), w.min(1), w.max(1), np.abs(diffs).mean(1), diffs.std(1)], axis=1)


class ConvAutoencoder(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.enc = nn.Sequential(nn.Conv1d(channels, 32, 5, stride=2, padding=2), nn.GELU(),
                                 nn.Conv1d(32, 16, 5, stride=2, padding=2), nn.GELU())
        self.dec = nn.Sequential(nn.ConvTranspose1d(16, 32, 4, stride=2, padding=1), nn.GELU(),
                                 nn.ConvTranspose1d(32, channels, 4, stride=2, padding=1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # (batch, time, channels)
        return self.dec(self.enc(x.transpose(1, 2))).transpose(1, 2)


def train_autoencoder(windows: np.ndarray, epochs: int, seed: int) -> ConvAutoencoder:
    torch.manual_seed(seed)
    model = ConvAutoencoder(windows.shape[2])
    opt = torch.optim.AdamW(model.parameters(), lr=2e-3)
    data = torch.tensor(windows)
    for _ in range(epochs):
        order = torch.randperm(len(data))
        for i in range(0, len(data), 128):
            b = data[order[i:i + 128]]
            opt.zero_grad()
            nn.functional.mse_loss(model(b), b).backward()
            opt.step()
    return model.eval()


@torch.no_grad()
def reconstruction_error(model: ConvAutoencoder, windows: np.ndarray) -> np.ndarray:
    """Per window, the worst channel's error, so a fault in one channel is not averaged away."""
    x = torch.tensor(windows)
    return ((model(x) - x) ** 2).mean(1).max(1).values.numpy()


def windows_to_events(flags: np.ndarray, starts: np.ndarray) -> list[tuple[int, int]]:
    """Merge overlapping flagged windows into (start, end) events, in sample indices."""
    events: list[tuple[int, int]] = []
    for flagged, s in zip(flags, starts, strict=True):
        if not flagged:
            continue
        end = s + WINDOW
        if events and s <= events[-1][1]:
            events[-1] = (events[-1][0], max(events[-1][1], end))
        else:
            events.append((int(s), int(end)))
    return events


def event_scores(pred: list[tuple[int, int]], truth: list[tuple[int, int]]) -> dict:
    def overlaps(a, b):
        return a[0] < b[1] and b[0] < a[1]

    correct = sum(any(overlaps(p, t) for t in truth) for p in pred)
    found = sum(any(overlaps(t, p) for p in pred) for t in truth)
    precision = correct / len(pred) if pred else 0.0
    recall = found / len(truth) if truth else 0.0
    return {"precision": precision, "recall": recall, "f0_5": f_beta_from(precision, recall, 0.5),
            "predicted_events": len(pred), "true_events": len(truth)}


def f_beta_from(precision: float, recall: float, beta: float) -> float:
    if precision == 0 or recall == 0:
        return 0.0
    b2 = beta * beta
    return (1 + b2) * precision * recall / (b2 * precision + recall)


def label_index(frame: pd.DataFrame, labels: pd.DataFrame) -> list[tuple[int, int]]:
    idx = frame.index
    return [(int(idx.searchsorted(a)), int(idx.searchsorted(b))) for a, b in zip(labels["StartTime"], labels["EndTime"], strict=True)]


def run(frame: pd.DataFrame, labels: pd.DataFrame, out_dir: Path, data_name: str, epochs: int = 15, seed: int = 0) -> dict:
    seed_everything(seed)
    x = frame.interpolate(limit=10).ffill().bfill().to_numpy(np.float64)
    n = len(x)
    train_end, val_end = int(0.3 * n), int(0.5 * n)
    mean, std = x[:train_end].mean(0), x[:train_end].std(0) + 1e-6
    x = (x - mean) / std
    truth = label_index(frame, labels)
    nominal_train = np.ones(n, bool)
    for a, b in truth:
        nominal_train[a:b] = False

    w, starts = make_windows(x)
    in_train = (starts + WINDOW <= train_end) & np.array([nominal_train[s:s + WINDOW].all() for s in starts])
    in_val = (starts >= train_end) & (starts + WINDOW <= val_end)
    in_test = starts >= val_end
    test_truth = [(a, b) for a, b in truth if a >= val_end]

    def threshold(scores: np.ndarray) -> float:
        return float(np.quantile(scores[in_val], 0.995))

    iso = IsolationForest(n_estimators=300, random_state=seed).fit(window_stats(w[in_train]))
    iso_scores = -iso.score_samples(window_stats(w))
    ae = train_autoencoder(w[in_train], epochs, seed)
    ae_scores = reconstruction_error(ae, w)

    results = {}
    for name, scores in (("isolation_forest", iso_scores), ("conv_autoencoder", ae_scores)):
        flags = scores >= threshold(scores)
        events = windows_to_events(flags[in_test], starts[in_test])
        results[name] = event_scores(events, test_truth)
    ModelCard(
        name="telemetry-anomalies", task="Flag anomalous periods in multichannel satellite telemetry.", data=data_name,
        split="By time: first 30% (nominal windows only) train, next 20% sets the alarm threshold, last 50% test.",
        baseline="Isolation Forest on window statistics", results=results,
        beats_baseline=results["conv_autoencoder"]["f0_5"] > results["isolation_forest"]["f0_5"],
        limitations=["Simple event-wise scoring; ESA-ADB's corrected event-wise F-score is stricter.",
                     "The threshold is a fixed quantile of validation scores, so the alarm rate depends on how clean that period was."],
        extra={"window": WINDOW, "stride": STRIDE},
    ).save(out_dir)
    log_run("telemetry", {"data": data_name}, results)
    return results


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--channels", type=Path)
    p.add_argument("--labels", type=Path)
    p.add_argument("--synthetic", action="store_true")
    p.add_argument("--out", type=Path, default=Path("models"))
    p.add_argument("--epochs", type=int, default=15)
    args = p.parse_args()
    if args.synthetic:
        frame, labels = synthetic_telemetry()
        name = "synthetic telemetry (smoke test only)"
    elif args.channels and args.labels:
        frame = pd.read_parquet(args.channels) if args.channels.suffix == ".parquet" else pd.read_csv(args.channels, parse_dates=["time"])
        frame = frame.set_index("time").sort_index()
        labels = pd.read_csv(args.labels, parse_dates=["StartTime", "EndTime"])
        name = f"ESA-ADB export: {args.channels.name}"
    else:
        p.error("give --channels and --labels, or --synthetic")
    for k, v in run(frame, labels, args.out, name, args.epochs).items():
        print(k, v)


if __name__ == "__main__":
    main()
