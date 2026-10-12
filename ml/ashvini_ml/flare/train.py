"""Flare forecasting: will an M-class or larger flare start in the next 24 hours?

Input is a folder of downsampled solar images saved as .npy (for example from the
SDO machine-learning dataset of Galvez et al., 2019) and a CSV with `file`, `time`
and `label` (1 when a flare of M class or above began within 24 hours). The CNN here
is the from-scratch baseline. Fine-tuning IBM and NASA's open Surya model on the
same labels is the stretch step; its weights are on Hugging Face.

Scored with the true skill statistic, which does not reward always saying "no flare".

    python -m ashvini_ml.flare.train --images sdo_npy --labels labels.csv --out ../models
    python -m ashvini_ml.flare.train --synthetic --out /tmp/models
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn

from ..common import ModelCard, log_run, seed_everything, time_split, torch_device, true_skill_statistic

SIZE = 64


def synthetic_disks(n: int = 1500, seed: int = 0) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Disks with active regions; bigger, brighter, more twisted regions flare more often."""
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:SIZE, 0:SIZE].astype(np.float32) - SIZE / 2
    disk = (np.hypot(xx, yy) < SIZE / 2 - 2).astype(np.float32)
    images, labels = [], []
    for _ in range(n):
        img = 0.3 * disk + rng.normal(0, 0.03, (SIZE, SIZE))
        complexity = 0.0
        for _ in range(int(rng.integers(0, 5))):
            cx, cy, s, b = rng.uniform(-20, 20), rng.uniform(-20, 20), rng.uniform(1.5, 5), rng.uniform(0.2, 1.0)
            img += b * np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * s * s)) * disk
            complexity += b * s
        labels.append(rng.random() < 1 / (1 + np.exp(-(complexity - 6))))
        images.append(img.astype(np.float32))
    times = pd.date_range("2012-01-01", periods=n, freq="12h").to_numpy()
    return np.stack(images)[:, None], np.array(labels), times


class FlareCnn(nn.Module):
    def __init__(self):
        super().__init__()

        def block(i, o):
            return nn.Sequential(nn.Conv2d(i, o, 3, padding=1), nn.BatchNorm2d(o), nn.ReLU(), nn.MaxPool2d(2))

        self.net = nn.Sequential(block(1, 16), block(16, 32), block(32, 64), nn.AdaptiveAvgPool2d(1), nn.Flatten(),
                                 nn.Dropout(0.2), nn.Linear(64, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


def train_cnn(x: np.ndarray, y: np.ndarray, epochs: int, seed: int) -> FlareCnn:
    torch.manual_seed(seed)
    dev = torch_device()
    xs, ys = torch.tensor(x), torch.tensor(y, dtype=torch.float32)
    pos_weight = torch.tensor((len(y) - y.sum()) / max(y.sum(), 1), dtype=torch.float32).to(dev)
    model = FlareCnn().to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    for _ in range(epochs):
        model.train()
        order = torch.randperm(len(ys))
        for i in range(0, len(ys), 64):
            b = order[i:i + 64]
            xb = torch.flip(xs[b], dims=[3]) if torch.rand(1).item() < 0.5 else xs[b]  # the Sun has no left and right
            opt.zero_grad()
            loss_fn(model(xb.to(dev)), ys[b].to(dev)).backward()
            opt.step()
    return model.eval()


@torch.no_grad()
def probabilities(model: FlareCnn, x: np.ndarray) -> np.ndarray:
    dev = next(model.parameters()).device
    return np.concatenate([torch.sigmoid(model(torch.tensor(x[i:i + 256]).to(dev))).cpu().numpy() for i in range(0, len(x), 256)])


def run(x: np.ndarray, y: np.ndarray, times: np.ndarray, out_dir: Path, data_name: str, epochs: int = 15, seed: int = 0) -> dict:
    seed_everything(seed)
    q = np.quantile(times, [0.6, 0.8])
    train, val, test = time_split(times, q[0], q[1])
    model = train_cnn(x[train], y[train], epochs, seed)
    p_val = probabilities(model, x[val])
    cut = max(np.linspace(0.05, 0.95, 19), key=lambda c: true_skill_statistic(y[val], p_val >= c))
    p_test = probabilities(model, x[test])
    previous = np.roll(y, 1)[test]  # persistence: flare in the previous window means flare in the next
    results = {"climatology_no_flare": {"tss": true_skill_statistic(y[test], np.zeros(test.sum(), bool))},
               "persistence": {"tss": true_skill_statistic(y[test], previous)},
               "cnn": {"tss": true_skill_statistic(y[test], p_test >= cut), "cut": float(cut)},
               "test_flare_rate": float(y[test].mean())}
    ModelCard(
        name="flare-forecast", task="Predict whether an M-class or larger flare starts within 24 hours.", data=data_name,
        split="By time: first 60% train, next 20% choose the cut, last 20% test.", baseline="Persistence",
        results=results, beats_baseline=results["cnn"]["tss"] > results["persistence"]["tss"],
        limitations=["Flaring is rare and clustered in time; a few active regions dominate any test period.",
                     "Fine-tuning Surya is the planned stronger model and is not included in these numbers."],
    ).save(out_dir)
    log_run("flare", {"data": data_name, "epochs": epochs}, results)
    return results


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--images", type=Path)
    p.add_argument("--labels", type=Path)
    p.add_argument("--synthetic", action="store_true")
    p.add_argument("--epochs", type=int, default=15)
    p.add_argument("--out", type=Path, default=Path("models"))
    args = p.parse_args()
    if args.synthetic:
        x, y, t = synthetic_disks()
        name = "synthetic solar disks (smoke test only)"
    elif args.images and args.labels:
        lab = pd.read_csv(args.labels, parse_dates=["time"]).sort_values("time")
        x = np.stack([np.load(args.images / f) for f in lab["file"]]).astype(np.float32)[:, None]
        y, t, name = lab["label"].to_numpy(bool), lab["time"].to_numpy(), "SDO images with GOES flare labels"
    else:
        p.error("give --images and --labels, or --synthetic")
    for k, v in run(x, y, t, args.out, name, args.epochs).items():
        print(k, v)


if __name__ == "__main__":
    main()
