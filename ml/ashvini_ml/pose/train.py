"""Spacecraft pose estimation: a direct-regression CNN, scored the way ESA's challenge scored it.

SPEED+ trains on synthetic renders of the Tango spacecraft and tests on lab images
(hardware-in-the-loop), so the real question is the domain gap. This module gives a
plain baseline: a CNN that regresses the attitude quaternion and the position, with
heavy augmentation. Keypoint heatmaps plus PnP (see `solve_pnp`) are the stronger
published approach and the next step.

Layout expected under --root: synthetic/train.json, synthetic/validation.json,
synthetic/images/*.jpg and lightbox/test.json, lightbox/images/*.jpg.

    python -m ashvini_ml.pose.train --root speedplus --epochs 30 --out ../models
    python -m ashvini_ml.pose.train --synthetic --out /tmp/models
"""

import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch
from torch import nn

from ..common import ModelCard, log_run, seed_everything, torch_device

SIZE = 128
ROT_FLOOR_DEG, POS_FLOOR = 0.169, 2.173e-3  # errors below the lab's measurement accuracy count as zero


def pose_score(q_true: np.ndarray, t_true: np.ndarray, q_pred: np.ndarray, t_pred: np.ndarray) -> dict:
    """ESA score: rotation error (radians) plus position error relative to distance, averaged over images."""
    q_true = q_true / np.linalg.norm(q_true, axis=1, keepdims=True)
    q_pred = q_pred / np.linalg.norm(q_pred, axis=1, keepdims=True)
    rot = 2 * np.arccos(np.clip(np.abs(np.sum(q_true * q_pred, axis=1)), 0, 1))
    pos = np.linalg.norm(t_true - t_pred, axis=1) / np.linalg.norm(t_true, axis=1)
    rot = np.where(np.degrees(rot) < ROT_FLOOR_DEG, 0, rot)
    pos = np.where(pos < POS_FLOOR, 0, pos)
    return {"score": float(np.mean(rot + pos)), "rotation_deg": float(np.degrees(np.mean(rot))), "position_rel": float(np.mean(pos))}


def solve_pnp(points_3d: np.ndarray, points_2d: np.ndarray, camera_matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Pose from matched keypoints (EPnP, then refinement). Needs opencv-python-headless."""
    import cv2

    ok, rvec, tvec = cv2.solvePnP(points_3d.astype(np.float64), points_2d.astype(np.float64), camera_matrix, None,
                                  flags=cv2.SOLVEPNP_EPNP)
    if not ok:
        raise RuntimeError("PnP did not converge")
    rvec, tvec = cv2.solvePnPRefineLM(points_3d, points_2d, camera_matrix, None, rvec, tvec)
    return rvec.ravel(), tvec.ravel()


class PoseNet(nn.Module):
    def __init__(self):
        super().__init__()

        def block(i, o):
            return nn.Sequential(nn.Conv2d(i, o, 3, padding=1), nn.BatchNorm2d(o), nn.ReLU(), nn.MaxPool2d(2))

        self.body = nn.Sequential(block(1, 32), block(32, 64), block(64, 128), block(128, 128), nn.AdaptiveAvgPool2d(1))
        self.rot = nn.Linear(128, 4)
        self.pos = nn.Linear(128, 3)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.body(x).flatten(1)
        return nn.functional.normalize(self.rot(h), dim=1), self.pos(h)


def quaternion_loss(q_pred: torch.Tensor, q_true: torch.Tensor) -> torch.Tensor:
    return (1 - torch.abs((q_pred * q_true).sum(1))).mean()  # sign-invariant: q and -q are the same attitude


def load_split(root: Path, split_json: str, image_dir: str, limit: int | None = None):
    from PIL import Image

    items = json.loads((root / split_json).read_text())[:limit]
    images = np.stack([np.asarray(Image.open(root / image_dir / it["filename"]).convert("L").resize((SIZE, SIZE)), np.float32) / 255
                       for it in items])
    q = np.array([it["q_vbs2tango_true"] for it in items], np.float32)
    t = np.array([it["r_Vo2To_vbs_true"] for it in items], np.float32)
    return images[:, None], q, t


def synthetic_pose_data(n: int = 600, seed: int = 0):
    """A bright ellipse whose position, size and angle follow the pose. Enough to exercise the code."""
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:SIZE, 0:SIZE].astype(np.float32)
    images, qs, ts = [], [], []
    for _ in range(n):
        angle = rng.uniform(-math.pi, math.pi)
        q = np.array([math.cos(angle / 2), 0, 0, math.sin(angle / 2)], np.float32)
        t = np.array([rng.uniform(-1, 1), rng.uniform(-1, 1), rng.uniform(5, 12)], np.float32)
        cx, cy, size = SIZE / 2 + t[0] * 25, SIZE / 2 + t[1] * 25, 60 / t[2] * 6
        u = (xx - cx) * math.cos(angle) + (yy - cy) * math.sin(angle)
        v = -(xx - cx) * math.sin(angle) + (yy - cy) * math.cos(angle)
        img = np.exp(-((u / size) ** 2 + (v / (size / 3)) ** 2)) + rng.normal(0, 0.05, (SIZE, SIZE))
        images.append(img.astype(np.float32))
        qs.append(q)
        ts.append(t)
    return np.stack(images)[:, None], np.stack(qs), np.stack(ts)


def augment(x: torch.Tensor) -> torch.Tensor:
    """Brightness, contrast and noise: the cheap half of closing the render-to-lab gap."""
    gain = torch.empty(x.shape[0], 1, 1, 1).uniform_(0.6, 1.4)
    bias = torch.empty(x.shape[0], 1, 1, 1).uniform_(-0.1, 0.1)
    return (x * gain + bias + 0.03 * torch.randn_like(x)).clamp(0, 1)


def train(images, q, t, epochs: int, seed: int) -> tuple[PoseNet, np.ndarray, np.ndarray]:
    torch.manual_seed(seed)
    t_mean, t_std = t.mean(0), t.std(0) + 1e-6
    dev = torch_device()
    x, qq, tt = torch.tensor(images), torch.tensor(q).to(dev), torch.tensor((t - t_mean) / t_std).to(dev)
    model = PoseNet().to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    for _ in range(epochs):
        model.train()
        order = torch.randperm(len(x))
        for i in range(0, len(x), 32):
            b = order[i:i + 32]
            q_hat, t_hat = model(augment(x[b]).to(dev))
            loss = quaternion_loss(q_hat, qq[b]) + nn.functional.smooth_l1_loss(t_hat, tt[b])
            opt.zero_grad()
            loss.backward()
            opt.step()
    return model.eval(), t_mean, t_std


@torch.no_grad()
def predict(model: PoseNet, images, t_mean, t_std) -> tuple[np.ndarray, np.ndarray]:
    dev = next(model.parameters()).device
    qs, ts = [], []
    for i in range(0, len(images), 256):
        q_hat, t_hat = model(torch.tensor(images[i:i + 256]).to(dev))
        qs.append(q_hat.cpu().numpy())
        ts.append(t_hat.cpu().numpy())
    return np.concatenate(qs), np.concatenate(ts) * t_std + t_mean


def run(train_data, test_sets: dict, out_dir: Path, data_name: str, epochs: int, seed: int = 0) -> dict:
    seed_everything(seed)
    model, t_mean, t_std = train(*train_data, epochs, seed)
    results = {}
    mean_q = np.array([[1, 0, 0, 0]], np.float32)
    for name, (images, q, t) in test_sets.items():
        q_hat, t_hat = predict(model, images, t_mean, t_std)
        results[name] = {"posenet": pose_score(q, t, q_hat, t_hat),
                         "mean_pose": pose_score(q, t, np.repeat(mean_q, len(q), 0), np.repeat(t_mean[None], len(q), 0))}
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save({k: v.cpu() for k, v in model.state_dict().items()}, out_dir / "pose-posenet.pt")
    first = next(iter(results.values()))
    ModelCard(
        name="spacecraft-pose", task="Estimate a spacecraft's attitude and position from one greyscale image.", data=data_name,
        split="Train on synthetic renders; test on held-out renders and, with SPEED+, on lab images.",
        baseline="Mean pose of the training set", results=results,
        beats_baseline=first["posenet"]["score"] < first["mean_pose"]["score"],
        limitations=["Direct regression is far weaker than keypoints plus PnP on SPEED+; it is the starting point.",
                     "Lab-image results are the honest number; synthetic-test results flatter every model."],
    ).save(out_dir)
    log_run("pose", {"data": data_name, "epochs": epochs}, results)
    return results


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--root", type=Path)
    p.add_argument("--synthetic", action="store_true")
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--out", type=Path, default=Path("models"))
    args = p.parse_args()
    if args.synthetic:
        images, q, t = synthetic_pose_data()
        cut = int(0.8 * len(q))
        tests = {"synthetic_test": (images[cut:], q[cut:], t[cut:])}
        results = run((images[:cut], q[:cut], t[:cut]), tests, args.out, "synthetic ellipses (smoke test only)", args.epochs)
    elif args.root:
        train_data = load_split(args.root, "synthetic/train.json", "synthetic/images")
        tests = {"synthetic_validation": load_split(args.root, "synthetic/validation.json", "synthetic/images"),
                 "lightbox": load_split(args.root, "lightbox/test.json", "lightbox/images")}
        results = run(train_data, tests, args.out, "SPEED+ (Stanford SLAB / ESA)", args.epochs)
    else:
        p.error("give --root or --synthetic")
    for k, v in results.items():
        print(k, v)


if __name__ == "__main__":
    main()
