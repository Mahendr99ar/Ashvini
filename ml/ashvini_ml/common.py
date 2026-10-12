"""Things every module needs: seeds, splits, metrics, conformal intervals, model cards and run logging."""

import json
import os
import platform
import random
import subprocess
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

import numpy as np


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch

        torch.manual_seed(seed)
        torch.use_deterministic_algorithms(True, warn_only=True)
    except ImportError:
        pass


def torch_device():
    """The GPU when there is one (Colab, Kaggle), otherwise the CPU."""
    import torch

    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def group_split(groups: np.ndarray, fractions: tuple[float, float, float], seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Split by group (event, object, satellite) so nothing from one group leaks across sets."""
    unique = np.unique(groups)
    rng = np.random.default_rng(seed)
    rng.shuffle(unique)
    n_train = int(len(unique) * fractions[0])
    n_val = int(len(unique) * fractions[1])
    train_g, val_g = set(unique[:n_train]), set(unique[n_train:n_train + n_val])
    train = np.array([g in train_g for g in groups])
    val = np.array([g in val_g for g in groups])
    return train, val, ~(train | val)


def time_split(times: np.ndarray, train_end, val_end) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Train before `train_end`, validate up to `val_end`, test after. The future never trains the past."""
    train = times < train_end
    val = (times >= train_end) & (times < val_end)
    return train, val, times >= val_end


# ---------- metrics ----------

def rmse(y: np.ndarray, p: np.ndarray) -> float:
    return float(np.sqrt(np.mean((np.asarray(y) - np.asarray(p)) ** 2)))


def mae(y: np.ndarray, p: np.ndarray) -> float:
    return float(np.mean(np.abs(np.asarray(y) - np.asarray(p))))


def fbeta(tp: int, fp: int, fn: int, beta: float) -> float:
    if tp == 0:
        return 0.0
    precision, recall = tp / (tp + fp), tp / (tp + fn)
    b2 = beta * beta
    return (1 + b2) * precision * recall / (b2 * precision + recall)


def true_skill_statistic(y: np.ndarray, p: np.ndarray) -> float:
    """Hit rate minus false alarm rate. Used for storms and flares because it ignores class balance."""
    y, p = np.asarray(y, bool), np.asarray(p, bool)
    tp, fn = int((y & p).sum()), int((y & ~p).sum())
    fp, tn = int((~y & p).sum()), int((~y & ~p).sum())
    hit = tp / (tp + fn) if tp + fn else 0.0
    false_alarm = fp / (fp + tn) if fp + tn else 0.0
    return hit - false_alarm


def interval_coverage(y: np.ndarray, low: np.ndarray, high: np.ndarray) -> float:
    y = np.asarray(y)
    return float(np.mean((y >= low) & (y <= high)))


# ---------- conformal ----------

def conformal_half_width(residuals: np.ndarray, coverage: float = 0.9) -> float:
    """Split conformal: the finite-sample quantile of absolute calibration residuals."""
    r = np.sort(np.abs(np.asarray(residuals)))
    n = len(r)
    k = min(n - 1, int(np.ceil((n + 1) * coverage)) - 1)
    return float(r[max(k, 0)])


# ---------- model cards ----------

@dataclass
class ModelCard:
    name: str
    task: str
    data: str
    split: str
    baseline: str
    results: dict
    beats_baseline: bool
    limitations: list[str] = field(default_factory=list)
    features: list[str] = field(default_factory=list)
    extra: dict = field(default_factory=dict)

    def save(self, out_dir: Path) -> Path:
        out_dir.mkdir(parents=True, exist_ok=True)
        body = {**self.__dict__, "created": datetime.now(UTC).isoformat(timespec="seconds"), "environment": environment()}
        path = out_dir / f"{self.name}.json"
        path.write_text(json.dumps(body, indent=2, default=float))
        (out_dir / f"{self.name}.md").write_text(render_card(body))
        return path


def render_card(card: dict) -> str:
    rows = "\n".join(f"| {k} | {v} |" for k, v in flatten(card["results"]).items())
    lims = "\n".join(f"- {x}" for x in card["limitations"]) or "- None recorded yet."
    verdict = "beats" if card["beats_baseline"] else "does not beat"
    return (f"# {card['name']}\n\n{card['task']}\n\n**Data:** {card['data']}\n\n**Split:** {card['split']}\n\n"
            f"**Baseline:** {card['baseline']}. This model {verdict} it on held-out data.\n\n"
            f"| Metric | Value |\n|---|---|\n{rows}\n\n## Limitations\n\n{lims}\n")


def flatten(d: dict, prefix: str = "") -> dict:
    out = {}
    for k, v in d.items():
        key = f"{prefix}{k}"
        if isinstance(v, dict):
            out.update(flatten(v, key + "."))
        else:
            out[key] = round(v, 4) if isinstance(v, float) else v
    return out


def environment() -> dict:
    try:
        commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, timeout=5).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        commit = ""
    return {"python": platform.python_version(), "commit": commit or None}


def log_run(name: str, params: dict, metrics: dict) -> None:
    """Log to MLflow when it is installed and MLFLOW_TRACKING_URI is set; otherwise do nothing."""
    if not os.environ.get("MLFLOW_TRACKING_URI"):
        return
    try:
        import mlflow
    except ImportError:
        return
    with mlflow.start_run(run_name=name):
        mlflow.log_params({k: str(v) for k, v in params.items()})
        mlflow.log_metrics({k: float(v) for k, v in flatten(metrics).items() if isinstance(v, int | float)})
