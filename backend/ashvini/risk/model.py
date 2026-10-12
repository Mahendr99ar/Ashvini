"""The trained risk model, if one is configured.

`ml/collision` exports an ONNX regressor that predicts the final log10 risk of an
event from its current features, plus a JSON file with the feature order and the
conformal interval width measured on the calibration set (see `serving_meta_path`). Without a model the
product falls back to the current Pc, and says so.
"""

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class RiskPrediction:
    log_risk: float
    low: float
    high: float
    source: str  # "model:<version>" or "current-pc"
    high_risk: bool | None = None  # the model's call against its tuned cut; None for the fallback


def serving_meta_path(onnx_path: Path) -> Path:
    """`collision-risk.onnx` comes with `collision-risk.serving.json`: feature order, interval width, version."""
    return onnx_path.with_name(onnx_path.stem + ".serving.json")


class RiskModel:
    def __init__(self, onnx_path: Path):
        import onnxruntime as ort

        meta = json.loads(serving_meta_path(onnx_path).read_text())
        self.features: list[str] = meta["features"]
        self.half_width: float = float(meta.get("conformal_half_width", 1.0))
        self.version: str = meta.get("version", onnx_path.stem)
        # From collision-risk-0.2 the model predicts the change from the current risk, then applies
        # a shrink and a high-risk cut that were tuned on validation events.
        self.target: str = meta.get("target", "risk")
        self.shrink: float = float(meta.get("shrink", 1.0))
        self.cut: float = float(meta.get("cut", -6.0))
        self.session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
        self.input_name = self.session.get_inputs()[0].name

    def predict(self, features: dict[str, float]) -> RiskPrediction:
        row = np.array([[features.get(name, np.nan) for name in self.features]], dtype=np.float32)
        raw = float(np.asarray(self.session.run(None, {self.input_name: row})[0]).ravel()[0])
        value = combine(self.target, features.get("risk", np.nan), raw, self.shrink)
        value = float(np.clip(value, -30, 0))
        return RiskPrediction(value, value - self.half_width, min(0.0, value + self.half_width), f"model:{self.version}",
                              high_risk=value >= self.cut)


def combine(target: str, current_log_risk: float, raw: float, shrink: float) -> float:
    """Turn the model output into a log10 risk. Older models predict it directly; newer ones predict a change.
    Unlike training, the product does not floor low values: an operator still wants to see 1e-9 versus 1e-7."""
    if target != "delta_from_risk":
        return raw
    if not np.isfinite(current_log_risk):
        return raw
    return float(max(current_log_risk, -30.0) + shrink * raw)


def fallback_prediction(log_pc: float) -> RiskPrediction:
    """What the ESA challenge called the naive baseline: the latest risk, carried forward."""
    return RiskPrediction(log_pc, log_pc, log_pc, "current-pc")


def model_card_allows(onnx_path: Path) -> bool:
    """Serve the model only if its card says the served model beat carrying the latest risk forward.
    No card (a hand-placed model) is allowed, so local experiments still work."""
    card = onnx_path.with_suffix(".json")
    if not card.exists():
        return True
    info = json.loads(card.read_text())
    return bool(info.get("extra", {}).get("serving_beats_baseline", info.get("beats_baseline", False)))


def load_model(path: str) -> RiskModel | None:
    if not path:
        return None
    p = Path(path)
    if not p.exists() or not model_card_allows(p):
        return None
    return RiskModel(p)
