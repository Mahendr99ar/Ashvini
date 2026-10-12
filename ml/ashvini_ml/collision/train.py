"""Train and compare every collision-risk model, then export the one the API serves.

    python -m ashvini_ml.collision.train --data kelvins/train_data.csv --out ../models
    python -m ashvini_ml.collision.train --synthetic --out /tmp/models      # no download, CPU, a minute
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from ..common import ModelCard, conformal_half_width, group_split, interval_coverage, log_run, seed_everything
from .data import FLOOR, SERVING_FEATURES, feature_table, load_kelvins, prepare_events, synthetic_cdms
from .models import (
    Scaler,
    adjust,
    baseline_predict,
    challenge_score,
    fit_lgbm_fixed,
    predict_ensemble,
    predict_transformer,
    residual_target,
    sequences,
    train_ensemble,
    train_lgbm,
    train_transformer,
    tune_decision,
)

VERSION = "collision-risk-0.3"


def split_events(event_ids: np.ndarray, seed: int) -> dict[str, np.ndarray]:
    train, val, rest = group_split(event_ids, (0.6, 0.15, 0.25), seed)
    rest_ids = event_ids[rest]
    rng = np.random.default_rng(seed + 1)
    cal_ids = set(rng.choice(rest_ids, size=len(rest_ids) // 2, replace=False))
    cal = np.array([e in cal_ids for e in event_ids]) & rest
    return {"train": train, "val": val, "cal": cal, "test": rest & ~cal}


def export_onnx(model, n_features: int, path: Path) -> None:
    import onnxmltools
    from onnxmltools.convert.common.data_types import FloatTensorType

    onx = onnxmltools.convert_lightgbm(model.booster_, initial_types=[("features", FloatTensorType([None, n_features]))],
                                       target_opset=15)
    path.write_bytes(onx.SerializeToString())


def onnx_matches(path: Path, model, x: pd.DataFrame) -> float:
    import onnxruntime as ort

    sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    onnx_pred = sess.run(None, {"features": x.to_numpy(np.float32)})[0].ravel()
    return float(np.max(np.abs(onnx_pred - model.predict(x))))


def out_of_fold(dev: np.ndarray, fit_predict, folds: int, seed: int) -> np.ndarray:
    """Predictions for every development event from a model that never saw it.

    Run 4 tuned the decision on the 15% validation split, which holds only about 16 high-risk
    events. The chosen line did well there and worse on test. Tuning on out-of-fold predictions
    over train and validation together sees about six times as many high-risk events.
    """
    rng = np.random.default_rng(seed)
    fold_of = rng.permutation(len(dev)) % folds
    out = np.zeros(len(dev))
    for k in range(folds):
        hold = fold_of == k
        out[hold] = fit_predict(dev[~hold], dev[hold])
    return out


def run(cdms: pd.DataFrame, out_dir: Path, data_name: str, epochs: int = 40, seed: int = 0, folds: int = 4) -> dict:
    seed_everything(seed)
    visible, target, latest = prepare_events(cdms)
    events = target.index.to_numpy()
    parts = split_events(events, seed)
    y = target.to_numpy()
    base = baseline_predict(latest)  # latest visible risk, clipped at the floor
    delta = residual_target(y, base)

    full = feature_table(visible, latest).replace([np.inf, -np.inf], np.nan)
    serving = latest.reindex(columns=SERVING_FEATURES).replace([np.inf, -np.inf], np.nan)
    m = parts
    dev = np.flatnonzero(m["train"] | m["val"])  # tuning uses out-of-fold predictions over these
    test = np.flatnonzero(m["test"])
    results: dict = {"baseline": challenge_score(y[test], base[test])}
    decisions: dict = {}

    def score_model(name: str, d_dev: np.ndarray, d_test: np.ndarray) -> np.ndarray:
        decision = tune_decision(y[dev], base[dev], d_dev)
        decisions[name] = decision
        p_test = adjust(base[test], d_test, decision["shrink"], decision["cut"])
        results[name] = {**challenge_score(y[test], p_test), "shrink": decision["shrink"], "cut": decision["cut"]}
        return p_test

    # The baseline with only the line moved: shows how much of any gain comes from the line alone.
    score_model("baseline_tuned_line", np.zeros(len(dev)), np.zeros(len(test)))

    gbms = {}
    for name, x in (("lightgbm_full", full), ("lightgbm_serving", serving)):
        first = train_lgbm(x[m["train"]], delta[m["train"]], x[m["val"]], delta[m["val"]], seed)
        trees = first.best_iteration_ or first.n_estimators

        def gbm_fp(tr, ho, x=x, trees=trees):
            return fit_lgbm_fixed(x.iloc[tr], delta[tr], trees, seed).predict(x.iloc[ho])

        oof = out_of_fold(dev, gbm_fp, folds, seed)
        gbms[name] = fit_lgbm_fixed(x.iloc[dev], delta[dev], trees, seed)
        score_model(name, oof, gbms[name].predict(x.iloc[test]))
    gbm_serve = gbms["lightgbm_serving"]
    serve_test = adjust(base[test], gbm_serve.predict(serving.iloc[test]),
                        decisions["lightgbm_serving"]["shrink"], decisions["lightgbm_serving"]["cut"])

    x_seq, pad = sequences(visible, events)

    def transformer_fp(tr, ho):
        sc = Scaler.fit(x_seq[tr].reshape(-1, x_seq.shape[-1])[~pad[tr].ravel()])
        net = train_transformer(x_seq[tr], pad[tr], delta[tr], sc, epochs, seed)
        return predict_transformer(net, x_seq[ho], pad[ho], sc)

    score_model("transformer", out_of_fold(dev, transformer_fp, folds, seed), transformer_fp(dev, test))

    fill = full.median(numeric_only=True)
    tab = full.fillna(fill).clip(-1e30, 1e30).to_numpy(np.float64)
    ens_std = []

    def ensemble_fp(tr, ho):
        members, sc = train_ensemble(tab[tr], delta[tr], epochs=epochs)
        mean, std = predict_ensemble(members, sc, tab[ho])
        ens_std.append(float(std.mean()))
        return mean

    score_model("deep_ensemble", out_of_fold(dev, ensemble_fp, folds, seed), ensemble_fp(dev, test))
    results["deep_ensemble"]["mean_predictive_std"] = ens_std[-1]

    # Conformal interval for the served model. Events where both the truth and the forecast sit on the
    # floor would make the interval look tiny, so calibrate and check only on events that are not.
    serve_dec = decisions["lightgbm_serving"]
    cal_pred = adjust(base[m["cal"]], gbm_serve.predict(serving[m["cal"]]), serve_dec["shrink"], serve_dec["cut"])
    cal_live = (y[m["cal"]] > FLOOR) | (cal_pred > FLOOR)
    test_live = (y[m["test"]] > FLOOR) | (serve_test > FLOOR)
    half = conformal_half_width(y[m["cal"]][cal_live] - cal_pred[cal_live], 0.9) if cal_live.sum() >= 10 else 1.0
    results["serving_interval"] = {
        "half_width": half, "calibration_events": int(cal_live.sum()), "test_events": int(test_live.sum()),
        "coverage_90": interval_coverage(y[m["test"]][test_live], serve_test[test_live] - half, serve_test[test_live] + half)
        if test_live.any() else None,
    }

    out_dir.mkdir(parents=True, exist_ok=True)
    onnx_path = out_dir / "collision-risk.onnx"
    export_onnx(gbm_serve, len(SERVING_FEATURES), onnx_path)
    results["onnx_max_abs_diff"] = onnx_matches(onnx_path, gbm_serve, serving[m["test"]])
    onnx_path.with_name(onnx_path.stem + ".serving.json").write_text(json.dumps(
        {"features": SERVING_FEATURES, "conformal_half_width": half, "version": VERSION,
         "target": "delta_from_risk", "shrink": serve_dec["shrink"], "cut": serve_dec["cut"],
         "floor": FLOOR}, indent=2))

    model_names = ("lightgbm_full", "lightgbm_serving", "transformer", "deep_ensemble")  # not the tuned-line baseline
    best = min(model_names, key=lambda k: results[k]["score"])
    beats = results[best]["score"] < results["baseline"]["score"]
    serving_beats = results["lightgbm_serving"]["score"] < results["baseline"]["score"]
    ModelCard(
        name="collision-risk",
        task="Predict the final log10 collision risk of an event from CDMs issued at least two days before TCA.",
        data=data_name, split="By event: 75% development (4-fold out-of-fold tuning), 12.5% conformal calibration, 12.5% test.",
        baseline="Latest visible CDM risk carried forward", results=results, beats_baseline=beats, features=SERVING_FEATURES,
        limitations=["Live events are scored from public TLE data, whose covariance is modelled, not measured.",
                     "High-risk events are rare; the test set holds few of them, so the score moves between seeds.",
                     "The serving model uses only features Ashvini can compute live, and is weaker than the full model.",
                     "Models predict the change from the latest risk; shrink and line are chosen on out-of-fold predictions. "
                     "shrink 0 with cut -6 is the baseline, so a tie means the model added nothing on validation."],
        extra={"best_model": best, "serving_model": "lightgbm_serving", "serving_beats_baseline": serving_beats,
               "target": "delta_from_risk", "version": VERSION},
    ).save(out_dir)
    log_run("collision-risk", {"data": data_name, "epochs": epochs, "seed": seed}, results)
    return results


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", type=Path, help="ESA challenge train_data.csv")
    p.add_argument("--synthetic", action="store_true")
    p.add_argument("--out", type=Path, default=Path("models"))
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    if args.synthetic:
        cdms, name = synthetic_cdms(), "synthetic CDMs (smoke test only)"
    elif args.data:
        cdms, name = load_kelvins(args.data), "ESA Collision Avoidance Challenge (Zenodo 4463683)"
    else:
        p.error("give --data or --synthetic")
    results = run(cdms, args.out, name, args.epochs, args.seed)
    for k, v in results.items():
        print(f"{k:>18}: {v}")


if __name__ == "__main__":
    main()
