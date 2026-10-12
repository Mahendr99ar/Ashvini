"""Run every module and collect the model cards in one folder.

    python -m ashvini_ml.train_all --synthetic --out /tmp/models        # the CI smoke run
    python -m ashvini_ml.train_all --config data.json --out ../models   # real data; see README for the file

`data.json` maps module names to their inputs, for example:

    {"collision": {"data": "kelvins/train_data.csv"},
     "kp": {"omni": ["omni/omni2_all_years.dat"]},
     "telemetry": {"channels": "esa-adb/mission1.parquet", "labels": "esa-adb/labels.csv"}}

Modules missing from the file are skipped, so you can train them one at a time.
"""

import argparse
import json
import time
from pathlib import Path

import pandas as pd

from .collision import data as collision_data
from .collision import train as collision
from .debris import train as debris
from .density import train as density
from .flare import train as flare
from .kp import data as kp_data
from .kp import train as kp
from .manoeuvre import train as manoeuvre
from .orbit_error import train as orbit_error
from .pose import train as pose
from .rl import env as rl_env
from .rl import train as rl
from .telemetry import train as telemetry


def synthetic_jobs(out: Path, quick: bool) -> dict:
    e = 3 if quick else 20
    return {
        "collision": lambda: collision.run(collision_data.synthetic_cdms(300 if quick else 1500), out, "synthetic CDMs", epochs=e),
        "kp": lambda: kp.run(kp_data.synthetic_omni(24 * 365 if quick else 24 * 365 * 3), out, "synthetic solar wind",
                             "2015-08-01" if quick else "2016-10-01", "2015-10-15" if quick else "2017-04-01", epochs=e),
        "density": lambda: density.run(density.synthetic_density(8000 if quick else 40000), out, "synthetic densities", epochs=e),
        "telemetry": lambda: telemetry.run(*telemetry.synthetic_telemetry(12000 if quick else 40000), out, "synthetic telemetry", epochs=e),
        "orbit_error": lambda: orbit_error.run(orbit_error.synthetic_history(30 if quick else 120, 20 if quick else 40), out, "synthetic histories"),
        "manoeuvre": lambda: manoeuvre.run(manoeuvre.synthetic_sets(120 if quick else 300), out, "synthetic histories"),
        "debris": lambda: debris.run(debris.synthetic_breakups(5 if quick else 8), out, "synthetic break-ups"),
        "rl": lambda: rl.run(rl_env.sequences_from_cdms(collision_data.synthetic_cdms(400 if quick else 1500)), out,
                             "synthetic CDM sequences", steps=2048 if quick else 30000),
        "pose": lambda: pose.run(*_pose_sets(quick), out, "synthetic ellipses", epochs=1 if quick else 10),
        "flare": lambda: flare.run(*flare.synthetic_disks(300 if quick else 1500), out, "synthetic disks", epochs=e),
    }


def _pose_sets(quick: bool):
    images, q, t = pose.synthetic_pose_data(120 if quick else 600)
    cut = int(0.8 * len(q))
    return (images[:cut], q[:cut], t[:cut]), {"synthetic_test": (images[cut:], q[cut:], t[cut:])}


def real_jobs(cfg: dict, out: Path) -> dict:
    jobs = {}
    if "collision" in cfg:
        jobs["collision"] = lambda: collision.run(collision_data.load_kelvins(Path(cfg["collision"]["data"])), out,
                                                  "ESA Collision Avoidance Challenge")
    if "kp" in cfg:
        jobs["kp"] = lambda: kp.run(kp_data.load_omni([Path(p) for p in cfg["kp"]["omni"]]), out, "NASA OMNI",
                                    "2017-01-01", "2020-01-01")
    if "density" in cfg:
        c = cfg["density"]
        jobs["density"] = lambda: density.run(
            density.add_drivers(pd.read_csv(c["densities"], parse_dates=["time"]), kp_data.load_omni([Path(p) for p in c["omni"]])),
            out, "accelerometer densities with OMNI drivers")
    if "telemetry" in cfg:
        c = cfg["telemetry"]

        def tele():
            frame = pd.read_parquet(c["channels"]).set_index("time").sort_index()
            labels = pd.read_csv(c["labels"], parse_dates=["StartTime", "EndTime"])
            return telemetry.run(frame, labels, out, "ESA-ADB")

        jobs["telemetry"] = tele
    if "orbit_error" in cfg:
        jobs["orbit_error"] = lambda: orbit_error.run(orbit_error.read_history(Path(cfg["orbit_error"]["history"]).read_text()),
                                                      out, "Space-Track history")
    if "manoeuvre" in cfg:
        jobs["manoeuvre"] = lambda: manoeuvre.run(
            manoeuvre.inject(manoeuvre.tle_history_frame(Path(cfg["manoeuvre"]["history"]).read_text()), 0.35, 0),
            out, "Space-Track history with injected burns")
    if "debris" in cfg:
        jobs["debris"] = lambda: debris.run(debris.from_gp_csv(Path(cfg["debris"]["catalogue"])), out, "CelesTrak GP debris")
    if "rl" in cfg:
        jobs["rl"] = lambda: rl.run(rl_env.sequences_from_cdms(pd.read_csv(cfg["rl"]["cdms"])), out, "ESA CDM sequences",
                                    steps=int(cfg["rl"].get("steps", 200000)))
    if "pose" in cfg:
        root = Path(cfg["pose"]["root"])
        jobs["pose"] = lambda: pose.run(pose.load_split(root, "synthetic/train.json", "synthetic/images"),
                                        {"lightbox": pose.load_split(root, "lightbox/test.json", "lightbox/images")},
                                        out, "SPEED+", epochs=int(cfg["pose"].get("epochs", 30)))
    if "flare" in cfg:
        c = cfg["flare"]

        def fl():
            lab = pd.read_csv(c["labels"], parse_dates=["time"]).sort_values("time")
            import numpy as np

            x = np.stack([np.load(Path(c["images"]) / f) for f in lab["file"]]).astype("float32")[:, None]
            return flare.run(x, lab["label"].to_numpy(bool), lab["time"].to_numpy(), out, "SDO with GOES labels")

        jobs["flare"] = fl
    return jobs


CARD_NAMES = {
    "collision": "collision-risk", "kp": "kp-forecast", "density": "density", "telemetry": "telemetry-anomalies",
    "orbit_error": "orbit-error", "manoeuvre": "manoeuvre-detection", "debris": "debris-families",
    "rl": "avoidance-policy", "pose": "spacecraft-pose", "flare": "flare-forecast",
}


def already_done(name: str, out: Path) -> bool:
    """A module is done when its model card exists, so a run that was cut off can pick up where it stopped."""
    return (out / f"{CARD_NAMES[name]}.json").exists()


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--synthetic", action="store_true")
    p.add_argument("--quick", action="store_true", help="tiny sizes, for CI")
    p.add_argument("--config", type=Path)
    p.add_argument("--only", nargs="*", help="module names to run")
    p.add_argument("--out", type=Path, default=Path("models"))
    p.add_argument("--skip-done", action="store_true", help="skip modules whose model card is already in --out")
    args = p.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    jobs = synthetic_jobs(args.out, args.quick) if args.synthetic else real_jobs(json.loads(args.config.read_text()), args.out)
    summary = {}
    for name, job in jobs.items():
        if args.only and name not in args.only:
            continue
        if args.skip_done and already_done(name, args.out):
            print(f"{name:>12}: already done, skipping")
            continue
        t0 = time.perf_counter()
        job()
        summary[name] = round(time.perf_counter() - t0, 1)
        print(f"{name:>12}: done in {summary[name]} s")
    (args.out / "index.json").write_text(json.dumps(sorted(p.stem for p in args.out.glob("*.json")
                                                           if not p.name.endswith(".serving.json") and p.name != "index.json"), indent=2))


if __name__ == "__main__":
    main()
