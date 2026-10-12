"""Train a PPO policy on conjunction episodes and compare it with the fixed rule.

    python -m ashvini_ml.rl.train --cdms kelvins/train_data.csv --steps 200000 --out ../models
    python -m ashvini_ml.rl.train --synthetic --steps 20000 --out /tmp/models
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from ..collision.data import synthetic_cdms
from ..common import ModelCard, log_run, seed_everything
from .env import WAIT, ConjunctionEnv, fixed_rule, sequences_from_cdms


def evaluate(env: ConjunctionEnv, policy, episodes: int, seed: int) -> dict:
    costs, dv, collisions = [], [], []
    for i in range(episodes):
        obs, _ = env.reset(seed=seed + i)
        total, done, info = 0.0, False, {}
        while not done:
            obs, reward, done, _, info = env.step(policy(obs))
            total += reward
        costs.append(-total)
        dv.append(info["delta_v"])
        collisions.append(info["expected_collisions"])
    return {"mean_cost": float(np.mean(costs)), "delta_v_per_event_ms": float(np.mean(dv)),
            "expected_collisions_per_1000_events": float(np.sum(collisions) / episodes * 1000)}


MONITOR_LINE = -6.0  # log10 Pc; below this all through an event, no operator would consider a burn


def relevant(sequences: list, line: float = MONITOR_LINE) -> list:
    """Keep the events whose risk reaches the monitor line at some point.

    Run 3 trained on every event. Almost all of them stay far below 1e-6, so waiting is
    always right, and every policy tied with "never burn". The decision is only
    interesting on events that could plausibly need a burn.
    """
    return [(t, r) for t, r in sequences if np.nanmax(r) >= line]


def run(sequences: list, out_dir: Path, data_name: str, steps: int = 20000, seed: int = 0) -> dict:
    from stable_baselines3 import PPO

    seed_everything(seed)
    total_events = len(sequences)
    sequences = relevant(sequences)
    if len(sequences) < 20:
        raise ValueError(f"only {len(sequences)} of {total_events} events reach the monitor line; too few to train on")
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(sequences))
    cut = int(0.8 * len(order))
    # Rewards run from 0 to about -1e6 on the worst events. At that scale PPO's value estimates are swamped
    # by a few rare episodes and run 4 collapsed to never burning. Scaling by 1e-2 and an entropy bonus
    # keep the burn actions in play long enough to learn when they pay.
    # Rewards run from 0 to about -1e6 on the worst events, and run 5 collapsed to never burning.
    # Training uses scaled rewards with potential-based shaping, which leaves the best policy unchanged
    # but credits a burn on the step it is made. A slice of the training events picks the checkpoint.
    n_val = max(20, (cut) // 8)
    fit_ids, val_ids = order[:cut - n_val], order[cut - n_val:cut]
    train_env = ConjunctionEnv([sequences[i] for i in fit_ids], seed, reward_scale=1e-3, shaping=True)
    val_env = ConjunctionEnv([sequences[i] for i in val_ids], seed + 2)
    test_env = ConjunctionEnv([sequences[i] for i in order[cut:]], seed + 1)
    model = PPO("MlpPolicy", train_env, n_steps=2048, batch_size=256, learning_rate=3e-4, gamma=1.0, ent_coef=0.01,
                seed=seed, verbose=0, device="cpu")  # a small MLP policy runs faster on CPU

    def ppo_policy(obs):
        return int(model.predict(obs, deterministic=True)[0])

    best_cost, best_params, history = float("inf"), None, []
    chunk = max(steps // 8, 2048)
    for done_steps in range(chunk, steps + 1, chunk):
        model.learn(total_timesteps=chunk, reset_num_timesteps=False)
        cost = evaluate(val_env, ppo_policy, len(val_ids), 20_000)["mean_cost"]
        history.append({"steps": done_steps, "validation_cost": round(cost, 4)})
        if cost < best_cost:
            best_cost, best_params = cost, model.get_parameters()
    model.set_parameters(best_params)

    episodes = min(500, len(order) - cut)
    results = {"never_burn": evaluate(test_env, lambda o: WAIT, episodes, 10_000),
               "fixed_rule": evaluate(test_env, fixed_rule, episodes, 10_000),
               "ppo": evaluate(test_env, ppo_policy, episodes, 10_000),
               "events": {"total": total_events, "reach_monitor_line": len(sequences), "test_episodes": episodes},
               "ppo_checkpoints": history}
    out_dir.mkdir(parents=True, exist_ok=True)
    model.save(out_dir / "avoidance-ppo")
    ModelCard(
        name="avoidance-policy", task="Decide when and how much to burn as CDMs for an event arrive.", data=data_name,
        split="Events whose risk reaches 1e-6 at some point; by event, 70% train, 10% checkpoint choice, 20% test.", baseline="One small burn at the last CDM if Pc is above 1e-4",
        results=results, beats_baseline=results["ppo"]["mean_cost"] < min(results["fixed_rule"]["mean_cost"], results["never_burn"]["mean_cost"]),
        limitations=["The burn model is a fixed drop in log risk, not a propagated orbit change.",
                     "Costs are relative: collision 1e6, propellant 200 per m/s. Changing them changes the policy.",
                     "Research only. Ashvini never plans or commands burns.",
                     "Trained and tested only on events that reach 1e-6; on the rest, waiting is trivially right."],
    ).save(out_dir)
    log_run("avoidance-policy", {"data": data_name, "steps": steps}, results)
    return results


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--cdms", type=Path)
    p.add_argument("--synthetic", action="store_true")
    p.add_argument("--steps", type=int, default=20000)
    p.add_argument("--out", type=Path, default=Path("models"))
    args = p.parse_args()
    if args.synthetic:
        seqs, name = sequences_from_cdms(synthetic_cdms(1500)), "synthetic CDM sequences (smoke test only)"
    elif args.cdms:
        seqs, name = sequences_from_cdms(pd.read_csv(args.cdms)), "ESA Collision Avoidance Challenge CDM sequences"
    else:
        p.error("give --cdms or --synthetic")
    for k, v in run(seqs, args.out, name, args.steps).items():
        print(k, v)


if __name__ == "__main__":
    main()
