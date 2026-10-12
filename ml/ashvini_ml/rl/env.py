"""A conjunction as a decision problem.

An episode is one close approach. Every step a new CDM arrives (risk, time to TCA)
and the operator can wait, make a small burn, or make a large one. Burning early is
cheap in risk reduction per m/s but may be wasted on an event that would have gone
away; burning late is certain but costs more. The episode ends at the last CDM, and
the cost is propellant plus the expected cost of a collision at the risk that
remains.

The risk sequences come from the same generator as the collision-risk module, or
from real ESA CDM sequences when they are passed in.
"""

import gymnasium as gym
import numpy as np
from gymnasium import spaces

WAIT, SMALL, LARGE = 0, 1, 2
DELTA_V = {WAIT: 0.0, SMALL: 0.05, LARGE: 0.20}  # m/s
RISK_DROP = {WAIT: 0.0, SMALL: 2.0, LARGE: 4.0}  # orders of magnitude, if done early enough
LATE_PENALTY_DAYS = 1.0  # burns closer than this to TCA buy only half the reduction
COLLISION_COST = 1e6
PROPELLANT_COST_PER_MS = 200.0


class ConjunctionEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, sequences: list[tuple[np.ndarray, np.ndarray]], seed: int | None = None, reward_scale: float = 1.0,
                 shaping: bool = False):
        """`sequences`: per event, (time_to_tca days descending, observed log10 risk), the last value being final.

        Training-only options (evaluation uses the defaults, so reported costs are the true ones):
        `reward_scale` shrinks the numbers PPO sees. `shaping` adds potential-based shaping
        (Ng, Harada and Russell, 1999) with the potential set to minus the expected collision
        cost at the current risk. A burn then pays off on the step it is made instead of at
        the end of the episode, and the best policy is unchanged.
        """
        super().__init__()
        self.reward_scale = reward_scale
        self.shaping = shaping
        self.sequences = sequences
        self.rng = np.random.default_rng(seed)
        self.observation_space = spaces.Box(low=-10, high=10, shape=(5,), dtype=np.float32)
        self.action_space = spaces.Discrete(3)

    def _raw(self) -> np.ndarray:
        """Current state in natural units: residual log risk, days to TCA, risk slope per day, CDMs left, orders removed."""
        t, r = self.times, self.risks - self.reduction
        k = self.k
        slope = (r[k] - r[k - 1]) / max(t[k - 1] - t[k], 0.1) if k > 0 else 0.0
        return np.array([r[k], t[k], slope, len(t) - 1 - k, self.reduction], dtype=np.float64)

    def _obs(self) -> np.ndarray:
        """The same state scaled to roughly unit range. ESA writes tiny risks as -30, and CDMs that share a
        time give huge slopes; unscaled, a few such values dominate the policy network."""
        risk, days, slope, left, reduced = self._raw()
        return np.clip(np.array([np.clip(risk, -30, 0) / 10, days / 7, np.clip(slope, -20, 20) / 10, left / 20, reduced / 4]),
                       -10, 10).astype(np.float32)

    def _potential(self) -> float:
        return -COLLISION_COST * 10 ** float(np.clip(self.risks[self.k] - self.reduction, -30, 0))

    def reset(self, *, seed: int | None = None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self.times, self.risks = self.sequences[int(self.rng.integers(len(self.sequences)))]
        self.k, self.reduction, self.spent = 0, 0.0, 0.0
        return self._obs(), {}

    def step(self, action: int):
        action = int(action)
        before = self._potential() if self.shaping else 0.0
        if action != WAIT and self.reduction < RISK_DROP[LARGE]:
            drop = RISK_DROP[action] * (0.5 if self.times[self.k] < LATE_PENALTY_DAYS else 1.0)
            self.reduction = max(self.reduction, drop)
            self.spent += DELTA_V[action]
        reward = -PROPELLANT_COST_PER_MS * DELTA_V[action] if action != WAIT else 0.0
        self.k += 1
        done = self.k >= len(self.times) - 1
        info = {}
        if done:
            final = self.risks[-1] - self.reduction
            expected_loss = COLLISION_COST * 10 ** final
            reward -= expected_loss
            info = {"final_log_risk": float(final), "delta_v": self.spent, "expected_collisions": float(10 ** final)}
        if self.shaping:
            reward += (0.0 if done else self._potential()) - before  # potential of the end state is zero
        return self._obs(), float(reward) * self.reward_scale, done, False, info


def fixed_rule(obs: np.ndarray, action_line: float = -4.0) -> int:
    """Typical practice: one small burn at the last opportunity if the risk is over the action line.
    Reads the scaled observation, so it undoes the scaling first."""
    risk, steps_left, reduced = obs[0] * 10, round(float(obs[3]) * 20), obs[4] * 4
    return SMALL if steps_left <= 1 and risk >= action_line and reduced == 0 else WAIT


def sequences_from_cdms(cdms) -> list[tuple[np.ndarray, np.ndarray]]:
    out = []
    for _, g in cdms.sort_values(["event_id", "time_to_tca"], ascending=[True, False]).groupby("event_id"):
        if len(g) >= 3:
            out.append((g["time_to_tca"].to_numpy(float), g["risk"].to_numpy(float)))
    return out
