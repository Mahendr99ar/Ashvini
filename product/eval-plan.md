# Evaluation plan

A model is launch-ready when all of these hold on held-out data, and the results are published in its model card whether it passes or not.

| Module | Held-out set | Must beat | Launch bar | Guardrail |
|---|---|---|---|---|
| Collision risk | ESA challenge test events, split by event | Latest CDM carried forward | Lower challenge score than baseline | No more high-risk events missed than baseline |
| Kp forecast | 2020 to 2025, split by time | Persistence and 27-day recurrence | Lower RMSE at 3 h and 24 h | Storm true skill statistic not below persistence |
| Density | Satellites and years not in training | NRLMSISE-00 | Lower log-density error | 90% interval covers 85 to 95% of truth |
| Telemetry anomalies | ESA-ADB test split | Isolation Forest | Higher event-wise F0.5 | Alarm rate on nominal days reported |
| Orbit error | Objects not in training | Plain SGP4 | Lower 3-day position error | Never worse than SGP4 on more than 20% of objects |
| Manoeuvres | Injected events plus public ISS reboosts | Mean-motion threshold | Higher F1 | Median detection delay reported |
| Debris families | Break-ups not in training | Nearest orbit plane | Higher top-1 parent accuracy | |
| RL policy | Simulated events with seeds not used in training | Fixed Pc rule | Fewer collisions at equal or lower propellant | |

## Product evaluation

- Five to ten sessions with people who match the personas. Record the task, where they hesitated, and what they did next.
- Copilot: every number in an answer must match a tool result. Evals run in CI in offline mode.
- After each session, update `personas.md` and `decision-log.md`.
