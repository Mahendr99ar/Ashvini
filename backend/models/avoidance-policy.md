# avoidance-policy

Decide when and how much to burn as CDMs for an event arrive.

**Data:** ESA CDM sequences

**Split:** Events whose risk reaches 1e-6 at some point; by event, 70% train, 10% checkpoint choice, 20% test.

**Baseline:** One small burn at the last CDM if Pc is above 1e-4. This model does not beat it on held-out data.

| Metric | Value |
|---|---|
| never_burn.mean_cost | 62.0706 |
| never_burn.delta_v_per_event_ms | 0.0 |
| never_burn.expected_collisions_per_1000_events | 0.0621 |
| fixed_rule.mean_cost | 2.4419 |
| fixed_rule.delta_v_per_event_ms | 0.0011 |
| fixed_rule.expected_collisions_per_1000_events | 0.0022 |
| ppo.mean_cost | 62.0706 |
| ppo.delta_v_per_event_ms | 0.0 |
| ppo.expected_collisions_per_1000_events | 0.0621 |
| events.total | 11731 |
| events.reach_monitor_line | 1866 |
| events.test_episodes | 374 |
| ppo_checkpoints | [{'steps': 25000, 'validation_cost': 1.7082}, {'steps': 50000, 'validation_cost': 1.7082}, {'steps': 75000, 'validation_cost': 1.7082}, {'steps': 100000, 'validation_cost': 1.7082}, {'steps': 125000, 'validation_cost': 1.7082}, {'steps': 150000, 'validation_cost': 1.7082}, {'steps': 175000, 'validation_cost': 1.7082}, {'steps': 200000, 'validation_cost': 1.7082}] |

## Limitations

- The burn model is a fixed drop in log risk, not a propagated orbit change.
- Costs are relative: collision 1e6, propellant 200 per m/s. Changing them changes the policy.
- Research only. Ashvini never plans or commands burns.
- Trained and tested only on events that reach 1e-6; on the rest, waiting is trivially right.
