"""The recommendation shown next to every event, with the reasons that produced it.

It is a recommendation, never an action. The thresholds belong to the operator and
are always shown with the result, so nobody has to guess why an event is red.
"""

import math
from dataclasses import dataclass, field

ACTION, MONITOR, DISMISS = "manoeuvre", "monitor", "dismiss"
LEVELS = {DISMISS: 0, MONITOR: 1, ACTION: 2}


@dataclass
class Thresholds:
    action_pc: float = 1e-4
    monitor_pc: float = 1e-6


@dataclass
class Recommendation:
    level: str
    reasons: list[str] = field(default_factory=list)


def _fmt_pc(log_value: float) -> str:
    return f"1e{log_value:.1f}" if log_value > -30 else "below 1e-30"


def recommend(log_pc: float, predicted_log_risk: float, predicted_high: float, prediction_source: str,
              miss_km: float, hours_to_tca: float, data_age_days: float, t: Thresholds) -> Recommendation:
    action_line, monitor_line = math.log10(t.action_pc), math.log10(t.monitor_pc)
    reasons = [f"Pc now {_fmt_pc(log_pc)}; action line {t.action_pc:.0e}, monitor line {t.monitor_pc:.0e}."]
    if prediction_source.startswith("model"):
        reasons.append(f"Risk model expects a final risk near {_fmt_pc(predicted_log_risk)} (upper bound {_fmt_pc(predicted_high)}).")
        deciding = max(predicted_log_risk, log_pc)
    else:
        reasons.append("No trained risk model loaded; using the current Pc.")
        deciding = log_pc

    if deciding >= action_line:
        level = ACTION
    elif deciding >= monitor_line or predicted_high >= action_line:
        level = MONITOR
        if predicted_high >= action_line > deciding:
            reasons.append("The upper end of the predicted range crosses the action line.")
    else:
        level = DISMISS

    if data_age_days > 3:
        reasons.append(f"Elements are {data_age_days:.1f} days old; expect the numbers to move with fresh tracking.")
        if level == DISMISS:
            level = MONITOR
    if miss_km < 0.5 and level == DISMISS:
        level = MONITOR
        reasons.append(f"Miss distance is only {miss_km * 1000:.0f} m, so it stays on the watch list.")
    if hours_to_tca < 12 and level == ACTION:
        reasons.append(f"Closest approach in {hours_to_tca:.0f} h: little time left to plan a manoeuvre.")
    return Recommendation(level, reasons)
