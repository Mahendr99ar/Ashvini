import math

import numpy as np

from ashvini_ml.collision.models import baseline_predict, challenge_score
from ashvini_ml.common import conformal_half_width, group_split, interval_coverage, true_skill_statistic
from ashvini_ml.pose.train import pose_score
from ashvini_ml.rl.env import SMALL, WAIT, fixed_rule
from ashvini_ml.telemetry.train import event_scores, windows_to_events


def test_perfect_risk_predictions_score_zero():
    y = np.array([-3.0, -5.0, -10.0, -20.0])
    s = challenge_score(y, y)
    assert s["score"] == 0 and s["f2"] == 1 and s["missed_high_risk"] == 0


def test_missing_every_high_risk_event_is_infinitely_bad():
    s = challenge_score(np.array([-3.0, -4.0]), np.array([-20.0, -20.0]))
    assert math.isinf(s["score"]) and s["missed_high_risk"] == 2


def test_baseline_clips_low_risks():
    import pandas as pd

    assert baseline_predict(pd.DataFrame({"risk": [-30.0, -4.0]})).tolist() == [-6.001, -4.0]


def test_group_split_never_shares_a_group():
    groups = np.repeat(np.arange(50), 4)
    tr, va, te = group_split(groups, (0.6, 0.2, 0.2), 0)
    assert not (set(groups[tr]) & set(groups[te])) and not (set(groups[tr]) & set(groups[va]))
    assert (tr | va | te).all()


def test_conformal_interval_covers_about_what_it_promises():
    rng = np.random.default_rng(0)
    cal, test = rng.normal(0, 1, 2000), rng.normal(0, 1, 20000)
    h = conformal_half_width(cal, 0.9)
    assert 0.88 < interval_coverage(test, -h, h) < 0.92


def test_tss_is_zero_for_always_no_and_one_for_perfect():
    y = np.array([1, 0, 0, 1, 0], bool)
    assert true_skill_statistic(y, np.zeros(5, bool)) == 0
    assert true_skill_statistic(y, y) == 1


def test_pose_score_is_zero_for_the_true_pose_and_sign_invariant():
    q = np.array([[0.5, 0.5, 0.5, 0.5]])
    t = np.array([[0.1, 0.2, 8.0]])
    assert pose_score(q, t, q, t)["score"] == 0
    assert pose_score(q, t, -q, t)["score"] == 0


def test_event_scoring_counts_overlaps():
    events = windows_to_events(np.array([0, 1, 1, 0, 0, 0, 0, 1]), np.arange(8) * 64)
    assert events == [(64, 192), (448, 512)]
    s = event_scores(events, [(100, 150), (1000, 1100)])
    assert s["precision"] == 0.5 and s["recall"] == 0.5


def test_fixed_rule_burns_only_at_the_end_and_only_above_the_line():
    # observations are scaled: risk / 10, days / 7, slope / 10, CDMs left / 20, orders removed / 4
    assert fixed_rule(np.array([-0.35, 0.5 / 7, 0, 1 / 20, 0])) == SMALL
    assert fixed_rule(np.array([-0.35, 3.0 / 7, 0, 5 / 20, 0])) == WAIT
    assert fixed_rule(np.array([-0.70, 0.5 / 7, 0, 1 / 20, 0])) == WAIT


def test_shaping_leaves_the_true_return_unchanged_up_to_the_start():
    from ashvini_ml.rl.env import ConjunctionEnv

    seq = [(np.array([5.0, 3.0, 1.5, 0.5]), np.array([-5.0, -4.0, -3.5, -3.0]))]
    totals = {}
    for shaping in (False, True):
        env = ConjunctionEnv(seq, 0, shaping=shaping)
        env.reset(seed=0)
        start, total, done = env._potential(), 0.0, False
        for action in (0, 1, 0):
            _, r, done, _, _ = env.step(action)
            total += r
        totals[shaping] = total + (start if shaping else 0.0)
    assert done and abs(totals[True] - totals[False]) < 1e-6


def test_decision_step_can_fall_back_to_the_baseline():
    from ashvini_ml.collision.models import adjust, tune_decision

    rng = np.random.default_rng(0)
    base = rng.uniform(-9, -3, 400)
    y = np.maximum(base + rng.normal(0, 0.3, 400), -6.001)
    noise = rng.normal(0, 3, 400)  # a useless model
    d = tune_decision(y, np.maximum(base, -6.001), noise)
    baseline = challenge_score(y, adjust(np.maximum(base, -6.001), noise, 0.0, -6.0))["score"]
    assert d["val_score"] <= baseline
