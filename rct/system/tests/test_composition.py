"""The admissible compositions and the policies (composition.py)."""

import numpy as np
import pytest

from cs_rct import composition, design, model

FOUR_PILLARS = [(2, 0), (1, 1), (0, 2), (2, 1), (1, 2), (0, 3), (2, 2), (1, 3), (0, 4)]


def test_admissible_table_by_pillar():
    for pillar in ("Smoking", "Alcohol", "Nutrition", "Physical_activity"):
        assert composition.compositions(pillar) == FOUR_PILLARS
    assert composition.compositions("Emotional_Wellbeing") == [(1, 1), (0, 2), (1, 2), (0, 3), (1, 3), (0, 4)]


def test_each_removal_rule():
    kept, removed = composition.admissible("Smoking", "task_marked", 2, 28)
    assert kept == [(2, 0)] and set(removed.values()) == {"task_marked"}
    kept, removed = composition.admissible("Smoking", None, 0, 28)
    assert kept == [(0, 2), (0, 3), (0, 4)] and set(removed.values()) == {"no_recommendation"}
    kept, removed = composition.admissible("Smoking", None, 1, 28)
    assert all(r <= 1 for r, _ in kept) and set(removed.values()) == {"recommendations_available"}
    kept, removed = composition.admissible("Smoking", None, 2, 3)
    assert all(r + q <= 3 for r, q in kept) and set(removed.values()) == {"weekly_cap"}
    assert composition.admissible("Emotional_Wellbeing", "count_reached", 1, 28)[0] == []


def posterior_with(effects, sd):
    """A posterior with the given means ({name: value}) and standard deviation sd on every coefficient."""
    mean = np.zeros(len(model.NAMES))
    for name, value in effects.items():
        mean[model.NAMES.index(name)] = value
    return model.Posterior(names=list(model.NAMES), mean=mean.tolist(), cov=(np.eye(len(model.NAMES)) * sd ** 2).tolist())


def test_flat_posterior_gives_the_starting_probabilities():
    probabilities, rule = composition.probabilities("conservative", FOUR_PILLARS, model.prior(), 3)
    explore = design.EXPLORE_SHARE
    assert rule == "conservative"
    assert probabilities["1,1"] == pytest.approx(1 - explore + explore / 9)
    assert all(p == pytest.approx(explore / 9) for k, p in probabilities.items() if k != "1,1")
    assert sum(probabilities.values()) == pytest.approx(1.0)


def test_a_composition_above_the_threshold_receives_the_thompson_share():
    # Values relative to nothing: (1,3) 0.15, (0,3) and (0,4) 0.10, (1,1) 0.05.
    posterior = posterior_with({"recommendation_1": 0.05, "reminder_3": 0.10}, 0.005)
    probabilities, _ = composition.probabilities("conservative", FOUR_PILLARS, posterior, 3)
    assert sum(probabilities.values()) == pytest.approx(1.0)
    # The posterior draws of the lag terms, times three days and three reminders, let a few draws
    # rank another composition first.
    assert probabilities["1,3"] > 0.9 * (1 - design.EXPLORE_SHARE) and probabilities["1,1"] < 0.1


def test_a_composition_below_the_threshold_returns_its_share_to_the_default():
    # The same direction, too uncertain to pass P > 0.9.
    posterior = posterior_with({"recommendation_1": 0.005, "reminder_3": 0.01}, 0.2)
    probabilities, _ = composition.probabilities("conservative", FOUR_PILLARS, posterior, 3)
    assert probabilities["1,1"] == pytest.approx(1 - design.EXPLORE_SHARE + design.EXPLORE_SHARE / 9)


def test_the_lag_effect_counts_three_times_in_the_value():
    # A gain of 0.05 for each reminder beyond the first, and a cost of 0.03 per reminder on each of
    # three later days: (2,0) has value 0, (1,1) -0.09, (0,4) 0.15 - 0.36. (2,0) receives the share.
    gains = {"reminder_2": 0.05, "reminder_3": 0.05, "reminder_4": 0.05}
    posterior = posterior_with({**gains, "lag_reminders": -0.03}, 0.001)
    probabilities, _ = composition.probabilities("conservative", FOUR_PILLARS, posterior, 3)
    assert probabilities["2,0"] == pytest.approx(1 - design.EXPLORE_SHARE + design.EXPLORE_SHARE / 9)
    # Without the cost, (0,4) has the highest value, 0.15.
    probabilities, _ = composition.probabilities("conservative", FOUR_PILLARS, posterior_with(gains, 0.001), 3)
    assert probabilities["0,4"] == pytest.approx(1 - design.EXPLORE_SHARE + design.EXPLORE_SHARE / 9)


def test_the_same_seed_gives_the_same_probabilities():
    posterior = posterior_with({"reminder_2": 0.02}, 0.02)
    first = composition.probabilities("conservative", FOUR_PILLARS, posterior, 11)
    assert composition.probabilities("conservative", FOUR_PILLARS, posterior, 11) == first


def test_default_not_admissible_puts_its_share_on_the_fallback():
    admissible = [(0, 2), (0, 3), (0, 4)]
    probabilities, _ = composition.probabilities("conservative", admissible, model.prior(), 3)
    assert probabilities["0,2"] == pytest.approx(1 - design.EXPLORE_SHARE + design.EXPLORE_SHARE / 3)
    assert composition.fallback([(2, 0)]) == (2, 0)
