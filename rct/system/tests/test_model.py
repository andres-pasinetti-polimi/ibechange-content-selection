"""The working model of the composition (model.py) and the training points (update.py)."""

from datetime import date, timedelta

import numpy as np

from cs_rct import composition, model, update

COMPOSITIONS = composition.compositions("Smoking")
# True effects of the generator: the first recommendation, each reminder, and a lag effect.
FIRST_RECOMMENDATION, PER_REMINDER, LAG_REMINDER = 0.08, 0.03, -0.02


def simulated_points(n_users=300, days=40, seed=0):
    """Days with compositions drawn uniformly; the outcome has a participant level, a baseline
    feature, the immediate effects and a burden of reminders on the three following days."""
    rng = np.random.default_rng(seed)
    probabilities = {composition.key(c): 1 / len(COMPOSITIONS) for c in COMPOSITIONS}
    points = []
    for user in range(n_users):
        level = rng.normal(0.4, 0.1)
        history = []
        for _ in range(days):
            x = rng.random()
            chosen = COMPOSITIONS[int(rng.integers(len(COMPOSITIONS)))]
            lag = [sum(float(h[0]) for h in history[-3:]), sum(float(h[1]) for h in history[-3:])]
            reminders_before = sum(c[1] for c in [h[2] for h in history[-3:]])
            p = (level + 0.1 * x + FIRST_RECOMMENDATION * (chosen[0] >= 1) + PER_REMINDER * chosen[1]
                 + LAG_REMINDER * reminders_before)
            outcome = int(rng.random() < np.clip(p, 0, 1))
            centred = composition.centred_counts(chosen, probabilities)
            history.append((centred[0], centred[1], chosen))
            points.append(model.DecisionPoint(f"u{user}", [1.0, x],
                                              composition.centred_terms(chosen, probabilities).tolist(), lag, outcome))
    return points


def test_prior_is_flat_and_round_trips_through_json():
    prior = model.prior()
    assert prior.names == composition.TERM_NAMES + composition.LAG_NAMES and set(prior.mean) == {0.0}
    again = model.Posterior.from_json(prior.to_json())
    assert again == prior
    # A posterior of the scheme of 14 October 2026 cannot be used by the composition policy.
    assert model.Posterior.from_json({"study": "usability", "theta_mean": [0.0], "theta_cov": [[1.0]]}) is None


def test_fit_recovers_known_effects():
    posterior = model.fit(simulated_points())
    mean, sd = np.array(posterior.mean), np.sqrt(np.diag(posterior.cov))
    names = posterior.names
    # One more reminder from (1,1) to (1,2): reminder_2 plus the interaction.
    contrast = composition.terms((1, 2)) - composition.terms((1, 1))
    estimate = contrast @ mean[:7]
    contrast_sd = np.sqrt(contrast @ np.array(posterior.cov)[:7, :7] @ contrast)
    assert abs(estimate - (PER_REMINDER + FIRST_RECOMMENDATION * 0)) < 3 * contrast_sd
    lag = names.index("lag_reminders")
    assert abs(mean[lag] - LAG_REMINDER) < 3 * sd[lag]
    assert abs(mean[names.index("lag_recommendations")]) < 3 * sd[names.index("lag_recommendations")]


def test_terms_are_centred_under_the_probabilities_of_the_day():
    probabilities = {"1,1": 0.7, "0,2": 0.1, "2,2": 0.2}
    expected = sum(p * composition.terms(composition.parse(k)) for k, p in probabilities.items())
    assert np.allclose(sum(p * composition.centred_terms(composition.parse(k), probabilities)
                           for k, p in probabilities.items()), 0)
    assert np.allclose(composition.centred_terms((1, 1), probabilities), composition.terms((1, 1)) - expected)
    # A composition with probability 1 contributes nothing.
    assert np.allclose(composition.centred_counts((2, 0), {"2,0": 1.0}), 0)


def row(day, status="decided", r=1, q=1, probabilities=None, process_id=1, user="u1"):
    return {"user_uuid": user, "decision_day": day, "status": status, "group_name": "learned",
            "mission_process_id": process_id, "composition_r": r, "composition_q": q,
            "composition_probabilities": probabilities or {"1,1": 0.5, "0,4": 0.5},
            "parameters": {"lag_days": 3}, "baseline": {}}


def test_lag_sums_read_the_three_preceding_days_of_every_status():
    start = date(2027, 1, 4)
    rows = [row(start, r=0, q=4), row(start + timedelta(days=1), status="no_choice", r=2, q=0,
                                      probabilities={"2,0": 1.0}),
            row(start + timedelta(days=2), r=1, q=1), row(start + timedelta(days=3))]
    centred = update.centred_counts_by_day(rows)
    # Day 0: (0,4) against an expectation of (0.5, 2.5); day 1 not randomised; day 2: (1,1) against it.
    assert update.lag_sums(centred, "u1", start + timedelta(days=3)) == [-0.5 + 0 + 0.5, 1.5 + 0 - 1.5]
    # Day 4 reads days 1 to 3, two of them (1,1).
    assert update.lag_sums(centred, "u1", start + timedelta(days=4)) == [1.0, -3.0]
    assert update.lag_sums(centred, "u1", start) == [0.0, 0.0]


def test_rows_without_a_composition_are_left_out_of_the_fit():
    start = date(2027, 1, 4)
    old = row(start)
    old.update(composition_r=None, composition_q=None, composition_probabilities=None)
    points, episodes = update.training_points([old], {}, None)
    assert points == [] and episodes == []


def test_a_model_version_of_the_old_scheme_is_replaced_at_the_next_run():
    from datetime import datetime

    from cs_rct import cycle
    from tests.helpers import load_catalogue
    legacy = cycle.ModelState(4, None, [], cycle.points_fingerprint([]))
    result = cycle.run_cycle(datetime(2027, 1, 5, 9), {}, load_catalogue(), [], {}, legacy, {}, {},
                             draw_seed=lambda: 1)
    assert result.new_model is not None and result.new_model.version == 5
    assert result.new_model.posterior == model.prior()
