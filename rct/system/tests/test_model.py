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
    n = len(composition.TERM_NAMES)
    estimate = contrast @ mean[:n]
    contrast_sd = np.sqrt(contrast @ np.array(posterior.cov)[:n, :n] @ contrast)
    assert abs(estimate - (PER_REMINDER + FIRST_RECOMMENDATION * 0)) < 3 * contrast_sd
    lag = names.index("lag_reminders")
    assert abs(mean[lag] - LAG_REMINDER) < 3 * sd[lag]
    assert abs(mean[names.index("lag_recommendations")]) < 3 * sd[names.index("lag_recommendations")]


def test_fit_recovers_a_moderator_of_the_reminders():
    """The effect of each reminder is PER_REMINDER + 0.1 m_i, with m_i the participant's centred
    marked share; the fit recovers the coefficient of days_marked_share_x_reminders."""
    rng = np.random.default_rng(1)
    probabilities = {composition.key(c): 1 / len(COMPOSITIONS) for c in COMPOSITIONS}
    points = []
    for user in range(600):
        m = float(rng.uniform(-0.4, 0.4))
        moderators = {"days_marked_share": m}
        for _ in range(40):
            chosen = COMPOSITIONS[int(rng.integers(len(COMPOSITIONS)))]
            p = 0.4 + (PER_REMINDER + 0.1 * m) * chosen[1]
            points.append(model.DecisionPoint(
                f"u{user}", [1.0], composition.centred_terms(chosen, probabilities, moderators).tolist(),
                [0.0, 0.0], int(rng.random() < np.clip(p, 0, 1))))
    posterior = model.fit(points)
    i = posterior.names.index("days_marked_share_x_reminders")
    sd = np.sqrt(posterior.cov[i][i])
    assert abs(posterior.mean[i] - 0.1) < 3 * sd


def test_a_version_without_moderated_terms_is_fitted_and_used_with_its_own_terms():
    names = composition.BASE_TERM_NAMES + composition.LAG_NAMES
    probabilities = {composition.key(c): 1 / len(COMPOSITIONS) for c in COMPOSITIONS}
    points = [model.DecisionPoint("u1", [1.0], composition.centred_terms(
        c, probabilities, names=composition.BASE_TERM_NAMES).tolist(), [0.0, 0.0], i % 2)
        for i, c in enumerate(COMPOSITIONS * 5)]
    posterior = model.fit(points, names=names)
    assert posterior.names == names and len(posterior.mean) == 9
    # The moderators do not change the values of a version without moderated terms.
    rng = lambda: np.random.default_rng(0)  # noqa: E731
    plain = composition.values(posterior, COMPOSITIONS, 3, rng(), 50)
    moderated = composition.values(posterior, COMPOSITIONS, 3, rng(), 50, {"days_marked_share": 0.4})
    assert np.allclose(plain, moderated)


def test_the_moderators_change_the_composition_of_a_participant():
    """Coefficients common to all participants give different probabilities to two participants
    whose moderators differ."""
    posterior = model.prior()
    mean = np.zeros(len(posterior.names))
    mean[posterior.names.index("days_marked_share_x_reminders")] = 0.2
    posterior = model.Posterior(names=posterior.names, mean=mean.tolist(),
                                cov=(np.eye(len(mean)) * 1e-6).tolist())
    parameters = {"explore_share": 0.0, "default_composition": [1, 1], "lag_days": 3, "thompson_draws": 200,
                  "beat_threshold": 0.9}
    high, _ = composition.probabilities("conservative", COMPOSITIONS, posterior, 7, parameters,
                                        {"days_marked_share": 0.3})
    low, _ = composition.probabilities("conservative", COMPOSITIONS, posterior, 7, parameters,
                                       {"days_marked_share": -0.3})
    # Each reminder is worth 0.2 m: the participant with m = 0.3 gets the most reminders, the one with
    # m = -0.3 the fewest.
    assert high["0,4"] == 1.0 and low["2,0"] == 1.0


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
