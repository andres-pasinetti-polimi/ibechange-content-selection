"""The daily composition (R, Q): which compositions a day admits, their probabilities, and the
terms of the model.

rct/design/rct_design_record_2026-10-27.md, Sections 1.1, 1.2 and 12, and
implementation_plan_composition_2026-11-04.md, Phases 1 to 3. A composition is a pair (R, Q), with R
the number of recommendations of the day (a resource counts as one) and Q the number of reminders.

A composition is written as the text "R,Q" where it is a key of a JSON object (logs, draws).
"""

import numpy as np

from cs_rct import design

# The treatment terms of the model (Phase 2): the increments of the first and second
# recommendation, of the first to fourth reminder, and the interaction R x Q.
BASE_TERM_NAMES = ["recommendation_1", "recommendation_2", "reminder_1", "reminder_2", "reminder_3", "reminder_4",
                   "recommendations_x_reminders"]
# The participant moderators (decided 20 November 2026): each multiplies R and Q, with coefficients
# common to all participants, so that the composition chosen differs between participants through
# the values of the moderators. Each value is a share of the participant's history, shrunk towards
# a centre and centred on it (features.moderators), so that the base terms are the effects of a
# participant at the centre.
MODERATOR_NAMES = ["recommendations_opened_share", "days_marked_share"]
MODERATED_NAMES = [f"{m}_x_{count}" for m in MODERATOR_NAMES for count in ("recommendations", "reminders")]
TERM_NAMES = BASE_TERM_NAMES + MODERATED_NAMES
# The lag terms: the centred sums of R and of Q over the design.LAG_DAYS preceding days.
LAG_NAMES = ["lag_recommendations", "lag_reminders"]


def key(composition):
    return f"{composition[0]},{composition[1]}"


def parse(text):
    r, q = text.split(",")
    return int(r), int(q)


def compositions(pillar):
    """Every composition of the pillar before the rules of the day, in a fixed order."""
    max_r = design.MAX_RECOMMENDATIONS_BY_PILLAR.get(pillar, design.MAX_RECOMMENDATIONS)
    return [(r, q) for total in range(design.DAILY_MINIMUM, design.DAILY_MAXIMUM + 1)
            for r in range(min(max_r, total), -1, -1) for q in [total - r] if q <= design.MAX_REMINDERS]


def admissible(pillar, reminders_closed, recommendations_available, cap_room):
    """(admissible compositions, {composition key: rule that removed it}).

    reminders_closed: None, or the rule that forbids reminders today ("task_marked",
    "count_reached"). recommendations_available: how many recommendations the Mission can supply
    today without sending one code twice (0, 1 or 2). cap_room: notifications left under the
    weekly cap. The first rule that removes a composition is the one logged."""
    kept, removed = [], {}
    for r, q in compositions(pillar):
        if reminders_closed and q > 0:
            removed[key((r, q))] = reminders_closed
        elif r > recommendations_available:
            removed[key((r, q))] = "no_recommendation" if recommendations_available == 0 else "recommendations_available"
        elif r + q > cap_room:
            removed[key((r, q))] = "weekly_cap"
        else:
            kept.append((r, q))
    return kept, removed


def fallback(admissible_set):
    """The default of the conservative rule when design.DEFAULT_COMPOSITION is not admissible:
    the default if admissible, else the admissible composition with the fewest notifications, and among those the
    one closest to the default. PROVISIONAL (design record, Section 8, item 2: (0,2) when no
    recommendation is available)."""
    default = design.DEFAULT_COMPOSITION
    if default in admissible_set:
        return default
    return min(admissible_set, key=lambda c: (c[0] + c[1], abs(c[0] - default[0]) + abs(c[1] - default[1]), c))


def terms(composition, moderators=None, names=None):
    """The treatment terms of a composition, before centring, in the order of names (TERM_NAMES by
    default; a model version fitted before 20 November 2026 has only BASE_TERM_NAMES).

    moderators: {name in MODERATOR_NAMES: value}; a missing moderator counts as 0, the centre."""
    r, q = composition
    moderators = moderators or {}
    value = dict(zip(BASE_TERM_NAMES, [float(r >= 1), float(r >= 2)] + [float(q >= k) for k in range(1, 5)]
                     + [float(r * q)]))
    for m in MODERATOR_NAMES:
        x = float(moderators.get(m) or 0.0)
        value[f"{m}_x_recommendations"] = x * r
        value[f"{m}_x_reminders"] = x * q
    return np.array([value[name] for name in (names or TERM_NAMES)])


def expected(probabilities, function):
    """The expectation of function(composition) under {composition key: probability}."""
    return sum(p * np.asarray(function(parse(k)), dtype=float) for k, p in probabilities.items())


def centred_terms(composition, probabilities, moderators=None, names=None):
    """The terms of the composition less their expectation under the day's probabilities. The
    moderators are fixed before the draw, so a moderated term is the moderator times the centred R
    or Q."""
    function = lambda c: terms(c, moderators, names)  # noqa: E731
    return function(composition) - expected(probabilities, function)


def centred_counts(composition, probabilities):
    """(R, Q) of the chosen composition less their expectation under the day's probabilities, the
    contribution of the day to the lag terms of the three following days."""
    return np.array(composition, dtype=float) - expected(probabilities, lambda c: np.array(c, dtype=float))


def starting_probabilities(policy, admissible_set, explore_share, default):
    """The probabilities before any data: uniform, or the conservative rule with a flat posterior
    (1 - E on the default, E spread uniformly)."""
    n = len(admissible_set)
    uniform = {key(c): 1 / n for c in admissible_set}
    if policy == "uniform":
        return uniform
    anchor = default if default in admissible_set else fallback(admissible_set)
    return {k: explore_share * p + (1 - explore_share) * (k == key(anchor)) for k, p in uniform.items()}


def values(posterior, admissible_set, lag_days, rng, draws, moderators=None):
    """Posterior draws of the value of each admissible composition for a participant with the given
    moderators: its effect on the day plus lag_days times its effect on one later day. Shape (draws,
    compositions). The terms are those the posterior names, so a version fitted without moderated
    terms gives the values it gave when it was used.

    The floating-point warnings of numpy are silenced because the BLAS of macOS raises them on
    finite inputs (as in model.fit); the result is checked for finiteness instead."""
    mean = np.asarray(posterior.mean)
    cov = np.asarray(posterior.cov)
    term_names = [name for name in posterior.names if name not in LAG_NAMES]
    term_index = [posterior.names.index(name) for name in term_names]
    lag_index = [posterior.names.index(name) for name in LAG_NAMES]
    rows = np.array([terms(c, moderators, term_names) for c in admissible_set])
    counts = np.array(admissible_set, dtype=float)
    with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
        sample = rng.multivariate_normal(mean, cov, size=draws, method="cholesky")
        value = sample[:, term_index] @ rows.T + lag_days * sample[:, lag_index] @ counts.T
    if not np.isfinite(value).all():
        raise ValueError("the posterior draws are not finite")
    return value


def probabilities(policy, admissible_set, posterior, seed, parameters=None, moderators=None):
    """({composition key: probability}, rule): the probabilities of the day.

    moderators: the participant's moderators of the day ({name: value}, logged in the state).

    rule is "uniform", "conservative", or "fallback" when the conservative rule has no posterior
    (the starting probabilities are used). The posterior draws come from a generator seeded by the
    decision seed and separate from the generator of the day's other draws, so that the replay
    reproduces the probabilities exactly. parameters: the values in force (design.parameters()),
    read from the decision row by the replay."""
    parameters = parameters or design.parameters()
    explore = parameters["explore_share"]
    default = tuple(parameters["default_composition"])
    start = starting_probabilities(policy, admissible_set, explore, default)
    if policy == "uniform":
        return start, "uniform"
    if posterior is None:
        return start, "fallback"
    anchor = default if default in admissible_set else fallback(admissible_set)
    rng = np.random.default_rng([seed, 1])
    value = values(posterior, admissible_set, parameters["lag_days"], rng, parameters["thompson_draws"], moderators)
    anchor_index = admissible_set.index(anchor)
    beats = (value > value[:, [anchor_index]]).mean(axis=0) > parameters["beat_threshold"]
    beats[anchor_index] = True
    best = np.argmax(np.where(beats[None, :], value, -np.inf), axis=1)
    thompson = np.bincount(best, minlength=len(admissible_set)) / len(best)
    n = len(admissible_set)
    return ({key(c): (1 - explore) * float(thompson[i]) + explore / n for i, c in enumerate(admissible_set)},
            "conservative")


def draw(rng, probabilities_by_key):
    """One composition drawn from {key: probability}, as (R, Q)."""
    keys = list(probabilities_by_key)
    p = np.array([probabilities_by_key[k] for k in keys])
    index = min(int((rng.random() > np.cumsum(p / p.sum())).sum()), len(keys) - 1)
    return parse(keys[index])
