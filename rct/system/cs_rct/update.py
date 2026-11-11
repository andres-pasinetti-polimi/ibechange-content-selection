"""The update of the model at the resolution of episodes.

The update reads the decision rows the service wrote, not a reconstruction: the baseline features,
the composition and the probabilities of the admissible compositions are taken as logged. For each
episode that has ended (completed, closed or expired) it reads the daily outcome of every decision
day, and the posterior is refitted on all decision points of all such episodes. A withdrawn
(replaced) episode contributes its days, with the marks entered by its replacement (8 November 2026).
The days of the fixed-probability group do not enter, so
that the group stays a sample on which the learned rule can be evaluated. Decisions of the scheme of
14 October 2026 (no composition logged) are left out of the fit.

The lag terms of a decision point are the sums, over the design.LAG_DAYS preceding days of the same
participant, of the chosen R and Q less their expectation under the probabilities of each day. They
are read from the logged decisions of every status, not from the deliveries EUT reports: a day that
was not randomised (one composition) contributes zero, and so does a day without a composition (past
the window, below the minimum).

Refitting on everything at each update gives the same posterior as updating episode by episode,
since the posterior depends on the set of points and not on their order, and it lets the replay
check refit any version from the episodes it lists.
"""

from datetime import timedelta

from cs_rct import composition, design, features, model
from cs_rct.history import daily_outcomes, local_time


def centred_counts_by_day(decision_rows):
    """{(user_uuid, day): centred (R, Q)} for every row with a composition."""
    centred = {}
    for row in decision_rows:
        if row.get("composition_r") is None or not row.get("composition_probabilities"):
            continue
        chosen = (row["composition_r"], row["composition_q"])
        centred[(row["user_uuid"], row["decision_day"])] = composition.centred_counts(
            chosen, row["composition_probabilities"])
    return centred


def lag_sums(centred, user_uuid, day, lag_days=None):
    lag_days = lag_days or design.LAG_DAYS
    total = [0.0, 0.0]
    for back in range(1, lag_days + 1):
        value = centred.get((user_uuid, day - timedelta(days=back)))
        if value is not None:
            total = [total[0] + float(value[0]), total[1] + float(value[1])]
    return total


def training_points(decision_rows, participants, now, study=None, lag_rows=None, baseline=None):
    """(decision points, ids of the episodes they come from), sorted for reproducibility.

    lag_rows: the rows the lag terms are read from, all decision rows by default (the replay passes
    every row while it restricts decision_rows to the episodes of one version). baseline: the names
    of psi, features.baseline_names(study) by default (the replay passes those of the version)."""
    baseline = baseline or features.baseline_names(study)
    centred = centred_counts_by_day(decision_rows if lag_rows is None else lag_rows)
    by_episode = {}
    for row in decision_rows:
        if row["status"] != "decided" or row["group_name"] == "fixed" or row.get("composition_r") is None:
            continue
        by_episode.setdefault((row["user_uuid"], row["mission_process_id"]), []).append(row)

    points, episodes = [], []
    for (user_uuid, process_id), rows in sorted(by_episode.items()):
        participant = participants.get(user_uuid)
        if participant is None:
            continue
        episode = next((e for e in participant.episodes if e["mission_process_id"] == process_id), None)
        if episode is None:
            continue
        rows = sorted(rows, key=lambda r: r["decision_day"])
        # `now` is UTC; the participant's history is in local time.
        # The decision times cut the marks for the day before counted by "mark_minus_backfill".
        decided_at = {r["decision_day"]: local_time(r["decided_at"], participant.timezone)
                      for r in rows if r.get("decided_at") is not None}
        outcomes = daily_outcomes(participant, episode, [r["decision_day"] for r in rows],
                                  local_time(now, participant.timezone), decided_at)
        if outcomes is None:
            continue
        episodes.append(process_id)
        for row in rows:
            chosen = (row["composition_r"], row["composition_q"])
            points.append(model.DecisionPoint(
                user_uuid=user_uuid,
                psi=[row["baseline"][name] for name in baseline],
                terms=composition.centred_terms(chosen, row["composition_probabilities"]).tolist(),
                lags=lag_sums(centred, user_uuid, row["decision_day"], row["parameters"].get("lag_days")),
                outcome=outcomes[row["decision_day"]],
            ))
    return points, sorted(episodes)


def refit(decision_rows, participants, now, study=None):
    """(posterior, ids of the episodes included)."""
    points, episodes = training_points(decision_rows, participants, now, study)
    return model.fit(points, features.baseline_names(study)), episodes
