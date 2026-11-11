"""One run of the service: update, group assignment, closing pass, decisions.

The run is a function of what the service holds (participants rebuilt from
raw_events, the decision log, the latest model version) and of the moment `now`,
and returns what is to be written; run.py reads and writes the database around
it, and simulate.py calls it on simulated participants. The order inside a run:

1. Update. The decision points of every episode that has ended are collected; if
   they differ from those of the latest model version, the posterior is refitted
   and a new version is written. A decision taken in this run reads the new version.
2. Groups. Each participant seen for the first time is assigned to the learned or
   the fixed-probability group (groups.py).
3. Closing pass. Each participant without a row for yesterday gets one that says
   why nothing was decided, if the service ran yesterday.
4. Decisions. Each participant without a row for today and with an open episode is
   decided, up to design.LAST_DECISION_HOUR.
5. Revisions. The plan of each participant decided earlier today is revised if the day's
   task was marked, the Mission's count reached, or the Mission replaced or closed
   (revise.py). Whether a Mission starts receiving
   deliveries on the day of its selection or the next morning is an open decision
   (design.FIRST_DELIVERY_DAY).
"""

import hashlib
import json
import logging
from dataclasses import dataclass, field
from datetime import timedelta

import numpy as np

from cs_rct import decide, design, features, groups, model, revise, update
from cs_rct.history import local_time

log = logging.getLogger("cycle")


@dataclass
class ModelState:
    version: int
    posterior: model.Posterior
    episodes: list
    fingerprint: str


@dataclass
class RunResult:
    new_model: ModelState | None = None
    new_groups: list = field(default_factory=list)     # (user_uuid, group, stratum, seed)
    decisions: list = field(default_factory=list)      # (decision row, delivery rows)
    # (decision id, revision number, reason, seed, ids of cancelled deliveries, new delivery rows)
    revisions: list = field(default_factory=list)
    summary: dict = field(default_factory=dict)


def run_cycle(now, participants, catalogue, decision_rows, deliveries_by_user, latest_model, group_of,
              groups_by_stratum, draw_seed, ran_yesterday=True, study=None,
              revisions_of=None):
    """Compute one run. draw_seed() returns a fresh seed for each random draw.

    latest_model: ModelState or None. group_of: {user_uuid: group}. groups_by_stratum:
    {stratum: [groups in assignment order]}. deliveries_by_user: each delivery with its id and "cancelled". revisions_of: {decision id:
    latest revision number}.
    """
    study = study or design.STUDY
    result = RunResult()

    # 1. Update.
    points, episodes = update.training_points(decision_rows, participants, now, study)
    fingerprint = points_fingerprint(points)
    current = latest_model
    fit_failed = False
    # A version whose posterior cannot be read (scheme of 14 October 2026) is replaced by a new one.
    if current is None or current.posterior is None or fingerprint != current.fingerprint:
        try:
            posterior = model.fit(points, features.baseline_names(study))
        except Exception:
            # The decisions of this run use the starting probabilities, flagged in their rows.
            log.exception("Model update failed; decisions fall back to the starting probabilities")
            fit_failed = True
        else:
            current = ModelState(version=(current.version + 1) if current else 1, posterior=posterior,
                                 episodes=episodes, fingerprint=fingerprint)
            result.new_model = current
    posterior_in_force = None if fit_failed or current is None else current.posterior
    version_in_force = None if posterior_in_force is None else current.version

    # 2. Groups.
    for user_uuid in sorted(participants):
        if user_uuid in group_of:
            continue
        participant = participants[user_uuid]
        seed = draw_seed()
        name = groups.stratum(participant)
        assigned = groups.assign(np.random.default_rng(seed), groups_by_stratum.setdefault(name, []))
        groups_by_stratum[name].append(assigned)
        group_of[user_uuid] = assigned
        result.new_groups.append((user_uuid, assigned, name, seed))

    rows_by_user_day = {(row["user_uuid"], row["decision_day"]) for row in decision_rows}
    # `now` is UTC; the day, its hour and yesterday are each participant's own.
    local_now = {user_uuid: local_time(now, participant.timezone) for user_uuid, participant in participants.items()}

    def stamped(row, participant):
        # The decision log keeps the time of the run in UTC and the zone the day was read in.
        row["decided_at"] = now
        row["timezone"] = participant.timezone
        return row

    # 3. Closing pass for yesterday.
    if ran_yesterday:
        for user_uuid, participant in sorted(participants.items()):
            yesterday = local_now[user_uuid].date() - timedelta(days=1)
            if (user_uuid, yesterday) in rows_by_user_day or not in_programme(participant, yesterday):
                continue
            row = closing_row(participant, yesterday, local_now[user_uuid], group_of[user_uuid])
            result.decisions.append((stamped(row, participant), []))

    # 4. Decisions for today.
    for user_uuid, participant in sorted(participants.items()):
        today = local_now[user_uuid].date()
        if (user_uuid, today) in rows_by_user_day or local_now[user_uuid].hour > design.LAST_DECISION_HOUR:
            continue
        decided = decide.decide(participant, catalogue, posterior_in_force, version_in_force,
                                group_of[user_uuid], today, local_now[user_uuid], draw_seed(),
                                deliveries_by_user.get(user_uuid, []), study)
        if decided is not None:
            result.decisions.append((stamped(decided[0], participant), decided[1]))

    # 5. Revisions of the plans decided in earlier runs today.
    revisions_of = revisions_of or {}
    for row in decision_rows:
        participant = participants.get(row["user_uuid"])
        if participant is None or row.get("id") is None:
            continue
        today = local_now[row["user_uuid"]].date()
        if row["decision_day"] != today:
            continue
        previous = deliveries_by_user.get(row["user_uuid"], [])
        day_deliveries = [d for d in previous if d.get("decision_id") == row["id"]]
        if not day_deliveries:
            continue
        seed = draw_seed()
        change = revise.revise(participant, catalogue, row, day_deliveries, previous, today,
                               local_now[row["user_uuid"]], seed)
        if change is not None:
            reason, cancelled, new_rows = change
            result.revisions.append((row["id"], revisions_of.get(row["id"], 0) + 1, reason, seed, cancelled, new_rows))

    statuses = {}
    for row, _ in result.decisions:
        statuses[row["status"]] = statuses.get(row["status"], 0) + 1
    result.summary = {"model_version": version_in_force, "fit_failed": fit_failed,
                      "new_model": result.new_model is not None,
                      "decision_points_in_model": len(points), "episodes_in_model": len(episodes),
                      "new_groups": len(result.new_groups), "rows_by_status": statuses,
                      "revisions": [reason for _, _, reason, _, _, _ in result.revisions]}
    return result


def in_programme(participant, day):
    """Whether `day` falls within the participant's programme, from enrolment for T days."""
    start = participant.enrolment_date
    if start is None:
        if not participant.episodes:
            return False
        start = participant.episodes[0]["selected_at"].date()
    return start <= day < start + timedelta(days=design.PROGRAMME_DAYS)


def closing_row(participant, day, now, group):
    row = decide.empty_row(day, now, participant.user_uuid, group, None)
    row["status"] = decide.closing_status(participant, day, now)
    del row["draws"]
    return row


def points_fingerprint(points):
    """A hash of the decision points, so that a new version is written exactly when they change."""
    text = json.dumps([[p.user_uuid, p.psi, p.terms, p.lags, p.outcome] for p in points])
    return hashlib.sha256(text.encode()).hexdigest()
