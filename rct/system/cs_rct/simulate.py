"""Pre-deployment simulation (rct/paper/main.tex, Section "Simulation study").

The service's own cycle (cycle.run_cycle) runs on simulated participants, so that
what is simulated is the code that will run. Each simulated day has two runs,
at 01:00 (closing pass of the day before, decisions for participants with an open
episode) and at 09:00 (decisions for Missions selected that morning); behaviour,
marks and outcomes of the day follow the decisions.

The world. A participant selects a Mission of the catalogue at random when none
is open. The probability that the target behaviour occurs on a day is
p = p_i + p_e + delta * k, with p_i a participant level, p_e an episode level and
k = R + Q - 2 the number of notifications of the day's composition beyond two, of either
type, clipped to [0.02, 0.98]. The two levels have
the shares of the pilot's variance components (participant 27.8%, episode 47.2%,
task 25.0%; pilot_numbers.tex) applied on the probability scale at a mean of 0.5,
an approximation, since the pilot's unit is a ticked task and not a day. A day
with the behaviour is marked with the participant's recording probability (1 on a
class A Mission). The effect delta follows the scenario: "null"; "common";
"heterogeneous", delta_i ~ N(delta, (delta/2)^2); "decaying", delta (1 - 0.02 d)
on programme day d and zero from day 28 (Klasnja et al. 2019).

Recording (rct/design/composition_model_2026-10-15.md, Section 8). On a class B Mission a
day with the behaviour that is not recorded on the day can be recorded the next
day with probability BACKFILL_BASE. With --recording-effect e, each notification
beyond two also records a share e of the behaviour of that day that would
otherwise go unrecorded, and, times --backfill-ratio, a share of the unrecorded
behaviour of the day before. --backfill-ratio 1 is the assumption under which the
outcome "mark_minus_backfill" removes the effect on recording.

Recorded: the expected number of notifications a day under the probabilities of the
decision days, by study week; the effect of one more notification estimated at the end of
the run, the contrast of (1,2) against (1,1) under the posterior mean with its 90% interval,
against the true average effect on the daily outcome (the effect on the behaviour times the
probability of marking it); and the expected number of marked days beyond those of two
notifications under the learned rule, under the uniform rule over the admissible compositions,
and under the largest admissible number of notifications.

Run from the system/ folder:
    python -m cs_rct.simulate --participants 10 --days 30 --scenario null --replications 20
"""

import argparse
import json
import math
from datetime import date, datetime, time, timedelta

import numpy as np

from cs_rct import catalogue as catalogue_module
from cs_rct import composition, cycle, design
from cs_rct.history import build_participants, marked_days, open_episode, start_day

START = date(2027, 1, 4)
MEAN_LEVEL = 0.5
SD_PARTICIPANT = math.sqrt(0.278 * 0.25)
SD_EPISODE = math.sqrt(0.472 * 0.25)
SCHEDULE_PROBABILITY = 0.3
REPLACEMENT_PROBABILITY = 0.05
LEFT_TO_EXPIRE_PROBABILITY = 0.1
RECOMMENDATIONS_PER_MISSION = 7
BACKFILL_BASE = 0.1


class World:
    def __init__(self, rng, participants, days, enrolment_days, scenario, effect, catalogue,
                 recording_effect=0.0, backfill_ratio=1.0):
        self.rng = rng
        self.recording_effect = recording_effect
        self.backfill_ratio = backfill_ratio
        self.unrecorded = set()  # (user, process id, day) with the behaviour and no mark yet
        self.days = days
        self.scenario = scenario
        self.effect = effect
        self.catalogue = catalogue
        self.missions = [m for m in catalogue.missions.values() if m.reminder_pools]
        self.tables = {"users": [], "episodes": [], "marks": [], "scheduled_tasks": [], "notification_events": []}
        self.level, self.recording, self.own_effect = {}, {}, {}
        self.process_id = 0
        for index in range(participants):
            user = f"sim-{index:04d}"
            enrolment = START + timedelta(days=int(rng.integers(enrolment_days)))
            self.tables["users"].append({"user_uuid": user, "language": "es", "enrolment_date": enrolment,
                                         "age": int(rng.integers(45, 81)), "recruitment_center": "SIM",
                                         "disabled_at": None})
            self.level[user] = rng.normal(0, SD_PARTICIPANT)
            self.recording[user] = rng.beta(8, 2)
            self.own_effect[user] = (rng.normal(effect, effect / 2) if scenario == "heterogeneous" else effect)
        self.episode_level = {}
        self.base_probability = {}  # (user, day) -> probability of the behaviour with two notifications
        self.model_versions = {}    # version -> (posterior, episodes), as model_versions holds them

    def true_effect(self, user, day):
        enrolment = next(u["enrolment_date"] for u in self.tables["users"] if u["user_uuid"] == user)
        programme_day = (day - enrolment).days
        if self.scenario == "decaying":
            return self.effect * max(0.0, 1 - 0.02 * programme_day) if programme_day < 28 else 0.0
        return self.own_effect[user]

    def observed_effect(self, row, extra):
        """The true effect on the daily outcome of a decision row of `extra` notifications beyond
        two, with the bounds on p."""
        base = self.base_probability.get((row["user_uuid"], row["decision_day"]))
        if base is None:
            return 0.0
        effect = self.true_effect(row["user_uuid"], row["decision_day"])
        mission = self.catalogue.missions[row["mission_id"]]
        recording = 1.0 if mission.mission_class == "A" else self.recording[row["user_uuid"]]
        return (bounded(base + effect * extra) - bounded(base)) * recording

    def next_id(self):
        self.process_id += 1
        return self.process_id

    def select_missions(self, day):
        """Participants in their programme without an open episode select a Mission at 08:00."""
        now = datetime.combine(day, time(8))
        participants = build_participants(self.tables)
        for user in self.tables["users"]:
            uuid = user["user_uuid"]
            if not (user["enrolment_date"] <= day < user["enrolment_date"] + timedelta(days=self.days)):
                continue
            if open_episode(participants[uuid], now) is not None:
                continue
            mission = self.missions[int(self.rng.integers(len(self.missions)))]
            process_id = self.next_id()
            self.tables["episodes"].append({
                "mission_process_id": process_id, "user_uuid": uuid, "mission_id": mission.mission_id,
                "selected_at": now, "selection_received_at": now, "outcome": None, "resolved_at": None,
                "resolution_received_at": None})
            self.episode_level[process_id] = self.rng.normal(0, SD_EPISODE)
            if mission.schedulable:
                for offset in range(design.WINDOW_DAYS):
                    if self.rng.random() < SCHEDULE_PROBABILITY:
                        slot = datetime.combine(day + timedelta(days=offset), time(18))
                        self.tables["scheduled_tasks"].append({
                            "user_uuid": uuid, "mission_process_id": process_id, "mission_id": mission.mission_id,
                            "step": None, "total_steps": None, "start_at": slot, "end_at": slot,
                            "entered_at": now, "process_id": self.next_id(), "received_at": now})

    def live_day(self, day, extra_of):
        """Behaviour, marks, sent notifications and outcomes of `day`."""
        evening = datetime.combine(day, time(21))
        participants = build_participants(self.tables)
        for uuid, participant in participants.items():
            episode = open_episode(participant, datetime.combine(day, time(12)))
            if episode is None:
                continue
            mission = self.catalogue.missions[episode["mission_id"]]
            k = extra_of.get((uuid, day), 0)
            base = MEAN_LEVEL + self.level[uuid] + self.episode_level[episode["mission_process_id"]]
            self.base_probability[(uuid, day)] = base
            p = bounded(base + self.true_effect(uuid, day) * k)
            recording = 1.0 if mission.mission_class == "A" else self.recording[uuid]
            pushed = min(1.0, self.recording_effect * k)
            process_id = episode["mission_process_id"]
            if self.rng.random() < p:
                if self.rng.random() < recording + (1 - recording) * pushed:
                    self.add_mark(uuid, process_id, mission, day, evening)
                else:
                    self.unrecorded.add((uuid, process_id, day))
            yesterday = day - timedelta(days=1)
            if (uuid, process_id, yesterday) in self.unrecorded:
                self.unrecorded.discard((uuid, process_id, yesterday))
                if self.rng.random() < min(1.0, BACKFILL_BASE + self.backfill_ratio * pushed):
                    self.add_mark(uuid, process_id, mission, yesterday, evening)
            self.resolve(participant, episode, mission, day)

    def add_mark(self, uuid, process_id, mission, reference_day, entered_at):
        self.tables["marks"].append({
            "user_uuid": uuid, "mission_process_id": process_id,
            "mission_id": mission.mission_id, "step": None, "total_steps": None,
            "reference_time": datetime.combine(reference_day, time(18)), "entered_at": entered_at,
            "process_id": self.next_id(), "received_at": entered_at})

    def resolve(self, participant, episode, mission, day):
        closing = datetime.combine(day, time(21, 30))
        if day == start_day(episode) and self.rng.random() < REPLACEMENT_PROBABILITY:
            outcome = "changed"
        else:
            marks = len(marked_days(participant, episode, closing))
            if mission.mission_class == "A" and mission.states_count and marks >= mission.count:
                outcome = "achieved"
            elif (day - start_day(episode)).days == design.WINDOW_DAYS - 1:
                if self.rng.random() < LEFT_TO_EXPIRE_PROBABILITY:
                    return
                share = marks / mission.count if mission.states_count else marks / design.WINDOW_DAYS
                outcome = "achieved" if self.rng.random() < min(1.0, share) else "not_achieved"
            else:
                return
        for row in self.tables["episodes"]:
            if row["mission_process_id"] == episode["mission_process_id"]:
                row.update(outcome=outcome, resolved_at=closing, resolution_received_at=closing)

    def record_sent(self, deliveries):
        for delivery in deliveries:
            if delivery["content_id"] is None:
                continue
            moment = datetime.combine(delivery["decision_day"], time(10))
            self.tables["notification_events"].append({
                "user_uuid": delivery["user_uuid"], "action": "sent", "content_id": delivery["content_id"],
                "content_type": delivery["format"], "mission_id": delivery["mission_id"],
                "mission_process_id": delivery["mission_process_id"], "rating": None, "created_at": moment,
                "process_id": self.next_id(), "received_at": moment})


def bounded(probability):
    return min(0.98, max(0.02, probability))


def simulate(participants, days, enrolment_days, scenario, effect, seed, recording_effect=0.0, backfill_ratio=1.0):
    """One replication; returns the decision rows, the final model state and the world."""
    rng = np.random.default_rng(seed)
    catalogue = catalogue_module.load()
    catalogue_module.set_recommendations(catalogue, {
        f"R-{m}-{j}": {"missions": [m]} for m in catalogue.missions for j in range(RECOMMENDATIONS_PER_MISSION)})
    world = World(rng, participants, days, enrolment_days, scenario, effect, catalogue,
                  recording_effect, backfill_ratio)
    decision_rows, deliveries_by_user, group_of, groups_by_stratum = [], {}, {}, {}
    latest = None

    ids = {"decision": 0, "delivery": 0}
    revisions_of, deliveries_by_id = {}, {}

    def add_deliveries(user_uuid, decision_id, deliveries, revision):
        for delivery in deliveries:
            ids["delivery"] += 1
            delivery.update(id=ids["delivery"], decision_id=decision_id, revision=revision, cancelled=False)
            deliveries_by_id[delivery["id"]] = delivery
        deliveries_by_user.setdefault(user_uuid, []).extend(deliveries)

    def run(now):
        nonlocal latest
        result = cycle.run_cycle(now, build_participants(world.tables), catalogue, decision_rows,
                                 deliveries_by_user, latest, group_of, groups_by_stratum,
                                 draw_seed=lambda: int(rng.integers(2 ** 62)), revisions_of=revisions_of)
        if result.new_model is not None:
            latest = result.new_model
            world.model_versions[latest.version] = (latest.posterior, latest.episodes)
        for row, deliveries in result.decisions:
            ids["decision"] += 1
            row["id"] = ids["decision"]
            decision_rows.append(row)
            add_deliveries(row["user_uuid"], row["id"], deliveries, 0)
            world.record_sent(deliveries)
        for decision_id, revision, reason, seed, cancelled, new_rows in result.revisions:
            revisions_of[decision_id] = revision
            for delivery_id in cancelled:
                deliveries_by_id[delivery_id]["cancelled"] = True
            user_uuid = next(r["user_uuid"] for r in decision_rows if r["id"] == decision_id)
            add_deliveries(user_uuid, decision_id, new_rows, revision)

    last_day = START + timedelta(days=enrolment_days + days + design.WINDOW_DAYS + design.EXPIRY_MARGIN_DAYS)
    day = START
    while day <= last_day:
        run(datetime.combine(day, time(1)))
        world.select_missions(day)
        run(datetime.combine(day, time(9)))
        extra_of = {(r["user_uuid"], r["decision_day"]): r["composition_r"] + r["composition_q"] - 2
                         for r in decision_rows if r["decision_day"] == day and r.get("composition_r") is not None}
        world.live_day(day, extra_of)
        day += timedelta(days=1)
    run(datetime.combine(day, time(1)))
    return decision_rows, latest, world


def summarise(replications, effect, scenario):
    """The quantities of the simulation study, over replications."""
    notifications_by_week, estimates, truths, covered = {}, [], [], []
    values = {"learned": [], "uniform": [], "most": []}
    contrast = composition.terms((1, 2)) - composition.terms((1, 1))
    n_terms = len(composition.TERM_NAMES)
    for decision_rows, latest, world in replications:
        learned = uniform = most = 0.0
        one_more = []
        for row in decision_rows:
            if row["status"] != "decided":
                continue
            probabilities = {composition.parse(k): p for k, p in row["composition_probabilities"].items()}
            extra = {c: c[0] + c[1] - 2 for c in probabilities}
            week = (row["decision_day"] - START).days // 7
            notifications_by_week.setdefault(week, []).append(sum(p * (extra[c] + 2) for c, p in probabilities.items()))
            learned += sum(p * world.observed_effect(row, extra[c]) for c, p in probabilities.items())
            uniform += float(np.mean([world.observed_effect(row, extra[c]) for c in probabilities]))
            most += world.observed_effect(row, max(extra.values()))
            one_more.append(world.observed_effect(row, 1))
        values["learned"].append(learned)
        values["uniform"].append(uniform)
        values["most"].append(most)
        mean = np.array(latest.posterior.mean[:n_terms])
        cov = np.array(latest.posterior.cov)[:n_terms, :n_terms]
        estimate = float(contrast @ mean)
        sd = math.sqrt(float(contrast @ cov @ contrast))
        truth = float(np.mean(one_more)) if one_more else 0.0
        estimates.append(estimate)
        truths.append(truth)
        covered.append(abs(estimate - truth) <= 1.645 * sd)
    weeks = {week: {"decisions": len(values_), **{f"q{int(q * 100):02d}": float(np.quantile(values_, q))
                                                   for q in (0.05, 0.25, 0.5, 0.75, 0.95)}}
             for week, values_ in sorted(notifications_by_week.items())}
    summary = {
        "scenario": scenario,
        "effect": effect,
        "replications": len(replications),
        "policy": design.POLICY,
        "daily_outcome": design.DAILY_OUTCOME,
        "expected_notifications_by_week": weeks,
        "marked_days_beyond_two_notifications_mean": {name: float(np.mean(v)) for name, v in values.items()},
    }
    summary["one_more_notification"] = {"estimate_mean": float(np.mean(estimates)),
                                        "truth_mean": float(np.mean(truths)),
                                        "bias": float(np.mean(np.array(estimates) - np.array(truths))),
                                        "estimate_sd": float(np.std(estimates, ddof=1)) if len(estimates) > 1 else None,
                                        "coverage_90": float(np.mean(covered))}
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--participants", type=int, default=10)
    parser.add_argument("--days", type=int, default=30, help="programme days per participant")
    parser.add_argument("--enrolment-days", type=int, default=7, help="span of enrolment dates")
    parser.add_argument("--scenario", choices=["null", "common", "heterogeneous", "decaying"], default="null")
    parser.add_argument("--effect", type=float, default=0.05)
    parser.add_argument("--policy", choices=["uniform", "conservative"], default=design.POLICY)
    parser.add_argument("--replications", type=int, default=5)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--recording-effect", type=float, default=0.0,
                        help="share of unrecorded behaviour recorded per notification beyond two")
    parser.add_argument("--backfill-ratio", type=float, default=1.0,
                        help="recording effect on the day before, relative to the day itself")
    parser.add_argument("--outcome", choices=["mark", "mark_minus_backfill"], default=design.DAILY_OUTCOME)
    arguments = parser.parse_args()
    design.POLICY = arguments.policy
    design.PROGRAMME_DAYS = arguments.days
    design.DAILY_OUTCOME = arguments.outcome
    effect = 0.0 if arguments.scenario == "null" else arguments.effect
    replications = [simulate(arguments.participants, arguments.days, arguments.enrolment_days, arguments.scenario,
                             effect, arguments.seed + r, arguments.recording_effect, arguments.backfill_ratio)
                    for r in range(arguments.replications)]
    summary = summarise(replications, effect, arguments.scenario)
    summary.update(recording_effect=arguments.recording_effect, backfill_ratio=arguments.backfill_ratio)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
