"""The decision of one participant-day: the composition (R, Q) of the day and its content.

rct/design/rct_design_record_2026-10-27.md, Sections 1 and 12; implementation_plan_composition_2026-11-04.md,
Phases 1 and 3. Every day of an open episode carries R recommendations (a resource counts as one) and Q
reminders. The compositions the day admits come from composition.admissible: 2 <= R + Q <= 4, R <= 2
(R <= 1 in Emotional Wellbeing), less those removed by the rules of the day:

- the day's task already marked, or the Mission's count reached: no reminder;
- no more recommendations than the Mission can supply that day without sending one code twice;
- the weekly cap.

The status of the day, in order:

- no open episode, or the participant disabled: nothing is sent and no decision is taken (the row
  of the day is written by the closing pass, close_day);
- the day of the selection, when design.FIRST_DELIVERY_DAY is "next_day" (an open decision): nothing
  is sent, and the closing pass writes the row as 'selection_day';
- the episode past its window of W days ('past_window'): nothing is sent, and the row is written
  so that the day is logged while the module waits for the Mission to be closed;
- no admissible composition ('below_minimum', open decision 8 of the design record): one
  recommendation alone if the Mission can supply one, as before the composition design;
- one admissible composition ('no_choice'): that composition, not randomised;
- otherwise ('decided') the composition is drawn from composition.probabilities, the policy in force
  (design.POLICY), or from the starting probabilities in the fixed group. Only these days are decision
  points of the model.

The probabilities of the admissible compositions are written before the draw, which is the first
draw of the day's generator. Nothing here touches the database. `previous` is the list of deliveries
the service handed out to the participant on earlier days, as written by store.py.
"""

import itertools
from datetime import datetime, time, timedelta

import numpy as np

from cs_rct import composition, content, design, features
from cs_rct.history import disliked_items, marked_days, open_episode, start_day, step3_profile


def decide(participant, catalogue, posterior, model_version, group, day, now, seed, previous,
           study=None):
    """Return (decision row, delivery rows) for `day`, or None if there is no open episode.

    posterior is None when the model could not be loaded; the conservative rule then uses its
    starting probabilities and the row is flagged as a fallback. group is "learned" or "fixed"
    (the fixed-probability group of the within-arm evaluation).
    """
    study = study or design.STUDY
    if participant.disabled_at is not None and participant.disabled_at <= now:
        return None
    episode = open_episode(participant, now)
    if episode is None:
        return None
    if design.FIRST_DELIVERY_DAY == "next_day" and start_day(episode) == day:
        # Deliveries start the morning after the selection (design.FIRST_DELIVERY_DAY, open decision).
        return None
    mission = catalogue.missions.get(episode["mission_id"])
    rng = np.random.default_rng(seed)

    row = empty_row(day, now, participant.user_uuid, group, seed)
    row.update(mission_process_id=episode["mission_process_id"], mission_id=episode["mission_id"],
               model_version=model_version, fallback=False)
    if mission is None:
        row["status"] = "unknown_mission"
        return row, []

    s, psi, context = features.compute(participant, episode, mission, day, now, study, previous)
    row["state"] = dict(zip(features.state_names(study), s))
    row["baseline"] = dict(zip(features.baseline_names(study), psi))
    row["context"] = context

    if context["episode_day"] >= design.WINDOW_DAYS:
        # No notification after the window: the module waits for the Mission to be closed.
        row["status"] = "past_window"
        return row, []

    marked = marked_days(participant, episode, now)
    if mission.states_count and len(marked) >= mission.count:
        closed = "count_reached"
    elif day in marked:
        closed = "task_marked"
    else:
        closed = None
    available = recommendation_supply(participant, catalogue, mission, episode, day, now, previous)
    admissible, removed = composition.admissible(mission.pillar, closed, available,
                                                 cap_room(day, previous))
    row["blocked_by"] = closed
    row["removed_compositions"] = removed
    scheduled_at = datetime.fromisoformat(context["scheduled_at"]) if context["scheduled_at"] else None

    if not admissible:
        row["status"] = "below_minimum"
        return row, deliveries(rng, participant, catalogue, mission, episode, day, now, previous,
                               recommendations=min(available, 1), reminders=0, scheduled_at=scheduled_at)

    if len(admissible) == 1:
        row["status"] = "no_choice"
        probabilities = {composition.key(admissible[0]): 1.0}
    else:
        row["status"] = "decided"
        if group == "fixed":
            probabilities = composition.starting_probabilities(design.POLICY, admissible, design.EXPLORE_SHARE,
                                                               design.DEFAULT_COMPOSITION)
            row["policy_rule"] = "fixed"
        else:
            probabilities, row["policy_rule"] = composition.probabilities(design.POLICY, admissible, posterior, seed)
        row["fallback"] = row["policy_rule"] == "fallback"
    row["composition_probabilities"] = probabilities
    # The composition is the first draw of the day's generator.
    chosen = composition.draw(rng, probabilities)
    row["composition_r"], row["composition_q"] = chosen
    row["draws"] = [content.record("composition", probabilities, composition.key(chosen),
                                   forced=len(probabilities) == 1)]
    return row, deliveries(rng, participant, catalogue, mission, episode, day, now, previous,
                           recommendations=chosen[0], reminders=chosen[1], scheduled_at=scheduled_at)


def empty_row(day, now, user_uuid, group, seed):
    """A decision row with every column empty. The columns of the scheme of 14 October 2026
    (propensity, propensity_raw, propensity_learned, admissible_draws, draw_1, draw_2, additional) stay
    in the table and are NULL on rows of the composition design."""
    return {
        "decision_day": day,
        "decided_at": now,
        "user_uuid": user_uuid,
        "mission_process_id": None,
        "mission_id": None,
        "group_name": group,
        "status": None,
        "state": None,
        "context": None,
        "baseline": None,
        "propensity": None,
        "propensity_raw": None,
        "propensity_learned": None,
        "fallback": None,
        "admissible_draws": None,
        "draw_1": None,
        "draw_2": None,
        "blocked_by": None,
        "additional": None,
        # The composition drawn, the probability of each admissible composition ({"R,Q": p}) and the
        # compositions removed by a rule of the day ({"R,Q": rule}).
        "composition_r": None,
        "composition_q": None,
        "composition_probabilities": None,
        "removed_compositions": None,
        "model_version": None,
        "parameters": design.parameters(),
        "rng_seed": seed,
        # Which rule gave the probabilities: 'uniform', 'conservative', 'fixed' or 'fallback'; NULL on
        # a day that is not a decision point.
        "policy_rule": None,
        # The day-level draws, written to the table draws with no delivery (store.save_decision).
        "draws": [],
    }


def cap_room(day, previous):
    """Notifications left today under the weekly cap: the cap less the deliveries of the six
    preceding days that were not withdrawn."""
    week_start = day - timedelta(days=6)
    earlier = sum(1 for delivery in previous
                  if week_start <= delivery["decision_day"] < day and delivery["content_id"] is not None
                  and not delivery.get("cancelled"))
    return design.WEEKLY_CAP - earlier


def content_state(participant, catalogue, mission, episode, day, now, previous):
    """What the content rule reads for the day's recommendations and resource."""
    earlier = [d for d in previous if d["decision_day"] < day]
    code_of = lambda item: catalogue.recommendation_code.get(item, item)  # noqa: E731
    # Recommendations withdrawn by a revision before they were sent do not count as sent.
    last_sent = {}
    for d in earlier:
        if d["format"] == "recommendation" and d["content_id"] is not None and not d.get("cancelled"):
            last_sent[d["content_id"]] = max(d["decision_day"], last_sent.get(d["content_id"], d["decision_day"]))
    yesterday = day - timedelta(days=1)
    prioritised, priority_probability = priority(catalogue, mission, participant, now)
    # One resource per episode, in the first plan of the episode with a recommendation position
    # (design record, Section 1.4, until the weekly resource is implemented). A resource withdrawn by
    # a revision before it was sent does not count as received.
    received = {d["content_id"] for d in previous if d["format"] == "resource" and not d.get("cancelled")}
    episode_has_resource = any(d["format"] == "resource" and not d.get("cancelled")
                               and d.get("mission_process_id") == episode["mission_process_id"] for d in previous)
    return {
        "tagged": catalogue.recommendations_of.get(mission.mission_id, []),
        "disliked": disliked_items(participant, now),
        "last_sent": last_sent,
        "code_of": code_of,
        "prioritised": prioritised,
        "priority_probability": priority_probability,
        "yesterday_codes": {code_of(item) for item, sent_on in last_sent.items() if sent_on == yesterday},
        "resources": [] if episode_has_resource else catalogue.resources_of.get(mission.mission_id, []),
        "received": received,
    }


def recommendation_draw(state, today_codes):
    return dict(tagged=state["tagged"], disliked=state["disliked"], last_sent=state["last_sent"],
                code_of=state["code_of"], prioritised=state["prioritised"],
                priority_probability=state["priority_probability"], today_codes=today_codes,
                yesterday_codes=state["yesterday_codes"])


def recommendation_supply(participant, catalogue, mission, episode, day, now, previous, used_today=()):
    """How many recommendations (a resource counts as one) the Mission can supply today, up to
    design.MAX_RECOMMENDATIONS, without sending one code twice. A separate generator, so that the
    draws of the day are unchanged."""
    state = content_state(participant, catalogue, mission, episode, day, now, previous)
    supply = int(any(item not in state["received"] for item in state["resources"]))
    today_codes = {state["code_of"](item) for item in used_today}
    probe = np.random.default_rng(0)
    while supply < design.MAX_RECOMMENDATIONS:
        choice = content.choose_recommendation(probe, **recommendation_draw(state, today_codes))
        if choice.content_id is None:
            break
        today_codes.add(state["code_of"](choice.content_id))
        supply += 1
    return supply


def deliveries(rng, participant, catalogue, mission, episode, day, now, previous, recommendations, reminders,
               scheduled_at, used_today=(), missing_as_reminders=False):
    """The day's deliveries: `recommendations` recommendation positions, then `reminders` reminders.

    The first recommendation position of an episode carries one of its resources when one is left.
    On a day whose task is scheduled, the first reminder is the one sent at the declared hour.
    used_today: content ids already handed out today, excluded from the draws (a revision that
    replaces the content of a day, revise.py). missing_as_reminders: a recommendation position that
    the Mission cannot fill becomes a reminder (revise.py, a Mission change); otherwise it is left out."""
    language = participant.language if participant.language in ("es", "en", "it", "ro") else "es"
    state = content_state(participant, catalogue, mission, episode, day, now, previous)
    earlier = [d for d in previous if d["decision_day"] < day]
    recent_reminders = {d["content_id"] for d in earlier
                        if d["format"] == "reminder"
                        and d["decision_day"] >= day - timedelta(days=design.REMINDER_NO_REPEAT_DAYS - 1)}
    recent_reminders |= set(used_today)

    chosen_today = set(used_today)
    rows = []
    missing = 0
    for position in range(recommendations):
        if position == 0 and state["resources"]:
            resource = content.choose_resource(rng, state["resources"], state["received"])
            if resource.content_id is not None:
                rows.append(delivery_row("resource", "resource", resource, {}))
                continue
        # A code already in today's plan is never drawn again the same day, even once every
        # code of the Mission has been sent.
        today_codes = {state["code_of"](item) for item in chosen_today}
        recommendation = content.choose_recommendation(rng, **recommendation_draw(state, today_codes))
        if recommendation.content_id is None:
            missing += 1
            continue
        chosen_today.add(recommendation.content_id)
        row = delivery_row("recommendation", "recommendation", recommendation, {})
        row["content_rule"] = recommendation.rule
        rows.append(row)
    if missing_as_reminders:
        reminders += missing

    hour_reminder = catalogue.hour_reminder_of.get(mission.mission_id)
    day_count = reminders
    if reminders and scheduled_at is not None and hour_reminder is not None:
        choice = content.Choice(hour_reminder.reminder_id, 1.0, [hour_reminder.reminder_id],
                                draws=[content.record("reminder_text", {hour_reminder.reminder_id: 1.0},
                                                      hour_reminder.reminder_id, forced=True)])
        rows.append(delivery_row("reminder", "reminder", choice,
                                 {"declared_at": scheduled_at, "deadline": scheduled_at},
                                 reminder=hour_reminder, language=language))
        day_count -= 1
    pool = [reminder.reminder_id for reminder in catalogue.day_reminder_pool(mission.mission_id)]
    # Once the pool less the texts sent within design.REMINDER_NO_REPEAT_DAYS is used up, a text
    # is repeated (content.choose_day_reminders, note "repeat_within_no_repeat_days").
    for choice in content.choose_day_reminders(rng, pool, recent_reminders, day_count):
        # A reminder is due by the scheduled time of the day's task, and by the latest time its
        # text allows (reminders/latest_times.py), whichever comes first.
        reminder = catalogue.reminders.get(choice.content_id)
        rows.append(delivery_row("reminder", "reminder", choice, {"deadline": earliest(scheduled_at, day, reminder),
                                                                  "day": day}, reminder=reminder, language=language))

    # The pop-up text of each recommendation and resource (content.choose_push_text).
    shown = [d for d in previous if d.get("push_text_id") is not None and not d.get("cancelled")]
    previous_push = max((d for d in shown if d["decision_day"] < day),
                        key=lambda d: (d["decision_day"], d["position"]), default={}).get("push_text_id")
    push_today = {d["push_text_id"] for d in shown if d["decision_day"] == day}
    for row in rows:
        if row["format"] in ("recommendation", "resource") and row["content_id"] is not None:
            push = content.choose_push_text(rng, catalogue.push_texts, push_today, previous_push)
            row["push_text_id"], row["push_text_probability"] = push.content_id, push.probability
            row["draws"] = row["draws"] + push.draws
            push_today.add(push.content_id)
    rows = random_order(rng, rows)
    for position, delivery in enumerate(rows):
        delivery["position"] = position
        delivery["decision_day"] = day
        delivery["user_uuid"] = participant.user_uuid
        delivery["mission_process_id"] = episode["mission_process_id"]
        delivery["mission_id"] = mission.mission_id
        # The scheduled time of the day's task, a fact of the day sent at plan level.
        delivery["task_scheduled_for"] = scheduled_at
    return rows


def priority(catalogue, mission, participant, now):
    """(ids of the prioritised class among the Mission's recommendations, its probability), or
    (None, None) when no priority applies.

    In Smoking and Alcohol a pre-intender or intender gets reflective-motivation content; an actor,
    and every participant in the other pillars, gets content whose COM-B sub-domain is a barrier
    flagged in Step 3 (email of the clinical team, 23 October 2026, points 2 and 4). A recommendation
    with several sub-domains is in the class if any of them is (PROVISIONAL: the rule for 58 such
    recommendations is asked to the clinical team)."""
    profile = step3_profile(participant, mission.pillar, now)
    if mission.pillar in design.HAPA_PILLARS and profile["hapa_stage"] in ("pre-intender", "intender"):
        targets, probability = {"reflective motivation"}, design.HAPA_PRIORITY_PROBABILITY
    elif profile["barriers"]:
        targets, probability = profile["barriers"], design.BARRIER_PRIORITY_PROBABILITY
    else:
        return None, None
    tagged = catalogue.recommendations_of.get(mission.mission_id, [])
    return {item for item in tagged if catalogue.recommendation_subdomains.get(item, frozenset()) & targets}, probability


def random_order(rng, rows):
    """The day's deliveries in a uniformly drawn order (decided 8 November 2026, to estimate an effect
    of the order after the trial). The reminder at the declared hour has its own time and goes last;
    every other delivery takes a position drawn uniformly among the n! orders. The draw is the last
    one of the day's generator, so the content draws before it are unchanged, and it is written with
    the draws of the first delivery (kind "order", the order as content ids joined by '>')."""
    hour = [row for row in rows if row["reminder_kind"] == "hour"]
    movable = [row for row in rows if row["reminder_kind"] != "hour"]
    if len(movable) < 2:
        return movable + hour
    permutation = [int(i) for i in rng.permutation(len(movable))]
    ordered = [movable[i] for i in permutation]
    label = lambda order: ">".join(str(movable[i]["content_id"]) for i in order)  # noqa: E731
    orders = list(itertools.permutations(range(len(movable))))
    ordered[0]["draws"] = ordered[0]["draws"] + [
        content.record("order", {label(order): 1 / len(orders) for order in orders}, label(permutation))]
    return ordered + hour


def earliest(scheduled_at, day, reminder):
    limits = [scheduled_at]
    if reminder is not None and reminder.latest_time is not None:
        limits.append(datetime.combine(day, reminder.latest_time))
    limits = [limit for limit in limits if limit is not None]
    return min(limits) if limits else None


def delivery_row(role, format_name, choice, timing, reminder=None, language=None):
    return {
        "role": role,
        "format": format_name,
        "content_id": choice.content_id,
        "content_probability": choice.probability,
        "candidates": choice.candidates,
        "excluded": choice.excluded,
        "note": choice.note,
        "reminder_kind": reminder.kind if reminder else None,
        # The ID of the pop-up text of a recommendation or resource, set in deliveries().
        "push_text_id": None,
        "push_text_probability": None,
        # Of a recommendation: its class in the content rule, the class probability and the
        # prioritised items (content.choose_recommendation).
        "content_rule": None,
        "bct": reminder.bct if reminder else None,
        "text": reminder.texts.get(language) if reminder else None,
        "language": language if reminder else None,
        # The declared hour of a scheduled task; otherwise OMI chooses the hour within the day.
        "declared_at": timing.get("declared_at"),
        "deadline": timing.get("deadline"),
        # The latest time the reminder's own text allows (reminders/latest_times.py), on its day.
        "latest_time": (datetime.combine(timing["day"], reminder.latest_time)
                        if reminder is not None and reminder.latest_time is not None and "day" in timing else None),
        # Every draw behind this delivery (format, content, pop-up text), written to the table draws.
        "draws": list(choice.draws),
    }


def closing_status(participant, day, now):
    """The status of a past day on which no decision was written: why nothing was decided."""
    if participant.disabled_at is not None and participant.disabled_at.date() <= day:
        return "disabled"
    end_of_day = datetime.combine(day, time.max)
    episode = open_episode(participant, min(end_of_day, now))
    if episode is None:
        return "no_open_episode"
    if design.FIRST_DELIVERY_DAY == "next_day" and start_day(episode) == day:
        return "selection_day"
    return "selected_after_last_decision_hour"
