"""Revisions of a day's plan after events of the day.

The deliveries of a day are decided once, at the decision of that day (decide.py). Three
events of the day change the deliveries not yet sent (architecture_doc_cs_omi_rct_section.md):

- the day's task is marked done: every reminder of the day is cancelled, and the
  recommendations stay;
- the Mission's count of days is reached (Missions that state one): the same;
- the Mission of the plan is replaced or closed: the deliveries still active are replaced
  by content of the newly selected Mission with the same composition, R recommendations and
  Q reminders (decide.deliveries), or cancelled if no Mission is open. A recommendation position
  the new Mission cannot fill becomes a reminder; when the new Mission admits no reminder today
  (task marked, count reached), only the recommendation positions are refilled.

A revision never adds notifications beyond those decided in the morning, so the probabilities
logged with the decision stay the probabilities of what was decided; the new content is drawn at
random under the content rule and logged with its own probability. Nothing here touches the
database.
"""

import numpy as np

from cs_rct import decide, design
from cs_rct.history import marked_days, open_episode, scheduled_slot

# The roles of reminders: "reminder" since the composition design, and the two roles of the
# scheme of 14 October 2026, for plans decided before it.
REMINDER_ROLES = ("reminder", "fixed_reminder", "additional_reminder")


def revise(participant, catalogue, decision, day_deliveries, previous, day, now, seed):
    """Return (reason, ids of deliveries to cancel, new delivery rows), or None if nothing changes.

    decision: the row of the day; day_deliveries: the day's deliveries with "cancelled"; previous:
    the participant's deliveries of every day, for the content rule.
    """
    active = [d for d in day_deliveries if not d.get("cancelled") and d["content_id"] is not None]
    if not active:
        return None
    plan_process_id = active[0]["mission_process_id"]
    episode = open_episode(participant, now)

    if episode is None or episode["mission_process_id"] != plan_process_id:
        # The Mission of the plan has ended. Replace by the open one, or cancel.
        cancelled = [d["id"] for d in active]
        if episode is None or episode["mission_id"] not in catalogue.missions:
            return "mission_ended", cancelled, []
        mission = catalogue.missions[episode["mission_id"]]
        # The same composition, refilled for the new Mission.
        recommendations = sum(1 for d in active if d["format"] in ("recommendation", "resource"))
        reminders = len(active) - recommendations
        # At most one recommendation a day in Emotional Wellbeing: the others become reminders.
        limit = design.MAX_RECOMMENDATIONS_BY_PILLAR.get(mission.pillar, design.MAX_RECOMMENDATIONS)
        reminders += max(recommendations - limit, 0)
        recommendations = min(recommendations, limit)
        reminders_allowed = not _reminders_closed(participant, episode, mission, day, now)
        rows = decide.deliveries(
            np.random.default_rng(seed), participant, catalogue, mission, episode, day, now, previous,
            recommendations=recommendations, reminders=reminders if reminders_allowed else 0,
            scheduled_at=scheduled_slot(participant, episode, day, now),
            used_today={d["content_id"] for d in day_deliveries}, missing_as_reminders=reminders_allowed)
        start = max(d["position"] for d in day_deliveries) + 1
        for offset, row in enumerate(rows):
            row["position"] = start + offset
        return "mission_changed", cancelled, rows

    reminders = [d for d in active if d["role"] in REMINDER_ROLES]
    if not reminders:
        return None
    mission = catalogue.missions.get(episode["mission_id"])
    if mission is None:
        return None
    marked = marked_days(participant, episode, now)
    if mission.states_count and len(marked) >= mission.count:
        return "count_reached", [d["id"] for d in reminders], []
    if day in marked:
        return "task_marked", [d["id"] for d in reminders], []
    return None


def _reminders_closed(participant, episode, mission, day, now):
    marked = marked_days(participant, episode, now)
    return day in marked or (mission.states_count and len(marked) >= mission.count)
