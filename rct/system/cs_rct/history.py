"""What the service knows about one participant, and the episode rules built on it.

The rows are those of the derived tables (parse_events.py). Every rule here takes
the moment of reading, `now`, and uses only the rows the service had received by
then, so that the same function serves the decision, the update of the model and
the replay of both.

Rules (rct/paper/main.tex, Section "Episodes and the daily outcome"):
- an episode is one selected Mission, open from its selection until its outcome
  or its expiry at t0 + W + delta, whichever comes first;
- an episode replaced by another ("changed") is withdrawn: it carries no outcome Y,
  but its days update the model like those of any other ended episode, with the
  marks entered by the replacement (decided 8 November 2026: a Task done on the
  day of a Mission replaced within a day is kept as a positive day);
- the daily outcome of a day is the calendar mark of that day, r = 1 if the
  episode carries a mark whose reference day is that day, entered by the end of
  the episode, and r = 0 otherwise; every day of an ended episode is observed
  (o = 1). With design.DAILY_OUTCOME = "mark_minus_backfill", 1 is
  subtracted on a day on which a mark naming the day before was entered.
"""

from dataclasses import dataclass, field
from datetime import datetime, time, timedelta, timezone
from zoneinfo import ZoneInfo

from cs_rct import design

# Y of an episode by its outcome; "finished" (mission_finished) ends the episode with a Y not
# yet known. Y is read only after the trial; the update uses the daily outcomes.
OUTCOME_VALUE = {"achieved": 1, "not_achieved": 0, "finished": None}


@dataclass
class Participant:
    user_uuid: str
    language: str | None = None
    enrolment_date: object = None   # date
    age: int | None = None
    recruitment_center: str | None = None
    timezone: str = "UTC"               # IANA name; every time of the participant is local to it
    sex: str | None = None              # attribute "sex" of new_user, else "gender"
    education: str | None = None
    disabled_at: datetime | None = None
    episodes: list = field(default_factory=list)       # rows of episodes, by selection time
    marks: list = field(default_factory=list)
    scheduled: list = field(default_factory=list)      # rows of scheduled_tasks
    notifications: list = field(default_factory=list)  # rows of notification_events
    health: list = field(default_factory=list)         # rows of health_assessments
    step3: list = field(default_factory=list)          # rows of psychological_assessments


def build_participants(tables):
    """Group the rows of the derived tables by participant."""
    participants = {}

    def get(user_uuid):
        if user_uuid not in participants:
            participants[user_uuid] = Participant(user_uuid=user_uuid)
        return participants[user_uuid]

    for row in tables.get("users", []):
        participant = get(row["user_uuid"])
        participant.language = row.get("language")
        participant.enrolment_date = row.get("enrolment_date")
        participant.age = row.get("age")
        participant.recruitment_center = row.get("recruitment_center")
        attributes = row.get("attributes") or {}
        participant.sex = attributes.get("sex") or attributes.get("gender")
        participant.education = attributes.get("education")
        participant.disabled_at = row.get("disabled_at")
    for row in tables.get("episodes", []):
        if row.get("user_uuid") is not None:
            get(row["user_uuid"]).episodes.append(row)
    for row in tables.get("marks", []):
        get(row["user_uuid"]).marks.append(row)
    for row in tables.get("scheduled_tasks", []):
        get(row["user_uuid"]).scheduled.append(row)
    for row in tables.get("notification_events", []):
        get(row["user_uuid"]).notifications.append(row)
    for row in tables.get("health_assessments", []):
        get(row["user_uuid"]).health.append(row)
    for row in tables.get("psychological_assessments", []):
        get(row["user_uuid"]).step3.append(row)

    for participant in participants.values():
        participant.timezone = design.CENTRE_TIMEZONE.get(participant.recruitment_center, design.DEFAULT_TIMEZONE)
        to_participant_time(participant)
        # An episode first seen through its outcome has no selection time and is never decided on.
        participant.episodes = sorted((e for e in participant.episodes if e["selected_at"] is not None),
                                      key=lambda e: e["selected_at"])
    return participants


def local_time(moment, zone):
    """A naive UTC datetime as naive local time in the IANA zone `zone`."""
    if moment is None:
        return None
    return moment.replace(tzinfo=timezone.utc).astimezone(ZoneInfo(zone)).replace(tzinfo=None)


def to_participant_time(participant):
    """Convert, in copies of the rows, every time of the participant from UTC to local time."""
    zone = participant.timezone
    participant.disabled_at = local_time(participant.disabled_at, zone)
    for name in ("episodes", "marks", "scheduled", "notifications", "health", "step3"):
        rows = []
        for row in getattr(participant, name):
            row = dict(row)
            for key, value in row.items():
                if isinstance(value, datetime):
                    row[key] = local_time(value, zone)
            rows.append(row)
        setattr(participant, name, rows)


def known(row, now, field_name="received_at"):
    """Whether the service had received the row by `now`. Rows without a reception time count as known."""
    received = row.get(field_name)
    return received is None or received <= now


# ---------------------------------------------------------------- episodes

def start_day(episode):
    return episode["selected_at"].date()


def expiry_time(episode):
    """Start of day t0 + W + delta."""
    return datetime.combine(start_day(episode) + timedelta(days=design.WINDOW_DAYS + design.EXPIRY_MARGIN_DAYS),
                            time.min)


def resolution(episode, now):
    """How the episode stands at `now`: (status, end time, outcome Y).

    status is "open", "withdrawn" (replaced by another Mission), "resolved" (completed
    or closed, Y = 1 or 0) or "expired" (Y = 0). For an open episode end time and Y are None.
    """
    expiry = expiry_time(episode)
    outcome_known = episode["outcome"] is not None and known(episode, now, "resolution_received_at")
    if outcome_known and episode["resolved_at"] is not None and episode["resolved_at"] < expiry:
        if episode["outcome"] == "changed":
            return "withdrawn", episode["resolved_at"], None
        return "resolved", episode["resolved_at"], OUTCOME_VALUE[episode["outcome"]]
    if now >= expiry:
        return "expired", expiry, 0
    if outcome_known:
        # An outcome without a time is read at the moment the service received it.
        end = episode["resolved_at"] or episode["resolution_received_at"] or now
        if episode["outcome"] == "changed":
            return "withdrawn", end, None
        return "resolved", end, OUTCOME_VALUE[episode["outcome"]]
    return "open", None, None


def open_episode(participant, now):
    """The episode open at `now`, or None. A participant has at most one: the latest selected."""
    for episode in reversed(participant.episodes):
        if episode["selected_at"] > now or not known(episode, now, "selection_received_at"):
            continue
        status, _, _ = resolution(episode, now)
        return episode if status == "open" else None
    return None


def episode_marks(participant, episode, now, entered_by=None):
    """The marks of the episode the service had received by `now`, entered by `entered_by` if given."""
    marks = []
    for mark in participant.marks:
        if mark["mission_process_id"] != episode["mission_process_id"] or not known(mark, now):
            continue
        if mark["reference_time"] is None:
            continue
        if entered_by is not None and mark["entered_at"] is not None and mark["entered_at"] > entered_by:
            continue
        marks.append(mark)
    return marks


def marked_days(participant, episode, now, entered_by=None):
    """The reference days of the episode's marks, as a set of dates."""
    return {mark["reference_time"].date() for mark in episode_marks(participant, episode, now, entered_by)}


def daily_outcomes(participant, episode, days, now, decided_at=None):
    """The daily outcome r of each given day of a finished episode, as {day: 0 or 1}.

    Returns None while the episode is open. A withdrawn episode ends at its replacement and its
    days are read like those of any other ended episode.
    Marks count if entered by the end of the episode, whatever the day they name
    (open decision 8 of the paper keeps marks entered at completion as the days they name).
    decided_at: {day: local time of the decision of that day}, used by "mark_minus_backfill"
    (backfilled_days).
    """
    status, end, _ = resolution(episode, now)
    if status == "open":
        return None
    marked = marked_days(participant, episode, now, entered_by=end)
    outcomes = {day: int(day in marked) for day in days}
    if design.DAILY_OUTCOME == "mark_minus_backfill":
        backfilled = backfilled_days(participant, episode, now, end, decided_at)
        outcomes = {day: value - int(day in backfilled) for day, value in outcomes.items()}
    return outcomes


def backfilled_days(participant, episode, now, end, decided_at=None):
    """The days t on which a mark naming day t-1 was entered after the decision of day t, as a set
    of dates.

    Only a mark entered after the decision of day t can be an effect of the deliveries of day t
    (decided 8 November 2026). The cut is the time of the decision and not the time of the first
    delivery: the decision time is fixed before the composition is drawn, whereas the time of the
    first delivery is chosen by OMI and can depend on the number of notifications of the day, so a
    cut there would make the window in which marks are counted longer on days with more
    notifications and create a difference between compositions with no effect on recording.
    A mark entered between the decision and the first delivery is counted: it cannot be an effect
    of the deliveries, but it does not depend on the composition either, so it adds variance and no
    bias. decided_at: {day: local time of the decision}; a day not in it (no decision row) counts
    every mark entered on it."""
    decided_at = decided_at or {}
    days = set()
    for mark in episode_marks(participant, episode, now, entered_by=end):
        entered = mark["entered_at"]
        if entered is None or entered.date() != mark["reference_time"].date() + timedelta(days=1):
            continue
        cut = decided_at.get(entered.date())
        if cut is None or entered > cut:
            days.add(entered.date())
    return days


def scheduled_slot(participant, episode, day, now):
    """The start of the task scheduled on `day` for the episode, or None. The latest plan entered wins."""
    slots = [row for row in participant.scheduled
             if row["mission_process_id"] == episode["mission_process_id"] and known(row, now)
             and row["start_at"] is not None and row["start_at"].date() == day]
    if not slots:
        return None
    return max(slots, key=lambda row: (row["entered_at"] or datetime.min))["start_at"]


def earlier_marked_share(participant, before_episode, now):
    """The participant's share of marked days over earlier non-withdrawn episodes finished by `now`.

    Returns None if there is none. Each episode counts W days.
    """
    marked, days = 0, 0
    for episode in participant.episodes:
        if episode is before_episode or episode["selected_at"] >= before_episode["selected_at"]:
            continue
        status, end, _ = resolution(episode, now)
        if status in ("open", "withdrawn"):
            continue
        window = {start_day(episode) + timedelta(days=offset) for offset in range(design.WINDOW_DAYS)}
        marked += len(marked_days(participant, episode, now, entered_by=end) & window)
        days += design.WINDOW_DAYS
    return marked / days if days else None


def latest_health_score(participant, pillar_id, now):
    """The score of the latest HHS assessment of the pillar received by `now`, or None."""
    rows = [row for row in participant.health
            if row["pillar_id"] == pillar_id and row["score"] is not None and known(row, now)]
    if not rows:
        return None
    return max(rows, key=lambda row: (row["assessed_at"] or datetime.min))["score"]


# The pillar_id of the EUT messages for each pillar of the Mission catalogue.
PILLAR_ID = {"Smoking": "SO", "Alcohol": "AO", "Nutrition": "NO", "Physical_activity": "PO",
             "Emotional_Wellbeing": "EO"}


def step3_profile(participant, pillar, now):
    """The Step 3 profile of the participant in a pillar of the catalogue, from the questionnaires
    received by `now`: {"barriers": set of COM-B sub-domains, "hapa_stage": str or None}.

    A pillar can have several Step 3 questionnaires (one per subpillar in Nutrition and Emotional
    Wellbeing); the latest answer of each counts, and the barriers are the union of those reported
    by EUT and those of the tagged answers. The HAPA stage is that of the latest answer that gives one.
    """
    rows = sorted((row for row in participant.step3 if row["pillar_id"] == PILLAR_ID.get(pillar) and known(row, now)),
                  key=lambda row: row["assessed_at"] or row["received_at"] or datetime.min)
    latest = {}
    for row in rows:
        latest[row["subpillar"]] = row
    barriers = set()
    for row in latest.values():
        barriers |= set(row["barriers_reported"] or []) | set(row["barriers_from_answers"] or [])
    stages = [row["hapa_stage"] for row in rows if row["hapa_stage"]]
    return {"barriers": barriers, "hapa_stage": stages[-1] if stages else None}


def disliked_items(participant, now):
    """Content ids the participant rated as disliked, by `now`."""
    return {row["content_id"] for row in participant.notifications
            if row["action"] == "rated" and row["rating"] == design.DISLIKE_RATING and known(row, now)}


def sent_count(participant, first_day, last_day, now):
    """Notifications sent to the participant on days first_day..last_day, as recorded by EUT."""
    return sum(1 for row in participant.notifications
               if row["action"] == "sent" and known(row, now) and row["created_at"] is not None
               and first_day <= row["created_at"].date() <= last_day)
