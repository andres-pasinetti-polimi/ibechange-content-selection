"""Turn the stored Artemis messages into rows of the tables the service works on.

Nothing here touches the database: the functions take message bodies and
return plain rows, so that they can be tested on their own. rebuild_tables.py
reads the bodies from raw_events and writes the rows.

Message formats: rct/design/iBeChange-modules-architecture.docx (30 September 2026).
Messages about pillars and escalation are not parsed yet; they stay in raw_events.
A psychological_assessments message (Step 3) is parsed only if it carries user_uuid, which the
version of the document of 30 September 2026 does not list (question 12 to EUT); one without it
is counted as malformed and stays in raw_events.
"""

import csv
import json
import logging
from collections import Counter
from datetime import datetime, timezone
from functools import lru_cache

from cs_rct.catalogue import CATALOGUE_FOLDER, hapa_stage_name, subdomain_names

log = logging.getLogger("parse_events")

# Outcome of an episode, as named in the outcome table of the EUT document.
OUTCOME_OF_EVENT = {
    "mission_accomplished": "achieved",
    "mission_not_achieved": "not_achieved",
    "mission_changed": "changed",
    # Sent on INT since 1 October 2026 and not listed in the EUT document; it ends the
    # episode, with an outcome whose meaning is to be confirmed with EUT.
    "mission_finished": "finished",
}

NOTIFICATION_ACTION_OF_EVENT = {
    "notification_sent": "sent",
    "notification_opened": "opened",
    "notification_rated": "rated",
    "notification_saved": "saved",
    "notification_unsaved": "unsaved",
}


def parse_events(messages):
    """Parse messages, given in the order in which they arrived.

    Each message is a body, or a pair (body, received_at) where received_at is the
    time the service stored it. Every row carries the received_at of the message it
    comes from, so that a decision can be computed from what the service had
    received at that moment and nothing later (paper, Section "The daily decision
    problem": every feature is computed from the information available at the decision).

    Returns one list of rows per table:
    users, episodes, marks, scheduled_tasks, notification_events, health_assessments,
    psychological_assessments.
    """
    users = {}  # user_uuid -> row
    episodes = {}  # mission_process_id -> row
    marks = []
    scheduled_tasks = []
    notification_events = []
    health_assessments = []
    psychological_assessments = []
    skipped = Counter()
    failed = Counter()

    seen_bodies = set()
    for message in messages:
        body, received_at = message if isinstance(message, tuple) else (message, None)
        # A message delivered twice has the same body: keep the first copy only.
        if body in seen_bodies:
            continue
        seen_bodies.add(body)

        event_type, data = read_body(body)

        if event_type is None or event_type == "null" or not isinstance(data, dict):
            # Seen on INT: bodies that are not one event, and events whose type is the string "null".
            skipped[str(event_type)] += 1
            continue
        try:
            if event_type == "new_user":
                user = get_user(users, data["uuid"])
                attributes = data.get("attributes", {})
                user["personal_id"] = data.get("personalId")
                user["language"] = data.get("language")
                user["enrolment_date"] = to_date(attributes.get("enrolmentDate"))
                user["age"] = to_int(attributes.get("userAge"))
                user["recruitment_center"] = attributes.get("recruitmentCenter")
                user["attributes"] = attributes

            elif event_type == "disabled_user":
                user = get_user(users, data["user_uuid"])
                user["disabled_at"] = to_datetime(data.get("date_disabled"))

            elif event_type == "mission_selected":
                episode = get_episode(episodes, int(data["mission_process_id"]))
                episode["user_uuid"] = data["user_uuid"]
                # This message names the field "mission"; the others name it "mission_id".
                episode["mission_id"] = data.get("mission_id") or data.get("mission")
                episode["selected_at"] = to_datetime(data.get("created_at"))
                episode["selection_received_at"] = received_at

            elif event_type in OUTCOME_OF_EVENT:
                episode = get_episode(episodes, int(data["mission_process_id"]))
                episode["user_uuid"] = data["user_uuid"]
                episode["mission_id"] = episode["mission_id"] or data.get("mission_id")
                # If more than one outcome arrives, the last one is kept.
                episode["outcome"] = OUTCOME_OF_EVENT[event_type]
                episode["resolved_at"] = to_datetime(data.get("created_at"))
                episode["resolution_received_at"] = received_at

            elif event_type == "task_step_done":
                properties = data.get("properties", {})
                marks.append(
                    {
                        "user_uuid": data["user_uuid"],
                        "mission_process_id": to_int(data.get("mission_process_id")),
                        "mission_id": properties.get("mission_id") or properties.get("mission"),
                        "step": to_text(properties.get("step")),
                        "total_steps": to_int(properties.get("total_steps")),
                        # The day the mark refers to, and the moment it was entered.
                        "reference_time": to_datetime(properties.get("finished_at")),
                        "entered_at": to_datetime(data.get("created_at")),
                        "process_id": to_int(data.get("process_id")),
                        "received_at": received_at,
                    }
                )

            elif event_type in ("task_step_scheduled", "mission_scheduled"):
                # mission_scheduled is the earlier name of the same message (seen on INT, September 2026).
                properties = data.get("properties", {})
                for slot in properties.get("scheduled_for", []):
                    scheduled_tasks.append(
                        {
                            "user_uuid": data["user_uuid"],
                            "mission_process_id": to_int(data.get("mission_process_id")),
                            "mission_id": properties.get("mission_id") or properties.get("mission"),
                            "step": to_text(properties.get("step")),
                            "total_steps": to_int(properties.get("total_steps")),
                            "start_at": to_datetime(slot.get("startDate")),
                            "end_at": to_datetime(slot.get("endDate")),
                            "entered_at": to_datetime(data.get("created_at")),
                            "process_id": to_int(data.get("process_id")),
                            "received_at": received_at,
                        }
                    )

            elif event_type in NOTIFICATION_ACTION_OF_EVENT:
                properties = data.get("properties", {})
                notification_events.append(
                    {
                        "user_uuid": data["user_uuid"],
                        "action": NOTIFICATION_ACTION_OF_EVENT[event_type],
                        "content_id": properties.get("content_id"),
                        "content_type": properties.get("content_type"),
                        "mission_id": properties.get("mission_id") or properties.get("mission"),
                        "mission_process_id": to_int(data.get("mission_process_id")),
                        "rating": to_int(properties.get("rating")),
                        "created_at": to_datetime(data.get("created_at")),
                        "process_id": to_int(data.get("process_id")),
                        "received_at": received_at,
                    }
                )

            elif event_type == "health_habit_assessments":
                # One message per HHS questionnaire answered; repeated over the programme.
                health_assessments.append(
                    {
                        "user_uuid": data["user_uuid"],
                        "pillar_id": data.get("pillar_id"),
                        "score": data.get("score"),
                        "colour": data.get("colour"),
                        "trend": data.get("trend"),
                        "subpillars": {item.get("name"): item.get("score") for item in data.get("subpillars", [])},
                        "assessed_at": to_datetime(data.get("assessment_timestamp")),
                        "received_at": received_at,
                    }
                )

            elif event_type == "psychological_assessments":
                # One message per Step 3 questionnaire answered.
                psychological_assessments.append(psychological_assessment(data, received_at))

            else:
                skipped[event_type] += 1
        except (KeyError, TypeError, ValueError, AttributeError) as error:
            # A malformed message (a field missing or of the wrong type, as in the July 2026
            # messages of INT without mission_process_id) is left out and counted; it stays in
            # raw_events and the rest of the history is still parsed.
            failed[event_type] += 1
            log.debug("Message of type %s not parsed: %r", event_type, error)

    if skipped:
        log.info("Messages that are not events, by type: %s", dict(skipped))
    if failed:
        log.warning("Malformed messages left out, by type: %s", dict(failed))

    return {
        "users": list(users.values()),
        "episodes": list(episodes.values()),
        "marks": marks,
        "scheduled_tasks": scheduled_tasks,
        "notification_events": notification_events,
        "health_assessments": health_assessments,
        "psychological_assessments": psychological_assessments,
    }


def psychological_assessment(data, received_at):
    """The row of one Step 3 questionnaire answered.

    barriers_reported are the COM-B sub-domains of the field "barriers", computed by EUT; on INT
    (322 messages, 14 September to 29 October 2026) it holds only empty strings in Smoking and
    Alcohol. The answers are in answer_tasklog.data.answers, one per question, with the option ids
    in "value" and, in the other pillars, their tags in "data" ([{"answer": "physical_opportunity"}]).
    barriers_from_answers joins those tags and the tags of catalogue/step3_options.csv for every
    string of answer_tasklog that is the id of a tagged option, which is how Smoking and Alcohol
    give their barriers and the HAPA stage. A "hapa_stage" field in the message, if EUT adds one,
    takes precedence over the stage read from the answers. The subpillar is in the field
    "suppillar" on INT, as a list in Nutrition.
    """
    options = step3_options()
    log_data = data.get("answer_tasklog") or {}
    answers = [value for value in strings_in(log_data) if value in options]
    stages = [options[value]["hapa_stage"] for value in answers if options[value]["hapa_stage"]]
    from_answers = {name for value in answers for name in options[value]["subdomains"].split(";") if name}
    for answer in ((log_data.get("data") or {}).get("answers") or []) if isinstance(log_data, dict) else []:
        for tag in answer.get("data") or []:
            from_answers |= subdomain_names(tag.get("answer") if isinstance(tag, dict) else tag)
    subpillar = data.get("subpillar", data.get("suppillar"))
    return {
        "user_uuid": data["user_uuid"],
        "pillar_id": data.get("pillar_id"),
        "subpillar": ", ".join(subpillar) if isinstance(subpillar, list) else subpillar,
        "barriers_reported": sorted(subdomain_names(data.get("barriers") or [])),
        "barriers_from_answers": sorted(from_answers),
        "hapa_stage": hapa_stage_name(data.get("hapa_stage")) or (stages[-1] if stages else None),
        "answers": answers,
        "assessed_at": to_datetime(data.get("assessment_timestamp")),
        "received_at": received_at,
    }


@lru_cache(maxsize=1)
def step3_options():
    """{option id: row of catalogue/step3_options.csv} for the tagged options."""
    path = CATALOGUE_FOLDER / "step3_options.csv"
    if not path.exists():
        return {}
    with path.open(newline="", encoding="utf-8") as handle:
        return {row["option_id"]: row for row in csv.DictReader(handle) if row["tag"]}


def strings_in(value):
    """Every string in a nested structure of dicts and lists, keys included."""
    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        for key, item in value.items():
            yield from strings_in(key)
            yield from strings_in(item)
    elif isinstance(value, list):
        for item in value:
            yield from strings_in(item)


def read_body(body):
    """Split a body such as {"mission_selected": {...}} into ("mission_selected", {...}).

    Returns (None, None) for a body that is not a JSON object with exactly one key.
    """
    try:
        parsed = json.loads(body)
        if isinstance(parsed, str):
            # Some INT messages of July 2026 are JSON encoded twice.
            parsed = json.loads(parsed)
    except (json.JSONDecodeError, TypeError):
        return None, None
    if isinstance(parsed, dict) and len(parsed) == 1:
        event_type = next(iter(parsed))
        return event_type, parsed[event_type]
    return None, None


def get_user(users, user_uuid):
    """Return the row of a user, creating an empty one the first time the user appears."""
    if user_uuid not in users:
        users[user_uuid] = {
            "user_uuid": user_uuid,
            "personal_id": None,
            "language": None,
            "enrolment_date": None,
            "age": None,
            "recruitment_center": None,
            "attributes": None,
            "disabled_at": None,
        }
    return users[user_uuid]


def get_episode(episodes, mission_process_id):
    """Return the row of an episode, creating an empty one the first time it appears.

    An episode can appear first through its outcome, for example when it was
    selected before the service started listening.
    """
    if mission_process_id not in episodes:
        episodes[mission_process_id] = {
            "mission_process_id": mission_process_id,
            "user_uuid": None,
            "mission_id": None,
            "selected_at": None,
            "selection_received_at": None,
            "outcome": None,
            "resolved_at": None,
            "resolution_received_at": None,
        }
    return episodes[mission_process_id]


def to_datetime(text):
    """A naive datetime in UTC.

    The Mission messages carry no offset and are in UTC (measured on INT, 20 October 2026); the
    notification messages carry the participant's offset (+00:00, +02:00, +03:00 seen on INT)
    and are converted to UTC. history.py converts every time to the participant's local time.
    """
    if not text:
        return None
    moment = datetime.fromisoformat(text)
    return moment.astimezone(timezone.utc).replace(tzinfo=None) if moment.tzinfo is not None else moment


def to_int(value):
    """An integer field, or None when the message carries something else (an id seen as a
    UUID string on INT, for instance), so that one odd value does not stop the rebuild."""
    try:
        return int(value) if value is not None and not isinstance(value, bool) else None
    except (TypeError, ValueError):
        return None


def to_text(value):
    return None if value is None else str(value)


def to_date(text):
    return datetime.fromisoformat(text).date() if text else None
