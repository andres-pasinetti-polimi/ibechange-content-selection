"""Small builders of derived-table rows for the tests."""

from datetime import date, datetime

from cs_rct import catalogue
from cs_rct.history import build_participants

USER = "u1"
START = datetime(2027, 1, 4, 8, 0)


def episode(process_id=1, mission_id="PM1", selected_at=START, outcome=None, resolved_at=None, user=USER):
    return {"mission_process_id": process_id, "user_uuid": user, "mission_id": mission_id,
            "selected_at": selected_at, "selection_received_at": selected_at, "outcome": outcome,
            "resolved_at": resolved_at, "resolution_received_at": resolved_at}


def mark(reference, entered, process_id=1, user=USER):
    return {"user_uuid": user, "mission_process_id": process_id, "mission_id": None, "step": None,
            "total_steps": None, "reference_time": reference, "entered_at": entered, "process_id": None,
            "received_at": entered}


def scheduled(start, entered=START, process_id=1, user=USER):
    return {"user_uuid": user, "mission_process_id": process_id, "mission_id": None, "step": None,
            "total_steps": None, "start_at": start, "end_at": start, "entered_at": entered, "process_id": None,
            "received_at": entered}


def notification(action, created, content_id="R1", rating=None, user=USER):
    return {"user_uuid": user, "action": action, "content_id": content_id, "content_type": "recommendation",
            "mission_id": None, "mission_process_id": 1, "rating": rating, "created_at": created,
            "process_id": None, "received_at": created}


def participant(episodes=(), marks=(), scheduled_tasks=(), notifications=(), enrolment=date(2027, 1, 4)):
    tables = {
        "users": [{"user_uuid": USER, "language": "es", "enrolment_date": enrolment, "age": 62,
                   "recruitment_center": "ICO", "disabled_at": None}],
        "episodes": list(episodes), "marks": list(marks), "scheduled_tasks": list(scheduled_tasks),
        "notification_events": list(notifications),
    }
    return build_participants(tables)[USER]


def load_catalogue(items_per_mission=7):
    cat = catalogue.load()
    catalogue.set_recommendations(cat, {f"R-{m}-{j}": {"missions": [m]}
                                        for m in cat.missions for j in range(items_per_mission)})
    return cat


def force_composition(monkeypatch, composition):
    """Every decision of the test takes `composition` when it is admissible: the conservative rule
    with no exploration and that composition as default (with the prior or no posterior, the
    starting probabilities put probability 1 on it)."""
    from cs_rct import design
    monkeypatch.setattr(design, "POLICY", "conservative")
    monkeypatch.setattr(design, "EXPLORE_SHARE", 0.0)
    monkeypatch.setattr(design, "DEFAULT_COMPOSITION", composition)
