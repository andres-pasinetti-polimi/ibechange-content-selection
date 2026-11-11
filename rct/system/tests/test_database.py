"""End-to-end run on PostgreSQL: raw events -> hourly runs -> REST -> replay check.

Uses a database cs_rct_test on the server of docker-compose.yml, created and emptied
here, so the development database is left alone. Skipped when the server is down.
"""

import json
import math
from datetime import date, datetime, timedelta, timezone

import pytest

from cs_rct import design, replay, run, store
from cs_rct.history import build_participants
from tests.helpers import force_composition

USERS = ["user-a", "user-b"]
DAY1 = date(2027, 1, 4)


def at(day_offset, hour, minute=0):
    return datetime.combine(DAY1 + timedelta(days=day_offset), datetime.min.time()) + timedelta(hours=hour,
                                                                                             minutes=minute)


def store_event(db, event, received):
    """Insert a message as the consumer would, received at a time given in UTC."""
    body = json.dumps(event, default=str)
    db.execute("INSERT INTO raw_events (message_id, destination, headers, body, event_type, received_at) "
               "VALUES (%s, %s, %s, %s, %s, %s)",
               (None, "/queue/test", "{}", body, next(iter(event)), received.replace(tzinfo=timezone.utc)))


def events_of_day(db, day_offset):
    """Selections on day 0, a mark every evening, completion of user-a's Mission on day 3."""
    for index, user in enumerate(USERS):
        if day_offset == 0:
            store_event(db, {"new_user": {"uuid": user, "language": "es", "attributes": {
                "enrolmentDate": DAY1.isoformat(), "userAge": 55 + 10 * index, "recruitmentCenter": "ICO"}}},
                at(0, 7))
            store_event(db, {"mission_selected": {"user_uuid": user, "mission": "PM3",
                                                  "created_at": at(0, 8).isoformat(), "process_id": 1,
                                                  "mission_process_id": 100 + index}}, at(0, 8))
            store_event(db, {"health_habit_assessments": {"pillar_id": "PO", "score": 40.0 + index,
                                                          "subpillars": [], "user_uuid": user,
                                                          "assessment_timestamp": at(0, 7, 30).isoformat()}},
                        at(0, 7, 30))
        store_event(db, {"task_step_done": {"user_uuid": user, "created_at": at(day_offset, 21).isoformat(),
                                            "process_id": 2, "mission_process_id": 100 + index,
                                            "properties": {"mission_id": "PM3",
                                                           "finished_at": at(day_offset, 18).isoformat()}}},
                    at(day_offset, 21))
    if day_offset == 3:
        store_event(db, {"mission_accomplished": {"user_uuid": "user-a", "mission_id": "PM3",
                                                  "created_at": at(3, 21, 30).isoformat(), "process_id": 3,
                                                  "mission_process_id": 100}}, at(3, 21, 30))


def test_hourly_runs_rest_and_replay(db, monkeypatch):
    # The scenario below starts deliveries on the day of the selection (an open decision).
    monkeypatch.setattr(design, "FIRST_DELIVERY_DAY", "selection_day")
    # Position 1 holds a recommendation, so that the placement below has one to match.
    force_composition(monkeypatch, (1, 1))
    store.save_recommendations(db, {f"R{j}": {"missions": ["PM3"]} for j in range(7)})
    for day_offset in range(6):
        run.run_job(db, at(day_offset, 1))
        events_of_day(db, day_offset)  # selections arrive before the 09:00 run on day 0
        run.run_job(db, at(day_offset, 9))

    decisions = store.load_decisions(db)
    by_user_day = {(row["user_uuid"], row["decision_day"]): row for row in decisions}
    assert len(by_user_day) == len(decisions)
    # Day 0: selection at 08:00, decided at 09:00. PM3 asks for three days, so the third
    # mark (day 2) reaches the count and day 3 admits only (2,0).
    assert by_user_day[("user-a", DAY1)]["status"] == "decided"
    assert by_user_day[("user-b", DAY1)]["baseline"]["hhs_pillar"] == 0.41
    day3 = by_user_day[("user-a", DAY1 + timedelta(days=3))]
    assert day3["status"] == "no_choice" and day3["blocked_by"] == "count_reached"
    assert day3["composition_probabilities"] == {"2,0": 1.0}
    # After the completion on day 3, user-a has no open episode: closing rows from day 4.
    assert by_user_day[("user-a", DAY1 + timedelta(days=4))]["status"] == "no_open_episode"
    # The completed episode entered a model version, which later decisions read.
    versions = store.load_model_versions(db)
    assert any(100 in episodes for _, episodes in versions.values())
    # Every delivery has its draws, and the content draws give the logged content probability.
    deliveries = store.rows(db, "SELECT * FROM deliveries WHERE content_id IS NOT NULL")
    draws_of = {}
    for draw in store.rows(db, "SELECT * FROM draws ORDER BY delivery_id, seq"):
        draws_of.setdefault(draw["delivery_id"], []).append(draw)
    assert deliveries and all(d["id"] in draws_of for d in deliveries)
    for delivery in deliveries:
        content = [x for x in draws_of[delivery["id"]] if x["kind"] not in ("push_text", "order")]
        assert math.prod(x["options"][x["chosen"]] for x in content) == pytest.approx(delivery["content_probability"])
    for row in decisions:
        if row["status"] == "decided":
            assert row["policy_rule"] is not None
            [day_draw] = [x for x in draws_of.get(None, []) if x["decision_id"] == row["id"]]
            assert day_draw["kind"] == "composition" and day_draw["options"] == row["composition_probabilities"]
        assert {"missions.csv", "reminders.csv", "recommendations"} <= set(row["catalogue_version"] or {}) \
            or row["status"] != "decided"

    from cs_rct.api import app
    client = app.test_client()
    response = client.get("/selected_contents", query_string={"day": DAY1.isoformat()})
    assert response.status_code == 200
    selected = response.get_json()
    assert set(selected) == set(USERS)
    plan = selected["user-a"]
    first = next(c for c in plan["contents"] if c["type"] == "recommendation")
    assert first["id"].startswith("R")
    assert set(first) == {"id", "type", "latest_time", "push_text_id"} and first["latest_time"] is None
    assert first["push_text_id"] in range(1, 43)

    placement = {"recommendation_plans": [{"user_id": "user-a", "plans": [
        {"content_id": first["id"], "plan_id": plan["plan_id"], "mission_id": "PM3",
         "scheduled_for": at(0, 10).isoformat()}]}]}
    assert client.post("/recommendation_plans", json=placement).get_json() == {"stored": 1, "matched": 1}
    assert db.execute("SELECT count(*) FROM placements").fetchone()[0] == 1

    checked, mismatches = replay.check_probabilities(decisions, versions)
    assert checked == sum(row["status"] == "decided" for row in decisions) and mismatches == []
    participants = build_participants(store.load_tables(db))
    checked, mismatches = replay.check_versions(decisions, versions, participants, at(6, 1))
    assert checked == len(versions) and mismatches == []
