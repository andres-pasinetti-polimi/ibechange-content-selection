import json
from datetime import date, datetime
from pathlib import Path

from cs_rct.parse_events import parse_events, read_body

SAMPLES_FILE = Path(__file__).parent.parent / "scripts" / "sample_events.json"


def sample_bodies():
    return [json.dumps(event) for event in json.loads(SAMPLES_FILE.read_text())]


def test_read_body_splits_type_and_data():
    assert read_body('{"mission_selected": {"mission": "NM1"}}') == ("mission_selected", {"mission": "NM1"})


def test_read_body_rejects_bodies_that_are_not_one_event():
    assert read_body("not json") == (None, None)
    assert read_body('{"a": 1, "b": 2}') == (None, None)


def test_sample_events_give_one_user_one_resolved_episode_and_one_mark():
    tables = parse_events(sample_bodies())

    [user] = tables["users"]
    assert user["user_uuid"] == "3a6f5cbc-3e91-4a7a-8b41-7ce6bbd3372f"
    assert user["enrolment_date"] == date(2026, 7, 22)
    assert user["age"] == 52

    [episode] = tables["episodes"]
    assert episode["mission_process_id"] == 209425
    assert episode["mission_id"] == "NM1"  # read from the "mission" field of mission_selected
    assert episode["outcome"] == "achieved"

    [mark] = tables["marks"]
    assert mark["reference_time"] == datetime(2026, 7, 23, 18, 15)
    assert mark["entered_at"] == datetime(2026, 7, 23, 21, 5)

    assert len(tables["scheduled_tasks"]) == 1
    assert [row["action"] for row in tables["notification_events"]] == ["sent", "opened"]


def test_a_message_delivered_twice_is_counted_once():
    bodies = sample_bodies()
    tables = parse_events(bodies + bodies)
    assert len(tables["marks"]) == 1
    assert len(tables["notification_events"]) == 2


def test_an_outcome_without_a_selection_still_gives_an_episode():
    body = json.dumps(
        {"mission_changed": {"user_uuid": "u1", "mission_id": "NM5", "created_at": "2026-07-22T08:00:00", "process_id": 1, "mission_process_id": 7}}
    )
    [episode] = parse_events([body])["episodes"]
    assert episode["outcome"] == "changed"
    assert episode["selected_at"] is None
    assert episode["mission_id"] == "NM5"


def test_bodies_that_are_not_events_are_skipped():
    tables = parse_events(["not json", '{"new_pillar": {"user_uuid": "u1", "pillars": "NO"}}'])
    assert all(rows == [] for rows in tables.values())


def test_health_assessment_gives_one_row_with_subpillars():
    body = json.dumps({"health_habit_assessments": {
        "pillar_id": "NO", "score": 34.9, "colour": "orange", "trend": 0,
        "subpillars": [{"score": 1, "colour": "green", "name": "fruit"}, {"score": 0.5, "colour": "orange", "name": "bakery"}],
        "assessment_timestamp": "2026-07-22T08:17:52.905150", "user_uuid": "u1"}})
    [row] = parse_events([body])["health_assessments"]
    assert row["pillar_id"] == "NO" and row["score"] == 34.9
    assert row["subpillars"] == {"fruit": 1, "bakery": 0.5}


def test_step3_assessment_reads_barriers_and_the_hapa_stage_of_the_answers():
    # 3386e8bd-... is "I have decided to quit and am preparing to do so" (Intender) and
    # dc5b6c52-... "I do not currently have access to professional support" (physical_opportunity),
    # in questionnaire-comb-hapa-smoking-2.json.
    body = json.dumps({"psychological_assessments": {
        "user_uuid": "u1", "pillar_id": "SO", "barriers": ["social_opportunity"], "subpillar": None,
        "assessment_timestamp": "2026-07-22T08:17:52.905150",
        "answer_tasklog": {"8c9f963b89a74ec89e4b7a554b6419b1": "3386e8bd-a804-4e56-aaec-5d716930b074",
                           "ee0b79f047124d1393e9c7afcc540326": ["dc5b6c52-164c-4eef-9172-501c77a1e452"]}}})
    [row] = parse_events([body])["psychological_assessments"]
    assert row["hapa_stage"] == "intender" and row["barriers_reported"] == ["social opportunity"]
    assert row["barriers_from_answers"] == ["physical opportunity"]


def test_step3_assessment_in_the_format_seen_on_int():
    # The body of a message received on INT on 29 October 2026, with a user_uuid added.
    body = json.dumps({"psychological_assessments": {
        "user_uuid": "u1", "pillar_id": "NO", "barriers": ["psychological_capability", ""],
        "suppillar": ["fruit", "vegetables"], "assessment_timestamp": "2026-10-29T12:06:54.105859",
        "answer_tasklog": {"data": {"answers": [{"code": "ddf5823da02a4005a69512278cb48e70",
                                                 "data": [{"answer": "reflecitve_motivation"}],
                                                 "questionData": None, "value": ["7903fa7e"]}]},
                           "draft": False, "id": 3660}}})
    [row] = parse_events([body])["psychological_assessments"]
    assert row["subpillar"] == "fruit, vegetables" and row["hapa_stage"] is None
    assert row["barriers_reported"] == ["psychological capability"]
    assert row["barriers_from_answers"] == ["reflective motivation"]


def test_step3_assessment_without_user_uuid_is_left_out():
    body = json.dumps({"psychological_assessments": {"pillar_id": "NO", "barriers": ["physical_opportunity"],
                                                     "subpillar": "fruits", "answer_tasklog": {}}})
    assert parse_events([body])["psychological_assessments"] == []


def test_messages_seen_on_int_do_not_stop_the_parsing():
    old_format = json.dumps(json.dumps({"mission_selected": {"user_uuid": "u1", "mission": "PM13",
                                                             "selection_timestamp": "2026-07-22T11:08:51",
                                                             "process_id": 1}}))
    null_type = json.dumps({"null": {"user_uuid": "u1", "mission_id": "NM102", "mission_process_id": 5}})
    finished = json.dumps({"mission_finished": {"user_uuid": "u1", "mission_id": "NM105",
                                                "created_at": "2026-10-01T14:04:28", "mission_process_id": 7}})
    scheduled = json.dumps({"mission_scheduled": {"user_uuid": "u1", "created_at": "2026-09-14T12:43:25",
                                                  "mission_process_id": 7, "properties": {
                                                      "mission_id": "NM105", "scheduled_for": [
                                                          {"startDate": "2026-09-17T10:00:00",
                                                           "endDate": "2026-09-17T10:20:00"}]}}})
    tables = parse_events([old_format, null_type, finished, scheduled])
    [episode] = tables["episodes"]
    assert episode["mission_process_id"] == 7 and episode["outcome"] == "finished"
    assert len(tables["scheduled_tasks"]) == 1


def test_a_timestamp_with_an_offset_becomes_utc():
    body = json.dumps({"notification_opened": {"user_uuid": "u1", "created_at": "2026-10-15T10:50:21+03:00",
                                               "properties": {"content_id": "R1"}}})
    [row] = parse_events([body])["notification_events"]
    # Messages without an offset are in UTC (measured on INT), so one with an offset is stored in UTC too.
    assert row["created_at"] == datetime(2026, 10, 15, 7, 50, 21)
