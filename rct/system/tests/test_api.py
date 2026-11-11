import json
"""Every endpoint of cs_rct/api.py on PostgreSQL (database cs_rct_test, as in test_database.py)."""

from cs_rct import design, run, store
from tests.helpers import force_composition
from tests.test_database import DAY1, at, events_of_day, store_event


def iso(moment):
    """An API time: in these tests every participant's zone is UTC (conftest.py)."""
    return moment.isoformat() + "+00:00"


def client():
    from cs_rct.api import app
    return app.test_client()


def test_health(db):
    response = client().get("/health")
    assert response.status_code == 200 and response.get_json() == {"status": "ok"}


def test_recommendations_are_stored_and_read_by_the_next_run(db, monkeypatch):
    monkeypatch.setattr(design, "FIRST_DELIVERY_DAY", "selection_day")
    force_composition(monkeypatch, (1, 1))
    api = client()
    assert api.post("/recommendations", json=[]).status_code == 400
    response = api.post("/recommendations", json={"R1": {"missions": ["PM3"]}, "R2": {"missions": []}})
    assert response.status_code == 201
    assert response.get_json() == {"stored": 2, "without_missions": ["R2"], "missions_not_in_catalogue": []}
    assert store.latest_recommendations(db) == {"R1": {"missions": ["PM3"]}, "R2": {"missions": []}}
    # The format VU returns: a list with rec_id and mission.
    listed = api.post("/recommendations", json=[{"rec_id": "R1", "mission": ["PM3"]},
                                                {"rec_id": "R3", "mission": ["NM92"]}]).get_json()
    assert listed["stored"] == 2 and listed["missions_not_in_catalogue"] == ["NM92"]
    assert store.latest_recommendations(db)["R1"]["missions"] == ["PM3"]
    # The format GET /internal/recommendations returns on INT (29 October 2026).
    wrapped = api.post("/recommendations", json={"recommendations": [
        {"rec_id": "R1", "mission": ["PM3"], "comb": ["Psychological capability"], "hapa": [],
         "type": "recommendation"}]}).get_json()
    assert wrapped["stored"] == 1 and store.latest_recommendations(db)["R1"]["comb"] == ["Psychological capability"]
    api.post("/recommendations", json={"R1": {"missions": ["PM3"]}})

    events_of_day(db, 0)
    run.run_job(db, at(0, 9))
    selected = api.get("/selected_contents", query_string={"day": DAY1.isoformat()}).get_json()
    recommendations = [c for c in selected["user-a"]["contents"] if c["type"] == "recommendation"]
    assert recommendations[0]["id"] == "R1"
    assert {k: v for k, v in recommendations[0].items() if k != "push_text_id"} == \
        {"id": "R1", "type": "recommendation", "latest_time": None}
    assert recommendations[0]["push_text_id"] in range(1, 43)
    assert selected["user-a"]["delivery_date"] == DAY1.isoformat()


def test_missions_are_compared_with_the_catalogue(db):
    api = client()
    assert api.post("/missions", json={"PM3": 3}).status_code == 400
    body = api.post("/missions", json=[{"mission_id": "PM3", "weekly_frequency": 3},
                                       {"mission_id": "PM2", "weekly_frequency": 5},
                                       {"mission_id": "XX1", "weekly_frequency": 1}]).get_json()
    assert body["unknown"] == ["XX1"]
    assert body["different_count"] == [{"mission_id": "PM2", "vu": 5, "catalogue": 2}]


def test_selected_contents_and_plans(db, monkeypatch):
    monkeypatch.setattr(design, "FIRST_DELIVERY_DAY", "selection_day")
    api = client()
    assert api.get("/selected_contents", query_string={"day": "16-10-2026"}).status_code == 400
    assert api.get("/selected_contents", query_string={"day": DAY1.isoformat()}).get_json() == {}

    events_of_day(db, 0)
    run.run_job(db, at(0, 9))
    selected = api.get("/selected_contents", query_string={"day": DAY1.isoformat()}).get_json()
    assert set(selected) == {"user-a", "user-b"}
    plan = selected["user-a"]
    assert plan["plan_id"].endswith(".0") and plan["mission_start_time"] == iso(at(0, 8))
    assert plan["mission_end_time"] is None
    # Without recommendations in the catalogue only reminders are handed out.
    first = plan["contents"][0]
    assert first["type"] == "day_reminder" and plan["delivery_date"] == DAY1.isoformat()
    assert plan["mission_id"] == "PM3" and plan["mission_process_id"] == 100
    assert plan["task_scheduled_for"] is None and all(set(c) == {"id", "type", "latest_time", "push_text_id"} for c in plan["contents"])
    assert all((c["push_text_id"] is None) == c["type"].endswith("reminder") for c in plan["contents"])
    # The same call twice returns the same plans.
    assert api.get("/selected_contents", query_string={"day": DAY1.isoformat()}).get_json() == selected

    assert api.post("/recommendation_plans", json=[]).status_code == 400
    plans = {"recommendation_plans": [{"user_id": "user-a", "plans": [
        {"content_id": first["id"], "plan_id": plan["plan_id"], "mission_id": "PM3",
         "scheduled_for": at(0, 11).isoformat(), "occurrence_index": 0},
        {"content_id": "unknown", "plan_id": "999999", "mission_id": "PM3", "scheduled_for": at(0, 12).isoformat()},
    ]}]}
    assert api.post("/recommendation_plans", json=plans).get_json() == {"stored": 2, "matched": 1}
    delivery_id = db.execute("SELECT id FROM deliveries WHERE decision_id = %s AND content_id = %s",
                             (int(plan["plan_id"].split(".")[0]), first["id"])).fetchone()[0]
    rows = db.execute("SELECT delivery_id, user_uuid, scheduled_for FROM placements ORDER BY id").fetchall()
    assert rows == [(delivery_id, "user-a", at(0, 11)), (None, "user-a", at(0, 12))]


def test_run_and_replay(db, monkeypatch):
    monkeypatch.setattr(design, "FIRST_DELIVERY_DAY", "selection_day")
    api = client()
    events_of_day(db, 0)
    response = api.post("/run")
    assert response.status_code == 200 and response.get_json()["status"] == "ok"
    replay = api.post("/replay").get_json()
    assert replay["probability_mismatches"] == 0 and replay["version_mismatches"] == 0


def test_a_mark_during_the_day_revises_the_plan(db, monkeypatch):
    monkeypatch.setattr(design, "FIRST_DELIVERY_DAY", "selection_day")
    # One recommendation and one reminder.
    force_composition(monkeypatch, (1, 1))
    api = client()
    store.save_recommendations(db, {"R1": {"missions": ["PM3"]}})
    events_of_day(db, 0)
    run.run_job(db, at(0, 9))
    before = api.get("/selected_contents", query_string={"day": DAY1.isoformat()}).get_json()["user-a"]
    assert before["plan_id"].endswith(".0") and any(c["type"] == "day_reminder" for c in before["contents"])

    # user-a marks the day's task at 10:00; the 11:00 run revises the plan.
    store_event(db, {"task_step_done": {"user_uuid": "user-a", "created_at": at(0, 10).isoformat(),
                                        "mission_process_id": 100,
                                        "properties": {"mission_id": "PM3", "finished_at": at(0, 9, 30).isoformat()}}},
                at(0, 10))
    summary = run.run_job(db, at(0, 11))
    assert summary["revisions"] == ["task_marked"]
    after = api.get("/selected_contents", query_string={"day": DAY1.isoformat()}).get_json()["user-a"]
    assert after["plan_id"] == before["plan_id"].replace(".0", ".1")
    assert before["created_at"] == iso(at(0, 9)) and after["created_at"] == iso(at(0, 11))
    assert [c["type"] for c in after["contents"]] == ["recommendation"]
    # A second run changes nothing.
    assert run.run_job(db, at(0, 12))["revisions"] == []


def test_a_scheduled_task_gives_an_hour_reminder_with_its_deadline(db, monkeypatch):
    monkeypatch.setattr(design, "FIRST_DELIVERY_DAY", "selection_day")
    events_of_day(db, 0)
    store_event(db, {"task_step_scheduled": {"user_uuid": "user-a", "created_at": at(0, 8, 30).isoformat(),
                                             "mission_process_id": 100, "properties": {
                                                 "mission_id": "PM3", "scheduled_for": [
                                                     {"startDate": at(0, 18).isoformat(),
                                                      "endDate": at(0, 18, 20).isoformat()}]}}}, at(0, 8, 30))
    run.run_job(db, at(0, 9))
    plan = client().get("/selected_contents", query_string={"day": DAY1.isoformat()}).get_json()["user-a"]
    hour = [c for c in plan["contents"] if c["type"] == "hour_reminder"]
    assert len(hour) == 1 and hour[0]["id"].startswith("PRh")
    # The scheduled time is a fact of the day, in the plan; latest_time carries only the limit of
    # a reminder's own text, which hour reminders and content do not have.
    assert plan["task_scheduled_for"] == iso(at(0, 18))
    assert hour[0]["latest_time"] is None
    from cs_rct import catalogue
    texts = catalogue.load().reminders
    for c in plan["contents"]:
        if c["type"] == "day_reminder" and texts[c["id"]].latest_time is not None:
            assert c["latest_time"] == f"{DAY1.isoformat()}T{texts[c['id']].latest_time.isoformat()}+00:00"
        else:
            assert c["latest_time"] is None


def test_resources_are_stored_and_sent_with_their_type(db, monkeypatch):
    monkeypatch.setattr(design, "FIRST_DELIVERY_DAY", "selection_day")
    # A composition with a recommendation position, which the resource takes (without it the test
    # failed whenever the drawn composition had R = 0).
    force_composition(monkeypatch, (1, 1))
    api = client()
    listed = api.post("/resources", json=[{"rec_id": "PRs1", "mission": ["PM3"], "type": "resource"}]).get_json()
    assert listed == {"stored": 1, "without_missions": [], "missions_not_in_catalogue": []}
    events_of_day(db, 0)
    run.run_job(db, at(0, 9))
    contents = api.get("/selected_contents", query_string={"day": DAY1.isoformat()}).get_json()["user-a"]["contents"]
    [resource] = [c for c in contents if c["type"] == "resource"]
    assert {k: v for k, v in resource.items() if k != "push_text_id"} == \
        {"id": "PRs1", "type": "resource", "latest_time": None}
    assert resource["push_text_id"] in range(1, 43)


def test_the_daily_report_counts_plans_placements_and_sends(db, monkeypatch):
    monkeypatch.setattr(design, "FIRST_DELIVERY_DAY", "selection_day")
    api = client()
    events_of_day(db, 0)
    run.run_job(db, at(0, 9))
    plan = api.get("/selected_contents", query_string={"day": DAY1.isoformat()}).get_json()["user-a"]
    first = plan["contents"][0]
    api.post("/recommendation_plans", json={"recommendation_plans": [{"user_id": "user-a", "plans": [
        {"content_id": first["id"], "plan_id": plan["plan_id"], "mission_id": "PM3",
         "scheduled_for": at(0, 11).isoformat()}]}]})
    store_event(db, {"notification_sent": {"user_uuid": "user-a", "created_at": at(0, 11).isoformat(),
                                           "properties": {"content_id": first["id"]}}}, at(0, 11))
    run.run_job(db, at(0, 12))
    report = api.get("/report", query_string={"day": DAY1.isoformat()}).get_json()
    [centre] = report["by_centre"]
    assert centre["centre"] == "ICO" and centre["participants_with_a_plan"] == 2
    assert centre["placed_by_omi"] == 1 and centre["sent_by_eut"] == 1
    assert report["operations"]["failed_runs_last_24h"] == []


def test_a_catalogue_of_vu_replaces_ours_only_if_it_is_the_rct_one(db, monkeypatch, tmp_path):
    from cs_rct import api as api_module
    workbook = tmp_path / "recommendations_workbook.json"
    workbook.write_text(json.dumps({f"SRc{j}a": {"missions": ["SM1"]} for j in range(20)}))
    monkeypatch.setitem(api_module.WORKBOOK_CATALOGUE, "recommendations", workbook)
    api = client()
    ours = json.loads(workbook.read_text())
    assert api.post("/recommendations", json=ours).status_code == 201
    # The same catalogue again: not stored twice.
    again = api.post("/recommendations", json=ours)
    assert again.status_code == 200 and again.get_json()["unchanged"]
    # A catalogue of VU with 18 of the 20 ids (90%): refused, ours kept.
    pilot_like = [{"rec_id": f"SRc{j}a", "mission": ["SM1"]} for j in range(18)]
    refused = api.post("/recommendations?source=vu", json={"recommendations": pilot_like})
    assert refused.status_code == 409
    assert store.latest_recommendations(db) == ours
    # All 20 ids and one more: stored, and read as the latest.
    rct = [{"rec_id": f"SRc{j}a", "mission": ["SM2"]} for j in range(21)]
    assert api.post("/recommendations?source=vu", json={"recommendations": rct}).status_code == 201
    assert set(store.latest_recommendations(db)) == {f"SRc{j}a" for j in range(21)}
    assert db.execute("SELECT count(*) FROM recommendation_catalogue").fetchone()[0] == 2
