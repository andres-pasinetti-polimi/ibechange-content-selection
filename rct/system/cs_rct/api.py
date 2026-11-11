"""REST interface of the service, called by the orchestrator (Cronicle scripts).

Endpoints follow rct/design/iBeChange-modules-architecture.docx (30 September 2026), with
the payloads of the RCT, in which CS decides per participant-day:

GET  /health
POST /recommendations      the recommendations catalogue, in the format of the EUT document,
                           {content_id: {"missions": [...], ...}}, or in the one VU returns
                           (16 October 2026), [{"rec_id": ..., "mission": [...], ...}]
POST /resources            the resources catalogue, in the same formats. With ?source=vu (the Cronicle
                           job), the catalogue of VU is stored only if it holds at least VU_MIN_COVERAGE
                           of the ids of the catalogue built from the ICO workbook
                           (catalogue/*_workbook.json); otherwise the current catalogue is kept (409).
                           A catalogue equal to the latest stored is not stored again.
POST /missions             the Missions from VU, [{"mission_id", "weekly_frequency"}], checked against
                           the catalogue the service reads (catalogue/missions.csv)
GET  /selected_contents    ?day=YYYY-MM-DD: the day's deliveries, one plan per participant, in the
                           format OMI reads (the pilot's); idempotent, plan_id = decision id
POST /recommendation_plans the placements OMI returns ({"recommendation_plans": [...]}), stored
                           and matched to our deliveries by plan_id and content_id
POST /run                  one run of the service (cs_rct/run.py), for Cronicle, which runs in
                           its own container and reaches the service over HTTP
POST /replay               the replay check (cs_rct/replay.py)
GET  /report               ?day=YYYY-MM-DD: the daily reconciliation report (cs_rct/report.py)
POST /backup               a copy of every table, gzipped CSV with a manifest (cs_rct/backup.py)

Run from the system/ folder:
    python -m cs_rct.api
"""

import json
import logging
from datetime import date, datetime, timezone
from zoneinfo import ZoneInfo

from flask import Flask, jsonify, request
from psycopg.types.json import Jsonb

from cs_rct import backup, catalogue, database, design, replay, report, run, store

log = logging.getLogger("api")
app = Flask(__name__)


def connect():
    db = database.connect()
    database.create_tables(db)
    return db


@app.get("/health")
def health():
    return jsonify({"status": "ok"}), 200


@app.post("/recommendations")
def post_recommendations():
    return store_catalogue("recommendations")


@app.post("/resources")
def post_resources():
    return store_catalogue("resources")


# The catalogues built from the ICO workbooks (scripts/build_*_catalogue.py), which a catalogue of VU
# must cover to replace them: VU then serves the RCT catalogue, not the pilot's or a partial one.
WORKBOOK_CATALOGUE = {"recommendations": catalogue.CATALOGUE_FOLDER / "recommendations_workbook.json",
                      "resources": catalogue.CATALOGUE_FOLDER / "resources_workbook.json"}
VU_MIN_COVERAGE = 0.95


def store_catalogue(kind):
    body = as_catalogue(request.get_json(silent=True))
    if not body:
        return jsonify({"error": "expected a non-empty JSON object {content_id: {...}} "
                                 "or a list [{\"rec_id\": ..., \"mission\": [...]}]"}), 400
    if request.args.get("source") == "vu" and WORKBOOK_CATALOGUE[kind].exists():
        workbook_ids = set(json.loads(WORKBOOK_CATALOGUE[kind].read_text()))
        coverage = len(workbook_ids & set(body)) / len(workbook_ids) if workbook_ids else 1.0
        if coverage < VU_MIN_COVERAGE:
            return jsonify({"stored": 0, "reason": "not the RCT catalogue: the current one is kept",
                            "items": len(body), "share_of_workbook_ids": round(coverage, 4)}), 409
    without_missions = sorted(content_id for content_id, item in body.items() if not item.get("missions"))
    # Missions named by the recommendations but absent from the catalogue the service reads,
    # which is what a catalogue of another study (the pilot's) produces.
    known = catalogue.load().missions
    unknown = sorted({m for item in body.values() for m in item.get("missions", []) if m not in known})
    with connect() as db:
        if store.latest_recommendations(db, kind) == body:
            return jsonify({"stored": 0, "unchanged": True, "items": len(body)}), 200
        store.save_recommendations(db, body, kind)
    return jsonify({"stored": len(body), "without_missions": without_missions,
                    "missions_not_in_catalogue": unknown}), 201


def as_catalogue(body):
    """{content_id: {"missions": [...], ...}} from any of the formats, or None: the object of the
    EUT document, a list [{"rec_id", "mission", ...}], or that list inside {"recommendations": [...]},
    which is what GET /internal/recommendations returns on INT (29 October 2026, for resources too)."""
    if isinstance(body, dict) and len(body) == 1 and isinstance(body.get("recommendations"), list):
        body = body["recommendations"]
    if isinstance(body, list):
        items = {}
        for item in body:
            if not isinstance(item, dict) or not item.get("rec_id"):
                return None
            items[item["rec_id"]] = {**item, "missions": item.get("mission", [])}
        return items or None
    if isinstance(body, dict) and body and all(isinstance(item, dict) for item in body.values()):
        return body
    return None


@app.post("/missions")
def post_missions():
    body = request.get_json(silent=True)
    if not isinstance(body, list):
        return jsonify({"error": "expected a JSON list of Missions"}), 400
    missions = catalogue.load().missions
    unknown, different = [], []
    for item in body:
        mission = missions.get(item.get("mission_id"))
        if mission is None:
            unknown.append(item.get("mission_id"))
        elif mission.states_count and item.get("weekly_frequency") != mission.count:
            different.append({"mission_id": mission.mission_id, "vu": item.get("weekly_frequency"),
                              "catalogue": mission.count})
    missing = sorted(set(missions) - {item.get("mission_id") for item in body})
    if unknown or different or missing:
        log.warning("Missions from VU differ from the catalogue: unknown %s, different %s, missing %s",
                    unknown, different, missing)
    return jsonify({"received": len(body), "unknown": unknown, "different_count": different,
                    "missing_from_vu": missing}), 201


@app.get("/selected_contents")
def get_selected_contents():
    """The current plan of each participant for the day, as in the shared architecture document
    (section "CS and OMI (RCT)", 16 October 2026):

    {user_uuid: {plan_id, created_at, mission_id, mission_process_id, mission_start_time,
                 mission_end_time, delivery_date, task_scheduled_for,
                 contents: [{id, type, latest_time, push_text_id}]}}

    A plan covers one participant, one day and one Mission. plan_id is "<decision id>.<revision>",
    so a revised plan has a new plan_id, and created_at is the time of the decision or of the
    latest revision; a plan whose deliveries were all cancelled has an empty contents list and
    null Mission fields. task_scheduled_for is the time the participant scheduled the day's task
    for, null if not scheduled; every reminder of the day is sent before it, and the hour_reminder
    close to it. latest_time is the latest time a reminder's own text allows
    (reminders/latest_times.py), null if none and on recommendations and resources. mission_end_time is null while the Mission is open, as in the pilot. type is
    recommendation, resource, hour_reminder or day_reminder."""
    try:
        day = date.fromisoformat(request.args.get("day") or date.today().isoformat())
    except ValueError:
        return jsonify({"error": "day must be YYYY-MM-DD"}), 400
    with connect() as db:
        rows = store.rows(db, "SELECT d.*, e.selected_at, (c.delivery_id IS NOT NULL) AS cancelled, "
                              "coalesce(dec.timezone, %s) AS timezone, "
                              "coalesce(r.revision, 0) AS plan_revision, "
                              "coalesce(r.created_at, dec.decided_at) AS plan_created_at "
                              "FROM deliveries d "
                              "JOIN decisions dec ON dec.id = d.decision_id "
                              "LEFT JOIN LATERAL (SELECT revision, created_at FROM plan_revisions "
                              "                   WHERE decision_id = d.decision_id "
                              "                   ORDER BY revision DESC LIMIT 1) r ON true "
                              "LEFT JOIN episodes e ON e.mission_process_id = d.mission_process_id "
                              "LEFT JOIN delivery_cancellations c ON c.delivery_id = d.id "
                              "WHERE d.decision_day = %s AND d.content_id IS NOT NULL "
                              "ORDER BY d.user_uuid, d.position", (design.DEFAULT_TIMEZONE, day))
    by_user = {}
    for row in rows:
        by_user.setdefault(row["user_uuid"], []).append(row)
    return jsonify({user: omi_plan(user_rows, day) for user, user_rows in by_user.items()}), 200


def omi_plan(rows, day):
    zone = rows[0]["timezone"]
    active = [row for row in rows if not row["cancelled"]]
    first = active[0] if active else None
    # The scheduled time of the day's task for the plan's Mission, read from every delivery of
    # that Mission, cancelled ones included, so that it stays known after reminders are withdrawn.
    of_mission = [row for row in rows if first and row["mission_process_id"] == first["mission_process_id"]]
    scheduled = max((row["task_scheduled_for"] for row in of_mission if row["task_scheduled_for"]), default=None)
    return {
        "plan_id": f"{rows[0]['decision_id']}.{rows[0]['plan_revision']}",
        "created_at": from_utc(rows[0]["plan_created_at"], zone),
        "mission_id": first["mission_id"] if first else None,
        "mission_process_id": first["mission_process_id"] if first else None,
        "mission_start_time": from_utc(first["selected_at"], zone) if first else None,
        "mission_end_time": None,
        "delivery_date": day.isoformat(),
        "task_scheduled_for": with_offset(scheduled, zone),
        "contents": [omi_content(row) for row in active],
    }


def omi_content(row):
    return {
        "id": row["content_id"],
        "type": omi_type(row),
        # The latest time the reminder's own text allows; null if none, and on content.
        "latest_time": with_offset(row["latest_time"], row["timezone"]),
        # The ID of the pop-up text EUT sends with a recommendation or resource; null on reminders.
        "push_text_id": row.get("push_text_id"),
    }


def with_offset(local, zone):
    """A naive local time in `zone` as ISO 8601 with the zone's offset, e.g. 2026-11-05T18:00:00+01:00."""
    return local.replace(tzinfo=ZoneInfo(zone)).isoformat() if local is not None else None


def from_utc(moment, zone):
    """A naive UTC time as ISO 8601 in the local time of `zone`, with its offset."""
    if moment is None:
        return None
    return moment.replace(tzinfo=timezone.utc).astimezone(ZoneInfo(zone)).isoformat()


def omi_type(row):
    return f"{row['reminder_kind']}_reminder" if row["format"] == "reminder" else row["format"]


@app.post("/recommendation_plans")
def post_recommendation_plans():
    """The placements OMI decided, as GET /recommendation_plans/ of omi-app returns them,
    {"recommendation_plans": [{user_id, plans: [{content_id, scheduled_for, plan_id, ...}]}]}.
    Each placement is matched to our delivery by plan_id ("<decision id>.<revision>") and
    content_id; a placement that matches nothing is stored with delivery_id null."""
    body = request.get_json(silent=True)
    if not isinstance(body, dict) or not isinstance(body.get("recommendation_plans"), list):
        return jsonify({"error": "expected {\"recommendation_plans\": [{\"user_id\": ..., \"plans\": [...]}]}"}), 400
    placements = [{**plan, "user_id": entry.get("user_id")}
                  for entry in body["recommendation_plans"] if isinstance(entry, dict)
                  for plan in entry.get("plans", []) if isinstance(plan, dict)]
    matched = 0
    with connect() as db, db.transaction():
        for plan in placements:
            delivery_id = None
            decision_id = str(plan.get("plan_id", "")).split(".")[0]
            if decision_id.isdigit():
                found = db.execute("SELECT id FROM deliveries WHERE decision_id = %s AND content_id = %s "
                                   "ORDER BY revision DESC, position",
                                   (int(decision_id), plan.get("content_id"))).fetchall()
                if found:
                    delivery_id = found[min(plan.get("occurrence_index") or 0, len(found) - 1)][0]
                    matched += 1
            scheduled_for = plan.get("scheduled_for")
            db.execute(
                "INSERT INTO placements (delivery_id, user_uuid, scheduled_for, status, body) "
                "VALUES (%s, %s, %s, %s, %s)",
                (delivery_id, plan.get("user_id"), to_utc(scheduled_for), plan.get("status"), Jsonb(plan)),
            )
    return jsonify({"stored": len(placements), "matched": matched}), 201


def to_utc(text):
    """A placement time from OMI, stored in UTC like every other time; one without an offset is
    kept as sent (its zone is to be agreed with TU/e)."""
    if not text:
        return None
    moment = datetime.fromisoformat(text)
    return moment.astimezone(timezone.utc).replace(tzinfo=None) if moment.tzinfo is not None else moment


@app.post("/run")
def post_run():
    with connect() as db:
        summary = run.run_job(db, design.utc_now())
    if summary is None:
        return jsonify({"status": "skipped", "reason": "another run holds the lock"}), 409
    return jsonify({"status": "ok", "summary": summary}), 200


@app.get("/report")
def get_report():
    try:
        day = date.fromisoformat(request.args.get("day") or design.utc_now().date().isoformat())
    except ValueError:
        return jsonify({"error": "day must be YYYY-MM-DD"}), 400
    with connect() as db:
        return jsonify(report.daily_report(db, day)), 200


@app.post("/replay")
def post_replay():
    with connect() as db:
        summary = replay.run_replay(db)
    return jsonify(summary), 200


@app.post("/backup")
def post_backup():
    with connect() as db:
        manifest = backup.backup(db, design.utc_now())
    return jsonify(manifest), 200


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    app.run(host="0.0.0.0", port=8000)


if __name__ == "__main__":
    main()
