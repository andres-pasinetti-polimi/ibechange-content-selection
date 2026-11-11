"""Reading and writing the tables of the decision log (schema.sql, second part)."""

import json

from psycopg.rows import dict_row
from psycopg.types.json import Jsonb

from cs_rct import design
from cs_rct.cycle import ModelState
from cs_rct.model import Posterior

DERIVED_TABLES = ["users", "episodes", "marks", "scheduled_tasks", "notification_events", "health_assessments",
                  "psychological_assessments"]


def rows(db, query, params=()):
    with db.cursor(row_factory=dict_row) as cursor:
        cursor.execute(query, params)
        return cursor.fetchall()


def load_tables(db):
    return {table: rows(db, f"SELECT * FROM {table}") for table in DERIVED_TABLES}


def load_decisions(db):
    return rows(db, "SELECT * FROM decisions ORDER BY id")


def load_deliveries(db):
    """Every delivery, grouped by participant, with "cancelled" set by a revision."""
    by_user = {}
    for row in rows(db, "SELECT d.*, (c.delivery_id IS NOT NULL) AS cancelled FROM deliveries d "
                        "LEFT JOIN delivery_cancellations c ON c.delivery_id = d.id "
                        "ORDER BY d.decision_day, d.position"):
        by_user.setdefault(row["user_uuid"], []).append(row)
    return by_user


def decided_days(db, day):
    """Participants who already have a row for `day`."""
    return {row["user_uuid"] for row in rows(db, "SELECT user_uuid FROM decisions WHERE decision_day = %s", (day,))}


def save_decision(db, decision, delivery_rows, data_cutoff, code_version, catalogue_version=None):
    """Write a decision, its deliveries and their draws in one transaction; return the decision id."""
    day_draws = decision.get("draws", [])
    decision = {key: value for key, value in decision.items() if key != "draws"}
    decision = {**decision, "data_cutoff": data_cutoff, "code_version": code_version,
                "catalogue_version": catalogue_version}
    with db.transaction():
        decision_id = insert_returning_id(db, "decisions", decision)
        save_draws(db, decision_id, 0, None, day_draws)
        for delivery in delivery_rows:
            save_delivery(db, decision_id, 0, delivery)
    return decision_id


def save_delivery(db, decision_id, revision, delivery):
    """Write one delivery and the draws behind it."""
    row = {key: value for key, value in delivery.items() if key != "draws"}
    delivery_id = insert_returning_id(db, "deliveries", {**row, "decision_id": decision_id, "revision": revision})
    save_draws(db, decision_id, revision, delivery_id, delivery.get("draws", []))
    return delivery_id


def save_draws(db, decision_id, revision, delivery_id, draws):
    for seq, draw in enumerate(draws):
        db.execute("INSERT INTO draws (decision_id, revision, delivery_id, seq, kind, options, chosen, forced) "
                   "VALUES (%s, %s, %s, %s, %s, %s, %s, %s)",
                   (decision_id, revision, delivery_id, seq, draw["kind"], Jsonb(draw["options"]), draw["chosen"],
                    draw["forced"]))


def load_revisions(db):
    """{decision id: latest revision number}."""
    return {row["decision_id"]: row["revision"]
            for row in rows(db, "SELECT decision_id, max(revision) AS revision FROM plan_revisions GROUP BY 1")}


def save_revision(db, decision_id, revision, reason, seed, cancelled, new_rows, now):
    with db.transaction():
        db.execute("INSERT INTO plan_revisions (decision_id, revision, created_at, reason, rng_seed) "
                   "VALUES (%s, %s, %s, %s, %s)", (decision_id, revision, now, reason, seed))
        for delivery_id in cancelled:
            db.execute("INSERT INTO delivery_cancellations (delivery_id, revision, cancelled_at, reason) "
                       "VALUES (%s, %s, %s, %s)", (delivery_id, revision, now, reason))
        for delivery in new_rows:
            save_delivery(db, decision_id, revision, delivery)


def insert_returning_id(db, table, row):
    columns = list(row)
    query = (f"INSERT INTO {table} ({', '.join(columns)}) VALUES ({', '.join(['%s'] * len(columns))}) "
             f"RETURNING id")
    values = [as_value(row[column]) for column in columns]
    return db.execute(query, values).fetchone()[0]


def as_value(value):
    # Lists and dicts go to JSONB columns.
    return Jsonb(value) if isinstance(value, (list, dict)) else value


def latest_model(db):
    """The latest model version as a cycle.ModelState, or None."""
    found = rows(db, "SELECT * FROM model_versions ORDER BY version DESC LIMIT 1")
    if not found:
        return None
    row = found[0]
    # A version of the scheme of 14 October 2026 has no posterior the composition policy can use
    # (Posterior.from_json gives None); the next update writes a new version.
    return ModelState(row["version"], Posterior.from_json(row["posterior"]), row["episodes"], row["fingerprint"])


def load_model_versions(db):
    return {row["version"]: (Posterior.from_json(row["posterior"]), row["episodes"])
            for row in rows(db, "SELECT * FROM model_versions ORDER BY version")}


def save_model(db, state, parameters, data_cutoff, code_version, now, catalogue_version=None):
    db.execute(
        "INSERT INTO model_versions (version, created_at, episodes, fingerprint, posterior, parameters, "
        "data_cutoff, code_version, catalogue_version) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)",
        (state.version, now, Jsonb(state.episodes), state.fingerprint, Jsonb(state.posterior.to_json()),
         Jsonb(parameters), data_cutoff, code_version, Jsonb(catalogue_version) if catalogue_version else None),
    )


def load_groups(db):
    return rows(db, "SELECT * FROM participant_groups ORDER BY assigned_at, user_uuid")


def save_group(db, user_uuid, group_name, stratum, now, seed):
    db.execute("INSERT INTO participant_groups (user_uuid, group_name, stratum, assigned_at, rng_seed) "
               "VALUES (%s, %s, %s, %s, %s)", (user_uuid, group_name, stratum, now, seed))


CATALOGUE_TABLE = {"recommendations": "recommendation_catalogue", "resources": "resource_catalogue"}


def latest_recommendations(db, kind="recommendations"):
    found = rows(db, f"SELECT body FROM {CATALOGUE_TABLE[kind]} ORDER BY id DESC LIMIT 1")
    return found[0]["body"] if found else {}


def save_recommendations(db, body, kind="recommendations"):
    db.execute(f"INSERT INTO {CATALOGUE_TABLE[kind]} (body) VALUES (%s)", (Jsonb(body),))


def start_job(db, job, now):
    return db.execute("INSERT INTO job_runs (job, started_at) VALUES (%s, %s) RETURNING id", (job, now)).fetchone()[0]


def finish_job(db, run_id, status, summary=None, error=None):
    db.execute("UPDATE job_runs SET finished_at = %s, status = %s, summary = %s, error = %s WHERE id = %s",
               (design.utc_now(), status, json.dumps(summary, default=str) if summary else None, error, run_id))
