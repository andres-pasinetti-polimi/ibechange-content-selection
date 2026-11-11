"""The scheduled job of the service: rebuild the derived tables, then run one cycle.

Run every hour by Cronicle (scripts/cronicle/cs_hourly.sh). One run at a time is
enforced with a PostgreSQL advisory lock, so a run that overlaps a slow previous
one exits without doing anything. Every run is recorded in job_runs.

Run from the system/ folder:
    python -m cs_rct.run
    python -m cs_rct.run --now 2026-11-03T01:00   # for tests on a copy of the database
"""

import argparse
import logging
import os
import secrets
import subprocess
from datetime import datetime, timedelta

from cs_rct import catalogue, cycle, database, design, store
from cs_rct.history import build_participants
from cs_rct.rebuild_tables import rebuild

log = logging.getLogger("run")

LOCK_ID = 20261201  # any constant shared by all runs


CODE_VERSION_FILE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "CODE_VERSION")


def code_version():
    """The commit the service runs from: CS_CODE_VERSION, else the file CODE_VERSION written at deploy
    (scripts/write_code_version.sh; "-dirty" when system/ had changes not committed), else git."""
    if os.getenv("CS_CODE_VERSION"):
        return os.getenv("CS_CODE_VERSION")
    if os.path.exists(CODE_VERSION_FILE):
        with open(CODE_VERSION_FILE) as file:
            return file.read().strip() or None
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True,
                              cwd=os.path.dirname(__file__), check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def run(db, now):
    data_cutoff = rebuild(db)
    participants = build_participants(store.load_tables(db))
    cat = catalogue.load(store.latest_recommendations(db), store.latest_recommendations(db, "resources"))
    group_rows = store.load_groups(db)
    group_of = {row["user_uuid"]: row["group_name"] for row in group_rows}
    groups_by_stratum = {}
    for row in group_rows:
        groups_by_stratum.setdefault(row["stratum"], []).append(row["group_name"])
    yesterday = now.date() - timedelta(days=1)
    ran_yesterday = bool(db.execute(
        "SELECT 1 FROM job_runs WHERE job = 'cs_run' AND status = 'ok' AND started_at::date = %s LIMIT 1",
        (yesterday,)).fetchone())

    result = cycle.run_cycle(
        now=now,
        participants=participants,
        catalogue=cat,
        decision_rows=store.load_decisions(db),
        revisions_of=store.load_revisions(db),
        deliveries_by_user=store.load_deliveries(db),
        latest_model=store.latest_model(db),
        group_of=group_of,
        groups_by_stratum=groups_by_stratum,
        draw_seed=lambda: secrets.randbits(63),
        ran_yesterday=ran_yesterday,
    )

    version = code_version()
    if result.new_model is not None:
        store.save_model(db, result.new_model, design.parameters(), data_cutoff, version, now, cat.version)
    for user_uuid, group_name, stratum, seed in result.new_groups:
        store.save_group(db, user_uuid, group_name, stratum, now, seed)
    for decision, deliveries in result.decisions:
        store.save_decision(db, decision, deliveries, data_cutoff, version, cat.version)
    for decision_id, revision, reason, seed, cancelled, new_rows in result.revisions:
        store.save_revision(db, decision_id, revision, reason, seed, cancelled, new_rows, now)
    return {**result.summary, "data_cutoff": data_cutoff}


def run_job(db, now):
    """One run under the lock, recorded in job_runs; returns the summary, or None if another run holds the lock."""
    if not db.execute("SELECT pg_try_advisory_lock(%s)", (LOCK_ID,)).fetchone()[0]:
        log.warning("Another run holds the lock; exiting")
        return None
    run_id = store.start_job(db, "cs_run", now)
    try:
        summary = run(db, now)
    except Exception as error:
        store.finish_job(db, run_id, "failed", error=repr(error))
        raise
    finally:
        db.execute("SELECT pg_advisory_unlock(%s)", (LOCK_ID,))
    store.finish_job(db, run_id, "ok", summary)
    return summary


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    parser = argparse.ArgumentParser()
    parser.add_argument("--now", help="the moment of the run in UTC, ISO format; default the current time")
    arguments = parser.parse_args()
    now = datetime.fromisoformat(arguments.now) if arguments.now else design.utc_now()

    db = database.connect()
    database.create_tables(db)
    summary = run_job(db, now)
    log.info("Run finished: %s", summary)


if __name__ == "__main__":
    main()
