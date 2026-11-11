"""Checks of the service on a server (INT or production), one line per check.

Without options the checks only read: the catalogue files, the database (tables
and counts), the Artemis broker (connection only: nothing is subscribed, so no
message is consumed) and the REST API. With --run, one cycle of the service is
run and the replay check after it, which writes decision rows: use it only where
the participants are test users or the trial has started.

Run from the system/ folder, with the environment of .env loaded:
    python -m scripts.server_check
    python -m scripts.server_check --api-url http://localhost:8000 --run
"""

import argparse
import sys
from datetime import datetime

import requests
import stomp

from cs_rct import catalogue, config, database, design, replay, run, store
from cs_rct.history import build_participants

results = []


def check(name, function):
    try:
        detail = function()
        results.append(True)
        print(f"PASS  {name}: {detail}")
    except Exception as error:
        results.append(False)
        print(f"FAIL  {name}: {error!r}")


def catalogue_files():
    cat = catalogue.load()
    pools = sum(1 for m in cat.missions.values() if m.reminder_pools)
    return (f"{len(cat.missions)} Missions ({pools} with a reminder pool), {len(cat.reminders)} reminders, "
            f"study {design.STUDY}, first delivery day "
            f"{design.FIRST_DELIVERY_DAY}")


def code_identity():
    """The commit the decisions are logged with; a run without it cannot be tied to its code."""
    version = run.code_version()
    if not version:
        raise AssertionError("no code version: write CODE_VERSION with scripts/write_code_version.sh before deploying")
    if version.endswith("-dirty"):
        raise AssertionError(f"{version}: the code was deployed with changes not committed")
    return version


def database_tables():
    with database.connect() as db:
        database.create_tables(db)
        counts = {table: db.execute(f"SELECT count(*) FROM {table}").fetchone()[0]
                  for table in ["raw_events", "decisions", "deliveries", "model_versions", "job_runs"]}
        by_type = db.execute("SELECT event_type, count(*) FROM raw_events GROUP BY 1 ORDER BY 2 DESC").fetchall()
        recommendations = store.latest_recommendations(db)
        resources = store.latest_recommendations(db, "resources")
    return (f"{counts}; events by type {dict(by_type)}; recommendations in catalogue {len(recommendations)}, "
            f"resources {len(resources)}")


def broker_connection():
    broker = stomp.Connection(host_and_ports=[(config.ARTEMIS_HOST, config.ARTEMIS_PORT)])
    if config.ARTEMIS_CA_CERT:
        broker.set_ssl(for_hosts=[(config.ARTEMIS_HOST, config.ARTEMIS_PORT)], ca_certs=config.ARTEMIS_CA_CERT)
    broker.connect(config.ARTEMIS_USER, config.ARTEMIS_PASSWORD, wait=True)
    broker.disconnect()
    return f"connected to {config.ARTEMIS_HOST}:{config.ARTEMIS_PORT} (queue {config.ARTEMIS_QUEUE} not read)"


def api(url):
    health = requests.get(f"{url}/health", timeout=10)
    health.raise_for_status()
    day = datetime.now().date().isoformat()
    selected = requests.get(f"{url}/selected_contents", params={"day": day}, timeout=30)
    selected.raise_for_status()
    return f"health {health.json()}; {len(selected.json())} participants with deliveries on {day}"


def one_cycle():
    with database.connect() as db:
        summary = run.run_job(db, design.utc_now())
    return summary


def replay_check():
    with database.connect() as db:
        decisions = store.load_decisions(db)
        versions = store.load_model_versions(db)
        participants = build_participants(store.load_tables(db))
    checked, mismatches = replay.check_probabilities(decisions, versions)
    versions_checked, version_mismatches = replay.check_versions(decisions, versions, participants, design.utc_now())
    if mismatches or version_mismatches:
        raise AssertionError(f"{len(mismatches)} probability and {len(version_mismatches)} version mismatches")
    return f"{checked} probabilities and {versions_checked} versions reproduced"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--api-url", default=None, help="base URL of the CS API; skipped if not given")
    parser.add_argument("--run", action="store_true", help="also run one cycle and the replay check")
    arguments = parser.parse_args()

    print(f"Database {config.DATABASE_URL.split('@')[-1]}, broker {config.ARTEMIS_HOST}:{config.ARTEMIS_PORT}")
    check("catalogue files", catalogue_files)
    check("code version", code_identity)
    check("database", database_tables)
    check("Artemis broker", broker_connection)
    if arguments.api_url:
        check("REST API", lambda: api(arguments.api_url.rstrip("/")))
    if arguments.run:
        check("one cycle", one_cycle)
        check("replay check", replay_check)
    sys.exit(0 if all(results) else 1)


if __name__ == "__main__":
    main()
