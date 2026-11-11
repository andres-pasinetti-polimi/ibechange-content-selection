"""Receive the EUT events from the POLIMI queue on Artemis and store them.

Each message is written to the raw_events table and acknowledged to the broker
only after the write has succeeded. If a write fails, the consumer disconnects
and connects again after a pause; the broker then delivers again every message
that was not acknowledged, so no event is lost while the database is down.

Turning the stored events into Missions, marks and outcomes is a separate step
that reads raw_events. When a mission_selected message is stored, the consumer also
starts a run of the service (cs_rct/run.py) in a background thread, so that a
participant who selects a Mission receives that day's plan within minutes rather than
at the next hourly run, if CS_RUN_ON_SELECTION=1 (off by default).

Run from the system/ folder:
    python -m cs_rct.artemis_consumer
"""

import logging
import os
import threading
import time

import psycopg
import stomp
from stomp.exception import ConnectFailedException

from cs_rct import config, database, design, run
from cs_rct.parse_events import read_body

log = logging.getLogger("artemis_consumer")

SUBSCRIPTION_ID = "cs-rct"
SECONDS_BEFORE_RETRY = 10
# Off by default: plans reach OMI with the hourly exchange (Cronicle), so a run right after the
# selection would not make them reach OMI earlier.
RUN_ON_SELECTION = os.getenv("CS_RUN_ON_SELECTION", "0") == "1"
SECONDS_BEFORE_RUN_RETRY = 30


class SelectionRunner:
    """Runs the service once after one or more Mission selections, in a background thread.

    Selections that arrive while a run is pending or running lead to one more run, not one each.
    If the hourly run holds the lock, the run is tried again after SECONDS_BEFORE_RUN_RETRY.
    """

    def __init__(self):
        self.requested = threading.Event()
        threading.Thread(target=self.loop, daemon=True, name="selection-runner").start()

    def request(self):
        self.requested.set()

    def loop(self):
        while True:
            self.requested.wait()
            self.requested.clear()
            try:
                with database.connect() as db:
                    summary = run.run_job(db, design.utc_now())
                if summary is None:
                    time.sleep(SECONDS_BEFORE_RUN_RETRY)
                    self.requested.set()
                else:
                    log.info("Run after a Mission selection: %s", summary.get("rows_by_status"))
            except Exception:
                log.exception("Run after a Mission selection failed; the hourly run will cover it")


class StoreListener(stomp.ConnectionListener):
    """Stores each message and acknowledges it once it is stored."""

    def __init__(self, broker, db, runner=None):
        self.broker = broker
        self.db = db
        self.runner = runner
        # Set when a message could not be stored; the main loop then reconnects.
        self.store_failed = False

    def on_message(self, frame):
        message_id = frame.headers.get("message-id")
        event_type = read_body(frame.body)[0]
        try:
            database.save_raw_event(
                self.db,
                message_id=message_id,
                destination=frame.headers.get("destination"),
                headers=dict(frame.headers),
                body=frame.body,
                event_type=event_type,
            )
        except Exception:
            # Not acknowledged: the broker will deliver it again after we reconnect.
            log.exception("Could not store message %s", message_id)
            self.store_failed = True
            return
        self.broker.ack(message_id, SUBSCRIPTION_ID)
        log.info("Stored message %s", message_id)
        if event_type == "mission_selected" and self.runner is not None:
            self.runner.request()

    def on_connected(self, frame):
        log.info("Connected to the broker")

    def on_disconnected(self):
        log.warning("Disconnected from the broker")

    def on_error(self, frame):
        log.error("Broker error: %s", frame.body)


def open_broker_connection():
    broker = stomp.Connection(
        host_and_ports=[(config.ARTEMIS_HOST, config.ARTEMIS_PORT)],
        heartbeats=(10000, 10000),
    )
    if config.ARTEMIS_CA_CERT:
        broker.set_ssl(
            for_hosts=[(config.ARTEMIS_HOST, config.ARTEMIS_PORT)],
            ca_certs=config.ARTEMIS_CA_CERT,
        )
    return broker


def connect_and_subscribe(broker):
    broker.connect(config.ARTEMIS_USER, config.ARTEMIS_PASSWORD, wait=True)
    # "client-individual": each message stays on the queue until we acknowledge it.
    broker.subscribe(
        destination=f"/queue/{config.ARTEMIS_QUEUE}",
        id=SUBSCRIPTION_ID,
        ack="client-individual",
    )
    log.info("Subscribed to /queue/%s", config.ARTEMIS_QUEUE)


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")

    db = database.connect()
    database.create_tables(db)

    broker = open_broker_connection()
    listener = StoreListener(broker, db, SelectionRunner() if RUN_ON_SELECTION else None)
    broker.set_listener("store", listener)

    while True:
        if listener.store_failed:
            log.warning("Reconnecting to the broker and the database after a failed write")
            broker.disconnect()
            time.sleep(SECONDS_BEFORE_RETRY)
            try:
                listener.db = database.connect()
            except psycopg.Error as error:
                log.error("Database still unavailable: %s", error)
                continue
            listener.store_failed = False

        if not broker.is_connected():
            try:
                connect_and_subscribe(broker)
            except ConnectFailedException:
                log.error("Could not connect to %s:%s", config.ARTEMIS_HOST, config.ARTEMIS_PORT)
                time.sleep(SECONDS_BEFORE_RETRY)
                continue

        time.sleep(1)


if __name__ == "__main__":
    main()
