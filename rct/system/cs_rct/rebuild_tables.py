"""Rebuild the derived tables (users, episodes, marks, ...) from raw_events.

Every run empties the derived tables and fills them again from the full history
of messages, inside one transaction, so that readers never see a half-built
state. A change to the parsing rules therefore takes effect on the whole
history at the next run.

Run from the system/ folder:
    python -m cs_rct.rebuild_tables
"""

import logging
from datetime import timezone

from cs_rct import database
from cs_rct.parse_events import parse_events

log = logging.getLogger("rebuild_tables")


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")

    db = database.connect()
    database.create_tables(db)

    rebuild(db)


def rebuild(db):
    """Rebuild the derived tables; return the id of the last raw event read (the data cut-off)."""
    rows = db.execute("SELECT id, body, received_at FROM raw_events ORDER BY id").fetchall()
    # Every time is stored in UTC, as the messages carry it (parse_events.to_datetime).
    tables = parse_events([(body, to_naive_utc(received_at)) for _, body, received_at in rows])

    with db.transaction():
        for table, table_rows in tables.items():
            db.execute(f"DELETE FROM {table}")
            database.insert_rows(db, table, table_rows)
            log.info("%s: %d rows", table, len(table_rows))
    return rows[-1][0] if rows else 0


def to_naive_utc(moment):
    return moment.astimezone(timezone.utc).replace(tzinfo=None) if moment is not None else None


if __name__ == "__main__":
    main()
