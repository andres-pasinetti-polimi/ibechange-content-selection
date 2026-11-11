"""Access to the PostgreSQL database of the service."""

import json
from pathlib import Path

import psycopg
from psycopg.types.json import Jsonb

from cs_rct import config

SCHEMA_FILE = Path(__file__).with_name("schema.sql")


def connect():
    """Open a connection in autocommit mode, so that every statement is committed at once."""
    return psycopg.connect(config.DATABASE_URL, autocommit=True)


def create_tables(db):
    """Create the tables that do not exist yet. Existing tables are left as they are."""
    db.execute(SCHEMA_FILE.read_text())


def save_raw_event(db, message_id, destination, headers, body, event_type):
    db.execute(
        """
        INSERT INTO raw_events (message_id, destination, headers, body, event_type)
        VALUES (%s, %s, %s, %s, %s)
        """,
        (message_id, destination, json.dumps(headers), body, event_type),
    )


def insert_rows(db, table, rows):
    """Insert rows given as dicts whose keys are the column names."""
    if not rows:
        return
    columns = list(rows[0].keys())
    placeholders = ", ".join(["%s"] * len(columns))
    query = f"INSERT INTO {table} ({', '.join(columns)}) VALUES ({placeholders})"
    values = [[as_database_value(row[column]) for column in columns] for row in rows]
    with db.cursor() as cursor:
        cursor.executemany(query, values)


def as_database_value(value):
    # Lists and dicts go to JSONB columns (the schema has no array column); everything else is
    # passed as it is. A list passed as it is becomes a Postgres array literal, which a JSONB column
    # accepts only when empty ('{}'): the Step 3 barriers stopped every run on 3 November 2026.
    return Jsonb(value) if isinstance(value, (list, dict)) else value
