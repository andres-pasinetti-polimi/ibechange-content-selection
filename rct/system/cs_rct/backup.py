"""Daily copy of the database (rct_logging_specification.md, 1.3).

The image of the service has no pg_dump, and Cronicle runs in another container that does not reach the
database, so the copy is made by the service itself: each table of the database is written with COPY as a
gzipped CSV file with a header, in one folder per copy, with a manifest that gives for each table the number
of rows and the SHA-256 of its file, the schema the tables were created with, the commit of the code and the
id of the last raw event. All tables are read in one transaction, so the files agree with each other.

To restore: create the tables with schema.sql of the copy, then COPY <table> FROM each file (CSV, HEADER).

After a copy succeeds, only the latest KEEP copies are kept (folders named cs_rct_<UTC time>Z; every other
file of the folder, such as the copies of code or the pg_dump files taken by hand, is left alone), and every
copy is given to the owner of the backup folder (the user polimi on the server), since the service runs as
root in its container and the copies would otherwise need sudo to be moved or removed.
"""

import gzip
import hashlib
import json
import os
import re
import shutil
from pathlib import Path

from cs_rct import database, run

BACKUP_FOLDER = Path(os.getenv("CS_BACKUP_FOLDER", Path(__file__).resolve().parent.parent / "backups"))
KEEP = 7  # copies kept (user, 5 November 2026)
COPY_NAME = re.compile(r"cs_rct_\d{8}T\d{6}Z")


def tables(db):
    return [row[0] for row in db.execute(
        "SELECT table_name FROM information_schema.tables "
        "WHERE table_schema = 'public' AND table_type = 'BASE TABLE' ORDER BY table_name").fetchall()]


def backup(db, now, folder=None):
    """Write one copy of every table under `folder`; return the manifest."""
    target = Path(folder or BACKUP_FOLDER) / f"cs_rct_{now:%Y%m%dT%H%M%SZ}"
    target.mkdir(parents=True, exist_ok=False)
    manifest = {"taken_at": now.isoformat(), "code_version": run.code_version(), "tables": {}}
    with db.transaction():
        db.execute("SET TRANSACTION ISOLATION LEVEL REPEATABLE READ")
        manifest["last_raw_event_id"] = db.execute("SELECT max(id) FROM raw_events").fetchone()[0]
        for table in tables(db):
            path = target / f"{table}.csv.gz"
            rows = 0
            with gzip.open(path, "wb") as file, db.cursor() as cursor:
                with cursor.copy(f"COPY {table} TO STDOUT WITH (FORMAT csv, HEADER)") as copy:
                    for chunk in copy:
                        file.write(chunk)
                        rows += bytes(chunk).count(b"\n")
            # The header is one line; a field with a line break inside quotes would add lines, so the
            # count is checked against the table.
            count = db.execute(f"SELECT count(*) FROM {table}").fetchone()[0]
            manifest["tables"][table] = {"rows": count, "lines": rows - 1,
                                         "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
    (target / "schema.sql").write_text(database.SCHEMA_FILE.read_text())
    (target / "manifest.json").write_text(json.dumps(manifest, indent=1, default=str))
    manifest["folder"] = str(target)
    manifest["removed"] = prune(target.parent)
    give_to_folder_owner(target.parent)
    return manifest


def copies(folder):
    """The copies in `folder`, oldest first (the names sort by time)."""
    return sorted(path for path in Path(folder).iterdir() if path.is_dir() and COPY_NAME.fullmatch(path.name))


def prune(folder, keep=KEEP):
    """Remove the copies beyond the latest `keep`; return the names removed."""
    old = copies(folder)[:-keep] if keep > 0 else []
    for path in old:
        shutil.rmtree(path)
    return [path.name for path in old]


def give_to_folder_owner(folder):
    """Give every copy, and every file in it, to the owner of `folder`."""
    owner = Path(folder).stat()
    for copy in copies(folder):
        for path in [copy, *copy.rglob("*")]:
            if path.stat().st_uid != owner.st_uid or path.stat().st_gid != owner.st_gid:
                os.chown(path, owner.st_uid, owner.st_gid)
