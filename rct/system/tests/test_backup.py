"""The daily copy of the database (cs_rct/backup.py): every table, restorable, with a manifest."""

import gzip
import hashlib
import json
from datetime import datetime

from cs_rct import backup, database


def test_backup_copies_every_table_and_restores(db, tmp_path):
    database.save_raw_event(db, "m1", "/queue/x", {"a": 1}, '{"new_user": {"user_uuid": "u1"}}', "new_user")
    database.save_raw_event(db, "m2", "/queue/x", {}, 'line one\nline two, "quoted"', None)
    manifest = backup.backup(db, datetime(2026, 11, 5, 3, 30), tmp_path)
    folder = tmp_path / "cs_rct_20261105T033000Z"
    assert manifest["last_raw_event_id"] == 2
    assert set(manifest["tables"]) == set(backup.tables(db))
    for table, entry in manifest["tables"].items():
        path = folder / f"{table}.csv.gz"
        assert hashlib.sha256(path.read_bytes()).hexdigest() == entry["sha256"]
    assert manifest["tables"]["raw_events"]["rows"] == 2
    assert json.loads((folder / "manifest.json").read_text())["tables"] == manifest["tables"]

    # Restore raw_events into an empty copy of the table and compare.
    db.execute("CREATE TABLE restored (LIKE raw_events)")
    with db.cursor() as cursor, cursor.copy("COPY restored FROM STDIN WITH (FORMAT csv, HEADER)") as copy:
        copy.write(gzip.open(folder / "raw_events.csv.gz").read())
    original = db.execute("SELECT id, message_id, headers, body FROM raw_events ORDER BY id").fetchall()
    restored = db.execute("SELECT id, message_id, headers, body FROM restored ORDER BY id").fetchall()
    assert original == restored


def test_only_the_latest_copies_are_kept_and_other_files_are_left(tmp_path):
    for day in range(1, 10):
        (tmp_path / f"cs_rct_202611{day:02d}T033000Z").mkdir()
    (tmp_path / "code_before_7837bb7_20261105_1527").mkdir()
    (tmp_path / "cs_rct_before_reset_2026-10-20.sql.gz").write_bytes(b"x")
    (tmp_path / "cs_rct_cronicle.py.bak_20261105").write_text("x")
    (tmp_path / "cs_rct_20261101T0330Z").mkdir()  # not the name of a copy
    removed = backup.prune(tmp_path, keep=7)
    assert removed == ["cs_rct_20261101T033000Z", "cs_rct_20261102T033000Z"]
    assert [p.name for p in backup.copies(tmp_path)] == [f"cs_rct_202611{d:02d}T033000Z" for d in range(3, 10)]
    for name in ("code_before_7837bb7_20261105_1527", "cs_rct_before_reset_2026-10-20.sql.gz",
                 "cs_rct_cronicle.py.bak_20261105", "cs_rct_20261101T0330Z"):
        assert (tmp_path / name).exists()


def test_a_backup_removes_the_oldest_copy_beyond_seven(db, tmp_path):
    for day in range(1, 8):
        (tmp_path / f"cs_rct_202610{day:02d}T033000Z").mkdir()
    manifest = backup.backup(db, datetime(2026, 11, 5, 3, 30), tmp_path)
    assert manifest["removed"] == ["cs_rct_20261001T033000Z"]
    assert len(backup.copies(tmp_path)) == 7
