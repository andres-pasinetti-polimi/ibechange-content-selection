import psycopg
import pytest

from cs_rct import config, database, design

ADMIN_URL = config.DATABASE_URL
TEST_URL = ADMIN_URL.rsplit("/", 1)[0] + "/cs_rct_test"


@pytest.fixture(autouse=True)
def utc_participants(monkeypatch):
    """The tests are written in UTC: every participant's local time is UTC unless a test sets
    a centre's zone itself (test_timezones.py)."""
    monkeypatch.setattr(design, "CENTRE_TIMEZONE", {})
    monkeypatch.setattr(design, "DEFAULT_TIMEZONE", "UTC")


@pytest.fixture()
def db(monkeypatch):
    """A fresh database cs_rct_test on the PostgreSQL of docker-compose.yml; skipped if it is down."""
    try:
        admin = psycopg.connect(ADMIN_URL, autocommit=True, connect_timeout=2)
    except psycopg.OperationalError:
        pytest.skip("PostgreSQL of docker-compose.yml is not running")
    admin.execute("DROP DATABASE IF EXISTS cs_rct_test WITH (FORCE)")
    admin.execute("CREATE DATABASE cs_rct_test")
    monkeypatch.setattr(config, "DATABASE_URL", TEST_URL)
    connection = database.connect()
    database.create_tables(connection)
    yield connection
    connection.close()
