import os
import numpy as np
from datetime import datetime
import psycopg2
import time
import json
import logging
from psycopg2.extras import Json


def sanitize_for_json(obj):
    """Recursively convert data into JSON-serializable format."""
    if isinstance(obj, dict):
        return {k: sanitize_for_json(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [sanitize_for_json(v) for v in obj]
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, datetime):
        return obj.isoformat()
    else:
        return obj


class DataStorage:
    """
    DataStorage handles storing various module outputs into a structured PostgreSQL schema.
    Connection parameters are read from environment variables:
      DB_HOST, DB_NAME, DB_USER, DB_PASSWORD
    """

    def __init__(self):
        # DB connection settings
        self.db_params = {
            "dbname": os.getenv("DB_NAME"),
            "user": os.getenv("DB_USER"),
            "password": os.getenv("DB_PASSWORD"),
            "host": os.getenv("DB_HOST"),
            "port": "5432",
        }
        # Ensure tables exist
        self._ensure_tables()

        # Create a single run identifier
        self.run_id = self._create_run()

    def _get_conn(self, retries=5, delay=3):
        for i in range(retries):
            try:
                return psycopg2.connect(**self.db_params)
            except psycopg2.OperationalError as e:
                print(f"[DB INIT] Attempt {i + 1} failed: {e}")
                time.sleep(delay)
        raise Exception("Failed to connect to the database after several retries")

    def _ensure_tables(self):
        """
        Create all required tables with structured schemas.
        """
        table_creations = [
            """
            CREATE TABLE IF NOT EXISTS runs (
                run_id        SERIAL PRIMARY KEY,
                started_at    TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT now(),
                description   TEXT
            );
            """,
            # Intervention runs
            """
            CREATE TABLE IF NOT EXISTS intervention_mab_runs (
                run_id      INTEGER PRIMARY KEY REFERENCES runs(run_id),
                started_at  TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT now(),
                bandit_type TEXT    NOT NULL,        -- e.g. "GaussianInverseGammaTS"
                initial_params      JSONB                       -- store INTERVENTION_MAB_initial_params
            );
            """,
            # Recommendation runs
            """
            CREATE TABLE IF NOT EXISTS recommendation_mab_runs (
                run_id      INTEGER PRIMARY KEY REFERENCES runs(run_id),
                started_at  TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT now(),
                bandit_type TEXT    NOT NULL,
                initial_params      JSONB
            );
            """,
            # Resource runs
            """
            CREATE TABLE IF NOT EXISTS resource_mab_runs (
                run_id      INTEGER PRIMARY KEY REFERENCES runs(run_id),
                started_at  TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT now(),
                bandit_type TEXT    NOT NULL,
                initial_params      JSONB
            );
            """,
            # MAB updates tables
            """
            CREATE TABLE IF NOT EXISTS intervention_mab_updates (
                run_id INTEGER NOT NULL REFERENCES runs(run_id),
                id SERIAL NOT NULL,
                user_id TEXT NOT NULL,
                process_id INTEGER,
                timestamp TIMESTAMP WITH TIME ZONE,
                reward INTEGER,
                params JSONB,
                PRIMARY KEY (run_id, id)
            );
            """,
            """
            CREATE TABLE IF NOT EXISTS recommendation_mab_updates (
                run_id INTEGER NOT NULL REFERENCES runs(run_id),
                id SERIAL NOT NULL,
                user_id TEXT NOT NULL,
                process_id INTEGER,
                timestamp TIMESTAMP WITH TIME ZONE,
                reward INTEGER,
                params JSONB,
                PRIMARY KEY (run_id, id)
            );
            """,
            """
            CREATE TABLE IF NOT EXISTS resource_mab_updates (
                run_id INTEGER NOT NULL REFERENCES runs(run_id),
                id SERIAL NOT NULL,
                user_id TEXT NOT NULL,
                process_id INTEGER,
                timestamp TIMESTAMP WITH TIME ZONE,
                reward INTEGER,
                params JSONB,
                PRIMARY KEY (run_id, id)
            );
            """,
            # MAB samples tables
            """
            CREATE TABLE IF NOT EXISTS intervention_mab_samples (
                run_id INTEGER NOT NULL REFERENCES runs(run_id),
                id SERIAL NOT NULL,
                user_id TEXT NOT NULL,
                plan_count INTEGER NOT NULL,
                content_count INTEGER NOT NULL,
                timestamp TIMESTAMP WITH TIME ZONE,
                sample JSONB,
                PRIMARY KEY (run_id, id)
            );
            """,
            """
            CREATE TABLE IF NOT EXISTS recommendation_mab_samples (
                run_id INTEGER NOT NULL REFERENCES runs(run_id),
                id SERIAL NOT NULL,
                user_id TEXT NOT NULL,
                plan_count INTEGER NOT NULL,
                content_count INTEGER NOT NULL,
                timestamp TIMESTAMP WITH TIME ZONE,
                sample JSONB,
                PRIMARY KEY (run_id, id)
            );
            """,
            """
            CREATE TABLE IF NOT EXISTS resource_mab_samples (
                run_id INTEGER NOT NULL REFERENCES runs(run_id),
                id SERIAL NOT NULL,
                user_id TEXT NOT NULL,
                plan_count INTEGER NOT NULL,
                content_count INTEGER NOT NULL,
                timestamp TIMESTAMP WITH TIME ZONE,
                sample JSONB,
                PRIMARY KEY (run_id, id)
            );
            """,
            # users
            """
            CREATE TABLE IF NOT EXISTS users (
                run_id INTEGER NOT NULL REFERENCES runs(run_id),
                id SERIAL NOT NULL,
                user_id TEXT NOT NULL,
                gender TEXT,
                userAge INTEGER,
                height INTEGER,
                weight INTEGER,
                recruitmentCenter TEXT,
                enrolmentDate TIMESTAMP WITH TIME ZONE,
                wearable TEXT,
                voiceRecording TEXT,
                occupation TEXT,
                education TEXT,
                digitalLiteracy TEXT,
                level INTEGER,
                PRIMARY KEY (run_id, id),
                UNIQUE (run_id, user_id)
            );
            """,
            # disabled_users
            # users might be re-enabled...?
            """
            CREATE TABLE IF NOT EXISTS disabled_users (
                run_id INTEGER NOT NULL REFERENCES runs(run_id),
                id SERIAL NOT NULL, 
                user_id TEXT NOT NULL,
                date_disabled TIMESTAMP WITH TIME ZONE,
                PRIMARY KEY (run_id, user_id),
                FOREIGN KEY (run_id, user_id) REFERENCES users(run_id, user_id)
            );
            """
            # escalation_levels
            """
            CREATE TABLE IF NOT EXISTS escalation_levels (
                run_id INTEGER NOT NULL REFERENCES runs(run_id),
                id SERIAL NOT NULL,
                user_id TEXT NOT NULL,
                timestamp TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT now(),
                escalation_level INTEGER NOT NULL,
                pillar TEXT,
                PRIMARY KEY (run_id, id),
                FOREIGN KEY (run_id, user_id) REFERENCES users(run_id, user_id)
            );
            """,
            # health_habit_assessments
            """
            CREATE TABLE IF NOT EXISTS health_habit_assessments (
                run_id INTEGER NOT NULL REFERENCES runs(run_id),
                id SERIAL NOT NULL,
                user_id TEXT NOT NULL,
                assessment_timestamp TIMESTAMP WITH TIME ZONE,
                alcohol FLOAT,
                nutrition FLOAT,
                physical_activity FLOAT,
                smoking FLOAT,
                emotional_wellbeing FLOAT,
                nutrition_components JSONB,
                emotional_wellbeing_components JSONB,
                PRIMARY KEY (run_id, id),
                FOREIGN KEY (run_id, user_id) REFERENCES users(run_id, user_id)
            );
            """,
            # new_missions_and_contents flattened to new_missions
            """
            CREATE TABLE IF NOT EXISTS new_missions_and_contents (
                run_id INTEGER NOT NULL REFERENCES runs(run_id),
                id SERIAL NOT NULL,
                user_id TEXT NOT NULL,
                update_timestamp TIMESTAMP WITH TIME ZONE,
                mission_id TEXT,
                recommendations TEXT[],
                resources TEXT[],
                prescribed BOOLEAN,
                selection_timestamp TIMESTAMP WITH TIME ZONE,
                finish_timestamp TIMESTAMP WITH TIME ZONE,
                PRIMARY KEY (run_id, id),
                FOREIGN KEY (run_id, user_id) REFERENCES users(run_id, user_id)
            );
            """,
            # recommendation_plans + plan_contents
            """
            CREATE TABLE IF NOT EXISTS recommendation_plans (
                run_id INTEGER NOT NULL REFERENCES runs(run_id),
                id SERIAL NOT NULL,
                plan_id SERIAL NOT NULL,
                user_id TEXT NOT NULL,
                created_at TIMESTAMP WITH TIME ZONE,
                PRIMARY KEY (run_id, id),
                UNIQUE (run_id, plan_id),
                FOREIGN KEY (run_id, user_id) REFERENCES users(run_id, user_id)
            );
            """,
            """
            CREATE TABLE IF NOT EXISTS plan_contents (
                run_id INTEGER NOT NULL REFERENCES runs(run_id),
                id SERIAL NOT NULL,
                plan_id INTEGER NOT NULL,
                content_id TEXT,
                mission_id TEXT,
                send_timestamp TIMESTAMP WITH TIME ZONE,
                type TEXT,
                PRIMARY KEY (run_id, id),
                FOREIGN KEY (run_id, plan_id) REFERENCES recommendation_plans(run_id, plan_id)
            );
            """,
            # selected_contents
            """
            CREATE TABLE IF NOT EXISTS selected_contents (
                run_id INTEGER NOT NULL REFERENCES runs(run_id),
                id SERIAL NOT NULL,
                user_id TEXT,
                plan_count INTEGER NOT NULL,
                timestamp TIMESTAMP WITH TIME ZONE,
                mission_start_time TIMESTAMP WITH TIME ZONE,
                mission_end_time TIMESTAMP WITH TIME ZONE,
                content_ids TEXT[],
                mission_id TEXT,
                PRIMARY KEY (run_id, id)
            );
            """,
            # user_feedback
            """
            CREATE TABLE IF NOT EXISTS user_feedback (
                run_id INTEGER NOT NULL REFERENCES runs(run_id),
                id SERIAL NOT NULL,
                user_id TEXT NOT NULL,
                process_id INTEGER,
                timestamp TIMESTAMP WITH TIME ZONE,
                event_name TEXT,
                properties JSONB,
                PRIMARY KEY (run_id, id),
                FOREIGN KEY (run_id, user_id) REFERENCES users(run_id, user_id)
            );
            """,
        ]

        conn = self._get_conn()
        cur = conn.cursor()
        for stmt in table_creations:
            cur.execute(stmt)
        conn.commit()
        cur.close()
        conn.close()

    def _create_run(self):
        """Insert a new runs entry and return its run_id."""
        conn = self._get_conn()
        cur = conn.cursor()
        cur.execute("INSERT INTO runs(description) VALUES (%s) RETURNING run_id;", (os.getenv("PIPELINE_DESCRIPTION"),))
        run_id = cur.fetchone()[0]
        conn.commit()
        cur.close()
        conn.close()
        return run_id

    def initialize_bandit(self, table, bandit_type, initial_params):
        conn = self._get_conn()
        cur = conn.cursor()
        clean_params = sanitize_for_json(initial_params)

        cur.execute(
            f"INSERT INTO {table} (run_id, bandit_type, initial_params) VALUES (%s, %s, %s);",
            (self.run_id, bandit_type, json.dumps(clean_params)),
        )
        conn.commit()
        cur.close()
        conn.close()

    def add_mab_update(self, table: str, update: dict):
        conn = self._get_conn()
        cur = conn.cursor()
        user_id = update.get("user_id")
        process_id = update.get("process_id")
        ts = update.get("timestamp")
        reward = update.get("reward")
        params = update.get("params")
        clean = sanitize_for_json(params)
        cur.execute(
            f"INSERT INTO {table}(run_id, user_id, process_id, timestamp, reward, params) VALUES (%s,%s,%s,%s,%s,%s);",
            (self.run_id, user_id, process_id, ts, reward, json.dumps(clean)),
        )
        conn.commit()
        cur.close()
        conn.close()

    def add_mab_sample(self, table: str, record: dict):
        conn = self._get_conn()
        cur = conn.cursor()
        uid = record.get("user_id")
        pc = record.get("plan_count")
        cc = record.get("content_count")
        ts = record.get("timestamp")
        sample = sanitize_for_json(record.get("sample"))
        cur.execute(
            f"INSERT INTO {table}(run_id, user_id, plan_count, content_count, timestamp, sample) VALUES (%s,%s,%s,%s,%s,%s);",
            (self.run_id, uid, pc, cc, ts, json.dumps(sample)),
        )
        conn.commit()
        cur.close()
        conn.close()

    def add_disabled_users(self, disabled_users):
        conn = self._get_conn()
        cur = conn.cursor()
        for user_id, items in disabled_users.items():
            cur.execute(
                "INSERT INTO disabled_users(run_id, user_id, date_disabled) VALUES (%s, %s, %s) ON CONFLICT DO NOTHING;",
                (
                    self.run_id,
                    user_id,
                    items["date_disabled"],
                ),
            )
        conn.commit()
        cur.close()
        conn.close()

    def add_escalation_levels(self, escalation_levels):
        conn = self._get_conn()
        cur = conn.cursor()
        for user_id, levels_list in escalation_levels.items():
            for esc_level in levels_list:
                cur.execute(
                    "INSERT INTO escalation_levels(run_id, user_id, timestamp, escalation_level, pillar) VALUES (%s, %s, %s, %s, %s);",
                    (
                        self.run_id,
                        user_id,
                        esc_level["update_timestamp"],
                        esc_level["level"],
                        esc_level.get("pillar", None),
                    ),
                )
        conn.commit()
        cur.close()
        conn.close()

    def add_users(self, users):
        conn = self._get_conn()
        cur = conn.cursor()

        for user_id, u in users.items():
            cur.execute(
                """
                INSERT INTO users (
                    run_id, user_id, gender, userAge, height, weight,
                    recruitmentCenter, enrolmentDate, wearable, voiceRecording,
                    occupation, education, digitalLiteracy, level
                )
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (run_id, user_id) DO UPDATE SET
                    gender = EXCLUDED.gender,
                    userAge = EXCLUDED.userAge,
                    height = EXCLUDED.height,
                    weight = EXCLUDED.weight,
                    recruitmentCenter = EXCLUDED.recruitmentCenter,
                    enrolmentDate = EXCLUDED.enrolmentDate,
                    wearable = EXCLUDED.wearable,
                    voiceRecording = EXCLUDED.voiceRecording,
                    occupation = EXCLUDED.occupation,
                    education = EXCLUDED.education,
                    digitalLiteracy = EXCLUDED.digitalLiteracy,
                    level = EXCLUDED.level
                """,
                (
                    self.run_id,
                    user_id,
                    u.get("gender"),
                    u.get("userAge"),
                    u.get("height"),
                    u.get("weight"),
                    u.get("recruitmentCenter"),
                    u.get("enrolmentDate"),
                    u.get("wearable"),
                    u.get("voiceRecording"),
                    u.get("occupation"),
                    u.get("education"),
                    u.get("digitalLiteracy"),
                    u.get("level"),
                ),
            )

        conn.commit()
        cur.close()
        conn.close()

    def add_health_habit_assessments(self, assessments):
        conn = self._get_conn()
        cur = conn.cursor()
        for user_id, hhs_list in assessments.items():
            for entry in hhs_list:
                h = entry["hhs"]

                alcohol = h.get("alcohol")
                physical_activity = h.get("physical_activity")
                smoking = h.get("smoking")

                nutrition = h.get("nutrition") if "nutrition" in h else None
                nutrition_components = Json(h["components"]) if "nutrition" in h and "components" in h else None

                ew = h.get("emotional_wellbeing") if "emotional_wellbeing" in h else None
                ew_components = Json(h["components"]) if "emotional_wellbeing" in h and "components" in h else None

                cur.execute(
                    "INSERT INTO health_habit_assessments"
                    " (run_id, user_id, assessment_timestamp, alcohol, nutrition,"
                    "  physical_activity, smoking, emotional_wellbeing, nutrition_components, emotional_wellbeing_components)"
                    " VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s);",
                    (
                        self.run_id,
                        user_id,
                        entry["assessment_timestamp"],
                        alcohol,
                        nutrition,
                        physical_activity,
                        smoking,
                        ew,
                        nutrition_components,
                        ew_components,
                    ),
                )
        conn.commit()
        cur.close()
        conn.close()

    def add_new_missions_and_contents(self, entries):
        conn = self._get_conn()
        cur = conn.cursor()
        for user_id, entry in entries.items():
            ts = entry["update_timestamp"]
            for m in entry.get("new_missions", []):
                cur.execute(
                    "INSERT INTO new_missions_and_contents"
                    " (run_id, update_timestamp, user_id, mission_id, recommendations, resources, prescribed, selection_timestamp, finish_timestamp)"
                    " VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s);",
                    (
                        self.run_id,
                        ts,
                        user_id,
                        m["mission"],
                        m.get("recommendations", []),
                        m.get("resources", []),
                        m["prescribed"],
                        m["selection_timestamp"],
                        m["finish_timestamp"],
                    ),
                )
        conn.commit()
        cur.close()
        conn.close()

    def add_recommendation_plans(self, plans):
        conn = self._get_conn()
        cur = conn.cursor()
        for user_id, p in plans.items():
            cur.execute(
                "INSERT INTO recommendation_plans (run_id, user_id, created_at) VALUES (%s, %s, %s) RETURNING plan_id;",
                (self.run_id, user_id, p["created_at"]),
            )
            plan_id = cur.fetchone()[0]
            for c in p.get("contents", []):
                cur.execute(
                    "INSERT INTO plan_contents"
                    " (run_id, plan_id, content_id, mission_id, send_timestamp, type)"
                    " VALUES (%s, %s, %s, %s, %s, %s);",
                    (self.run_id, plan_id, c["id"], c["mission"], c["send_timestamp"], c["type"]),
                )
        conn.commit()
        cur.close()
        conn.close()

    # for simplicity, we assume 1 mission per user
    def add_selected_contents(self, selections):
        conn = self._get_conn()
        cur = conn.cursor()
        ts = selections["timestamp"]
        mission_ids = selections["mission_ids"]
        for uid, items in selections["selected_contents"].items():
            cur.execute(
                "INSERT INTO selected_contents"
                " (run_id, user_id, plan_count, timestamp, mission_start_time, mission_end_time, content_ids, mission_id)"
                " VALUES (%s, %s, %s, %s, %s, %s, %s, %s);",
                (
                    self.run_id,
                    uid,
                    items["plan_count"],
                    ts,
                    items["mission_start_time"],
                    items["mission_end_time"],
                    [item["id"] for item in items["contents"]],
                    mission_ids[uid],
                ),
            )
        conn.commit()
        cur.close()
        conn.close()

    def add_user_feedback(self, feedback):
        conn = self._get_conn()
        cur = conn.cursor()
        for user_id, user_feedback in feedback.items():
            for event in user_feedback["events"]:
                cur.execute(
                    "INSERT INTO user_feedback"
                    " (run_id, user_id, process_id, timestamp, event_name, properties)"
                    " VALUES (%s, %s, %s, %s, %s, %s);",
                    (
                        self.run_id,
                        user_id,
                        event["process_id"],
                        event["timestamp"],
                        event["event_name"],
                        Json(event["properties"]),
                    ),
                )
        conn.commit()
        cur.close()
        conn.close()
