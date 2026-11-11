# cs_module/utils/data_storage.py
from __future__ import annotations

import os
import json
import time
import logging
import hashlib
from datetime import datetime
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np
import psycopg2
from psycopg2.extras import Json, execute_values

from cs_module.utils.logging_utils import get_logger

# Try to import cs_module.config to read MAB_TYPE, etc.
try:
    from cs_module import config as cs_config  # simulator & prod both have this
except Exception:  # pragma: no cover
    cs_config = None


# -----------------------------------------------------------------------------
# Utilities
# -----------------------------------------------------------------------------
def sanitize_for_json(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {k: sanitize_for_json(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [sanitize_for_json(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, datetime):
        return obj.isoformat()
    return obj


def _u64_from_key(*parts: Any) -> int:
    s = "|".join(map(str, parts)).encode("utf-8")
    h = hashlib.blake2b(s, digest_size=8).digest()
    return int.from_bytes(h, "big", signed=False)


def _hex_key(*parts: Any, digest_size: int = 16) -> str:
    s = "|".join(map(str, parts)).encode("utf-8")
    return hashlib.blake2b(s, digest_size=digest_size).hexdigest()


def _chunked(seq: Sequence[Any], n: int) -> Iterable[Sequence[Any]]:
    for i in range(0, len(seq), n):
        yield seq[i : i + n]


# -----------------------------------------------------------------------------
# DataStorage (fast, idempotent, batch-oriented)
# -----------------------------------------------------------------------------
class DataStorage:
    """
    Fast Postgres writer.

    Key changes vs row-by-row approach:
      1) One connection per DataStorage instance (reused across calls).
      2) Bulk inserts via psycopg2.extras.execute_values.
      3) Idempotency via deterministic *row_key/event_key* UNIQUE constraints
         + ON CONFLICT DO NOTHING (safe under Airflow retries/backfills).

    Operational note:
      - In Airflow, create DataStorage inside each task invocation.
        Do NOT share the same instance across tasks/workers.
    """

    # Sensible default batch sizes
    DEFAULT_BATCH = int(os.getenv("DS_BATCH_SIZE", "2000"))

    def __init__(self):
        self.enable_feature_audit = os.getenv("DS_ENABLE_FEATURE_AUDIT", "0").lower() in ("1", "true", "yes")
        self.store_feature_vector = os.getenv("DS_STORE_FEATURE_VECTOR", "1").lower() in ("1", "true", "yes")
        self.store_param_var = os.getenv("DS_STORE_PARAM_VAR", "1").lower() in ("1", "true", "yes")
        self.log = get_logger(self)

        # DB connection settings
        self.db_params = {
            "dbname": os.getenv("DB_NAME"),
            "user": os.getenv("DB_USER"),
            "password": os.getenv("DB_PASSWORD"),
            "host": os.getenv("DB_HOST"),
            "port": os.getenv("DB_PORT", "5432"),
        }

        self.conn = self._get_conn()
        self._set_utc_session()
        self._ensure_tables()

        # Create a single run identifier
        self.run_id = self._create_run()

    # ------------------------------------------------------------------
    # Connection lifecycle
    # ------------------------------------------------------------------
    def _get_conn(self, retries: int = 5, delay: float = 1.5):
        last = None
        for i in range(retries):
            try:
                conn = psycopg2.connect(**self.db_params)
                conn.autocommit = False
                return conn
            except psycopg2.OperationalError as e:
                last = e
                time.sleep(delay)
        raise RuntimeError(f"Failed to connect to Postgres after {retries} retries: {last}")

    def _set_utc_session(self) -> None:
        with self.conn.cursor() as cur:
            cur.execute("SET TIME ZONE 'UTC';")
        self.conn.commit()

    def close(self) -> None:
        try:
            if getattr(self, "conn", None) is not None:
                try:
                    self.conn.commit()
                except Exception:
                    pass
                self.conn.close()
        finally:
            self.conn = None

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Schema (idempotent keys + indexes)
    # ------------------------------------------------------------------
    def _ensure_tables(self) -> None:
        table_creations = [
            """
            CREATE TABLE IF NOT EXISTS runs (
                run_id        SERIAL PRIMARY KEY,
                created_at    TIMESTAMPTZ NOT NULL DEFAULT clock_timestamp(),
                description   TEXT,

                env_name      TEXT,
                world_id      INTEGER,
                policy_name   TEXT,
                seed          INTEGER,
                algo_seed     INTEGER,
                study_tag     TEXT,
                finished_at   TIMESTAMPTZ
            );
            """,
            # Bandit run metadata
            """
            CREATE TABLE IF NOT EXISTS intervention_mab_runs (
                run_id        INTEGER PRIMARY KEY REFERENCES runs(run_id),
                created_at    TIMESTAMPTZ NOT NULL DEFAULT clock_timestamp(),
                bandit_type   TEXT NOT NULL,
                initial_params JSONB
            );
            """,
            """
            CREATE TABLE IF NOT EXISTS recommendation_mab_runs (
                run_id        INTEGER PRIMARY KEY REFERENCES runs(run_id),
                created_at    TIMESTAMPTZ NOT NULL DEFAULT clock_timestamp(),
                bandit_type   TEXT NOT NULL,
                initial_params JSONB
            );
            """,
            """
            CREATE TABLE IF NOT EXISTS resource_mab_runs (
                run_id        INTEGER PRIMARY KEY REFERENCES runs(run_id),
                created_at    TIMESTAMPTZ NOT NULL DEFAULT clock_timestamp(),
                bandit_type   TEXT NOT NULL,
                initial_params JSONB
            );
            """,
            # Users
            """
            CREATE TABLE IF NOT EXISTS users (
                run_id INTEGER NOT NULL REFERENCES runs(run_id),
                id SERIAL NOT NULL,
                created_at TIMESTAMPTZ NOT NULL DEFAULT clock_timestamp(),
                user_id TEXT NOT NULL,
                gender TEXT,
                userAge INTEGER,
                height INTEGER,
                weight INTEGER,
                recruitmentCenter TEXT,
                enrolmentDate TIMESTAMPTZ,
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
            """
            CREATE TABLE IF NOT EXISTS disabled_users (
                run_id INTEGER NOT NULL REFERENCES runs(run_id),
                user_id TEXT NOT NULL,
                created_at TIMESTAMPTZ NOT NULL DEFAULT clock_timestamp(),
                date_disabled TIMESTAMPTZ,
                PRIMARY KEY (run_id, user_id),
                FOREIGN KEY (run_id, user_id) REFERENCES users(run_id, user_id)
            );
            """,
            """
            CREATE TABLE IF NOT EXISTS escalation_levels (
                run_id INTEGER NOT NULL REFERENCES runs(run_id),
                id SERIAL NOT NULL,
                created_at TIMESTAMPTZ NOT NULL DEFAULT clock_timestamp(),
                user_id TEXT NOT NULL,
                timestamp TIMESTAMPTZ NOT NULL DEFAULT clock_timestamp(),
                escalation_level INTEGER NOT NULL,
                pillar TEXT,
                row_key TEXT NOT NULL,
                PRIMARY KEY (run_id, id),
                UNIQUE (run_id, row_key),
                FOREIGN KEY (run_id, user_id) REFERENCES users(run_id, user_id)
            );
            """,
            # HHS: idempotent by (user_id, assessment_timestamp)
            """
            CREATE TABLE IF NOT EXISTS health_habit_assessments (
                run_id INTEGER NOT NULL REFERENCES runs(run_id),
                id SERIAL NOT NULL,
                created_at TIMESTAMPTZ NOT NULL DEFAULT clock_timestamp(),
                user_id TEXT NOT NULL,
                assessment_timestamp TIMESTAMPTZ,
                alcohol FLOAT,
                nutrition FLOAT,
                physical_activity FLOAT,
                smoking FLOAT,
                emotional_wellbeing FLOAT,
                nutrition_components JSONB,
                emotional_wellbeing_components JSONB,
                row_key TEXT NOT NULL,
                PRIMARY KEY (run_id, id),
                UNIQUE (run_id, row_key),
                FOREIGN KEY (run_id, user_id) REFERENCES users(run_id, user_id)
            );
            """,
            # Missions & contents: idempotent row_key
            """
            CREATE TABLE IF NOT EXISTS new_missions_and_contents (
                run_id INTEGER NOT NULL REFERENCES runs(run_id),
                id SERIAL NOT NULL,
                created_at TIMESTAMPTZ NOT NULL DEFAULT clock_timestamp(),
                user_id TEXT NOT NULL,
                update_timestamp TIMESTAMPTZ,
                mission_id TEXT,
                recommendations TEXT[],
                resources TEXT[],
                prescribed BOOLEAN,
                selection_timestamp TIMESTAMPTZ,
                finish_timestamp TIMESTAMPTZ,
                row_key TEXT NOT NULL,
                PRIMARY KEY (run_id, id),
                UNIQUE (run_id, row_key),
                FOREIGN KEY (run_id, user_id) REFERENCES users(run_id, user_id)
            );
            """,
            # Plans
            """
            CREATE TABLE IF NOT EXISTS recommendation_plans (
                run_id INTEGER NOT NULL REFERENCES runs(run_id),
                id SERIAL NOT NULL,
                created_at TIMESTAMPTZ NOT NULL DEFAULT clock_timestamp(),
                plan_id TEXT NOT NULL,
                user_id TEXT NOT NULL,
                PRIMARY KEY (run_id, id),
                UNIQUE (run_id, plan_id),
                FOREIGN KEY (run_id, user_id) REFERENCES users(run_id, user_id)
            );
            """,
            # Plan contents: idempotent by (plan_id, content_id, scheduled_for)
            """
            CREATE TABLE IF NOT EXISTS plan_contents (
                run_id INTEGER NOT NULL REFERENCES runs(run_id),
                id SERIAL NOT NULL,
                created_at TIMESTAMPTZ NOT NULL DEFAULT clock_timestamp(),

                plan_id TEXT NOT NULL,
                content_id TEXT,
                content_type TEXT,              -- "recommendation" | "resource" | NULL
                slot_idx INTEGER,               -- position in CS planned list (for planned rows)
                planned_ts TIMESTAMPTZ,         -- when CS computed the plan (for planned rows)

                scheduled_for TIMESTAMPTZ,      -- when OMI says it scheduled (for OMI rows)
                omi_idx INTEGER,                -- index in the OMI "plans" list (for OMI rows)

                source TEXT NOT NULL,           -- "planned" or "omi"
                row_key TEXT NOT NULL,

                PRIMARY KEY (run_id, id),
                UNIQUE (run_id, row_key),
                FOREIGN KEY (run_id, plan_id) REFERENCES recommendation_plans(run_id, plan_id)
            );
            """,
            # Selected contents: already idempotent
            """
            CREATE TABLE IF NOT EXISTS selected_contents (
                run_id INTEGER NOT NULL REFERENCES runs(run_id),
                id SERIAL NOT NULL,
                created_at TIMESTAMPTZ NOT NULL DEFAULT clock_timestamp(),
                user_id TEXT NOT NULL,
                plan_id TEXT NOT NULL,
                timestamp TIMESTAMPTZ,
                mission_start_time TIMESTAMPTZ,
                content_ids TEXT[],
                mission_id TEXT,
                PRIMARY KEY (run_id, id),
                UNIQUE (run_id, user_id, plan_id)
            );
            """,
            # User feedback: idempotent event_key
            """
            CREATE TABLE IF NOT EXISTS user_feedback (
                run_id INTEGER NOT NULL REFERENCES runs(run_id),
                id SERIAL NOT NULL,
                created_at TIMESTAMPTZ NOT NULL DEFAULT clock_timestamp(),
                user_id TEXT NOT NULL,
                process_id INTEGER,
                timestamp TIMESTAMPTZ,
                event_name TEXT,
                properties JSONB,
                event_key TEXT NOT NULL,
                PRIMARY KEY (run_id, id),
                UNIQUE (run_id, event_key),
                FOREIGN KEY (run_id, user_id) REFERENCES users(run_id, user_id)
            );
            """,
            # Bandit updates: idempotent row_key
            """
            CREATE TABLE IF NOT EXISTS intervention_mab_updates (
                run_id INTEGER NOT NULL REFERENCES runs(run_id),
                id SERIAL NOT NULL,
                created_at TIMESTAMPTZ NOT NULL DEFAULT clock_timestamp(),
                user_id TEXT NOT NULL,
                process_id INTEGER,
                timestamp TIMESTAMPTZ,         
                processed_at TIMESTAMPTZ,      
                feature_vector DOUBLE PRECISION[],
                reward INTEGER,
                params JSONB,
                row_key TEXT NOT NULL,
                PRIMARY KEY (run_id, id),
                UNIQUE (run_id, row_key)
            );
            """,
            """
            CREATE TABLE IF NOT EXISTS recommendation_mab_updates (
                run_id INTEGER NOT NULL REFERENCES runs(run_id),
                id SERIAL NOT NULL,
                created_at TIMESTAMPTZ NOT NULL DEFAULT clock_timestamp(),
                user_id TEXT NOT NULL,
                process_id INTEGER,
                timestamp TIMESTAMPTZ,
                processed_at TIMESTAMPTZ,      
                reward INTEGER,
                params JSONB,
                row_key TEXT NOT NULL,
                PRIMARY KEY (run_id, id),
                UNIQUE (run_id, row_key)
            );
            """,
            """
            CREATE TABLE IF NOT EXISTS resource_mab_updates (
                run_id INTEGER NOT NULL REFERENCES runs(run_id),
                id SERIAL NOT NULL,
                created_at TIMESTAMPTZ NOT NULL DEFAULT clock_timestamp(),
                user_id TEXT NOT NULL,
                process_id INTEGER,
                timestamp TIMESTAMPTZ,
                processed_at TIMESTAMPTZ,      
                reward INTEGER,
                params JSONB,
                row_key TEXT NOT NULL,
                PRIMARY KEY (run_id, id),
                UNIQUE (run_id, row_key)
            );
            """,
            # Bandit samples: idempotent row_key
            """
            CREATE TABLE IF NOT EXISTS intervention_mab_samples (
                run_id INTEGER NOT NULL REFERENCES runs(run_id),
                id SERIAL NOT NULL,
                created_at TIMESTAMPTZ NOT NULL DEFAULT clock_timestamp(),
                user_id TEXT NOT NULL,
                plan_id TEXT NOT NULL,
                slot_idx INTEGER NOT NULL,
                feature_vector DOUBLE PRECISION[] NOT NULL,
                selected_rec_ids TEXT[] NOT NULL,
                timestamp TIMESTAMPTZ,
                processed_at TIMESTAMPTZ,      
                sample JSONB,
                row_key TEXT NOT NULL,
                PRIMARY KEY (run_id, id),
                UNIQUE (run_id, row_key)
            );
            """,
            """
            CREATE TABLE IF NOT EXISTS recommendation_mab_samples (
                run_id INTEGER NOT NULL REFERENCES runs(run_id),
                id SERIAL NOT NULL,
                created_at TIMESTAMPTZ NOT NULL DEFAULT clock_timestamp(),
                user_id TEXT NOT NULL,
                plan_id TEXT NOT NULL,
                slot_idx INTEGER NOT NULL,
                timestamp TIMESTAMPTZ,
                processed_at TIMESTAMPTZ,      
                sample JSONB,
                row_key TEXT NOT NULL,
                PRIMARY KEY (run_id, id),
                UNIQUE (run_id, row_key)
            );
            """,
            """
            CREATE TABLE IF NOT EXISTS resource_mab_samples (
                run_id INTEGER NOT NULL REFERENCES runs(run_id),
                id SERIAL NOT NULL,
                created_at TIMESTAMPTZ NOT NULL DEFAULT clock_timestamp(),
                user_id TEXT NOT NULL,
                plan_id TEXT NOT NULL,
                slot_idx INTEGER NOT NULL,
                timestamp TIMESTAMPTZ,
                processed_at TIMESTAMPTZ,      
                sample JSONB,
                row_key TEXT NOT NULL,
                PRIMARY KEY (run_id, id),
                UNIQUE (run_id, row_key)
            );
            """,
            # New-version bandit tables (same schema, separate storage)
            """
            CREATE TABLE IF NOT EXISTS intervention_mab_runs_new (
                run_id        INTEGER PRIMARY KEY REFERENCES runs(run_id),
                created_at    TIMESTAMPTZ NOT NULL DEFAULT clock_timestamp(),
                bandit_type   TEXT NOT NULL,
                initial_params JSONB
            );
            """,
            """
            CREATE TABLE IF NOT EXISTS recommendation_mab_runs_new (
                run_id        INTEGER PRIMARY KEY REFERENCES runs(run_id),
                created_at    TIMESTAMPTZ NOT NULL DEFAULT clock_timestamp(),
                bandit_type   TEXT NOT NULL,
                initial_params JSONB
            );
            """,
            """
            CREATE TABLE IF NOT EXISTS resource_mab_runs_new (
                run_id        INTEGER PRIMARY KEY REFERENCES runs(run_id),
                created_at    TIMESTAMPTZ NOT NULL DEFAULT clock_timestamp(),
                bandit_type   TEXT NOT NULL,
                initial_params JSONB
            );
            """,
            """
            CREATE TABLE IF NOT EXISTS intervention_mab_updates_new (
                run_id INTEGER NOT NULL REFERENCES runs(run_id),
                id SERIAL NOT NULL,
                created_at TIMESTAMPTZ NOT NULL DEFAULT clock_timestamp(),
                user_id TEXT NOT NULL,
                process_id INTEGER,
                timestamp TIMESTAMPTZ,
                processed_at TIMESTAMPTZ,
                feature_vector DOUBLE PRECISION[],
                reward INTEGER,
                params JSONB,
                row_key TEXT NOT NULL,
                PRIMARY KEY (run_id, id),
                UNIQUE (run_id, row_key)
            );
            """,
            """
            CREATE TABLE IF NOT EXISTS recommendation_mab_updates_new (
                run_id INTEGER NOT NULL REFERENCES runs(run_id),
                id SERIAL NOT NULL,
                created_at TIMESTAMPTZ NOT NULL DEFAULT clock_timestamp(),
                user_id TEXT NOT NULL,
                process_id INTEGER,
                timestamp TIMESTAMPTZ,
                processed_at TIMESTAMPTZ,
                reward INTEGER,
                params JSONB,
                row_key TEXT NOT NULL,
                PRIMARY KEY (run_id, id),
                UNIQUE (run_id, row_key)
            );
            """,
            """
            CREATE TABLE IF NOT EXISTS resource_mab_updates_new (
                run_id INTEGER NOT NULL REFERENCES runs(run_id),
                id SERIAL NOT NULL,
                created_at TIMESTAMPTZ NOT NULL DEFAULT clock_timestamp(),
                user_id TEXT NOT NULL,
                process_id INTEGER,
                timestamp TIMESTAMPTZ,
                processed_at TIMESTAMPTZ,
                reward INTEGER,
                params JSONB,
                row_key TEXT NOT NULL,
                PRIMARY KEY (run_id, id),
                UNIQUE (run_id, row_key)
            );
            """,
            """
            CREATE TABLE IF NOT EXISTS intervention_mab_samples_new (
                run_id INTEGER NOT NULL REFERENCES runs(run_id),
                id SERIAL NOT NULL,
                created_at TIMESTAMPTZ NOT NULL DEFAULT clock_timestamp(),
                user_id TEXT NOT NULL,
                plan_id TEXT NOT NULL,
                slot_idx INTEGER NOT NULL,
                feature_vector DOUBLE PRECISION[] NOT NULL,
                selected_rec_ids TEXT[] NOT NULL,
                timestamp TIMESTAMPTZ,
                processed_at TIMESTAMPTZ,
                sample JSONB,
                row_key TEXT NOT NULL,
                PRIMARY KEY (run_id, id),
                UNIQUE (run_id, row_key)
            );
            """,
            """
            CREATE TABLE IF NOT EXISTS recommendation_mab_samples_new (
                run_id INTEGER NOT NULL REFERENCES runs(run_id),
                id SERIAL NOT NULL,
                created_at TIMESTAMPTZ NOT NULL DEFAULT clock_timestamp(),
                user_id TEXT NOT NULL,
                plan_id TEXT NOT NULL,
                slot_idx INTEGER NOT NULL,
                timestamp TIMESTAMPTZ,
                processed_at TIMESTAMPTZ,
                sample JSONB,
                row_key TEXT NOT NULL,
                PRIMARY KEY (run_id, id),
                UNIQUE (run_id, row_key)
            );
            """,
            """
            CREATE TABLE IF NOT EXISTS resource_mab_samples_new (
                run_id INTEGER NOT NULL REFERENCES runs(run_id),
                id SERIAL NOT NULL,
                created_at TIMESTAMPTZ NOT NULL DEFAULT clock_timestamp(),
                user_id TEXT NOT NULL,
                plan_id TEXT NOT NULL,
                slot_idx INTEGER NOT NULL,
                timestamp TIMESTAMPTZ,
                processed_at TIMESTAMPTZ,
                sample JSONB,
                row_key TEXT NOT NULL,
                PRIMARY KEY (run_id, id),
                UNIQUE (run_id, row_key)
            );
            """,
            """
            CREATE TABLE IF NOT EXISTS user_version_upgrades (
                run_id INTEGER NOT NULL REFERENCES runs(run_id),
                id SERIAL NOT NULL,
                created_at TIMESTAMPTZ NOT NULL DEFAULT clock_timestamp(),
                user_id TEXT NOT NULL,
                version TEXT NOT NULL,
                timestamp TIMESTAMPTZ,
                upgrade_key TEXT NOT NULL,
                PRIMARY KEY (run_id, id),
                UNIQUE (run_id, upgrade_key)
            );
            """,
            # Helpful indexes (optional but usually worth it)
            "CREATE INDEX IF NOT EXISTS idx_user_feedback_run_ts ON user_feedback(run_id, timestamp);",
            "CREATE INDEX IF NOT EXISTS idx_inter_updates_run_ts ON intervention_mab_updates(run_id, timestamp);",
            "CREATE INDEX IF NOT EXISTS idx_rec_updates_run_ts ON recommendation_mab_updates(run_id, timestamp);",
            "CREATE INDEX IF NOT EXISTS idx_res_updates_run_ts ON resource_mab_updates(run_id, timestamp);",
            "CREATE INDEX IF NOT EXISTS idx_inter_updates_new_run_ts ON intervention_mab_updates_new(run_id, timestamp);",
            "CREATE INDEX IF NOT EXISTS idx_rec_updates_new_run_ts ON recommendation_mab_updates_new(run_id, timestamp);",
            "CREATE INDEX IF NOT EXISTS idx_res_updates_new_run_ts ON resource_mab_updates_new(run_id, timestamp);",
            "CREATE INDEX IF NOT EXISTS idx_runs_world_policy ON runs(world_id, policy_name);",
        ]

        if self.enable_feature_audit:
            # Feature audit: idempotent row_key
            table_creations.append(
                """CREATE TABLE IF NOT EXISTS feature_audit (
                run_id  INTEGER NOT NULL REFERENCES runs(run_id),
                id      SERIAL NOT NULL,
                created_at TIMESTAMPTZ NOT NULL DEFAULT clock_timestamp(),

                user_id TEXT NOT NULL,
                mission_id TEXT,
                rec_id TEXT,
                process_id INTEGER,
                slot_idx INTEGER,
                prompted BOOLEAN,

                sel_ts TIMESTAMPTZ,
                finish_ts TIMESTAMPTZ,
                rating_ts TIMESTAMPTZ,

                reward INTEGER,
                fv DOUBLE PRECISION[],

                planned_so_far JSONB,
                recent_exposure_history_for_slot JSONB,
                available_recs_at_sel_ts JSONB,
                sent_seq JSONB,
                context_snapshot JSONB,

                row_key TEXT NOT NULL,

                PRIMARY KEY (run_id, id),
                UNIQUE (run_id, row_key)
            );
            """
            )

        with self.conn.cursor() as cur:
            # best-effort persistent TZ defaults (non-fatal)
            try:
                cur.execute("ALTER DATABASE cs_data SET timezone TO 'UTC';")
            except Exception:
                pass
            try:
                role = os.getenv("DB_USER") or "cs_user"
                cur.execute(f"ALTER ROLE {role} SET timezone TO 'UTC';")
            except Exception:
                pass

            for stmt in table_creations:
                cur.execute(stmt)

        self.conn.commit()

    # ------------------------------------------------------------------
    # Run creation
    # ------------------------------------------------------------------
    def _create_run(self) -> int:
        description = os.getenv("PIPELINE_DESCRIPTION")

        env_name = os.getenv("ENV_NAME") or os.getenv("CS_ENV_NAME") or os.getenv("SUITE_NAME")

        def _safe_int(keys: List[str]) -> Optional[int]:
            for k in keys:
                v = os.getenv(k)
                if v is None:
                    continue
                try:
                    return int(v)
                except Exception:
                    return None
            return None

        world_id = _safe_int(["WORLD_ID", "SCENARIO_ID"])
        seed = _safe_int(["WORLD_SEED", "SCENARIO_SEED"])
        algo_seed = _safe_int(["ALGO_SEED"])
        if algo_seed is None:
            algo_seed = seed

        study_tag = os.getenv("STUDY_TAG")

        config_mab_type = getattr(cs_config, "MAB_TYPE", None) if cs_config is not None else None
        policy_name = os.getenv("POLICY_NAME") or os.getenv("CS_POLICY_NAME") or config_mab_type

        if not description:
            if env_name or world_id is not None or policy_name:
                description = f"env={env_name}, world_id={world_id}, policy={policy_name}"
            else:
                description = "cs_module_run"

        with self.conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO runs(
                    description, env_name, world_id, policy_name, seed, algo_seed, study_tag
                )
                VALUES (%s,%s,%s,%s,%s,%s,%s)
                RETURNING run_id;
                """,
                (description, env_name, world_id, policy_name, seed, algo_seed, study_tag),
            )
            run_id = int(cur.fetchone()[0])
        self.conn.commit()
        return run_id

    # ------------------------------------------------------------------
    # Small helpers
    # ------------------------------------------------------------------
    def _to_float_array(self, x):
        if x is None:
            return None
        if isinstance(x, np.ndarray):
            return [float(v) for v in x.tolist()]
        if isinstance(x, (list, tuple)):
            return [float(v) for v in x]
        if isinstance(x, set):
            raise TypeError("feature_vector was a set (unordered). Use list/tuple to preserve order.")
        try:
            return [float(x)]
        except Exception as e:
            raise TypeError(f"feature_vector must be sequence of numbers, got {type(x).__name__}") from e

    def _bulk_insert(
        self,
        sql: str,
        rows: Sequence[Tuple[Any, ...]],
        page_size: Optional[int] = None,
        commit: bool = True,
    ) -> int:
        if not rows:
            return 0
        page_size = page_size or self.DEFAULT_BATCH
        with self.conn.cursor() as cur:
            execute_values(cur, sql, rows, page_size=page_size)
        if commit:
            self.conn.commit()
        return len(rows)

    # ------------------------------------------------------------------
    # Bandit initialization (idempotent)
    # ------------------------------------------------------------------
    def initialize_bandit(self, table: str, bandit_type: str, initial_params: dict):
        clean_params = sanitize_for_json(initial_params)
        with self.conn.cursor() as cur:
            cur.execute(
                f"""
                INSERT INTO {table} (run_id, bandit_type, initial_params)
                VALUES (%s, %s, %s)
                ON CONFLICT (run_id) DO UPDATE
                  SET bandit_type = EXCLUDED.bandit_type,
                      initial_params = EXCLUDED.initial_params;
                """,
                (self.run_id, bandit_type, json.dumps(clean_params)),
            )
        self.conn.commit()

    # ------------------------------------------------------------------
    # Users (bulk upsert)
    # ------------------------------------------------------------------
    def add_users(self, users: Dict[str, dict]) -> int:
        if not users:
            return 0

        rows = []
        for user_id, u in users.items():
            rows.append(
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
                )
            )

        sql = """
        INSERT INTO users (
            run_id, user_id, gender, userAge, height, weight,
            recruitmentCenter, enrolmentDate, wearable, voiceRecording,
            occupation, education, digitalLiteracy, level
        )
        VALUES %s
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
            level = EXCLUDED.level;
        """
        return self._bulk_insert(sql, rows)

    def add_disabled_users(self, disabled_users: Dict[str, dict]) -> int:
        if not disabled_users:
            return 0
        rows = []
        for user_id, items in disabled_users.items():
            rows.append((self.run_id, user_id, items.get("date_disabled")))

        sql = """
        INSERT INTO disabled_users(run_id, user_id, date_disabled)
        VALUES %s
        ON CONFLICT (run_id, user_id) DO NOTHING;
        """
        return self._bulk_insert(sql, rows)

    def add_escalation_levels(self, escalation_levels: Dict[str, list]) -> int:
        if not escalation_levels:
            return 0
        rows = []
        for user_id, levels_list in escalation_levels.items():
            for esc in levels_list:
                ts = esc.get("update_timestamp")
                level = esc.get("level")
                pillar = esc.get("pillar")
                row_key = _hex_key(self.run_id, "esc", user_id, ts, level, pillar)
                rows.append((self.run_id, user_id, ts, level, pillar, row_key))

        sql = """
        INSERT INTO escalation_levels(run_id, user_id, timestamp, escalation_level, pillar, row_key)
        VALUES %s
        ON CONFLICT (run_id, row_key) DO NOTHING;
        """
        return self._bulk_insert(sql, rows)

    # ------------------------------------------------------------------
    # HHS (bulk, idempotent)
    # ------------------------------------------------------------------
    def add_health_habit_assessments(self, assessments: Dict[str, list]) -> int:
        if not assessments:
            return 0

        def _sanitize_components_dict(d, pillar_name):
            if not isinstance(d, dict):
                logging.warning("HHS components for %s not a dict: %r", pillar_name, d)
                return None
            clean = {}
            for k, v in d.items():
                try:
                    clean[k] = float(v)
                except Exception:
                    logging.warning("Component %s.%s has non-numeric value %r; skipping.", pillar_name, k, v)
            return clean if clean else None

        rows = []
        for user_id, hhs_list in assessments.items():
            for entry in hhs_list:
                ts = entry.get("assessment_timestamp")
                h = entry.get("hhs", {}) or {}

                alcohol = h.get("alcohol")
                physical_activity = h.get("physical_activity")
                smoking = h.get("smoking")

                nutrition = h.get("nutrition") if "nutrition" in h else None
                nutrition_components = None
                if "nutrition" in h and "components" in h:
                    nc = _sanitize_components_dict(h.get("components"), "nutrition")
                    nutrition_components = nc

                ew = h.get("emotional_wellbeing") if "emotional_wellbeing" in h else None
                ew_components = None
                if "emotional_wellbeing" in h and "components" in h:
                    ec = _sanitize_components_dict(h.get("components"), "emotional_wellbeing")
                    ew_components = ec
                elif "emotional_distress" in h:
                    try:
                        ed_val = float(h.get("emotional_distress"))
                        ew_components = {"emotional_distress": ed_val}
                    except Exception:
                        logging.warning(
                            "emotional_distress has non-numeric value %r; skipping.", h.get("emotional_distress")
                        )

                row_key = _hex_key(self.run_id, "hhs", user_id, ts)

                rows.append(
                    (
                        self.run_id,
                        user_id,
                        ts,
                        alcohol,
                        nutrition,
                        physical_activity,
                        smoking,
                        ew,
                        json.dumps(sanitize_for_json(nutrition_components))
                        if nutrition_components is not None
                        else None,
                        json.dumps(sanitize_for_json(ew_components)) if ew_components is not None else None,
                        row_key,
                    )
                )

        sql = """
        INSERT INTO health_habit_assessments(
            run_id, user_id, assessment_timestamp,
            alcohol, nutrition, physical_activity, smoking, emotional_wellbeing,
            nutrition_components, emotional_wellbeing_components,
            row_key
        )
        VALUES %s
        ON CONFLICT (run_id, row_key) DO NOTHING;
        """
        return self._bulk_insert(sql, rows)

    # ------------------------------------------------------------------
    # Missions (bulk, idempotent)
    # ------------------------------------------------------------------
    def add_new_missions_and_contents(self, entries: Dict[str, dict]) -> int:
        if not entries:
            return 0

        rows = []
        for user_id, entry in entries.items():
            update_ts = entry.get("update_timestamp")
            for m in entry.get("new_missions", []) or []:
                mission_id = m.get("mission")
                selection_ts = m.get("selection_timestamp")
                row_key = _hex_key(self.run_id, "mission", user_id, mission_id, selection_ts)

                rows.append(
                    (
                        self.run_id,
                        user_id,
                        update_ts,
                        mission_id,
                        m.get("recommendations", []),
                        m.get("resources", []),
                        m.get("prescribed", False),
                        selection_ts,
                        m.get("finish_timestamp"),
                        row_key,
                    )
                )

        sql = """
        INSERT INTO new_missions_and_contents(
            run_id, user_id, update_timestamp, mission_id,
            recommendations, resources, prescribed,
            selection_timestamp, finish_timestamp,
            row_key
        )
        VALUES %s
        ON CONFLICT (run_id, row_key) DO NOTHING;
        """
        return self._bulk_insert(sql, rows)

    # ------------------------------------------------------------------
    # Plans (bulk, idempotent)
    # ------------------------------------------------------------------
    def add_recommendation_plans(self, plans: dict) -> int:
        """
        Writes:
          - recommendation_plans (upsert/ignore)
          - plan_contents (idempotent by row_key)
        """
        if not plans:
            return 0

        # Preload allowed users for FK safety (optional; remove if you guarantee users exist)
        with self.conn.cursor() as cur:
            cur.execute("SELECT user_id FROM users WHERE run_id = %s", (self.run_id,))
            known_users = {r[0] for r in cur.fetchall()}

        plan_rows = []
        content_rows = []

        for user_plan in (plans or {}).get("recommendation_plans", []) or []:
            user_id = user_plan.get("user_id")
            items = user_plan.get("plans", []) or []
            if not user_id or not items:
                continue
            if user_id not in known_users:
                logging.warning(
                    "DS skip recommendation_plan: unknown user (run_id=%s, user_id=%s)", self.run_id, user_id
                )
                continue

            for i, it in enumerate(items):
                pid = it.get("plan_id")
                if not pid:
                    continue
                # plan header
                plan_rows.append((self.run_id, user_id, pid))
                # plan content (idempotent)
                cid = it.get("content_id")
                sched = it.get("scheduled_for")
                omi_idx = i
                row_key = _hex_key(self.run_id, "plan_content", "omi", pid, omi_idx, cid, sched)
                content_rows.append((self.run_id, pid, cid, None, None, None, sched, omi_idx, "omi", row_key))

        # Insert plans
        if plan_rows:
            sql_plans = """
            INSERT INTO recommendation_plans(run_id, user_id, plan_id)
            VALUES %s
            ON CONFLICT (run_id, plan_id) DO NOTHING
            """
            self._bulk_insert(sql_plans, plan_rows, commit=False)

        # Insert contents
        applied = 0
        if content_rows:
            sql_contents = """
            INSERT INTO plan_contents(
                run_id, plan_id, content_id, content_type, slot_idx, planned_ts,
                scheduled_for, omi_idx, source, row_key
            )
            VALUES %s
            ON CONFLICT (run_id, row_key) DO NOTHING;
            """
            applied = self._bulk_insert(sql_contents, content_rows, commit=False)

        self.conn.commit()
        return applied

    # for simplicity, we assume 1 mission per user (as in your original)
    def add_selected_contents(self, selections: dict) -> int:
        if not selections:
            return 0

        ts = selections.get("timestamp")
        user_to_mission_id = selections.get("mission_id") or {}

        rows = []
        for user_id, items in (selections.get("selected_contents") or {}).items():
            pid = items.get("plan_id")
            if not pid:
                logging.warning(
                    "DS skip selected_contents: missing plan_id (run_id=%s user_id=%s)", self.run_id, user_id
                )
                continue
            rows.append(
                (
                    self.run_id,
                    user_id,
                    pid,
                    ts,
                    items.get("mission_start_time"),
                    [c.get("id") for c in (items.get("contents") or [])],
                    user_to_mission_id.get(user_id),
                )
            )

        sql = """
        INSERT INTO selected_contents(
            run_id, user_id, plan_id, timestamp, mission_start_time, content_ids, mission_id
        )
        VALUES %s
        ON CONFLICT (run_id, user_id, plan_id) DO NOTHING;
        """
        return self._bulk_insert(sql, rows)

    # ------------------------------------------------------------------
    # User feedback (bulk, idempotent)
    # ------------------------------------------------------------------
    def add_user_feedback(self, feedback: Dict[str, dict], batch_size: Optional[int] = None) -> int:
        """
        feedback shape:
          { user_id: { "events": [ {process_id,timestamp,event_name,properties}, ... ] }, ... }
        """
        if not feedback:
            return 0
        batch_size = batch_size or self.DEFAULT_BATCH

        rows: List[Tuple[Any, ...]] = []
        for user_id, payload in feedback.items():
            for ev in payload.get("events") or []:
                pid = ev.get("process_id")
                ts = ev.get("timestamp")
                name = ev.get("event_name")
                props = ev.get("properties") or {}

                # deterministic event_key for idempotency
                # include the parts that define "same event on replay"
                # (content_id/type/mission/is_end_misison are the critical ones in your simulator)
                ek = _hex_key(
                    self.run_id,
                    "uf",
                    user_id,
                    pid,
                    ts,
                    name,
                    props.get("content_id"),
                    props.get("content_type"),
                    props.get("mission_id"),
                    props.get("is_end_misison"),
                )

                rows.append((self.run_id, user_id, pid, ts, name, json.dumps(sanitize_for_json(props)), ek))

        sql = """
        INSERT INTO user_feedback(run_id, user_id, process_id, timestamp, event_name, properties, event_key)
        VALUES %s
        ON CONFLICT (run_id, event_key) DO NOTHING;
        """

        inserted = 0
        for chunk in _chunked(rows, batch_size):
            inserted += self._bulk_insert(sql, chunk, commit=False, page_size=batch_size)
        self.conn.commit()
        return inserted

    # ------------------------------------------------------------------
    # User version upgrades
    # ------------------------------------------------------------------
    def add_user_version_upgrades(self, upgrades: Dict[str, dict]) -> int:
        """
        upgrades: {user_id: {"version": "new", "timestamp": "<iso>"}}
        Idempotent: one row per (run_id, user_id) — silently skips duplicates.
        """
        if not upgrades:
            return 0
        rows = []
        for user_id, rv in upgrades.items():
            uk = _hex_key(self.run_id, "uvu", user_id)
            rows.append((self.run_id, user_id, rv.get("version", "new"), rv.get("timestamp"), uk))
        sql = """
        INSERT INTO user_version_upgrades(run_id, user_id, version, timestamp, upgrade_key)
        VALUES %s
        ON CONFLICT (run_id, upgrade_key) DO NOTHING;
        """
        inserted = self._bulk_insert(sql, rows, commit=True)
        return inserted

    def get_upgraded_user_ids(self) -> set:
        """Return user_ids that have a recorded version upgrade to 'new' across all runs."""
        with self.conn.cursor() as cur:
            cur.execute("SELECT DISTINCT user_id FROM user_version_upgrades WHERE version = 'new';")
            return {row[0] for row in cur.fetchall()}

    # ------------------------------------------------------------------
    # MAB updates (bulk, idempotent)
    # ------------------------------------------------------------------
    def add_intervention_mab_update(self, update: dict, table: str = "intervention_mab_updates") -> int:
        return self.add_intervention_mab_updates([update], table=table)

    def add_intervention_mab_updates(self, updates: Sequence[dict], batch_size: Optional[int] = None, table: str = "intervention_mab_updates") -> int:
        if not updates:
            return 0
        batch_size = batch_size or self.DEFAULT_BATCH

        rows = []
        for u in updates:
            user_id = u.get("user_id")
            process_id = u.get("process_id")
            ts = u.get("timestamp")  # event_ts
            processed_at = u.get("processed_at")  # process_ts
            fv = self._to_float_array(u.get("feature_vector")) if self.store_feature_vector else None
            reward = u.get("reward")
            params = self._slim_params(sanitize_for_json(u.get("params") or {}))
            rec_id = u.get("rec_id")
            rec_id_s = str(rec_id) if rec_id is not None else ""
            row_key = _hex_key(self.run_id, table, user_id, process_id, ts, rec_id_s)

            rows.append((self.run_id, user_id, process_id, ts, processed_at, fv, reward, json.dumps(params), row_key))

        sql = f"""
        INSERT INTO {table}(
            run_id, user_id, process_id, timestamp, processed_at, feature_vector, reward, params, row_key
        )
        VALUES %s
        ON CONFLICT (run_id, row_key) DO NOTHING;
        """

        inserted = 0
        for chunk in _chunked(rows, batch_size):
            inserted += self._bulk_insert(sql, chunk, commit=False, page_size=batch_size)
        self.conn.commit()
        return inserted

    def add_mab_update(self, table: str, update: dict) -> int:
        return self.add_mab_updates(table, [update])

    def add_mab_updates(self, table: str, updates: Sequence[dict], batch_size: Optional[int] = None) -> int:
        if not updates:
            return 0
        batch_size = batch_size or self.DEFAULT_BATCH

        rows = []
        for u in updates:
            user_id = u.get("user_id")
            process_id = u.get("process_id")
            ts = u.get("timestamp")  # event_ts
            processed_at = u.get("processed_at")  # process_ts
            reward = u.get("reward")
            params = sanitize_for_json(u.get("params"))
            row_key = _hex_key(self.run_id, table, "upd", user_id, process_id, ts)
            rows.append((self.run_id, user_id, process_id, ts, processed_at, reward, json.dumps(params), row_key))

        sql = f"""
        INSERT INTO {table}(run_id, user_id, process_id, timestamp, processed_at, reward, params, row_key)
        VALUES %s
        ON CONFLICT (run_id, row_key) DO NOTHING;
        """

        inserted = 0
        for chunk in _chunked(rows, batch_size):
            inserted += self._bulk_insert(sql, chunk, commit=False, page_size=batch_size)
        self.conn.commit()
        return inserted

    # ------------------------------------------------------------------
    # MAB samples (bulk, idempotent)
    # ------------------------------------------------------------------
    def add_intervention_mab_sample(self, record: dict, table: str = "intervention_mab_samples") -> int:
        return self.add_intervention_mab_samples([record], table=table)

    def add_intervention_mab_samples(self, records: Sequence[dict], batch_size: Optional[int] = None, table: str = "intervention_mab_samples") -> int:
        if not records:
            return 0
        batch_size = batch_size or self.DEFAULT_BATCH

        rows = []
        for r in records:
            user_id = r.get("user_id")
            plan_id = r.get("plan_id")
            slot_idx = r.get("slot_idx")
            fv = self._to_float_array(r.get("feature_vector"))
            sr = r.get("selected_rec_ids")
            if sr is not None and not isinstance(sr, list):
                sr = list(sr)
            sr = [str(x) for x in sr] if sr is not None else []
            ts = r.get("timestamp")
            processed_at = r.get("processed_at")
            sample = sanitize_for_json(r.get("sample"))

            row_key = _hex_key(self.run_id, table, user_id, plan_id, slot_idx, ts)

            rows.append(
                (
                    self.run_id,
                    user_id,
                    plan_id,
                    slot_idx,
                    fv,
                    sr,
                    ts,
                    processed_at,
                    json.dumps(sample),
                    row_key,
                )
            )

        sql = f"""
        INSERT INTO {table}(
            run_id, user_id, plan_id, slot_idx, feature_vector, selected_rec_ids, timestamp, processed_at, sample, row_key
        )
        VALUES %s
        ON CONFLICT (run_id, row_key) DO NOTHING;
        """

        inserted = 0
        for chunk in _chunked(rows, batch_size):
            inserted += self._bulk_insert(sql, chunk, commit=False, page_size=batch_size)
        self.conn.commit()
        return inserted

    def add_mab_sample(self, table: str, record: dict) -> int:
        return self.add_mab_samples(table, [record])

    def add_mab_samples(self, table: str, records: Sequence[dict], batch_size: Optional[int] = None) -> int:
        if not records:
            return 0
        batch_size = batch_size or self.DEFAULT_BATCH

        rows = []
        for r in records:
            user_id = r.get("user_id")
            plan_id = r.get("plan_id")
            slot_idx = r.get("slot_idx")
            ts = r.get("timestamp")
            processed_at = r.get("processed_at")
            sample = sanitize_for_json(r.get("sample"))

            row_key = _hex_key(self.run_id, table, "samp", user_id, plan_id, slot_idx, ts)

            rows.append((self.run_id, user_id, plan_id, slot_idx, ts, processed_at, json.dumps(sample), row_key))

        sql = f"""
        INSERT INTO {table}(run_id, user_id, plan_id, slot_idx, timestamp, processed_at, sample, row_key)
        VALUES %s
        ON CONFLICT (run_id, row_key) DO NOTHING;
        """

        inserted = 0
        for chunk in _chunked(rows, batch_size):
            inserted += self._bulk_insert(sql, chunk, commit=False, page_size=batch_size)
        self.conn.commit()
        return inserted

    # ------------------------------------------------------------------
    # Feature audit (bulk, idempotent)
    # ------------------------------------------------------------------
    def add_feature_audit(self, audit_row: dict) -> int:
        return self.add_feature_audit_many([audit_row])

    def add_feature_audit_many(self, audit_rows: Sequence[dict], batch_size: Optional[int] = None) -> int:
        if not getattr(self, "enable_feature_audit", False):
            return 0
        if not audit_rows:
            return 0
        batch_size = batch_size or self.DEFAULT_BATCH

        rows = []
        for a in audit_rows:
            user_id = a.get("user_id")
            mission_id = a.get("mission_id")
            rec_id = a.get("rec_id")
            process_id = a.get("process_id")
            prompted = a.get("prompted")
            rating_ts = a.get("rating_ts")

            row_key = _hex_key(self.run_id, "feat_audit", user_id, mission_id, rec_id, process_id, prompted, rating_ts)

            rows.append(
                (
                    self.run_id,
                    user_id,
                    mission_id,
                    rec_id,
                    process_id,
                    a.get("slot_idx"),
                    prompted,
                    a.get("sel_ts"),
                    a.get("finish_ts"),
                    rating_ts,
                    a.get("reward"),
                    self._to_float_array(a.get("fv")),
                    json.dumps(sanitize_for_json(a.get("planned_so_far"))),
                    json.dumps(sanitize_for_json(a.get("recent_exposure_history_for_slot"))),
                    json.dumps(sanitize_for_json(a.get("available_recs_at_sel_ts"))),
                    json.dumps(sanitize_for_json(a.get("sent_seq"))),
                    json.dumps(sanitize_for_json(a.get("context_snapshot"))),
                    row_key,
                )
            )

        sql = """
        INSERT INTO feature_audit(
            run_id,
            user_id, mission_id, rec_id,
            process_id, slot_idx, prompted,
            sel_ts, finish_ts, rating_ts,
            reward, fv,
            planned_so_far,
            recent_exposure_history_for_slot,
            available_recs_at_sel_ts,
            sent_seq,
            context_snapshot,
            row_key
        )
        VALUES %s
        ON CONFLICT (run_id, row_key) DO NOTHING;
        """

        inserted = 0
        for chunk in _chunked(rows, batch_size):
            inserted += self._bulk_insert(sql, chunk, commit=False, page_size=batch_size)
        self.conn.commit()
        return inserted

    def _slim_params(self, params: Any) -> Any:
        if not isinstance(params, dict):
            return params
        out = {}
        if "rec_id" in params:
            out["rec_id"] = params["rec_id"]

        if "mu" in params:
            out["mu"] = params["mu"]

        # keep per-item bias update if present
        if "item_bias" in params:
            out["item_bias"] = params["item_bias"]

        if self.store_param_var:
            # only allow diagonal; never keep full P
            if "Pd" in params:
                out["Pd"] = params["Pd"]
            # if someone passes P, ignore it
        return out
