import os
import numpy as np
from datetime import datetime
import psycopg2
import time
import json
import logging
from psycopg2.extras import Json
from cs_module.utils.logging_utils import pretty, get_logger, ctx

# Try to import cs_module.config to read MAB_TYPE, etc.
try:
    from cs_module import config as cs_config  # simulator & prod both have this
except Exception:  # pragma: no cover – extremely defensive
    cs_config = None


def sanitize_for_json(obj):
    if isinstance(obj, dict):
        return {k: sanitize_for_json(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):  # add tuple
        return [sanitize_for_json(v) for v in obj]
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.floating,)):  # add NumPy scalars
        return float(obj)
    elif isinstance(obj, (np.integer,)):
        return int(obj)
    elif isinstance(obj, (np.bool_,)):
        return bool(obj)
    elif isinstance(obj, datetime):
        return obj.isoformat()
    return obj


class DataStorage:
    """
    DataStorage handles storing various module outputs into a structured PostgreSQL schema.
    Connection parameters are read from environment variables:
      DB_HOST, DB_NAME, DB_USER, DB_PASSWORD

    IMPORTANT:
    - In simulation, we use one DB per WORLD, and one runs.row per (env, world, policy).
    - In production (Airflow), the container is long-lived; we typically get one runs.row
      per CS rebuild (per container lifetime).
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
        self.log = get_logger(self)

    def _get_conn(self, retries=5, delay=3):
        for i in range(retries):
            try:
                conn = psycopg2.connect(**self.db_params)
                with conn.cursor() as cur:
                    cur.execute("SET TIME ZONE 'UTC';")
                return conn
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
                created_at    TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT clock_timestamp(),
                description   TEXT,

                -- NEW metadata (nullable, so safe in prod)
                env_name    TEXT,      -- e.g. "Core-Ref" or "prod"
                world_id    INTEGER,   -- seed index in simulator (or 0 in prod)
                policy_name TEXT,      -- e.g. "LearningFull", "Random", "prod_default"
                seed        INTEGER,   -- world seed (sim) or None
                algo_seed   INTEGER,   -- bandit seed (may match seed)
                study_tag   TEXT,      -- optional label like "paper_v1"
                finished_at TIMESTAMPTZ
            );
            """,
            # Intervention runs
            """
            CREATE TABLE IF NOT EXISTS intervention_mab_runs (
                run_id      INTEGER PRIMARY KEY REFERENCES runs(run_id),
                created_at  TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT clock_timestamp(),
                bandit_type TEXT    NOT NULL,        -- e.g. "GaussianInverseGammaTS"
                initial_params      JSONB                       -- store INTERVENTION_MAB_initial_params
            );
            """,
            # Recommendation runs
            """
            CREATE TABLE IF NOT EXISTS recommendation_mab_runs (
                run_id      INTEGER PRIMARY KEY REFERENCES runs(run_id),
                created_at  TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT clock_timestamp(),
                bandit_type TEXT    NOT NULL,
                initial_params      JSONB
            );
            """,
            # Resource runs
            """
            CREATE TABLE IF NOT EXISTS resource_mab_runs (
                run_id      INTEGER PRIMARY KEY REFERENCES runs(run_id),
                created_at  TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT clock_timestamp(),
                bandit_type TEXT    NOT NULL,
                initial_params      JSONB
            );
            """,
            # MAB updates tables
            """
            CREATE TABLE IF NOT EXISTS intervention_mab_updates (
                run_id INTEGER NOT NULL REFERENCES runs(run_id),
                id SERIAL NOT NULL,
                created_at TIMESTAMPTZ NOT NULL DEFAULT clock_timestamp(),
                user_id TEXT NOT NULL,
                process_id INTEGER,
                timestamp TIMESTAMP WITH TIME ZONE,
                feature_vector DOUBLE PRECISION[] NOT NULL,
                reward INTEGER,
                params JSONB,
                PRIMARY KEY (run_id, id)
            );
            """,
            """
            CREATE TABLE IF NOT EXISTS recommendation_mab_updates (
                run_id INTEGER NOT NULL REFERENCES runs(run_id),
                id SERIAL NOT NULL,
                created_at TIMESTAMPTZ NOT NULL DEFAULT clock_timestamp(),
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
                created_at TIMESTAMPTZ NOT NULL DEFAULT clock_timestamp(),
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
                created_at TIMESTAMPTZ NOT NULL DEFAULT clock_timestamp(),
                user_id TEXT NOT NULL,
                plan_id TEXT NOT NULL,
                content_count INTEGER NOT NULL,
                feature_vector DOUBLE PRECISION[] NOT NULL,
                selected_rec_ids TEXT[] NOT NULL,
                timestamp TIMESTAMP WITH TIME ZONE,
                sample JSONB,
                PRIMARY KEY (run_id, id)
            );
            """,
            """
            CREATE TABLE IF NOT EXISTS recommendation_mab_samples (
                run_id INTEGER NOT NULL REFERENCES runs(run_id),
                id SERIAL NOT NULL,
                created_at TIMESTAMPTZ NOT NULL DEFAULT clock_timestamp(),
                user_id TEXT NOT NULL,
                plan_id TEXT NOT NULL,
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
                created_at TIMESTAMPTZ NOT NULL DEFAULT clock_timestamp(),
                user_id TEXT NOT NULL,
                plan_id TEXT NOT NULL,
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
                created_at TIMESTAMPTZ NOT NULL DEFAULT clock_timestamp(),
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
                created_at TIMESTAMPTZ NOT NULL DEFAULT clock_timestamp(),
                user_id TEXT NOT NULL,
                date_disabled TIMESTAMP WITH TIME ZONE,
                PRIMARY KEY (run_id, user_id),
                FOREIGN KEY (run_id, user_id) REFERENCES users(run_id, user_id)
            );
            """,
            # escalation_levels
            """
            CREATE TABLE IF NOT EXISTS escalation_levels (
                run_id INTEGER NOT NULL REFERENCES runs(run_id),
                id SERIAL NOT NULL,
                created_at TIMESTAMPTZ NOT NULL DEFAULT clock_timestamp(),
                user_id TEXT NOT NULL,
                timestamp TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT clock_timestamp(),
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
                created_at TIMESTAMPTZ NOT NULL DEFAULT clock_timestamp(),
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
                created_at TIMESTAMPTZ NOT NULL DEFAULT clock_timestamp(),
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
                created_at TIMESTAMPTZ NOT NULL DEFAULT clock_timestamp(),
                plan_id TEXT NOT NULL,
                user_id TEXT NOT NULL,
                PRIMARY KEY (run_id, id),
                UNIQUE (run_id, plan_id),
                FOREIGN KEY (run_id, user_id) REFERENCES users(run_id, user_id)
            );
            """,
            """
            CREATE TABLE IF NOT EXISTS plan_contents (
                run_id INTEGER NOT NULL REFERENCES runs(run_id),
                id SERIAL NOT NULL,
                created_at TIMESTAMPTZ NOT NULL DEFAULT clock_timestamp(),
                plan_id TEXT NOT NULL,
                content_id TEXT,
                scheduled_for TIMESTAMP WITH TIME ZONE,
                PRIMARY KEY (run_id, id),
                FOREIGN KEY (run_id, plan_id) REFERENCES recommendation_plans(run_id, plan_id)
            );
            """,
            # selected_contents
            """
            CREATE TABLE IF NOT EXISTS selected_contents (
                run_id INTEGER NOT NULL REFERENCES runs(run_id),
                id SERIAL NOT NULL,
                created_at TIMESTAMPTZ NOT NULL DEFAULT clock_timestamp(),
                user_id TEXT NOT NULL,
                plan_id TEXT NOT NULL,
                timestamp TIMESTAMP WITH TIME ZONE,
                mission_start_time TIMESTAMP WITH TIME ZONE,
                content_ids TEXT[],
                mission_id TEXT,
                PRIMARY KEY (run_id, id),
                UNIQUE (run_id, user_id, plan_id)   
            );
            """,
            # user_feedback
            """
            CREATE TABLE IF NOT EXISTS user_feedback (
                run_id INTEGER NOT NULL REFERENCES runs(run_id),
                id SERIAL NOT NULL,
                created_at TIMESTAMPTZ NOT NULL DEFAULT clock_timestamp(),
                user_id TEXT NOT NULL,
                process_id INTEGER,
                timestamp TIMESTAMP WITH TIME ZONE,
                event_name TEXT,
                properties JSONB,
                PRIMARY KEY (run_id, id),
                FOREIGN KEY (run_id, user_id) REFERENCES users(run_id, user_id)
            );
            """,
            """
            CREATE TABLE IF NOT EXISTS feature_audit (
                run_id  INTEGER NOT NULL REFERENCES runs(run_id),
                id      SERIAL NOT NULL,
                created_at TIMESTAMPTZ NOT NULL DEFAULT clock_timestamp(),

                -- identifiers / provenance
                user_id TEXT NOT NULL,
                mission_id TEXT,
                rec_id TEXT,
                process_id INTEGER,
                slot_idx INTEGER,
                prompted BOOLEAN,

                -- timing context
                sel_ts TIMESTAMPTZ,
                finish_ts TIMESTAMPTZ,
                rating_ts TIMESTAMPTZ,

                -- bandit supervision
                reward INTEGER,
                fv DOUBLE PRECISION[],

                -- planning context snapshot (JSON blobs so we can evolve schema freely)
                planned_so_far JSONB,
                recent_exposure_history_for_slot JSONB,
                available_recs_at_sel_ts JSONB,
                sent_seq JSONB,
                context_snapshot JSONB, 

                PRIMARY KEY (run_id, id)
            );
            """,
        ]

        conn = self._get_conn()
        cur = conn.cursor()

        # 1) Persist UTC as the default display/session TZ for this DB & role
        try:
            cur.execute("ALTER DATABASE cs_data SET timezone TO 'UTC';")
            cur.execute("ALTER ROLE cs_user SET timezone TO 'UTC';")
        except Exception as e:
            # Non-fatal if role/db already configured or lacking perms
            logging.warning("Could not persist UTC timezone defaults: %s", e)

        # 2) Ensure this session is UTC right now
        try:
            cur.execute("SET TIME ZONE 'UTC';")
        except Exception as e:
            logging.warning("Could not SET TIME ZONE to UTC for this session: %s", e)

        # 3) Create tables as you already do
        for stmt in table_creations:
            cur.execute(stmt)
        conn.commit()
        cur.close()
        conn.close()

    # ------------------------------------------------------------------
    # Run creation (env/world/policy aware, but optional)
    # ------------------------------------------------------------------
    def _create_run(self):
        """
        Insert a new runs entry and return its run_id.

        We try to enrich it with:
          - env_name: ENV_NAME / CS_ENV_NAME / SUITE_NAME (sim)
          - world_id: WORLD_ID / SCENARIO_ID (sim)
          - policy_name: POLICY_NAME / CS_POLICY_NAME / config.MAB_TYPE
          - seed: WORLD_SEED / SCENARIO_SEED
          - algo_seed: ALGO_SEED (else seed)
          - study_tag: STUDY_TAG (optional)
        All of this is best-effort and safe in production if env vars are absent.
        """
        conn = self._get_conn()
        cur = conn.cursor()

        # ---- description (what you already had) ----
        description = os.getenv("PIPELINE_DESCRIPTION")

        # ---- env_name / world_id / seed (sim-oriented, but harmless in prod) ----
        env_name = (
            os.getenv("ENV_NAME") or os.getenv("CS_ENV_NAME") or os.getenv("SUITE_NAME")  # legacy simulator notion
        )

        def _safe_int(env_key_list):
            raw = None
            for k in env_key_list:
                v = os.getenv(k)
                if v is not None:
                    raw = v
                    break
            if raw is None:
                return None
            try:
                return int(raw)
            except Exception:
                return None

        world_id = _safe_int(["WORLD_ID", "SCENARIO_ID"])
        seed = _safe_int(["WORLD_SEED", "SCENARIO_SEED"])
        algo_seed = _safe_int(["ALGO_SEED"])
        if algo_seed is None:
            algo_seed = seed

        study_tag = os.getenv("STUDY_TAG")

        # ---- policy_name: prefer explicit env, then config.MAB_TYPE ----
        config_mab_type = getattr(cs_config, "MAB_TYPE", None) if cs_config is not None else None
        policy_name = os.getenv("POLICY_NAME") or os.getenv("CS_POLICY_NAME") or config_mab_type

        # Fallback description if none was provided
        if not description:
            if env_name or world_id is not None or policy_name:
                description = f"env={env_name}, world_id={world_id}, policy={policy_name}"
            else:
                description = "cs_module_run"

        cur.execute(
            """
            INSERT INTO runs(
                description,
                env_name,
                world_id,
                policy_name,
                seed,
                algo_seed,
                study_tag
            )
            VALUES (%s,%s,%s,%s,%s,%s,%s)
            RETURNING run_id;
            """,
            (
                description,
                env_name,
                world_id,
                policy_name,
                seed,
                algo_seed,
                study_tag,
            ),
        )
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

    def add_intervention_mab_update(self, update: dict):
        conn = self._get_conn()
        cur = conn.cursor()
        user_id = update.get("user_id")  # string
        process_id = update.get("process_id")
        ts = update.get("timestamp")
        feature_vector = self._to_float_array(update.get("feature_vector"))

        reward = update.get("reward")
        params = update.get("params")
        clean = sanitize_for_json(params)
        cur.execute(
            "INSERT INTO intervention_mab_updates(run_id, user_id, process_id, timestamp, feature_vector, reward, params) VALUES (%s, %s, %s, %s, %s, %s, %s);",
            (self.run_id, user_id, process_id, ts, feature_vector, reward, json.dumps(clean)),
        )
        conn.commit()
        cur.close()
        conn.close()

    def add_mab_update(self, table: str, update: dict):
        conn = self._get_conn()
        cur = conn.cursor()
        user_id = update.get("user_id")  # string
        process_id = update.get("process_id")
        ts = update.get("timestamp")
        reward = update.get("reward")
        params = update.get("params")
        clean = sanitize_for_json(params)
        cur.execute(
            f"INSERT INTO {table}(run_id, user_id, process_id, timestamp, reward, params) "
            f"VALUES (%s, %s, %s, %s, %s, %s);",
            (self.run_id, user_id, process_id, ts, reward, json.dumps(clean)),
        )
        conn.commit()
        cur.close()
        conn.close()

    def add_intervention_mab_sample(self, record: dict):
        conn = self._get_conn()
        cur = conn.cursor()
        user_id = record.get("user_id")  # string
        pc = record.get("plan_id")  # string
        cc = record.get("content_count")
        fv = self._to_float_array(record.get("feature_vector"))
        sr = record.get("selected_rec_ids")
        # ensure TEXT[] is a list of strings
        if sr is not None and not isinstance(sr, list):
            sr = list(sr)
        sr = [str(s) for s in sr] if sr is not None else None

        ts = record.get("timestamp")
        sample = sanitize_for_json(record.get("sample"))
        cur.execute(
            "INSERT INTO intervention_mab_samples(run_id, user_id, plan_id, content_count, feature_vector, selected_rec_ids, timestamp, sample) "
            "VALUES (%s, %s, %s, %s, %s, %s, %s, %s);",
            (self.run_id, user_id, pc, cc, fv, sr, ts, json.dumps(sample)),
        )
        conn.commit()
        cur.close()
        conn.close()

    def add_mab_sample(self, table: str, record: dict):
        conn = self._get_conn()
        cur = conn.cursor()
        user_id = record.get("user_id")  # string
        pc = record.get("plan_id")  # string
        cc = record.get("content_count")
        ts = record.get("timestamp")
        sample = sanitize_for_json(record.get("sample"))
        cur.execute(
            f"INSERT INTO {table}(run_id, user_id, plan_id, content_count, timestamp, sample) "
            f"VALUES (%s, %s, %s, %s, %s, %s);",
            (self.run_id, user_id, pc, cc, ts, json.dumps(sample)),
        )
        conn.commit()
        cur.close()
        conn.close()

    def add_disabled_users(self, disabled_users):
        conn = self._get_conn()
        cur = conn.cursor()
        for user_id, items in disabled_users.items():
            cur.execute(
                "INSERT INTO disabled_users(run_id, user_id, date_disabled) "
                "VALUES (%s, %s, %s) ON CONFLICT DO NOTHING;",
                (
                    self.run_id,
                    user_id,  # string
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
                    "INSERT INTO escalation_levels(run_id, user_id, timestamp, escalation_level, pillar) "
                    "VALUES (%s, %s, %s, %s, %s);",
                    (
                        self.run_id,
                        user_id,  # string
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
                    user_id,  # string
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

        def _sanitize_components_dict(d, pillar_name):
            """Return a new dict with float values; drop and warn on non-numeric."""
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

        for user_id, hhs_list in assessments.items():
            for entry in hhs_list:
                h = entry.get("hhs", {}) or {}

                # main pillars (as sent in this payload only)
                alcohol = h.get("alcohol")
                physical_activity = h.get("physical_activity")
                smoking = h.get("smoking")

                # nutrition + components (only attach components if pillar key present)
                nutrition = h.get("nutrition") if "nutrition" in h else None
                nutrition_components = None
                if "nutrition" in h and "components" in h:
                    nc = _sanitize_components_dict(h.get("components"), "nutrition")
                    nutrition_components = Json(nc) if nc is not None else None

                # emotional wellbeing + components (incl. emotional_distress-only case)
                ew = h.get("emotional_wellbeing") if "emotional_wellbeing" in h else None
                ew_components = None
                if "emotional_wellbeing" in h and "components" in h:
                    ec = _sanitize_components_dict(h.get("components"), "emotional_wellbeing")
                    ew_components = Json(ec) if ec is not None else None
                elif "emotional_distress" in h:
                    # Bi-weekly single-component update
                    try:
                        ed_val = float(h.get("emotional_distress"))
                        ew_components = Json({"emotional_distress": ed_val})
                    except Exception:
                        logging.warning(
                            "emotional_distress has non-numeric value %r; skipping.", h.get("emotional_distress")
                        )

                # If components appear without a recognized pillar, warn (don’t persist ambiguous components)
                if "components" in h and ("nutrition" not in h and "emotional_wellbeing" not in h):
                    logging.warning(
                        "Received 'components' without nutrition/emotional_wellbeing for user %s at %r; skipping components.",
                        user_id,
                        entry.get("assessment_timestamp"),
                    )

                cur.execute(
                    "INSERT INTO health_habit_assessments"
                    " (run_id, user_id, assessment_timestamp, alcohol, nutrition,"
                    "  physical_activity, smoking, emotional_wellbeing, nutrition_components, emotional_wellbeing_components)"
                    " VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s);",
                    (
                        self.run_id,
                        user_id,  # string
                        entry.get("assessment_timestamp"),
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
                        user_id,  # string
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
        """
        Accepts payload shaped like:
        {
        "recommendation_plans": [
            {
            "user_id": "...",
            "plans": [
                {"content_id": "...", "scheduled_for": "...", "plan_id": "...", "mission_id": "...",
                "mission_start_time": "..."},
                ...
            ]
            },
            ...
        ]
        }
        Groups by (user_id, plan_id) and writes:
        - recommendation_plans(run_id, user_id, plan_id) [UPSERT]
        - plan_contents(run_id, plan_id, content_id, scheduled_for)
        Returns the number of distinct (user_id, plan_id) pairs processed.
        """
        conn = self._get_conn()
        cur = conn.cursor()
        applied = 0

        # 0) Allow-list users for this run to avoid FK violations
        cur.execute("SELECT user_id FROM users WHERE run_id = %s", (self.run_id,))
        known_users = {row[0] for row in cur.fetchall()}

        for user_plan in (plans or {}).get("recommendation_plans", []):
            user_id = user_plan.get("user_id")
            items = user_plan.get("plans", []) or []
            if not user_id or not items:
                continue

            if user_id not in known_users:
                logging.warning(
                    "DS810 skip recommendation_plan: unknown user (run_id=%s, user_id=%s)", self.run_id, user_id
                )
                continue

            # Group items by plan_id for this user
            by_pid = {}
            for it in items:
                pid = it.get("plan_id")
                if not pid:
                    logging.warning("Skipping item without plan_id for user %s: %r", user_id, it)
                    continue
                by_pid.setdefault(pid, []).append(it)

            for pid, group in by_pid.items():
                # Upsert the plan row
                cur.execute(
                    """
                    INSERT INTO recommendation_plans (run_id, user_id, plan_id)
                    VALUES (%s, %s, %s)
                    ON CONFLICT (run_id, plan_id) DO NOTHING;
                    """,
                    (self.run_id, user_id, pid),
                )
                # Count as applied if we inserted OR it already existed (idempotent)
                applied += 1

                # Insert contents for this plan_id
                for c in group:
                    cur.execute(
                        """
                        INSERT INTO plan_contents (run_id, plan_id, content_id, scheduled_for)
                        VALUES (%s, %s, %s, %s);
                        """,
                        (self.run_id, pid, c.get("content_id"), c.get("scheduled_for")),
                    )

        conn.commit()
        cur.close()
        conn.close()
        return applied

    # for simplicity, we assume 1 mission per user
    def add_selected_contents(self, selections):
        conn = self._get_conn()
        cur = conn.cursor()
        ts = selections["timestamp"]
        user_to_mission_id = selections["mission_id"]
        for user_id, items in selections["selected_contents"].items():
            cur.execute(
                "INSERT INTO selected_contents"
                " (run_id, user_id, plan_id, timestamp, mission_start_time, content_ids, mission_id)"
                " VALUES (%s, %s, %s, %s, %s, %s, %s)"
                " ON CONFLICT (run_id, user_id, plan_id) DO NOTHING;",
                (
                    self.run_id,
                    user_id,  # string
                    items["plan_id"],  # string
                    ts,
                    items["mission_start_time"],
                    [item["id"] for item in items["contents"]],  # TEXT[] in schema
                    user_to_mission_id.get(user_id),
                ),
            )
        conn.commit()
        cur.close()
        conn.close()

    def add_user_feedback(self, feedback):
        t0 = time.time()
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
                        user_id,  # string
                        event["process_id"],
                        event["timestamp"],
                        event["event_name"],
                        Json(event["properties"]),
                    ),
                )
        conn.commit()
        cur.close()
        conn.close()

    def add_feature_audit(self, audit_row: dict):
        """
        Persist a snapshot of how we reconstructed the feature vector at rating time.
        This is used for offline debugging / fairness / reproducibility.

        Expected keys in audit_row (all already built in handle_recommendation_rating):
          user_id, mission_id, rec_id,
          process_id, slot_idx, prompted,
          sel_ts, finish_ts, rating_ts,
          reward, fv,
          planned_so_far,
          recent_exposure_history_for_slot,
          available_recs_at_sel_ts,
          sent_seq
        """
        t0 = time.time()

        conn = self._get_conn()
        cur = conn.cursor()

        cur.execute(
            """
            INSERT INTO feature_audit (
                run_id,
                user_id,
                mission_id,
                rec_id,
                process_id,
                slot_idx,
                prompted,
                sel_ts,
                finish_ts,
                rating_ts,
                reward,
                fv,
                planned_so_far,
                recent_exposure_history_for_slot,
                available_recs_at_sel_ts,
                sent_seq,
                context_snapshot
            )
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s);
            """,
            (
                self.run_id,
                audit_row.get("user_id"),
                audit_row.get("mission_id"),
                audit_row.get("rec_id"),
                audit_row.get("process_id"),
                audit_row.get("slot_idx"),
                audit_row.get("prompted"),
                audit_row.get("sel_ts"),
                audit_row.get("finish_ts"),
                audit_row.get("rating_ts"),
                audit_row.get("reward"),
                self._to_float_array(audit_row.get("fv")),
                json.dumps(sanitize_for_json(audit_row.get("planned_so_far"))),
                json.dumps(sanitize_for_json(audit_row.get("recent_exposure_history_for_slot"))),
                json.dumps(sanitize_for_json(audit_row.get("available_recs_at_sel_ts"))),
                json.dumps(sanitize_for_json(audit_row.get("sent_seq"))),
                json.dumps(sanitize_for_json(audit_row.get("context_snapshot"))),
            ),
        )

        conn.commit()
        cur.close()
        conn.close()

        dt = time.time() - t0
        self.log.info("PERF DS add_feature_audit: took=%.3fs", dt)

    def _to_float_array(self, x):
        if x is None:
            return None
        if isinstance(x, np.ndarray):
            return [float(v) for v in x.tolist()]
        if isinstance(x, (list, tuple)):
            return [float(v) for v in x]
        if isinstance(x, set):
            raise TypeError("feature_vector was a set (unordered). Pass list/tuple to preserve feature order.")
        try:
            return [float(x)]
        except Exception:
            raise TypeError(f"feature_vector must be sequence of numbers, got: {type(x).__name__}")
