# =============================================================================
# RUNBOOK — CS Module Airflow DAGs
#
# OVERVIEW
# --------
# We have a staged boot sequence. We rebuild state, learn from history,
# import anything OMI already sent, optionally force one plan push, and
# then go live. The "backfill" DAG is only meant to run temporarily,
# otherwise it would overlap with the live DAG.
#
# DAGS
# ----
# - cs_module_00_static_load                (manual, one-off per rebuild)
#   Loads static catalogs (missions / resources / recommendations) into CS.
#
# - cs_module_01_align_no_learn_no_intervene (manual, one-off, optional)
#   Replays all historical updates with:
#       is_learning = False
#       is_intervention = False
#   Uses FROZEN time. Mostly useful if you want CS to mirror past state
#   deterministically before training; you can skip this in many rebuilds.
#
# - cs_module_02_backfill_learn_no_intervene (hourly, FROZEN, catchup=True)
#   Replays hourly windows from BACKFILL_START → NOW-1h.
#   Sends updates to CS with:
#       is_learning = True          (update bandits from historical ratings)
#       is_intervention = False     (DO NOT generate historical plans)
#   This DAG is PAUSED by default. You UNPAUSE it, let it fully catch up,
#   then PAUSE it again. You do NOT leave it running alongside live.
#
#   NOTE on dual-bandit versioning:
#   During replay, historical events carry no new_version flag, so only the
#   old bandits are updated. Once the replay crosses CS_APP_RELEASE_DATE,
#   events with new_version=true will start upgrading users and training
#   the new bandits. No manual intervention needed.
#
# - cs_module_03_bootstrap_omi_once          (manual, one-off)
#   Pulls previously generated selected_contents from OMI and pushes them
#   into CS via /omi_fetched_contents. This is necessary because CS cannot
#   retrieve its own past plans from EUT — only OMI has them. Without this
#   step, CS would re-plan missions that OMI has already delivered.
#
# - cs_module_035_force_intervene_once       (manual, one-off)
#   Forces CS to produce a fresh plan "now":
#       is_learning = False
#       is_intervention = True
#   Uses REAL time. This is basically: "give me recommendations for
#   currently open missions immediately".
#
# - cs_module_04_live_learn_intervene        (hourly, REAL, catchup=False)
#   Normal live operation going forward:
#       is_learning = True
#       is_intervention = True
#   Learns from new feedback and emits new plans for new/ongoing missions.
#   This stays UNPAUSED in production.
#
#
# DEPLOYMENT WITH DUAL RATING VERSION (app update rollout)
# --------------------------------------------------------
# Use this sequence when deploying a new module version that introduces
# parallel old/new bandit sets for different app versions.
#
# Pre-requisites (before touching Airflow):
#   a) Download DB backup from production server.
#   b) Pause cs_module_04_live_learn_intervene.
#   c) Deploy new cs_module container image.
#   d) Set CS_APP_RELEASE_DATE env var (ISO UTC) to the timestamp when
#      the new app version was published to the store. This tells the
#      module which users default to the new bandit set at enrolment.
#      Example: CS_APP_RELEASE_DATE=2026-05-07T10:00:00Z
#   e) Set CS_CATCH_UP_LEARNING_DATE to the study start (or the earliest
#      date from which you want to retrain the bandits).
#
# Boot sequence:
#   1) Trigger cs_module_00_static_load
#      - Loads missions/resources/recommendations into CS.
#
#   2) Trigger cs_module_01_align_no_learn_no_intervene
#      - Replays all updates from CS_STUDY_START_UTC to CS_CATCH_UP_LEARNING_DATE
#        with no learning and no intervention, reconstructing user/mission state.
#      - Users enrolled before CS_APP_RELEASE_DATE are created as "old".
#        Users enrolled after are created as "new". No learning yet.
#
#   3) Unpause cs_module_02_backfill_learn_no_intervene
#      - Replays hourly windows with learning=True, intervention=False.
#      - Windows before CS_APP_RELEASE_DATE: old bandits trained only.
#      - Windows after CS_APP_RELEASE_DATE: users with new_version=true on
#        their events get upgraded; their ratings train the new bandits.
#      - Let it drain fully, then PAUSE it again.
#
#   4) Trigger cs_module_03_bootstrap_omi_once
#      - Imports plans OMI already delivered so CS does not re-plan
#        missions that have already been served.
#
#   5) Trigger cs_module_035_force_intervene_once
#      - Generates fresh plans immediately for all currently open missions.
#
#   6) Unpause cs_module_04_live_learn_intervene
#      - Resumes normal hourly operation.
#
#
# TYPICAL TEST / CLEAN REBUILD FLOW (no app update involved)
# ----------------------------------------------------------
# You do this anytime you want a clean restart of CS state after changing code:
#
#   1) Trigger cs_module_00_static_load
#   2) Trigger cs_module_01_align_no_learn_no_intervene  (optional)
#   3) Unpause cs_module_02_backfill_learn_no_intervene; let it drain; Pause
#   4) Trigger cs_module_03_bootstrap_omi_once
#   5) Trigger cs_module_035_force_intervene_once
#   6) Unpause cs_module_04_live_learn_intervene
#
#
# ROUTINE GAP RECOVERY (SHORT OUTAGE)
# -----------------------------------
# If live was running, something crashed for a few hours, and you just want
# to fill the hole:
#
#   1) Adjust CS_CATCH_UP_LEARNING_DATE / BACKFILL_START_DT so that
#      cs_module_02_backfill_learn_no_intervene covers the missed window.
#
#   2) Unpause cs_module_02_backfill_learn_no_intervene
#      - Let it run to catch up bandit learning for that gap (no intervention).
#
#   3) Pause cs_module_02_backfill_learn_no_intervene again
#      - Do NOT leave it on.
#
#   4) Leave cs_module_04_live_learn_intervene unpaused to resume normal ops.
#
#
# TUNABLES / VARIABLES
# --------------------
# - CS_CATCH_UP_LEARNING_DATE:
#       Earliest timestamp for backfill learning. Drives BACKFILL_START.
#
# - CS_STUDY_START_UTC:
#       Start-of-study floor used by the optional align DAG.
#
# - CS_APP_RELEASE_DATE (env var on cs_module container, not Airflow):
#       ISO UTC timestamp of when the new app version hit the store.
#       Users enrolled at or after this date default to the new bandit set.
#       Users enrolled before default to old. Absent = all users are "old".
#
# - BINDER_PROC_CAP:
#       Max ProcessBinder cache size in CS (default 200000).
#
# NOTES
# -----
# - cs_module_02_backfill_learn_no_intervene should never run at the same
#   time as cs_module_04_live_learn_intervene. Treat it like a one-time
#   catch-up job you manually unpause, let drain, then pause.
#
# - This runbook intentionally keeps CS in "FROZEN" clock mode while
#   replaying history, then switches to REAL time for live mode.
# =============================================================================

from __future__ import annotations
from functools import partial
from datetime import datetime, timedelta, timezone
import json, logging, requests, time

from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.models import Variable
from airflow.utils import timezone as aftz

# ────────────────────────────────────────────────────────────────────────
# Config & helpers (unchanged)
# ────────────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)sZ %(levelname)s %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)
logging.Formatter.converter = time.gmtime  # enforce UTC

CATCH_UP_LEARNING_DATE = Variable.get("CS_CATCH_UP_LEARNING_DATE", default_var="2026-05-01T00:00:00Z")
try:
    BACKFILL_START_DT = datetime.fromisoformat(CATCH_UP_LEARNING_DATE.replace("Z", "+00:00"))
except Exception as e:
    raise ValueError(f"Invalid CS_CATCH_UP_LEARNING_DATE: {CATCH_UP_LEARNING_DATE!r}") from e

STUDY_START = Variable.get("CS_STUDY_START_UTC", default_var="2026-03-01T00:00:00Z")


def get_cfg():
    cs_base = Variable.get("CS_BASE_URL", default_var="http://cs_module:8000")
    eut_base = Variable.get("EUT_PRE_BASE_URL", default_var="")
    omi_base = Variable.get("OMI_BASE_URL", default_var="")
    token = Variable.get("EUT_PRE_TOKEN", default_var="")
    return {
        "CS_BASE": cs_base,
        "OMI_BASE": omi_base,
        "EUT_BASE": eut_base,
        "HEADERS_EUT": {"Authorization": f"Basic {token}"},
        "HEADERS_OMI": {"Authorization": f"Basic {token}"},
        "HEADERS_CS": {"Content-Type": "application/json"},
    }


def _get_with_retry(url, headers, params=None, tries=5, delay=2.0):
    for i in range(tries):
        r = requests.get(url, headers=headers, params=params or {})
        if r.status_code == 200:
            return r
        logging.warning("Retry %d/%d for GET %s — status %d", i + 1, tries, url, r.status_code)
        if i < tries - 1:
            time.sleep(delay * (2 ** i))  # exponential backoff: 2s, 4s, 8s, 16s
    return r  # last response


def _eut_get(path: str, params: dict | None = None):
    cfg = get_cfg()
    url = f"{cfg['EUT_BASE']}{path}"
    logging.info("GET %s  params=%s", url, params or {})
    r = _get_with_retry(url, cfg["HEADERS_EUT"], params=params)
    if r.status_code != 200:
        raise RuntimeError(f"EUT GET {path} failed after retries: {r.status_code} {r.text}")
    return r.json()


def _omi_get(path: str, params: dict | None = None):
    cfg = get_cfg()
    if not cfg["OMI_BASE"]:
        logging.info("OMI_BASE not set; skipping %s", path)
        return None
    url = f"{cfg['OMI_BASE']}{path}"
    logging.info("GET %s  params=%s", url, params or {})
    r = _get_with_retry(url, cfg["HEADERS_OMI"], params=params)
    if r.status_code != 200:
        raise RuntimeError(f"OMI GET {path} failed after retries: {r.status_code} {r.text}")
    return r.json()


def _cs_get(path: str, params: dict | None = None):
    cfg = get_cfg()
    url = f"{cfg['CS_BASE']}{path}"
    logging.info("GET %s  params=%s", url, params or {})
    r = _get_with_retry(url, cfg["HEADERS_CS"], params=params)
    if r.status_code != 200:
        raise RuntimeError(f"CS GET {path} failed: {r.status_code} {r.text}")
    return r.json()


def _cs_post(endpoint: str, payload, extra_params: dict | None = None):
    cfg = get_cfg()
    url = f"{cfg['CS_BASE']}{endpoint}"
    if extra_params:
        payload = {"data": payload, **extra_params}
    logging.info("POST %s  bytes=%d", url, len(json.dumps(payload)))
    r = requests.post(url, json=payload, headers=cfg["HEADERS_CS"])
    if r.status_code not in (200, 201):
        raise RuntimeError(f"CS POST {endpoint} failed: {r.status_code} {r.text}")
    try:
        return r.json()
    except Exception:
        return None


def utc_iso(dt: datetime) -> str:
    if dt.tzinfo is None:
        raise ValueError("Naïve datetime received – timezone info required")
    return dt.astimezone(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def _prepare_cs_clock(window_end_utc: datetime, *, time_mode: str):
    time_mode = (time_mode or "REAL").upper()
    _cs_post("/set_time_mode", {"mode": time_mode})
    if time_mode == "FROZEN":
        _cs_post("/set_current_time", utc_iso(window_end_utc))


# ────────────────────────────────────────────────────────────────────────
# Tasks
# ────────────────────────────────────────────────────────────────────────


def fetch_static_objects():
    mappings = {
        "/recommendations": "/internal/recommendations?resources=false",
        "/resources": "/internal/recommendations?resources=true",
        "/missions": "/missions",
    }
    for endpoint, path in mappings.items():
        data = _eut_get(path)
        if data:
            _cs_post(endpoint, data)
        else:
            logging.warning("No data for %s", endpoint)


def _fetch_updates_window(start_dt: datetime, end_dt: datetime, *, is_learning: bool, is_intervention: bool):
    params = {"start_date": utc_iso(start_dt), "end_date": utc_iso(end_dt)}
    data = _eut_get("/updates", params=params)
    if data:
        _cs_post(
            "/updates",
            data,
            extra_params={
                "is_learning": is_learning,
                "is_intervention": is_intervention,
                "window_end": utc_iso(end_dt),
            },
        )
    else:
        logging.info("No updates for window %s → %s", params["start_date"], params["end_date"])


def _fetch_omi_recommendation_plans_window(start_dt: datetime, end_dt: datetime):
    """
    Pull recommendation plans from OMI for [start_dt, end_dt) and forward to CS.
    Uses start_time/end_time as requested.
    Returns number of plans applied on CS (if CS returns 'applied').
    """
    params = {"start_time": utc_iso(start_dt), "end_time": utc_iso(end_dt)}
    data = _omi_get("/recommendation_plans", params=params)
    if not data:
        logging.info("No OMI recommendation_plans for window %s → %s", params["start_time"], params["end_time"])
        return 0
    # POST to CS — choose the endpoint you exposed on CS; here we use /recommendation_plans
    res = _cs_post("/recommendation_plans", data)
    applied = (res or {}).get("applied", 0)
    logging.info("OMI recommendation_plans ingested: applied=%s", applied)
    return applied


def align_past_updates(*, time_mode: str):
    cursor = datetime.fromisoformat(STUDY_START.replace("Z", "+00:00"))
    stop_at = BACKFILL_START_DT

    while cursor < stop_at:
        window_end = min(cursor + timedelta(days=2), stop_at)
        _prepare_cs_clock(window_end, time_mode=time_mode)
        _fetch_updates_window(cursor, window_end, is_learning=False, is_intervention=False)
        _fetch_omi_recommendation_plans_window(cursor, window_end)
        cursor = window_end
        time.sleep(2)


def hourly_update(*, time_mode: str, is_learning: bool, is_intervention: bool, **context):
    end = context["logical_date"].astimezone(timezone.utc)
    start = (end - timedelta(hours=1)).astimezone(timezone.utc)
    _prepare_cs_clock(end, time_mode=time_mode)
    logging.info("INTERVAL %s → %s", start.isoformat(), end.isoformat())
    _fetch_updates_window(start, end, is_learning=is_learning, is_intervention=is_intervention)
    _fetch_omi_recommendation_plans_window(start, end)


def fetch_selected_contents(*, time_mode: str, **context):
    end = context["logical_date"].astimezone(timezone.utc)
    start = (end - timedelta(hours=1)).astimezone(timezone.utc)
    _prepare_cs_clock(end, time_mode=time_mode)
    params = {"start_time": utc_iso(start), "end_time": utc_iso(end)}
    data = _cs_get("/selected_contents", params=params)
    logging.info("Retrieved selected_contents payload:\n%s", json.dumps(data, indent=2))


# ────────────────────────────────────────────────────────────────────────
# DAG 0 — Static load only (manual one-off)
# ────────────────────────────────────────────────────────────────────────
with DAG(
    dag_id="cs_module_00_static_load",
    schedule=None,
    catchup=False,
    is_paused_upon_creation=True,
    tags=["cs_module", "static", "no_learn", "no_intervene"],
    default_args=dict(owner="POLIMI", start_date=datetime(2025, 1, 1, tzinfo=timezone.utc), retries=0),
) as dag_static:
    PythonOperator(task_id="fetch_static_objects", python_callable=fetch_static_objects)

# ────────────────────────────────────────────────────────────────────────
# DAG 1 — Alignment (optional; manual one-off; no learn / no intervene)
# ────────────────────────────────────────────────────────────────────────
with DAG(
    dag_id="cs_module_01_align_no_learn_no_intervene",
    schedule=None,
    catchup=False,
    is_paused_upon_creation=True,
    tags=["cs_module", "align", "frozen", "no_learn", "no_intervene"],
    default_args=dict(owner="POLIMI", start_date=datetime(2025, 1, 1, tzinfo=timezone.utc), retries=0),
) as dag_align:
    t_align = PythonOperator(
        task_id="align_past_updates", python_callable=partial(align_past_updates, time_mode="FROZEN")
    )

# ────────────────────────────────────────────────────────────────────────
# DAG 2 — Backfill (learn / no intervene; hourly; FROZEN; catchup=True)
# ────────────────────────────────────────────────────────────────────────
NOW = aftz.utcnow().replace(minute=0, second=0, microsecond=0)
if BACKFILL_START_DT > NOW:
    raise ValueError(
        f"CS_CATCH_UP_LEARNING_DATE ({BACKFILL_START_DT.isoformat()}) "
        f"is in the future relative to NOW ({NOW.isoformat()})."
    )
BACKFILL_START = BACKFILL_START_DT
BACKFILL_END = NOW

with DAG(
    dag_id="cs_module_02_backfill_learn_no_intervene",
    start_date=BACKFILL_START,
    end_date=BACKFILL_END,
    schedule="0 * * * *",
    catchup=True,
    max_active_runs=1,
    is_paused_upon_creation=True,
    tags=["cs_module", "backfill", "frozen", "learn", "no_intervene"],
    default_args=dict(owner="POLIMI", retries=0, retry_delay=timedelta(minutes=5)),
) as dag_backfill:
    t_updates = PythonOperator(
        task_id="hourly_update",
        python_callable=partial(hourly_update, time_mode="FROZEN", is_learning=True, is_intervention=False),
        depends_on_past=True,
        max_active_tis_per_dag=1,
    )
    t_selected = PythonOperator(
        task_id="fetch_selected_contents",
        python_callable=partial(fetch_selected_contents, time_mode="FROZEN"),
        depends_on_past=True,
    )
    t_updates >> t_selected


# ────────────────────────────────────────────────────────────────────────
# DAG 3 — Bootstrap selected contents from OMI
# ────────────────────────────────────────────────────────────────────────
with DAG(
    dag_id="cs_module_03_bootstrap_omi_once",
    schedule=None,
    catchup=False,
    is_paused_upon_creation=True,
    tags=["cs_module", "bootstrap", "omi"],
    default_args=dict(owner="POLIMI", start_date=datetime(2025, 1, 1, tzinfo=timezone.utc), retries=0),
) as dag_bootstrap:

    def _bootstrap_omi_once():
        # get everything OMI has (no window)
        data = _omi_get("/selected_contents", params=None)
        if not data:
            logging.info("OMI returned no selected_contents; nothing to ingest.")
        else:
            _cs_post("/omi_fetched_contents", data)

    PythonOperator(task_id="bootstrap_omi_selected_contents_once", python_callable=_bootstrap_omi_once)


# ────────────────────────────────────────────────────────────────────────
# DAG 3.5 — Force intervention (no learn / intervene; REAL; catchup=False)
# ────────────────────────────────────────────────────────────────────────
with DAG(
    dag_id="cs_module_035_force_intervene_once",
    schedule=None,
    catchup=False,
    is_paused_upon_creation=True,
    tags=["cs_module", "force", "real", "intervene", "once"],
    default_args=dict(owner="POLIMI", start_date=datetime(2025, 1, 1, tzinfo=timezone.utc), retries=0),
) as dag_force_intervene:

    def _force_intervene_once():
        # Use REAL time and nudge CS clock to "now" so selection windows are correct.
        now_utc = datetime.now(timezone.utc).replace(second=0, microsecond=0)
        _prepare_cs_clock(now_utc, time_mode="REAL")

        # Trigger planning without learning, immediately:
        # - empty updates payload
        # - is_intervention=True → CS will call _select_contents()
        # - is_learning=False → no MAB updates attempted here
        _cs_post(
            "/updates",
            {"_noop": True},
            extra_params={"is_learning": False, "is_intervention": True, "window_end": utc_iso(now_utc)},
        )

    PythonOperator(task_id="force_intervene_once", python_callable=_force_intervene_once)


# ────────────────────────────────────────────────────────────────────────
# DAG 4 — Live (learn / intervene; hourly; REAL; catchup=False)
# ────────────────────────────────────────────────────────────────────────
NEXT_HOUR = (aftz.utcnow() + timedelta(hours=1)).replace(minute=0, second=0, microsecond=0)
with DAG(
    dag_id="cs_module_04_live_learn_intervene",
    start_date=NEXT_HOUR,
    schedule="0 * * * *",
    catchup=False,
    max_active_runs=1,
    is_paused_upon_creation=True,
    tags=["cs_module", "live", "real", "learn", "intervene"],
    default_args=dict(owner="POLIMI", retries=3, retry_delay=timedelta(minutes=5)),
) as dag_live:
    t_updates = PythonOperator(
        task_id="hourly_update",
        python_callable=partial(hourly_update, time_mode="REAL", is_learning=True, is_intervention=True),
        depends_on_past=False,
        max_active_tis_per_dag=1,
    )
    t_selected = PythonOperator(
        task_id="fetch_selected_contents",
        python_callable=partial(fetch_selected_contents, time_mode="REAL"),
        depends_on_past=False,
    )
    t_updates >> t_selected
