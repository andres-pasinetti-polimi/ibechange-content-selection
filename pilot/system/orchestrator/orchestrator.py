import time
import requests
import logging
import os
from datetime import datetime, timedelta
from orchestrator.config_generator import generate_preferences, generate_config
from tqdm import tqdm
from dateutil import parser as dtp

logging.basicConfig(level=logging.WARNING, format="%(asctime)s - %(levelname)s - %(message)s")
logging.getLogger("werkzeug").setLevel(logging.ERROR)


# ----------------------------------------------------------------------
# Environment / world / policy configuration
# ----------------------------------------------------------------------

# High-level “environment”: describes how this world is built (catalogs, config profile, etc.)
ENV_NAME = os.environ.get("ENV_NAME", "baseline_v1")

# World index within this environment (0, 1, 2, …)
WORLD_ID = int(os.environ.get("WORLD_ID", "0"))

# Seed that defines this world (preferences, etc.); defaults to WORLD_ID
WORLD_SEED = int(os.environ.get("WORLD_SEED", str(WORLD_ID)))

# Root where we keep generated configs and preferences *inside* the orchestrator project
ENV_ROOT_DIR = os.environ.get("ENV_ROOT_DIR", "/app/generated_configs")

# World-local folders THIS orchestrator controls
WORLD_DIR = os.path.join(
    ENV_ROOT_DIR,
    ENV_NAME,
    f"world_{WORLD_ID:03d}",
)
WORLD_PREF_DIR = os.path.join(WORLD_DIR, "user_preferences")
WORLD_CONFIG_PATH = os.path.join(WORLD_DIR, "config.py")

os.makedirs(WORLD_PREF_DIR, exist_ok=True)

# Docker Compose project prefix (used to infer container names)
PROJECT_PREFIX = os.environ.get("PROJECT_PREFIX", f"{ENV_NAME}_s{WORLD_ID}")

CONTAINER_NAMES = [
    f"{PROJECT_PREFIX}-time_service-1",
    f"{PROJECT_PREFIX}-vu_api-1",
    f"{PROJECT_PREFIX}-cs_module-1",
    f"{PROJECT_PREFIX}-omi_module-1",
]


# Which “policies” (bandit regimes) to run in this world.
# You can override via env: MAB_TYPES="Random,TwoStageFull"
if "MAB_TYPES" in os.environ:
    MAB_TYPES = [m.strip() for m in os.environ["MAB_TYPES"].split(",") if m.strip()]
else:
    MAB_TYPES = [
        # "Random",
        # "ContextBase",
        # "TwoStageBase",
        "TwoStageFull",
        # "Oracle",
    ]

# How long each policy is run in this world (in simulated hours)
HOURS_PER_POLICY = int(os.environ.get("HOURS_PER_POLICY", str(24 * 7 * 12)))  # 12 simulated weeks


# ---- Configuration ----
TIME_SERVICE_URL = "http://time_service:5000"
VU_API_URL = "http://vu_api:5000"
CS_MODULE_URL = "http://cs_module:8000"
OMI_MODULE_URL = "http://omi_module:5000"


# --- add near the top, with other constants ---
MODULES_WITH_TIME = [
    ("VU", f"{VU_API_URL}/set_time_mode"),
    ("CS", f"{CS_MODULE_URL}/set_time_mode"),
    ("OMI", f"{OMI_MODULE_URL}/set_time_mode"),
]


# ----------------------------------------------------------------------
# HTTP helpers
# ----------------------------------------------------------------------
def post_and_wait(endpoint, data, label="data", timeout=900):
    start_time = time.time()
    while time.time() - start_time < timeout:
        try:
            response = requests.post(endpoint, json=data, timeout=timeout)
            try:
                response.raise_for_status()
            except requests.exceptions.HTTPError as e:
                logging.warning(
                    "[post_and_wait] %s HTTP %s: %s",
                    label,
                    response.status_code,
                    response.text[:500],  # truncate to avoid huge logs
                )
                raise  # re-raise so we go to the outer except

            try:
                return response.json()
            except ValueError:
                logging.warning(
                    "[post_and_wait] %s returned non-JSON body: %r",
                    label,
                    response.text[:200],
                )
                return None

        except requests.exceptions.RequestException as e:
            logging.warning("[post_and_wait] %s RequestException: %s", label, e)
            time.sleep(0.5)

    logging.error("[post_and_wait] %s timed out after %ss", label, timeout)
    return None


def fetch_and_wait(endpoint, label="data", params=None, timeout=300):
    """
    GET `endpoint` until it succeeds or times-out.

    Parameters
    ----------
    endpoint : str
    label    : str   – pretty-name for log lines
    params   : dict  – query-string parameters (default None)
    timeout  : int   – overall max wait in seconds
    """
    # print(f"\n🔍 Fetching {label} from {endpoint} …")
    start_time = time.time()

    while time.time() - start_time < timeout:
        try:
            # ← pass params straight to requests
            response = requests.get(endpoint, params=params, timeout=timeout)
            response.raise_for_status()

            # print(f"✅ Fetched {label} successfully")
            data = response.json()
            # print(f"📦 {label} content: {data!s}")
            return data  # (no need to call .json() twice)

        except requests.exceptions.RequestException:
            pass

        time.sleep(0.5)

    # print(f"❌ Timeout while fetching {label} from {endpoint}")
    return None


# ----------------------------------------------------------------------
# Orchestration helpers
# ----------------------------------------------------------------------
def set_modules_time_mode(mode: str = "FROZEN"):
    """Tell each module to use the requested time mode."""
    for short, endpoint in MODULES_WITH_TIME:
        post_and_wait(endpoint, {"mode": mode}, label=f"{short} time mode")


def restart_containers(container_names):
    for name in container_names:
        # print(f"🔄 Restarting container: {name}")
        os.system(f"docker restart {name}")
    # print("✅ All containers restarted.\n")


def seed_vu_for_world(seed: int):
    post_and_wait(f"{VU_API_URL}/run_seed", {"seed": int(seed)}, label="VU seed")


def run_hourly_update(last_printed_day):
    """
    One full simulation “tick”:

      - ask time_service for current time
      - pull updates from VU
      - push updates into CS & OMI
      - pull selected contents from CS, forward to OMI
      - pull recommendation plans from OMI, forward to CS & VU
      - advance global time by 1h and propagate set_current_time to all modules
    """
    current_time_data = fetch_and_wait(f"{TIME_SERVICE_URL}/get_time", label="current time")
    if not current_time_data:
        return last_printed_day

    now = dtp.isoparse(current_time_data["now"])
    prev = now - timedelta(hours=1)

    current_day = now.date()
    if current_day != last_printed_day:
        # print(f"\n# =======================\n# Current_time: {current_time_data['now']}\n# =======================")
        last_printed_day = current_day

    updates = fetch_and_wait(f"{VU_API_URL}/updates", label="VU updates")
    if not updates:
        return last_printed_day

    post_and_wait(
        f"{CS_MODULE_URL}/updates",
        {"data": updates, "window_end": now.isoformat().replace("+00:00", "Z")},
        label="CS updates",
    )
    post_and_wait(f"{OMI_MODULE_URL}/updates", updates, label="OMI updates")

    params = {
        "start_time": prev.isoformat().replace("+00:00", "Z"),
        "end_time": now.isoformat().replace("+00:00", "Z"),
    }
    selected_contents = fetch_and_wait(f"{CS_MODULE_URL}/selected_contents", label="Selected contents", params=params)

    if selected_contents:
        post_and_wait(f"{OMI_MODULE_URL}/selected_contents", selected_contents, label="OMI selected contents")

    recommendation_plans = fetch_and_wait(f"{OMI_MODULE_URL}/recommendation_plans", label="Recommendation plans")
    if recommendation_plans:
        post_and_wait(f"{CS_MODULE_URL}/recommendation_plans", recommendation_plans, label="CS recommendation plans")
        post_and_wait(f"{VU_API_URL}/recommendation_plans", recommendation_plans, label="VU recommendation plans")

    advance_result = post_and_wait(f"{TIME_SERVICE_URL}/advance", {"hours": 1}, label="advance time")
    if not advance_result:
        return last_printed_day

    current_time_data = fetch_and_wait(f"{TIME_SERVICE_URL}/get_time", label="current time after advancing")
    if not current_time_data:
        return last_printed_day

    current_time = current_time_data["now"]
    post_and_wait(f"{VU_API_URL}/set_current_time", current_time, label="VU set_current_time")
    post_and_wait(f"{CS_MODULE_URL}/set_current_time", current_time, label="CS set_current_time")
    post_and_wait(f"{OMI_MODULE_URL}/set_current_time", current_time, label="OMI set_current_time")

    return last_printed_day


def initialize_modules(current_time):
    """
    1) Set start_time for all modules.
    2) Load static catalogs (recs/resources/missions) from VU into CS and OMI.
    """
    post_and_wait(f"{VU_API_URL}/set_start_time", current_time, label="VU start time")
    post_and_wait(f"{CS_MODULE_URL}/set_start_time", current_time, label="CS start time")
    post_and_wait(f"{OMI_MODULE_URL}/set_start_time", current_time, label="OMI start time")

    recommendations = fetch_and_wait(f"{VU_API_URL}/recommendations", label="recommendations")
    resources = fetch_and_wait(f"{VU_API_URL}/resources", label="resources")
    missions = fetch_and_wait(f"{VU_API_URL}/missions", label="missions")

    post_and_wait(f"{CS_MODULE_URL}/recommendations", recommendations, label="CS recommendations")
    post_and_wait(f"{CS_MODULE_URL}/resources", resources, label="CS resources")
    post_and_wait(f"{CS_MODULE_URL}/missions", missions, label="CS missions")

    post_and_wait(f"{OMI_MODULE_URL}/recommendations", recommendations, label="OMI recommendations")
    post_and_wait(f"{OMI_MODULE_URL}/resources", resources, label="OMI resources")
    post_and_wait(f"{OMI_MODULE_URL}/missions", missions, label="OMI missions")


# ----------------------------------------------------------------------
# Entry point: run one WORLD across multiple POLICIES
# ----------------------------------------------------------------------
if __name__ == "__main__":
    seed = WORLD_SEED

    print("========================================")
    print(f" ENV_NAME   : {ENV_NAME}")
    print(f" WORLD_ID   : {WORLD_ID}")
    print(f" WORLD_SEED : {seed}")
    print(f" WORLD_DIR  : {WORLD_DIR}")
    print(f" MAB_TYPES  : {MAB_TYPES}")
    print("========================================")

    # Preferences depend only on world seed, not policy:
    int_prefs, theta = generate_preferences(seed=seed, output_dir=WORLD_PREF_DIR)

    total_hours = len(MAB_TYPES) * HOURS_PER_POLICY
    progress = tqdm(total=total_hours, desc="Simulated hours", unit="h", dynamic_ncols=True)

    for mab_type in MAB_TYPES:
        print()
        print(f"🔄 ENV={ENV_NAME}  WORLD={WORLD_ID}  SEED={seed}  POLICY={mab_type}")

        # Re-write config.py in WORLD_DIR for this policy
        generate_config(
            int_prefs,
            theta,
            mab_type=mab_type,
            output_path=WORLD_CONFIG_PATH,
            # feature_profile is inferred from mab_type by config_generator
        )
        snap_dir = os.path.join(WORLD_DIR, "config_snapshots")
        os.makedirs(snap_dir, exist_ok=True)

        snap_path = os.path.join(snap_dir, f"config__{mab_type}.py")
        import shutil

        shutil.copyfile(WORLD_CONFIG_PATH, snap_path)

        # Restart all containers so cs_module picks up the new config & re-inits DataStorage
        time.sleep(1)
        restart_containers(CONTAINER_NAMES)
        time.sleep(2)

        # FROZEN time during simulation
        set_modules_time_mode("FROZEN")

        # Virtual user seeded with the same world seed
        seed_vu_for_world(seed)

        # Initial time setup
        current_time_data = fetch_and_wait(f"{TIME_SERVICE_URL}/get_time", label="initial time")
        current_time = current_time_data["now"]
        time.sleep(1)
        post_and_wait(f"{VU_API_URL}/reset_counters", {}, label="VU reset counters")
        initialize_modules(current_time)

        # Run simulation loop for this policy
        last_printed_day = None
        for _ in range(HOURS_PER_POLICY):
            last_printed_day = run_hourly_update(last_printed_day)
            progress.update(1)

    progress.close()
    print()
    print(f"🎉 World {WORLD_ID} (env={ENV_NAME}, seed={seed}) complete for policies: {MAB_TYPES}")
