import logging
from flask import Flask, jsonify, request
from cs_module.content_selection.core import ContentSelection
from cs_module.services.time_handler import TimeHandler
import time
import os

# ---- Setup Logging ----
LOG_LEVEL = os.getenv("LOG_LEVEL", "DEBUG").upper()
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL, logging.DEBUG),
    format="%(asctime)s [%(levelname)s] %(message)s",
    force=True,
)
logging.Formatter.converter = time.gmtime  # all %(asctime)s in UTC

# Silence external noise
logging.getLogger("werkzeug").setLevel(logging.ERROR)
logging.getLogger("urllib3").setLevel(logging.WARNING)

# Optionally: only DEBUG inside cs_module, keep others at INFO
# logging.getLogger().setLevel(logging.INFO)
# logging.getLogger("cs_module").setLevel(logging.DEBUG)


app = Flask(__name__)


# Global Variables
time_handler = TimeHandler()

content_selection = ContentSelection(
    time_handler=time_handler,
)
recommendations_to_send = {}
resources_to_send = {}


def _as_bool(x, default=False):
    if isinstance(x, bool):
        return x
    if x is None:
        return default
    return str(x).strip().lower() in {"1", "true", "t", "yes", "y", "on"}


# ---- API Routes ----
@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "mode": time_handler.mode, "now": time_handler.now.isoformat()}), 200


# TimeHandler stays as you already have it (mode-aware)


@app.route("/set_time_mode", methods=["POST"])
def set_time_mode():
    body = request.get_json(silent=True) or {}
    mode = (body.get("mode") or "REAL").upper()
    time_handler.set_mode(mode)
    return jsonify({"mode": time_handler.mode, "now": time_handler.now.isoformat()}), 200


@app.route("/set_start_time", methods=["POST"])
def set_start_time():
    """Record an experiment epoch; if FROZEN, also move the cursor there."""
    iso = request.get_json(silent=True)
    dt = time_handler.parse_client_ts(iso)
    time_handler.set_start_time(dt)  # in FROZEN this ALSO sets current cursor
    return jsonify(
        {
            "start_time": iso,
            "mode": time_handler.mode,
            "now": time_handler.now.isoformat(),
        }
    ), 200


@app.route("/set_current_time", methods=["POST"])
def set_current_time():
    """Only moves time in FROZEN mode (no-op in REAL)."""
    iso = request.get_json(silent=True)
    dt = time_handler.parse_client_ts(iso)
    if time_handler.mode == "FROZEN":
        time_handler.set(dt)
        return jsonify({"current_time": iso, "mode": "FROZEN"}), 200
    else:
        # accept but do nothing (keeps code paths simple for simulator)
        logging.info("set_current_time called in REAL mode; ignoring.")
        return jsonify({"current_time": time_handler.now.isoformat(), "mode": "REAL"}), 200


@app.route("/recommendations", methods=["POST"])
def recommendations_endpoint():
    logging.info("Received request at /recommendations")
    recommendations = request.get_json()
    if not recommendations:
        logging.error("Invalid JSON data received at /recommendations")
        return jsonify({"error": "Invalid JSON data"}), 400

    global content_selection
    content_selection.initialise_recommendations(recommendations)
    logging.info("Recommendations successfully initialized")
    return jsonify({"message": "Recommendations initialised"}), 201


@app.route("/resources", methods=["POST"])
def resources_endpoint():
    logging.info("Received request at /resources")
    resources = request.get_json()
    if not resources:
        logging.error("Invalid JSON data received at /resources")
        return jsonify({"error": "Invalid JSON data"}), 400

    global content_selection
    content_selection.initialise_resources(resources)
    logging.info("Resources successfully initialized")
    return jsonify({"message": "Resources initialised"}), 201


@app.route("/missions", methods=["POST"])
def missions_endpoint():
    logging.info("Received request at /missions")
    missions = request.get_json()
    if not missions:
        logging.error("Invalid JSON data received at /missions")
        return jsonify({"error": "Invalid JSON data"}), 400

    global content_selection
    content_selection.initialise_missions(missions)
    logging.info("missions successfully initialized")
    return jsonify({"message": "missions initialised"}), 201


@app.route("/updates", methods=["POST"])
def updates_endpoint():
    logging.info("TIMESTAMP: %s", time_handler.now)
    logging.info("Received request at /updates")

    t0 = time.time()  # PERF: start total /updates timing

    # 1️⃣  Parse JSON body (empty dict if missing/invalid)
    body = request.get_json(silent=True) or {}

    # 2️⃣  Look for mode flag first in query-string, then in JSON
    is_intervention = _as_bool(request.args.get("is_intervention", body.get("is_intervention")), default=True)
    is_learning = _as_bool(request.args.get("is_learning", body.get("is_learning")), default=True)

    # 3️⃣  Actual updates payload lives in body["data"] if present,
    #     otherwise assume the whole body *is* the payload
    payload = body.get("data", body)

    if not payload:
        logging.error("Invalid JSON data received at /updates")
        return jsonify({"error": "Invalid JSON data"}), 400

    # 4️⃣  Update your in-memory store
    global content_selection

    t1 = time.time()
    window_end_raw = body.get("window_end") or request.args.get("window_end")
    window_end = time_handler.parse_client_ts(window_end_raw) if window_end_raw else time_handler.now
    content_selection.update(payload, is_learning=is_learning, is_intervention=is_intervention, window_end=window_end)
    t2 = time.time()

    logging.info(
        "PERF CS /updates: total=%.3fs, update()=%.3fs",
        t2 - t0,
        t2 - t1,
    )

    return jsonify({"message": "Updates received", "window_end": window_end.isoformat()}), 201


@app.route("/selected_contents", methods=["GET"])
def selected_contents_endpoint():
    logging.info("Received request at /selected_contents")

    start_time_str = request.args.get("start_time")
    end_time_str = request.args.get("end_time")

    start_time = None
    end_time = None

    try:
        if start_time_str:
            start_time = time_handler.parse_client_ts(start_time_str)
        if end_time_str:
            end_time = time_handler.parse_client_ts(end_time_str)
    except ValueError as e:
        logging.warning(f"Invalid datetime format: {e}")
        return jsonify({"error": "start_time and end_time must be in ISO 8601 format"}), 400

    # Get filtered selected contents
    selected_contents = content_selection.get_selected_contents(start_time, end_time)

    # logging.info(f"Selected contents: {selected_contents}")
    return jsonify(selected_contents), 200


@app.route("/omi_fetched_contents", methods=["POST"])
def omi_fetched_plans_endpoint():
    logging.info("Received request at /omi_fetched_contents")
    data = request.json
    if not data:
        logging.error("Invalid JSON data received at /omi_fetched_contents")
        return jsonify({"error": "Invalid JSON data"}), 400

    response = content_selection.get_omi_fetched_contents(data)
    logging.info("OMI fetched plans successfully processed")
    return jsonify(response), 201


@app.route("/recommendation_plans", methods=["POST"])
def recommendation_plans_endpoint():
    logging.info("Received request at /recommendation_plans")
    data = request.json
    if not data:
        logging.error("Invalid JSON data received at /recommendation_plans")
        return jsonify({"error": "Invalid JSON data"}), 400

    response = content_selection.save_recommendation_plans(data)
    logging.info("OMI recommendation plans successfully processed")
    return jsonify(response), 201


if __name__ == "__main__":
    logging.info("Starting CS API service...")
    app.run(host="0.0.0.0", port=8000)
