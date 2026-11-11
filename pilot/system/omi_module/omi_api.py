# omi_api.py
from flask import Flask, jsonify, request
import logging
from datetime import datetime, timedelta, timezone
from omi_module.services.time_handler import TimeHandler
import os
import hashlib
import numpy as np

WORLD_SEED = int(os.getenv("WORLD_SEED", "0"))
ENV_NAME = os.getenv("ENV_NAME", "baseline_v1")

# ──────────────────────────────────────────────────────────────
# Flask & logging
# ──────────────────────────────────────────────────────────────
app = Flask(__name__)
logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logging.getLogger("werkzeug").setLevel(logging.ERROR)

# ──────────────────────────────────────────────────────────────
# Globals (simulated stores)
# ──────────────────────────────────────────────────────────────
time_handler = TimeHandler()
users: dict = {}
selected_contents: dict = {}
new_missions_and_contents: dict = {}


# ──────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────
def _seed_from(*parts: str) -> int:
    s = "|".join(parts).encode("utf-8")
    h = hashlib.blake2b(s, digest_size=8).digest()
    return int.from_bytes(h, "big", signed=False)


def generate_plan_timestamps(num: int, user_id: str, contents: list[dict], mission_start_iso: str) -> list[str]:
    """
    Deterministic timestamps over next 7 days conditional on the received payload.
    Coupling key does NOT depend on plan_id.
    """
    start_dt = time_handler.parse_client_ts(mission_start_iso)
    end_dt = start_dt + timedelta(days=7)
    anchor = time_handler.utc_iso(start_dt)

    seed = _seed_from("omi_sched", ENV_NAME, str(WORLD_SEED), user_id, anchor, str(num))
    rng = np.random.default_rng(seed)

    ts_float = np.sort(rng.uniform(start_dt.timestamp(), end_dt.timestamp(), size=num))
    return [time_handler.utc_iso(datetime.fromtimestamp(float(ts), tz=timezone.utc)) for ts in ts_float]


# ──────────────────────────────────────────────────────────────
# Health & time mode
# ──────────────────────────────────────────────────────────────
@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "mode": time_handler.mode, "now": time_handler.now.isoformat()}), 200


@app.route("/set_time_mode", methods=["POST"])
def set_time_mode():
    body = request.get_json(silent=True) or {}
    mode = body.get("mode")
    if not mode:
        return jsonify({"error": "Missing 'mode'. Use REAL | FROZEN"}), 400
    try:
        time_handler.set_mode(mode)
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    return jsonify({"mode": time_handler.mode}), 200


# ──────────────────────────────────────────────────────────────
# Time control
# ──────────────────────────────────────────────────────────────
@app.route("/set_start_time", methods=["POST"])
def set_start_time():
    iso_time_str = request.get_json(silent=True)
    if not isinstance(iso_time_str, str):
        return jsonify({"error": "Body must be an ISO 8601 string"}), 400
    try:
        dt = time_handler.parse_client_ts(iso_time_str)
    except ValueError:
        return jsonify({"error": "Invalid datetime format. Use ISO 8601 like 2025-04-13T12:00:00Z"}), 400

    time_handler.set_start_time(dt)
    return jsonify({"start_time": time_handler.utc_iso(dt)}), 200


@app.route("/set_current_time", methods=["POST"])
def set_current_time():
    iso_time_str = request.get_json(silent=True)
    if not isinstance(iso_time_str, str):
        return jsonify({"error": "Body must be an ISO 8601 string"}), 400
    try:
        dt = time_handler.parse_client_ts(iso_time_str)
    except ValueError:
        return jsonify({"error": "Invalid datetime format. Use ISO 8601 like 2025-04-13T12:00:00Z"}), 400

    time_handler.set(dt)  # effective only in FROZEN
    return jsonify({"current_time": time_handler.utc_iso(time_handler.now)}), 200


# ──────────────────────────────────────────────────────────────
# Data ingestion
# ──────────────────────────────────────────────────────────────
@app.route("/recommendations", methods=["POST"])
def recommendations_endpoint():
    if not request.get_json(silent=True):
        return jsonify({"error": "Invalid JSON data"}), 400
    return jsonify({"message": "Recommendations initialised"}), 201


@app.route("/resources", methods=["POST"])
def resources_endpoint():
    if not request.get_json(silent=True):
        return jsonify({"error": "Invalid JSON data"}), 400
    return jsonify({"message": "Resources initialised"}), 201


@app.route("/missions", methods=["POST"])
def missions_endpoint():
    if not request.get_json(silent=True):
        return jsonify({"error": "Invalid JSON data"}), 400
    return jsonify({"message": "Missions initialised"}), 201


@app.route("/updates", methods=["POST"])
def updates_endpoint():
    logging.info("Received request at /updates")
    data = request.get_json(silent=True) or {}
    if not data:
        return jsonify({"error": "Invalid JSON data"}), 400

    global users, new_missions_and_contents
    users.update(data.get("new_users", {}))
    new_missions_and_contents = data.get("new_missions_and_contents", {})
    logging.info("Daily updates successfully processed")
    return jsonify({"message": "Daily updates received"}), 201


# ──────────────────────────────────────────────────────────────
# Planning I/O
# ──────────────────────────────────────────────────────────────
@app.route("/selected_contents", methods=["POST"])
def selected_contents_endpoint():
    global selected_contents
    payload = request.get_json(silent=True)
    if not isinstance(payload, dict):
        return jsonify({"error": "Invalid JSON data"}), 400
    selected_contents = payload
    return jsonify({"message": "Selected contents received"}), 201


@app.route("/recommendation_plans", methods=["GET"])
def process_recommendation_plans():
    global selected_contents
    logging.info("Generating recommendation plans …")

    recommendation_plans = {"recommendation_plans": []}

    if not selected_contents:
        return jsonify(recommendation_plans), 200

    for user_id, items in selected_contents.items():
        contents = items.get("contents", [])
        mission_start = items.get("mission_start_time")
        plan_id = items.get("plan_id")

        ts_list = generate_plan_timestamps(
            len(contents),
            user_id=user_id,
            contents=contents,
            mission_start_iso=mission_start,
        )

        user_plan = {
            "user_id": user_id,
            "plans": [
                {
                    "plan_id": plan_id,  # 👈 critical for CS/DataStorage
                    "content_id": content["id"],
                    "type": content["type"],  # "recommendation" | "resource"
                    "mission_id": content["mission_id"],
                    "scheduled_for": ts_list[i],
                }
                for i, content in enumerate(contents)
            ],
        }

        recommendation_plans["recommendation_plans"].append(user_plan)

    selected_contents = {}
    logging.info("✅ Recommendation plans ready")
    return jsonify(recommendation_plans), 200


# ──────────────────────────────────────────────────────────────
# Entrypoint
# ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    logging.info("🚀 Starting OMI API server …")
    app.run(host="0.0.0.0", port=5000, debug=False)
