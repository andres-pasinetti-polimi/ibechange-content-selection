from flask import Flask, jsonify, request
from virtual_user.virtual_user import VirtualUser
from cs_module.services.time_handler import TimeHandler
from virtual_user.services.feedback_manager import FeedbackManager
import logging
from numpy import random
import numpy as np

logging.basicConfig(level=logging.WARNING, format="%(asctime)s - %(levelname)s - %(message)s")
logging.getLogger("werkzeug").setLevel(logging.ERROR)

time_handler = TimeHandler()
service = VirtualUser(time_handler)
app = Flask(__name__)


@app.route("/run_seed", methods=["POST"])
def seed_endpoint():
    body = request.get_json(silent=True) or {}
    s = int(body.get("seed", 0))
    random.seed(s)
    np.random.seed(s)
    FeedbackManager.process_count = 0
    return jsonify({"seed": s}), 200


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
            "start_time": time_handler.utc_iso(time_handler.start_time),
            "mode": time_handler.mode,
            "now": time_handler.utc_iso(time_handler.now),
        }
    ), 200


@app.route("/set_current_time", methods=["POST"])
def set_current_time():
    """Only moves time in FROZEN mode (no-op in REAL)."""
    iso = request.get_json(silent=True)
    dt = time_handler.parse_client_ts(iso)
    if time_handler.mode == "FROZEN":
        time_handler.set(dt)
        return jsonify({"current_time": time_handler.utc_iso(time_handler.now), "mode": "FROZEN"}), 200
    else:
        # accept but do nothing (keeps code paths simple for simulator)
        logging.info("set_current_time called in REAL mode; ignoring.")
        return jsonify({"current_time": time_handler.utc_iso(time_handler.now), "mode": "REAL"}), 200


@app.route("/reset_counters", methods=["POST"])
def reset_counters():
    from virtual_user.services.feedback_manager import FeedbackManager

    FeedbackManager.process_count = 0
    return jsonify({"process_count": 0}), 200


@app.route("/recommendations", methods=["GET"])
def get_recommendations():
    return jsonify(service.raw_recommendations), 200


@app.route("/resources", methods=["GET"])
def get_resources():
    return jsonify(service.raw_resources), 200


@app.route("/missions", methods=["GET"])
def get_missions():
    return jsonify(service.raw_missions), 200


@app.route("/updates", methods=["GET"])
def get_updates():
    service.simulate_hour()
    return jsonify(service.get_updates()), 200


@app.route("/recommendation_plans", methods=["POST"])
def save_weekly_recommendation_plans():
    data = request.get_json(silent=True) or {}
    response = service.save_weekly_recommendation_plans(data)
    return jsonify(response), 201


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
