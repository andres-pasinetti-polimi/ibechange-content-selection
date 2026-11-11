import logging
import json
from flask import Flask, jsonify, request
from cs_module.content_selection.core import ContentSelection
import datetime
from cs_module.services.time_handler import TimeHandler


# ---- Setup Logging ----
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s", force=True)
logging.getLogger("werkzeug").setLevel(logging.ERROR)  # Silence Werkzeug request logs

# Per-module logging levels
logging.getLogger(__name__).setLevel(logging.WARNING)
logging.getLogger("content_selection.mab_updater").setLevel(logging.WARNING)
logging.getLogger("content_selection.user_manager").setLevel(logging.WARNING)


app = Flask(__name__)


# Global Variables
time_handler = TimeHandler()

content_selection = ContentSelection(
    time_handler=time_handler,
)
recommendations_to_send = {}
resources_to_send = {}


# ---- API Routes ----
@app.route("/set_start_time", methods=["POST"])
def set_start_time():
    """Get the current time from the time service."""
    iso_time_str = request.json
    try:
        dt = datetime.datetime.fromisoformat(iso_time_str)
    except ValueError:
        return (
            jsonify({"error": "Invalid datetime format. Use ISO 8601 (e.g., 2025-04-13T12:00:00)"}),
            400,
        )

    time_handler.set_start_time(dt)
    return jsonify({"start_time": dt.isoformat()}), 200


@app.route("/set_current_time", methods=["POST"])
def set_current_time():
    """Get the current time from the time service."""
    iso_time_str = request.json
    try:
        dt = datetime.datetime.fromisoformat(iso_time_str)
    except ValueError:
        return (
            jsonify({"error": "Invalid datetime format. Use ISO 8601 (e.g., 2025-04-13T12:00:00)"}),
            400,
        )

    time_handler.set(dt)
    return jsonify({"current_time": dt.isoformat()}), 200


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
    logging.info("Received request at /updates")

    # 1️⃣  Parse JSON body (empty dict if missing/invalid)
    body = request.get_json(silent=True) or {}

    # 2️⃣  Look for mode flag first in query-string, then in JSON, default to 'intervention'
    mode = request.args.get("mode") or body.get("mode", "intervention")

    # 3️⃣  Actual updates payload lives in body["data"] if present,
    #     otherwise assume the whole body *is* the payload
    payload = body.get("data", body)

    if not payload:
        logging.error("Invalid JSON data received at /updates")
        return jsonify({"error": "Invalid JSON data"}), 400

    # 4️⃣  Update your in-memory store
    global content_selection
    content_selection.update(payload, mode)

    logging.info("Updates successfully processed (mode=%s)", mode)
    return jsonify({"message": "Updates received"}), 201


@app.route("/selected_contents", methods=["GET"])
def selected_contents_endpoint():
    logging.info("Received request at /selected_contents")
    selected_contents = content_selection.get_selected_contents()
    logging.info(f"Selected contents: {selected_contents}")
    return jsonify(selected_contents), 200


@app.route("/recommendation_plans", methods=["POST"])
def recommendation_plans_endpoint():
    logging.info("Received request at /recommendation_plans")
    data = request.json
    if not data:
        logging.error("Invalid JSON data received at /recommendation_plans")
        return jsonify({"error": "Invalid JSON data"}), 400

    # Save feedback to a file
    with open("recommendation_plans.json", "w") as file:
        json.dump(data, file, indent=4)
    logging.info("Recommendation plans saved to recommendation_plans.json")

    response = content_selection.save_recommendation_plans(data)
    logging.info("Recommendation plans successfully processed")
    return jsonify(response), 201


if __name__ == "__main__":
    logging.info("Starting CS API service...")
    app.run(host="0.0.0.0", port=8000)
