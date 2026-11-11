from flask import Flask, jsonify, request
import time
import threading
import requests
import logging
from datetime import datetime, timezone


# Initialize Flask app
app = Flask(__name__)

# Configure Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("omi_api.log"),
        logging.StreamHandler()
    ]
)

VU_API_URL = "http://virtual_user_api:5000"

status = {"status": "idle"}
status_lock = threading.Lock()

users = {}
content_to_send = {}
recommendation_plans = {}

# ---- Utility Functions ----
def update_status(new_status):
    """Thread-safe status update."""
    with status_lock:
        status["status"] = new_status
        logging.info(f"🔄 Status updated: {new_status}")

def fetch_json(endpoint, timeout=10):
    """Fetch JSON data from an API endpoint."""
    try:
        response = requests.get(endpoint, timeout=timeout)
        response.raise_for_status()
        logging.info(f"✅ Successfully fetched data from {endpoint}")
        return response.json()
    except requests.exceptions.Timeout:
        logging.error(f"⏳ Timeout while fetching data from {endpoint}")
    except requests.exceptions.RequestException as e:
        logging.error(f"❌ Failed to fetch data from {endpoint}: {e}")
    return None

def simulate_processing(time_to_sleep=2):
    """Simulates processing time."""
    update_status("processing")
    time.sleep(time_to_sleep)
    update_status("ready")

# ---- API Routes ----
@app.route("/status", methods=["GET"])
def get_status():
    return jsonify(status)

@app.route("/initialise", methods=["GET"])
def initialise_endpoint():
    """Starts the initialization process in the background."""
    if status["status"] == "initialising":
        return jsonify({"message": "Already initialising"}), 409
    
    update_status("initialising")
    threading.Thread(target=simulate_processing).start()
    return jsonify({"message": "Initialisation started"}), 202

def fetch_new_users():
    """Fetches new users from VU API."""
    global users
    update_status("retrieving new users")
    new_users = fetch_json(f"{VU_API_URL}/new_users")
    if new_users:
        users.update(new_users)
        update_status("ready")

@app.route("/new_users", methods=["GET"])
def new_users_endpoint():
    if status["status"] == "retrieving new users":
        return jsonify({"message": "Already retrieving new users"}), 409
    
    threading.Thread(target=fetch_new_users).start()
    return jsonify({"message": "Retrieving new users"}), 202

@app.route("/user_feedback", methods=["GET"])
def user_feedback_endpoint():
    if status["status"] == "retrieving user feedback":
        return jsonify({"message": "Already retrieving user feedback"}), 409
    
    threading.Thread(target=simulate_processing).start()
    return jsonify({"message": "Processing user feedback"}), 202

@app.route("/newly_selected_missions", methods=["GET"])
def newly_selected_missions_endpoint():
    if status["status"] == "retrieving newly selected missions":
        return jsonify({"message": "Already retrieving newly selected missions"}), 409
    
    threading.Thread(target=simulate_processing).start()
    return jsonify({"message": "Processing newly selected missions"}), 202

@app.route("/timeslots", methods=["GET"])
def timeslots_endpoint():
    logging.info("Received request: /timeslots")
    timeslots = {user_id: 7 for user_id in users.keys()}
    return jsonify(timeslots)

@app.route("/selected_contents", methods=["POST"])
def selected_contents_endpoint():
    global content_to_send
    content_to_send = request.json
    logging.info(f"📩 Received content to send: {content_to_send}")
    return jsonify(content_to_send)

def process_recommendation_plans():
    """Process recommendation plans based on received content."""
    global recommendation_plans, content_to_send
    logging.info("Processing recommendation plans...")
    recommendations_to_send = content_to_send["recommendations"]
    resources_to_send = content_to_send["resources"]

    created_at = datetime.now(timezone.utc).isoformat()  

    resources_timing = {
        user_id: {
            "created_at": created_at,
            "contents": [
                {
                    "id": resource_id,
                    "type": "recommendation" if resource_id[2] == "c" else "resource",
                    "send_timestamp": ""
                }
            for mission, resource_id in mission_to_resource_ids.items()
            ]
        }
        for user_id, mission_to_resource_ids in resources_to_send.items()
    }

    recommendations_timing = {
        user_id: {
            "created_at": created_at,
            "contents": [
                {
                    "id": recommendation_id,
                    "type": "recommendation" if recommendation_id[2] == "c" else "resource",
                    "send_timestamp": ""
                }
            for mission, recommendation_ids_to_frequency in mission_to_recommendation_ids_to_frequency.items()
            for recommendation_id, frequency in recommendation_ids_to_frequency.items()
            for _ in range(frequency)
            ]
        }
        for user_id, mission_to_recommendation_ids_to_frequency in recommendations_to_send.items()
    }

    recommendation_plans = {
        user_id: {
            "created_at": resources_timing.get(user_id, {}).get("created_at", time.time()),
            "contents": resources_timing.get(user_id, {}).get("contents", []) + 
                        recommendations_timing.get(user_id, {}).get("contents", [])
        }
        for user_id in set(resources_timing.keys()).union(recommendations_timing.keys())
    }

    update_status("ready")
    logging.info(f"✅ Generated recommendation plans: {recommendation_plans}")

@app.route("/recommendation_plans", methods=["GET"])
def recommendation_plans_endpoint():
    if status["status"] == "processing recommendation plans":
        return jsonify({"message": "Already processing recommendation plans"}), 409
    
    update_status("processing recommendation plans")
    threading.Thread(target=process_recommendation_plans).start()
    return jsonify({"message": "Processing recommendation plans"}), 202

@app.route("/recommendation_plans_result", methods=["GET"])
def get_recommendation_plans_result():
    if status["status"] != "ready":
        return jsonify({"message": "Recommendation plans not ready yet"}), 409
    return jsonify(recommendation_plans)

if __name__ == "__main__":
    logging.info("🚀 Starting OMI API server...")
    app.run(host="0.0.0.0", port=5000, debug=True)
