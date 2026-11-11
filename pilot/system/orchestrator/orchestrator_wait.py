import time
import requests
import schedule
import threading
import logging
from flask import Flask, jsonify

app = Flask(__name__)

# Container URLs (Docker services)
VU_API_URL = "http://virtual_user_api:5000"
CS_MODULE_URL = "http://cs_module:5000"
OMI_MODULE_URL = "http://omi_module:5000"

# ---- Setup Logging ----
logging.basicConfig(
    level=logging.INFO,
    format="\n%(asctime)s - %(levelname)s - %(message)s",
    handleCS=[
        logging.FileHandler("orchestrator.log"),
        logging.StreamHandler()
    ]
)

def wait_for_completion(url, expected_status="ready", timeout=1800, check_interval=2):
    """Waits for an API endpoint to return a specific status."""
    start_time = time.time()
    while time.time() - start_time < timeout:
        #logging.info(f"Waiting for {url} to reach status '{expected_status}'...")
        try:
            response = requests.get(f"{url}/status")
            if response.status_code == 200:
                data = response.json()
                logging.debug(f"Current status from {url}: {data}")
                if data.get("status") == expected_status:
                    #logging.info(f"{url} is ready")
                    return True
        except requests.exceptions.RequestException as e:
            logging.error(f"Error reaching {url}: {e}")
        time.sleep(check_interval)
    
    logging.warning(f"Timeout reached while waiting for {url}. Task might still be running.")
    return False


def fetch_json(endpoint, timeout=10):
    """Fetch JSON data from an API endpoint and return the response."""
    try:
        response = requests.get(endpoint, timeout=timeout)
        response.raise_for_status()  # Raises an exception for 4xx/5xx erroCS
        logging.info(f"✅ Successfully fetched data from {endpoint}")
        return response.json()
    except requests.exceptions.Timeout:
        logging.error(f"⏳ Timeout while fetching data from {endpoint}")
    except requests.exceptions.RequestException as e:
        logging.error(f"❌ Failed to fetch data from {endpoint}: {e}")
    return None

def trigger_get_endpoint(endpoint, timeout=10):
    """Trigger an API endpoint via a GET request and log the outcome."""
    try:
        response = requests.get(endpoint, timeout=timeout)
        response.raise_for_status()
        logging.info(f"✅ Successfully triggered {endpoint}")
    except requests.exceptions.Timeout:
        logging.error(f"⏳ Timeout while triggering {endpoint}")
    except requests.exceptions.RequestException as e:
        logging.error(f"❌ Failed to trigger {endpoint}: {e}")

def post_json(endpoint, data):
    """Safely send JSON data to an endpoint."""
    try:
        requests.post(endpoint, json=data)
        logging.info(f"Successfully sent data to {endpoint}")
    except requests.exceptions.RequestException as e:
        logging.error(f"Failed to send data to {endpoint}: {e}")

def run_nightly_update():
    """Executes the full nightly update process."""
    logging.info("🔹 Orchestrator: Starting nightly update...")


    # Step 1: Update modules with daily information
    logging.info("➡️ Step 1: Updating CS and OMI modules...")
    daily_updates = fetch_json(f"{VU_API_URL}/daily_updates")

    if not daily_updates:
        logging.error("❌ Failed to fetch daily updates. Skipping nightly update.")
        return
    
    wait_for_completion(VU_API_URL)
    post_json(f"{CS_MODULE_URL}/daily_updates", daily_updates)
    wait_for_completion(CS_MODULE_URL)
    post_json(f"{OMI_MODULE_URL}/daily_updates", daily_updates)
    wait_for_completion(OMI_MODULE_URL)
    

    # Step 2: OMI module computes recommendation timeslots and sends them to CS module
    logging.info("➡️ Step 2: Computing recommendation timeslots...")
    num_timeslots = fetch_json(f"{OMI_MODULE_URL}/num_timeslots")
    wait_for_completion(OMI_MODULE_URL)

    if not num_timeslots:
        logging.error("❌ Failed to fetch recommendation timeslots. Skipping nightly update.")
        return

    post_json(f"{CS_MODULE_URL}/num_timeslots", num_timeslots)
    wait_for_completion(CS_MODULE_URL)

    # Step 3: CS module selects recommendations and resources. They are sent to the OMI module
    logging.info("➡️ Step 3: Computing recommendations and resources and sending to OMI module...")
    trigger_get_endpoint(f"{CS_MODULE_URL}/selected_contents")
    wait_for_completion(CS_MODULE_URL)
    selected_contents = fetch_json(f"{CS_MODULE_URL}/contents_result")

    if selected_contents:
        post_json(f"{OMI_MODULE_URL}/selected_contents", selected_contents)
        wait_for_completion(OMI_MODULE_URL)

    # Step 4: OMI module defines recommendation plans with timing information. They are sent to the CS and VU module
    logging.info("➡️ Step 4: Finalising recommendation plans with timing allocation...")
    trigger_get_endpoint(f"{OMI_MODULE_URL}/recommendation_plans")
    wait_for_completion(OMI_MODULE_URL)
    recommendation_plans = fetch_json(f"{OMI_MODULE_URL}/recommendation_plans_result")
    if recommendation_plans:
        post_json(f"{CS_MODULE_URL}/recommendation_plans", recommendation_plans)
        post_json(f"{VU_API_URL}/recommendation_plans", recommendation_plans)

    logging.info("✅ Orchestrator: Nightly update completed!")




# ---- Scheduler ----
schedule.every(20).seconds.do(run_nightly_update)

def schedule_runner():
    """Runs scheduled tasks in the background."""
    while True:
        schedule.run_pending()
        time.sleep(1)

# ---- Main Execution ----
if __name__ == "__main__":
    logging.info("🔹 Orchestrator: Starting initialisation...")

    # Step 0: Initialise modules with new useCS
    logging.info("➡️ Step 0: Initialising modules with new useCS...")
    for module in [OMI_MODULE_URL, CS_MODULE_URL]:
        requests.get(f"{module}/initialise")
        wait_for_completion(module)
    
    logging.info("✅ Orchestrator: Initialisation completed.")

    # Run Flask in one thread and scheduler in another
    threading.Thread(target=schedule_runner, daemon=True).start()
    
    while True:
        time.sleep(1)  # Keeps the script alive