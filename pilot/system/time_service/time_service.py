# time_service.py
from flask import Flask, request, jsonify
from datetime import datetime, timedelta, timezone
import logging

# ------------------------------------------------------------------- #
# Logging                                                             #
# ------------------------------------------------------------------- #
logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)sZ - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)
logging.Formatter.converter = lambda *args: datetime.now(timezone.utc).timetuple()  # force UTC in log header
logging.getLogger("werkzeug").setLevel(logging.ERROR)

# ------------------------------------------------------------------- #
# App + clock                                                         #
# ------------------------------------------------------------------- #
app = Flask(__name__)

# Store an *aware* UTC datetime
current_time = datetime(2025, 9, 1, 9, 0, 0, tzinfo=timezone.utc)


# ------------------------------------------------------------------- #
# Helpers                                                             #
# ------------------------------------------------------------------- #
def utc_iso(dt: datetime) -> str:
    """YYYY-MM-DDTHH:MM:SSZ string."""
    return dt.isoformat(timespec="seconds").replace("+00:00", "Z")


# ------------------------------------------------------------------- #
# Routes                                                              #
# ------------------------------------------------------------------- #
@app.route("/get_time", methods=["GET"])
def get_time():
    return jsonify({"now": utc_iso(current_time)})


@app.route("/advance", methods=["POST"])
def advance_time():
    global current_time
    try:
        hours = int(request.json.get("hours", 1))
        if hours < 0:
            raise ValueError
    except (TypeError, ValueError):
        return jsonify({"error": "hours must be a non-negative integer"}), 400

    current_time += timedelta(hours=hours)
    return jsonify({"now": utc_iso(current_time)})


# ------------------------------------------------------------------- #
# Entrypoint                                                          #
# ------------------------------------------------------------------- #
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
