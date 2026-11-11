"""The Cronicle jobs of the CS service, in one file that runs in the Cronicle container.

The Cronicle container has Python and requests but not the service's code, and it reaches the
services over the Docker networks it shares with them, so every job is an HTTP call:

    python3 cs_rct_cronicle.py hourly      # every hour: one CS run, then the exchange with OMI
    python3 cs_rct_cronicle.py daily       # once a day: a copy of the database, the catalogue of VU if it is the
                                           # RCT one, the replay check and yesterday's report
    python3 cs_rct_cronicle.py catalogue   # on catalogue changes: recommendations, resources, Missions

Hourly exchange (rct/design/iBeChange-modules-architecture.docx, section "CS and OMI (RCT)"):

    CS:  POST /run
    CS:  GET  /selected_contents?day=D   ->  OMI: POST /selected_contents/
    OMI: GET  /recommendation_plans/     ->  CS:  POST /recommendation_plans
                                         ->  VU:  POST /internal/recommendations (only if VU_RECEIVES_PLANS=1)

A plan belongs to the participant's local day, which can be the day after the UTC day (UTC+2 or
+3 in the evening), so D runs over the UTC day of the run and the day after; OMI recognises a
plan it already holds by its plan_id.

Settings, as environment variables of the Cronicle event, or as KEY=VALUE lines of the file .env-cs-rct next
to this script (the events have no variables, and they can be edited only from the web interface of
Cronicle); a variable of the event wins over the file:
    CS_URL      default http://cs-rct-api:8000
    OMI_URL     e.g. http://omi-app:3200; if empty, the exchange with OMI is skipped
    VU_URL, EUT_TOKEN, VU_RECEIVES_PLANS   for the catalogue job and the plans to VU
Every step prints one line; a failed call is retried (except when rejected with a 4xx status) and
then stops the job with exit status 1.
"""

import os
import sys
import time
from datetime import datetime, timedelta, timezone

import requests


def read_settings_file(path):
    """Set the KEY=VALUE lines of `path` as environment variables, unless already set."""
    if not os.path.exists(path):
        return
    with open(path) as file:
        for line in file:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, value = line.split("=", 1)
                os.environ.setdefault(key.strip(), value.strip())


read_settings_file(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env-cs-rct"))

CS_URL = os.getenv("CS_URL", "http://cs-rct-api:8000").rstrip("/")
OMI_URL = os.getenv("OMI_URL", "").rstrip("/")
VU_URL = os.getenv("VU_URL", "").rstrip("/")
EUT_TOKEN = os.getenv("EUT_TOKEN", "")
TRIES = 4


def call(method, url, payload=None, params=None, token=None):
    headers = {"Authorization": f"Basic {token}"} if token else {}
    for attempt in range(1, TRIES + 1):
        try:
            response = requests.request(method, url, json=payload, params=params, headers=headers, timeout=900)
            if response.status_code in (200, 201):
                return response.json() if response.content else None
            print(f"{method} {url}: HTTP {response.status_code} (attempt {attempt}): {response.text[:1000]}", flush=True)
            # A request the receiver rejects (4xx) is rejected again on a retry.
            if 400 <= response.status_code < 500:
                break
        except requests.RequestException as error:
            print(f"{method} {url}: {error!r} (attempt {attempt})", flush=True)
        if attempt < TRIES:
            time.sleep(5 * 2 ** (attempt - 1))
    raise SystemExit(f"{method} {url} failed after {attempt} attempts")


def hourly():
    result = call("POST", f"{CS_URL}/run")
    print(f"CS run: {result}", flush=True)
    if not OMI_URL:
        print("OMI_URL not set: exchange with OMI skipped", flush=True)
        return
    today = datetime.now(timezone.utc).date()
    for day in (today, today + timedelta(days=1)):
        plans = call("GET", f"{CS_URL}/selected_contents", params={"day": day.isoformat()})
        print(f"{day}: {len(plans)} plans from CS", flush=True)
        if plans:
            call("POST", f"{OMI_URL}/selected_contents/", payload=plans)
    window = {"start_time": f"{today}T00:00:00", "end_time": f"{today + timedelta(days=2)}T00:00:00"}
    placements = call("GET", f"{OMI_URL}/recommendation_plans/", params=window)
    if placements and placements.get("recommendation_plans"):
        stored = call("POST", f"{CS_URL}/recommendation_plans", payload=placements)
        print(f"placements: {stored}", flush=True)
        if os.getenv("VU_RECEIVES_PLANS") == "1":
            call("POST", f"{VU_URL}/internal/recommendations", payload=placements, token=EUT_TOKEN)
    else:
        print("no placements from OMI", flush=True)


def daily():
    # The copy comes first, so that it is taken even when the checks below fail.
    copy = call("POST", f"{CS_URL}/backup")
    print(f"backup: {copy['folder']}, {sum(t['rows'] for t in copy['tables'].values())} rows "
          f"in {len(copy['tables'])} tables", flush=True)
    catalogue_from_vu()
    replay = call("POST", f"{CS_URL}/replay")
    print(f"replay check: {replay}", flush=True)
    yesterday = (datetime.now(timezone.utc) - timedelta(days=1)).date().isoformat()
    report = call("GET", f"{CS_URL}/report", params={"day": yesterday})
    for line in report["by_centre"]:
        print(f"{yesterday} {line}", flush=True)
    print(f"operations: {report['operations']}", flush=True)
    # The job fails, and Cronicle reports it, when the log does not reproduce or a run failed.
    if replay["probability_mismatches"] or replay["version_mismatches"] or report["operations"]["failed_runs_last_24h"]:
        raise SystemExit("replay mismatches or failed runs in the last 24 hours")


def catalogue_from_vu():
    """The recommendations and resources of VU replace those of the service only when VU serves the RCT
    catalogue: the service stores them (POST ?source=vu) if they hold at least 95% of the ids of the
    catalogue built from the ICO workbook, and keeps its own otherwise. A failed or empty answer of VU
    leaves the service's catalogue as it is and does not fail the job."""
    if not VU_URL:
        print("catalogue: VU_URL not set, the service keeps its catalogue", flush=True)
        return
    for kind, flag in (("recommendations", "false"), ("resources", "true")):
        try:
            items = call("GET", f"{VU_URL}/internal/recommendations", params={"resources": flag}, token=EUT_TOKEN)
        except SystemExit as error:
            print(f"{kind}: VU did not answer ({error}); the service keeps its catalogue", flush=True)
            continue
        listed = items.get("recommendations") if isinstance(items, dict) else items
        if not listed:
            print(f"{kind}: VU returned none; the service keeps its catalogue", flush=True)
            continue
        response = requests.post(f"{CS_URL}/{kind}", params={"source": "vu"}, json=items, timeout=900)
        print(f"{kind}: HTTP {response.status_code} {response.text[:500]}", flush=True)


def catalogue():
    catalogue_from_vu()
    missions = call("GET", f"{VU_URL}/missions", token=EUT_TOKEN)
    print(f"missions: {call('POST', f'{CS_URL}/missions', payload=missions)}", flush=True)


if __name__ == "__main__":
    jobs = {"hourly": hourly, "daily": daily, "catalogue": catalogue}
    if len(sys.argv) != 2 or sys.argv[1] not in jobs:
        raise SystemExit(f"usage: {sys.argv[0]} {' | '.join(jobs)}")
    jobs[sys.argv[1]]()
