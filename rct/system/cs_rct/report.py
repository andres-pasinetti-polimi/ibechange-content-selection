"""The daily reconciliation report and the operational checks (rct_logging_specification.md, 9.1 and 9.4).

For one day it counts, by recruitment centre, what CS decided, what OMI placed and what EUT sent
and the participants opened, so that a gap between two components shows as a difference between
two counts. A sent or opened notification is matched to a delivery by participant, content id
and local day. The operational part reports the runs of the last 24 hours that failed, the last
run that succeeded and the last message received from Artemis.
"""

from datetime import timedelta

from cs_rct import design
from cs_rct.store import rows

PER_CENTRE = """
WITH plan AS (
    SELECT d.*, coalesce(u.recruitment_center, 'unknown') AS centre,
           (c.delivery_id IS NOT NULL) AS cancelled,
           EXISTS (SELECT 1 FROM placements p WHERE p.delivery_id = d.id) AS placed,
           EXISTS (SELECT 1 FROM notification_events n
                   WHERE n.user_uuid = d.user_uuid AND n.content_id = d.content_id AND n.action = 'sent'
                     AND (n.created_at AT TIME ZONE 'UTC' AT TIME ZONE coalesce(dec.timezone, %(zone)s))::date
                         = d.decision_day) AS sent,
           EXISTS (SELECT 1 FROM notification_events n
                   WHERE n.user_uuid = d.user_uuid AND n.content_id = d.content_id AND n.action = 'opened'
                     AND (n.created_at AT TIME ZONE 'UTC' AT TIME ZONE coalesce(dec.timezone, %(zone)s))::date
                         = d.decision_day) AS opened
    FROM deliveries d
    JOIN decisions dec ON dec.id = d.decision_id
    LEFT JOIN users u ON u.user_uuid = d.user_uuid
    LEFT JOIN delivery_cancellations c ON c.delivery_id = d.id
    WHERE d.decision_day = %(day)s AND d.content_id IS NOT NULL
)
SELECT centre,
       count(DISTINCT user_uuid) AS participants_with_a_plan,
       count(*) FILTER (WHERE format = 'recommendation') AS recommendations,
       count(*) FILTER (WHERE format = 'resource') AS resources,
       count(*) FILTER (WHERE format = 'reminder') AS reminders,
       count(*) FILTER (WHERE cancelled) AS cancelled,
       count(*) FILTER (WHERE NOT cancelled AND placed) AS placed_by_omi,
       count(*) FILTER (WHERE NOT cancelled AND sent) AS sent_by_eut,
       count(*) FILTER (WHERE NOT cancelled AND opened) AS opened
FROM plan GROUP BY centre ORDER BY centre
"""

STATUSES = """
SELECT coalesce(u.recruitment_center, 'unknown') AS centre, dec.status, count(*) AS days
FROM decisions dec LEFT JOIN users u ON u.user_uuid = dec.user_uuid
WHERE dec.decision_day = %(day)s GROUP BY 1, 2 ORDER BY 1, 2
"""


def daily_report(db, day):
    now = design.utc_now()
    failed = rows(db, "SELECT started_at, error FROM job_runs WHERE status = 'failed' AND started_at >= %s "
                      "ORDER BY started_at", (now - timedelta(hours=24),))
    last_ok = rows(db, "SELECT max(started_at) AS at FROM job_runs WHERE status = 'ok'")[0]["at"]
    last_message = rows(db, "SELECT max(received_at) AS at FROM raw_events")[0]["at"]
    return {
        "day": day.isoformat(),
        "by_centre": rows(db, PER_CENTRE, {"day": day, "zone": design.DEFAULT_TIMEZONE}),
        "decision_days_by_status": rows(db, STATUSES, {"day": day}),
        "operations": {
            "failed_runs_last_24h": [{"started_at": r["started_at"].isoformat(), "error": r["error"]} for r in failed],
            "last_successful_run_utc": last_ok.isoformat() if last_ok else None,
            "last_message_received": last_message.isoformat() if last_message else None,
        },
    }
