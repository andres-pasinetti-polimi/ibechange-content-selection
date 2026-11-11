"""Publish the sample events of sample_events.json to the local Artemis broker.

For local testing only: it plays the part of the EUT backend. The samples follow
the message formats of rct/design/iBeChange-modules-architecture.docx (30 September 2026).

Run from the system/ folder:
    python -m scripts.publish_sample_events
"""

import json
from pathlib import Path

import stomp

from cs_rct import config

SAMPLES_FILE = Path(__file__).with_name("sample_events.json")


def main():
    events = json.loads(SAMPLES_FILE.read_text())

    broker = stomp.Connection(host_and_ports=[(config.ARTEMIS_HOST, config.ARTEMIS_PORT)])
    broker.connect(config.ARTEMIS_USER, config.ARTEMIS_PASSWORD, wait=True)

    for event in events:
        broker.send(
            destination=f"/queue/{config.ARTEMIS_QUEUE}",
            body=json.dumps(event),
            content_type="application/json",
            # Makes the local broker create a point-to-point queue that keeps
            # messages until they are consumed. How the EUT broker is configured
            # is not known yet (rct/correspondence/eut_questions_rct_integration.md, question 6b).
            headers={"destination-type": "ANYCAST", "persistent": "true"},
        )
        print("sent", next(iter(event)))

    broker.disconnect()


if __name__ == "__main__":
    main()
