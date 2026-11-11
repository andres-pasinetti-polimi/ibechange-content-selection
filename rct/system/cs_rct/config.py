"""Settings read from environment variables.

Every value that depends on where the service runs is read here, so that the rest
of the code never reads the environment directly. The defaults match the local
test setup in docker-compose.yml.
"""

import os

# Artemis broker (STOMP protocol, as in the EUT example consumer)
ARTEMIS_HOST = os.getenv("ARTEMIS_HOST", "localhost")
ARTEMIS_PORT = int(os.getenv("ARTEMIS_PORT", "61613"))
ARTEMIS_USER = os.getenv("ARTEMIS_USER", "artemis")
ARTEMIS_PASSWORD = os.getenv("ARTEMIS_PASSWORD", "artemis")
ARTEMIS_QUEUE = os.getenv("ARTEMIS_QUEUE", "polimi-queue")

# TLS is used only when a CA certificate is given
ARTEMIS_CA_CERT = os.getenv("ARTEMIS_CA_CERT")

# PostgreSQL database of the service
DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://cs:cs@localhost:5433/cs_rct")
