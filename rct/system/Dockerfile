# Image of the CS service: the REST API (default command) and the Artemis consumer
# (docker-compose.yml, profile "service"). The hourly run and the replay check are
# started by Cronicle (scripts/cronicle/).
FROM python:3.13-slim

WORKDIR /app
ENV PYTHONPATH=/app PYTHONUNBUFFERED=1

COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

# CODE_VERSION (commit of the deployed code, written by scripts/write_code_version.sh before the
# copy to the server) is optional; the wildcard lets the build go on without it.
COPY requirements.txt CODE_VERSION* /app/
COPY cs_rct /app/cs_rct
COPY catalogue /app/catalogue
COPY scripts /app/scripts

EXPOSE 8000
# A run can take minutes once the trial has many participants, hence the long timeout.
CMD ["gunicorn", "--bind", "0.0.0.0:8000", "--workers", "2", "--timeout", "900", "cs_rct.api:app"]
