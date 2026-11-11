#!/usr/bin/env zsh
set -euo pipefail
setopt NULL_GLOB

#######################################
# Config (override via env)
#######################################

ENV_NAME=${ENV_NAME:-baseline_v1}        # logical name of environment
N_WORLDS=${N_WORLDS:-3}                  # how many worlds to run (0..N-1 *within this batch*)
COMPOSE_FILE=${COMPOSE_FILE:-docker-compose.yml}
TEMPLATE=${TEMPLATE:-generated_configs/config.py}
BASE_PROJECT_PREFIX=${BASE_PROJECT_PREFIX:-${ENV_NAME}}
WORLD_OFFSET=${WORLD_OFFSET:-0}          # global starting index for worlds

#######################################
# Helpers
#######################################

world_dir() {
  local idx=$((WORLD_OFFSET + $1))
  local padded
  padded=$(printf "%03d" "${idx}")
  echo "experiments/${ENV_NAME}/world_${padded}"
}

project_name() {
  local idx=$((WORLD_OFFSET + $1))
  echo "${BASE_PROJECT_PREFIX}_s${idx}"
}

ensure_template() {
  if [[ ! -f "$TEMPLATE" ]]; then
    echo "❌ TEMPLATE not found: $TEMPLATE" >&2
    exit 1
  fi
}

#######################################
# Commands
#######################################

cmd=${1:-help}

case "$cmd" in
  init)
    echo "📂 Initializing worlds for env: ${ENV_NAME}"
    ensure_template

    for ((i = 0; i < N_WORLDS; i++)); do
      local_global_id=$((WORLD_OFFSET + i))
      DIR=$(world_dir $i)
      echo "  • world_${local_global_id} → $DIR"
      mkdir -p "$DIR/user_preferences"
      mkdir -p "$DIR/postgres_data"
      cp "$TEMPLATE" "$DIR/config.py"
    done

    echo "✅ init done."
    ;;

  up|start)
    echo "🚀 Starting ${N_WORLDS} worlds for env: ${ENV_NAME}"
    echo "   (global WORLD_IDs from $WORLD_OFFSET to $((WORLD_OFFSET + N_WORLDS - 1)))"

    ensure_template

    echo "🧱 Building images (if needed)..."
    docker compose -f "${COMPOSE_FILE}" build

    for ((i = 0; i < N_WORLDS; i++)); do
      GLOBAL_ID=$((WORLD_OFFSET + i))
      PADDED_ID=$(printf "%03d" "${GLOBAL_ID}")
      DIR=$(world_dir $i)
      PROJECT=$(project_name $i)

      echo "------------------------------"
      echo "▶ World $GLOBAL_ID (padded: ${PADDED_ID})"
      echo "   project : $PROJECT"
      echo "   dir     : $DIR"
      mkdir -p "$DIR/user_preferences" "$DIR/postgres_data"

      if [[ ! -f "$DIR/config.py" ]]; then
        echo "   (no config.py found, copying template)"
        cp "$TEMPLATE" "$DIR/config.py"
      fi

      ENV_NAME="${ENV_NAME}" \
      WORLD_ID="${GLOBAL_ID}" \
      WORLD_ID_PADDED="${PADDED_ID}" \
      WORLD_SEED="${GLOBAL_ID}" \
      PROJECT_PREFIX="${PROJECT}" \
      docker compose -p "${PROJECT}" -f "${COMPOSE_FILE}" up -d --no-build
    done

    echo "✅ All worlds started."
    ;;

  down|stop)
    echo "🛑 Stopping worlds for env: ${ENV_NAME}"
    echo "   (global WORLD_IDs from $WORLD_OFFSET to $((WORLD_OFFSET + N_WORLDS - 1)))"

    # configurable timeouts
    STOP_TIMEOUT=${STOP_TIMEOUT:-8}          # seconds for docker stop
    KILL_GRACE=${KILL_GRACE:-2}              # seconds before kill (if needed)
    COMPOSE_TIMEOUT=${COMPOSE_TIMEOUT:-5}    # compose down timeout

    # helper: run a command but hard-timeout it
    run_with_timeout() {
      local seconds="$1"; shift
      if command -v gtimeout >/dev/null 2>&1; then
        # macOS coreutils (brew install coreutils)
        gtimeout "${seconds}" "$@"
      else
        # GNU timeout (linux)
        timeout "${seconds}" "$@"
      fi
    }

    stop_project() {
      local i="$1"
      local GLOBAL_ID=$((WORLD_OFFSET + i))
      local PADDED_ID=$(printf "%03d" "${GLOBAL_ID}")
      local PROJECT=$(project_name "$i")

      echo "------------------------------"
      echo "⏹  Stopping project $PROJECT ..."

      # 1) list containers for this compose project (fast, doesn't tear down)
      local cids
      cids=$(docker ps -q --filter "label=com.docker.compose.project=${PROJECT}" || true)

      if [[ -n "${cids}" ]]; then
        echo "   → docker stop (${STOP_TIMEOUT}s) ..."
        run_with_timeout $((STOP_TIMEOUT + 2)) docker stop -t "${STOP_TIMEOUT}" ${=cids} || true

        # refresh list of survivors
        local still
        still=$(docker ps -q --filter "label=com.docker.compose.project=${PROJECT}" || true)
        if [[ -n "${still}" ]]; then
          echo "   ⚠️ still running → docker kill ..."
          run_with_timeout $((KILL_GRACE + 2)) docker kill ${=still} || true
        fi
      else
        echo "   (no running containers found)"
      fi

      # 2) cleanup via compose down, but never block forever
      echo "   → compose down (${COMPOSE_TIMEOUT}s) ..."
      ENV_NAME="${ENV_NAME}" \
      WORLD_ID="${GLOBAL_ID}" \
      WORLD_ID_PADDED="${PADDED_ID}" \
      WORLD_SEED="${GLOBAL_ID}" \
      PROJECT_PREFIX="${PROJECT}" \
      run_with_timeout $((COMPOSE_TIMEOUT + 3)) \
        docker compose -p "${PROJECT}" -f "${COMPOSE_FILE}" down \
          --timeout "${COMPOSE_TIMEOUT}" --remove-orphans || true

      # 3) safety net: remove any leftovers by label (networks/containers)
      # (containers should already be gone; this is just belt & suspenders)
      docker ps -aq --filter "label=com.docker.compose.project=${PROJECT}" | xargs -r docker rm -f >/dev/null 2>&1 || true
      docker network ls -q --filter "label=com.docker.compose.project=${PROJECT}" | xargs -r docker network rm >/dev/null 2>&1 || true

      echo "   ✅ Project $PROJECT stopped."
    }

    # stop sequentially (safe) OR parallelize (faster) if you want:
    for ((i = 0; i < N_WORLDS; i++)); do
      stop_project "$i"
    done

    echo "✅ All worlds stopped (best-effort)."
    ;;

  clean)
    echo "🧨 Stopping + cleaning everything for env: ${ENV_NAME}"
    echo "   (global WORLD_IDs from $WORLD_OFFSET to $((WORLD_OFFSET + N_WORLDS - 1)))"

    for ((i = 0; i < N_WORLDS; i++)); do
      PROJECT=$(project_name $i)
      DIR=$(world_dir $i)

      echo "------------------------------"
      echo "⏹  Project: $PROJECT"
      docker compose -p "$PROJECT" -f "${COMPOSE_FILE}" down --remove-orphans || true

      echo "🧹 Removing world directory: $DIR"
      rm -rf "$DIR"
    done

    echo "✅ Env fully cleaned."
    ;;

  status)
    echo "ℹ️  docker compose projects for env: ${ENV_NAME}"
    docker ps --format 'table {{.Names}}\t{{.Status}}' | grep "${BASE_PROJECT_PREFIX}_" || echo "No running containers for this env."
    ;;

  *)
    cat <<EOF
Usage: $0 <command>

Commands:
  init    - create world directories + copy template config.py
  up      - start all worlds (or: start)
  down    - stop all worlds (or: stop)
  clean   - stop and delete world dirs + DB data
  status  - show running containers matching this env

Environment overrides:
  ENV_NAME            (default: baseline_v1)
  N_WORLDS            (default: 3)
  WORLD_OFFSET        (default: 0)
  COMPOSE_FILE        (default: docker-compose.yml)
  TEMPLATE            (default: generated_configs/config.py)
  BASE_PROJECT_PREFIX (default: \$ENV_NAME)

Example:
  N_WORLDS=10 ./manage_env.zsh init
  N_WORLDS=10 ./manage_env.zsh up
  N_WORLDS=10 ./manage_env.zsh status
  N_WORLDS=10 ./manage_env.zsh clean
EOF
    ;;
esac