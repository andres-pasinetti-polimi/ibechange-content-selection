#!/usr/bin/env zsh
set -u

# ============================
# Defaults
# ============================
ENV_NAME=${ENV_NAME:-baseline_v1}
STUDY_TAG=${STUDY_TAG:-study01}
N_WORLDS=${N_WORLDS:-40}
BATCH_SIZE=${BATCH_SIZE:-10}
POLL_SECONDS=${POLL_SECONDS:-30}
ORCH_SERVICE_NAME=${ORCH_SERVICE_NAME:-orchestrator}

# ============================
# CLI argument parsing
# ============================
while [[ $# -gt 0 ]]; do
  case "$1" in
    --env|--env_name)
      ENV_NAME="$2"; shift 2 ;;
    --study|--study_tag|--study-tag|--exp|--experiment)
      STUDY_TAG="$2"; shift 2 ;;
    --n_worlds|--n-worlds)
      N_WORLDS="$2"; shift 2 ;;
    --batch_size|--batch-size)
      BATCH_SIZE="$2"; shift 2 ;;
    --poll_seconds|--poll-seconds)
      POLL_SECONDS="$2"; shift 2 ;;
    --orchestrator|--orch)
      ORCH_SERVICE_NAME="$2"; shift 2 ;;
    *)
      echo "Unknown argument: $1" >&2
      exit 1 ;;
  esac
done

echo "======================================="
echo " Env          : ${ENV_NAME}"
echo " Study tag    : ${STUDY_TAG}"
echo " Total worlds : ${N_WORLDS}"
echo " Max parallel : ${BATCH_SIZE}"
echo " Poll every   : ${POLL_SECONDS}s"
echo " Orchestrator : ${ORCH_SERVICE_NAME}"
echo "======================================="
echo

# ============================
# Progress bar
# ============================
typeset -i COMPLETED=0

update_progress() {
  local width=40
  local done=$(( COMPLETED * width / N_WORLDS ))
  local todo=$(( width - done ))
  local percent=$(( COMPLETED * 100 / N_WORLDS ))

  local bar_done bar_todo
  bar_done=$(printf '%*s' "$done" '' | tr ' ' '#')
  bar_todo=$(printf '%*s' "$todo" '' | tr ' ' '-')

  printf "\r[%s%s] %3d%% (%d/%d)" \
    "$bar_done" "$bar_todo" \
    "$percent" "$COMPLETED" "$N_WORLDS"
}

update_progress

# ============================
# Helpers
# ============================

# Is orchestrator container for WORLD_ID $1 still running?
is_orch_running() {
  local idx=$1
  local project="${ENV_NAME}_s${idx}"
  local cont="${project}-${ORCH_SERVICE_NAME}-1"

  docker ps --format '{{.Names}}' | grep -q "^${cont}$"
}

# Start a single world (init + up)
start_world() {
  local idx=$1
  echo
  echo "🚀 Starting world ${idx} ..."
  echo "    → WORLD_OFFSET=${idx}, N_WORLDS=1, ENV_NAME=${ENV_NAME}"

  WORLD_OFFSET=${idx} \
  N_WORLDS=1 \
  ENV_NAME=${ENV_NAME} \
    ./manage_env.zsh init || {
      echo "    ❌ manage_env.zsh init FAILED for world ${idx}, continuing..." >&2
    }

  echo "    init for world ${idx} done, now up..."

  WORLD_OFFSET=${idx} \
  N_WORLDS=1 \
  ENV_NAME=${ENV_NAME} \
    ./manage_env.zsh up || {
      echo "    ❌ manage_env.zsh up FAILED for world ${idx}, continuing..." >&2
    }

  echo "    up for world ${idx} requested."

  # ⬅️ force function to return success so set -e doesn't kill run_study
  return 0
}

# Dump + down a single world, and update progress
finalize_world() {
  local idx=$1
  echo
  echo "📦 Dumping + stopping world ${idx} ..."

  WORLD_OFFSET=${idx} \
  N_WORLDS=1 \
  ENV_NAME=${ENV_NAME} \
    ./manage_results.zsh dump "${STUDY_TAG}"

  WORLD_OFFSET=${idx} \
  N_WORLDS=1 \
  ENV_NAME=${ENV_NAME} \
    ./manage_env.zsh down

  COMPLETED=$(( COMPLETED + 1 ))
  update_progress
}

# ============================
# Main scheduling loop
# ============================

typeset -i NEXT_WORLD=0
typeset -a RUNNING_WORLDS=()

echo
echo "▶️  Beginning streaming execution (up to ${BATCH_SIZE} parallel worlds)..."

while (( NEXT_WORLD < N_WORLDS || ${#RUNNING_WORLDS[@]} > 0 )); do
  # Launch new worlds while we have capacity
  while (( ${#RUNNING_WORLDS[@]} < BATCH_SIZE && NEXT_WORLD < N_WORLDS )); do
    echo "  INNER: starting world ${NEXT_WORLD} (len(RUNNING_WORLDS)=${#RUNNING_WORLDS[@]})"
    start_world "${NEXT_WORLD}"
    RUNNING_WORLDS+=("${NEXT_WORLD}")
    (( NEXT_WORLD++ ))
    echo "  INNER: after start, NEXT_WORLD=${NEXT_WORLD}, RUNNING_WORLDS=(${RUNNING_WORLDS[*]})"
  done

  if (( ${#RUNNING_WORLDS[@]} == 0 && NEXT_WORLD >= N_WORLDS )); then
    echo "BREAK: no running worlds and NEXT_WORLD>=N_WORLDS"
    break
  fi

  echo
  echo "⏳ Currently running worlds: ${RUNNING_WORLDS[*]} (sleeping ${POLL_SECONDS}s)"
  sleep "${POLL_SECONDS}"

  typeset -a STILL_RUNNING=()
  for idx in "${RUNNING_WORLDS[@]}"; do
    if is_orch_running "${idx}"; then
      STILL_RUNNING+=("${idx}")
    else
      echo
      echo "✅ World ${idx} finished."
      finalize_world "${idx}"
    fi
  done

  RUNNING_WORLDS=("${STILL_RUNNING[@]}")
done

echo

echo
echo "======================================="
echo "🎉 All worlds finished for env='${ENV_NAME}', study='${STUDY_TAG}'."
echo "DB dumps + preferences are under: results/${ENV_NAME}/${STUDY_TAG}"
echo
echo "To restore all ${N_WORLDS} worlds into local Postgres, run:"
echo
echo "  ENV_NAME=${ENV_NAME} N_WORLDS=${N_WORLDS} ./manage_results.zsh restore ${STUDY_TAG}"
echo