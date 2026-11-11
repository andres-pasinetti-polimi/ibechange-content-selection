#!/usr/bin/env zsh
set -euo pipefail
setopt NULL_GLOB

# ---------------------------
# Config (override via env)
# ---------------------------
ENV_NAME=${ENV_NAME:-baseline_v1}
N_WORLDS=${N_WORLDS:-10}
RESULTS_ROOT=${RESULTS_ROOT:-results}
BASE_PROJECT_PREFIX=${BASE_PROJECT_PREFIX:-${ENV_NAME}}
WORLD_OFFSET=${WORLD_OFFSET:-0}

# DB inside the *cs_db* containers
PG_DB=${PG_DB:-cs_data}
PG_USER=${PG_USER:-cs_user}
PG_PASSWORD=${PG_PASSWORD:-cs_password}

# Local Postgres (brew) user / host / port
LOCAL_PG_USER=${LOCAL_PG_USER:-$USER}
LOCAL_PG_HOST=${LOCAL_PG_HOST:-localhost}
LOCAL_PG_PORT=${LOCAL_PG_PORT:-5432}

# ---------------------------
# Helpers
# ---------------------------
project_name() {
  local idx=$((WORLD_OFFSET + $1))
  echo "${BASE_PROJECT_PREFIX}_s${idx}"
}

csdb_container_name() {
  local idx=$((WORLD_OFFSET + $1))
  echo "${BASE_PROJECT_PREFIX}_s${idx}-cs_db-1"
}

world_dir() {
  local idx=$((WORLD_OFFSET + $1))
  local padded
  padded=$(printf "%03d" "${idx}")
  echo "experiments/${ENV_NAME}/world_${padded}"
}

ensure_study_tag() {
  if [[ $# -lt 1 ]]; then
    echo "❌ Please provide STUDY_TAG, e.g.:"
    echo "   ./manage_results.zsh dump study01"
    exit 1
  fi
}

cmd=${1:-help}
shift || true

# ---------------------------
# CLI flags for this script
# ---------------------------
while [[ $# -gt 0 ]]; do
  case "$1" in
    --env|--env_name|--env-name)
      ENV_NAME="$2"; shift 2 ;;
    --n_worlds|--n-worlds)
      N_WORLDS="$2"; shift 2 ;;
    --results_root|--results-root)
      RESULTS_ROOT="$2"; shift 2 ;;
    --offset|--world_offset|--world-offset)
      WORLD_OFFSET="$2"; shift 2 ;;
    --base_project_prefix|--base-project-prefix)
      BASE_PROJECT_PREFIX="$2"; shift 2 ;;
    *)
      break ;;
  esac
done

case "$cmd" in
  dump)
    ensure_study_tag "$@"
    STUDY_TAG=$1

    # directory of this script (robust relative paths)
    SCRIPT_DIR=${0:A:h}

    OUT_BASE="${RESULTS_ROOT}/${ENV_NAME}/${STUDY_TAG}"
    DUMP_DIR="${OUT_BASE}/db_dumps"
    PREF_DIR="${OUT_BASE}/preferences"
    CFG_DIR="${OUT_BASE}/configs"
    CODE_DIR="${OUT_BASE}/code_snapshot"

    echo "💾 Dumping results for env=${ENV_NAME}, STUDY_TAG=${STUDY_TAG}"
    echo "   → ${OUT_BASE}"
    echo "   worlds: global WORLD_IDs ${WORLD_OFFSET} .. $((WORLD_OFFSET + N_WORLDS - 1))"

    mkdir -p "${DUMP_DIR}" "${PREF_DIR}" "${CFG_DIR}" "${CODE_DIR}"

    # ---- code snapshot (once per dump run; best-effort) ----
    echo "📌 Saving code snapshot..."
    cp -f "${SCRIPT_DIR}/docker-compose.yml" "${CODE_DIR}/docker-compose.yml" 2>/dev/null || true
    cp -f "${SCRIPT_DIR}/manage_env.zsh" "${CODE_DIR}/manage_env.zsh" 2>/dev/null || true
    cp -f "${SCRIPT_DIR}/manage_results.zsh" "${CODE_DIR}/manage_results.zsh" 2>/dev/null || true
    cp -f "${SCRIPT_DIR}/run_study.zsh" "${CODE_DIR}/run_study.zsh" 2>/dev/null || true
    cp -f "${SCRIPT_DIR}/orchestrator/config_generator.py" "${CODE_DIR}/config_generator.py" 2>/dev/null || true
    cp -f "${SCRIPT_DIR}/orchestrator/orchestrator.py" "${CODE_DIR}/orchestrator.py" 2>/dev/null || true

    # optional git provenance (won’t fail if not a git repo)
    (cd "${SCRIPT_DIR}" && git rev-parse HEAD > "${CODE_DIR}/git_commit.txt") 2>/dev/null || true
    (cd "${SCRIPT_DIR}" && git diff > "${CODE_DIR}/git_diff.patch") 2>/dev/null || true

    # ---- worlds loop (robust arithmetic loop) ----
    for ((i=0; i<N_WORLDS; i++)); do
      GLOBAL_ID=$((WORLD_OFFSET + i))

      cont=$(csdb_container_name $i)
      wdir=$(world_dir $i)
      wid=$(printf "%03d" "${GLOBAL_ID}")

      sql="${DUMP_DIR}/world_${GLOBAL_ID}.sql"

      echo "------------------------------"
      echo "World ${GLOBAL_ID}"
      echo "  container : ${cont}"
      echo "  dump file : ${sql}"

      if docker ps --format '{{.Names}}' | grep -q "^${cont}\$"; then
        echo "  → pg_dump from ${cont}"
        docker exec -e PGPASSWORD="${PG_PASSWORD}" "${cont}" \
          pg_dump -U "${PG_USER}" "${PG_DB}" > "${sql}"
      else
        echo "  ⚠️  Container ${cont} not running; skipping DB dump (but still copying files)."
      fi

      # --- preferences ---
      pref_src_dir="${wdir}/user_preferences"
      if [[ -d "${pref_src_dir}" ]]; then
        echo "  → copying user_preferences from ${pref_src_dir}"

        typeset -A PREF_MAP
        PREF_MAP=(
          rec_preferences "rec_preferences"
          res_preferences "res_preferences"
          int_preferences "int_preferences"
          int_theta       "int_theta"
        )

        for name suffix in ${(kv)PREF_MAP}; do
          src="${pref_src_dir}/${name}.json"
          dst="${PREF_DIR}/world_${wid}_${suffix}.json"
          if [[ -f "${src}" ]]; then
            cp "${src}" "${dst}"
          else
            echo "    ⚠️  Missing ${src}"
          fi
        done
      else
        echo "  ⚠️  No user_preferences dir at ${pref_src_dir}"
      fi

      # --- per-policy config snapshots ---
      snap_src_dir="${wdir}/config_snapshots"
      if [[ -d "${snap_src_dir}" ]]; then
        echo "  → copying config snapshots from ${snap_src_dir}"
        mkdir -p "${CFG_DIR}/world_${wid}"
        cp -R "${snap_src_dir}/" "${CFG_DIR}/world_${wid}/"
      else
        echo "  ⚠️  No config_snapshots dir at ${snap_src_dir}"
      fi

      # --- also copy the world-level current config.py as fallback ---
      config_src="${wdir}/config.py"
      if [[ -f "${config_src}" ]]; then
        mkdir -p "${CFG_DIR}/world_${wid}"
        cp "${config_src}" "${CFG_DIR}/world_${wid}/config__LAST.py"
      else
        echo "  ⚠️  No config.py at ${config_src}"
      fi
    done

    echo "✅ Dump complete. Base folder: ${OUT_BASE}"
    ;;

  restore)
    ensure_study_tag "$@"
    STUDY_TAG=$1

    OUT_BASE="${RESULTS_ROOT}/${ENV_NAME}/${STUDY_TAG}"
    DUMP_DIR="${OUT_BASE}/db_dumps"
    DB_PREFIX="${STUDY_TAG}"

    echo "🗄  Restoring dumps from ${DUMP_DIR} into local Postgres DBs:"
    echo "   worlds: global WORLD_IDs ${WORLD_OFFSET} .. $((WORLD_OFFSET + N_WORLDS - 1))"
    echo "   DB name pattern: ${DB_PREFIX}_w<world_id>"

    for ((i=0; i<N_WORLDS; i++)); do
      GLOBAL_ID=$((WORLD_OFFSET + i))
      sql="${DUMP_DIR}/world_${GLOBAL_ID}.sql"
      db="${DB_PREFIX}_w${GLOBAL_ID}"

      if [[ ! -f "${sql}" ]]; then
        echo "  ⚠️  Missing dump file: ${sql} (skipping)"
        continue
      fi

      echo "------------------------------"
      echo "World ${GLOBAL_ID}"
      echo "  dump : ${sql}"
      echo "  db   : ${db}"

      dropdb --if-exists "${db}" || true
      createdb "${db}"

      # If you want to respect LOCAL_PG_* variables, do:
      # psql -h "${LOCAL_PG_HOST}" -p "${LOCAL_PG_PORT}" -U "${LOCAL_PG_USER}" "${db}" < "${sql}"
      psql "${db}" < "${sql}"
    done

    echo "✅ Restore complete."
    echo "   You now have DBs: ${DB_PREFIX}_w${WORLD_OFFSET} ... _w$((WORLD_OFFSET + N_WORLDS - 1))"
    ;;

  *)
    cat <<EOF
Usage: $0 <command> [args] [--flags]

Commands:
  dump <STUDY_TAG>     - dump all world DBs + copy preferences
  restore <STUDY_TAG>  - restore each world_<WORLD_ID>.sql into local DB:
                         <STUDY_TAG>_w<WORLD_ID>

Flags (optional, override env vars):
  --env / --env_name            (default: baseline_v1)
  --n_worlds / --n-worlds       (default: 10)
  --results_root / --results-root (default: results)
  --offset / --world_offset     (default: 0)
  --base_project_prefix         (default: \$ENV_NAME)

Typical workflows:

  # Single-batch run (no offset):
  N_WORLDS=10 ./manage_env.zsh up
  N_WORLDS=10 ./manage_results.zsh dump study01
  N_WORLDS=10 ./manage_results.zsh restore study01

  # With explicit flags:
  ./manage_results.zsh dump study01 --env baseline_v1 --n_worlds 5 --offset 10
  ./manage_results.zsh restore study01 --env baseline_v1 --n_worlds 5 --offset 10

EOF
    ;;
esac