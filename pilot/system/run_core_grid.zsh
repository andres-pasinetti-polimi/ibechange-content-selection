#!/usr/bin/env zsh
set -eu

# -------------------------------
# Common settings for all envs
# -------------------------------
export N_WORLDS=10
export BATCH_SIZE=5
export POLL_SECONDS=30
export ORCH_SERVICE_NAME="orchestrator"

#export MAB_TYPES="Random,ContextBase,TwoStageBase,TwoStageFull,Oracle"
export MAB_TYPES="Random,ContextBase,ContextFull,TwoStageBase,TwoStageFull,HybridBase_bvar025,HybridFull_bvar025,Oracle"
#export MAB_TYPES="HybridBase,HybridFull"
export HOURS_PER_POLICY=$((24 * 7 * 12))

# -------------------------------
# (ENV_NAME, STUDY_TAG) pairs
# adjust STUDY_TAG strings as you like
# -------------------------------
typeset -a CORE_ENVS=(
  "core-topfb-bias10:blockD_topfb_bias10_900users_S10_v3"
  #"core-ref:core_ref_allpolicies_20seeds_v1"
  #"core-ref:core_ref_hybrid_bvar025_20seeds_v0"
  #"core-ref:core_ref_hybrid_20seeds_v0"
  #"core-ref-nobias:core_ref_nobias_allpolicies_20seeds_v0"
  #"core-ref-nobias:core_ref_nobias_hybrid_bvar025_20seeds_v0"
  #"core-ref-nobias:core_ref_nobias_hybrid_20seeds_v0"
  #"core-ref-highbias:core_ref_highbias_allpolicies_20seeds_v0"
  #"core-lowfb:core_lowfb_allpolicies_20seeds_v0"
  #"core-lowfb:core_lowfb_hybrid_bvar025_20seeds_v0"
  #"core-lowfb-nobias:core_lowfb_nobias_allpolicies_20seeds_v0"
  #"core-highfb:core_highfb_allpolicies_20seeds_v0"
  #"core-highfb-nobias:core_highfb_nobias_allpolicies_v0"
  #"core-highfb-highbias:core_highfb_highbias_hybrid_bvar025_20seeds_v0"
)

# -------------------------------
# Loop over environments
# -------------------------------
for pair in "${CORE_ENVS[@]}"; do
  IFS=':' read -r ENV_NAME STUDY_TAG <<< "$pair"

  export ENV_NAME
  export STUDY_TAG

  echo
  echo "======================================="
  echo "Starting env='${ENV_NAME}', study='${STUDY_TAG}'"
  echo "======================================="

  ./run_study.zsh

  echo
  echo "======================================="
  echo "Finished env='${ENV_NAME}', study='${STUDY_TAG}'"
  echo "Results under: results/${ENV_NAME}/${STUDY_TAG}"
  echo "======================================="
done

echo
echo "✅ All CORE_SCENARIO_GRID environments completed."