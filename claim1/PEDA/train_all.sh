#!/bin/bash

module purge
module load 2025
module load Anaconda3/2025.06-1

cd $HOME/fact-ai/PEDA
source activate peda_env

export MUJOCO_PY_MUJOCO_PATH="$HOME/.mujoco/mujoco210"
export MUJOCO_GL=egl
export D4RL_SUPPRESS_IMPORT_ERROR=1
export LD_LIBRARY_PATH="/usr/lib/nvidia:/usr/lib64/nvidia:$MUJOCO_PY_MUJOCO_PATH/bin:$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"

BASE_DIR="experiment_runs/uniform"
MODELS=(bc dt rvs)

ENVS=(
  "MO-Hopper-v2"
  "MO-Swimmer-v2"
  "MO-HalfCheetah-v2"
  "MO-Walker2d-v2"
  "MO-Ant-v2"
  "MO-Hopper-v3"
)

DATASETS=(expert_uniform amateur_uniform)
SEEDS=(1 2 3 4 5)

STEP=100000
ITER=1

for MODEL in "${MODELS[@]}"; do
  DIR="${BASE_DIR}/${MODEL}"

  for ENV in "${ENVS[@]}"; do

    if [ "${MODEL}" = "rvs" ]; then
      CSP=1; CAP=0; CRP=0
    elif [ "${MODEL}" = "dt" ]; then
      CSP=1; CAP=1; CRP=1
    else
      CSP=1; CAP=0; CRP=0
    fi

    if [ "${ENV}" = "MO-Hopper-v3" ]; then
      EVAL_PREF_MODE="dirichlet"
      NUM_EVAL_PREFS=50
      GRAN=1
      PREF_SEED=0
    else
      EVAL_PREF_MODE="grid"
      NUM_EVAL_PREFS=0
      GRAN=30
      PREF_SEED=0
    fi

    for DATASET in "${DATASETS[@]}"; do
      for SEED in "${SEEDS[@]}"; do
        echo "=================================================================="
        echo "MODEL=${MODEL} ENV=${ENV} DATASET=${DATASET} SEED=${SEED}"
        echo "steps/iter=${STEP}, iters=${ITER}, eval_episodes=10"
        echo "concat_state_pref=${CSP}, concat_act_pref=${CAP}, concat_rtg_pref=${CRP}"
        echo "eval_pref_mode=${EVAL_PREF_MODE}, granularity=${GRAN}, num_eval_prefs=${NUM_EVAL_PREFS}"
        echo "=================================================================="

        EXTRA_EVAL_ARGS=()
        if [ "${EVAL_PREF_MODE}" = "dirichlet" ]; then
          EXTRA_EVAL_ARGS+=(--eval_pref_mode dirichlet --num_eval_prefs "${NUM_EVAL_PREFS}" --pref_seed "${PREF_SEED}")
        else
          EXTRA_EVAL_ARGS+=(--eval_pref_mode grid --granularity "${GRAN}" --pref_seed "${PREF_SEED}")
        fi

        python run_with_cc.py \
          --cc project_name="PEDA" \
          --cc experiment_id="MODEL=${MODEL}|ENV=${ENV}|DATASET=${DATASET}" \
          --cc measure_power_secs=30 \
          --cc output_file=peda_runs.csv \
          experiment.py -- \
          --dir "${DIR}" \
          --env "${ENV}" \
          --data_mode _formal \
          --concat_state_pref "${CSP}" \
          --concat_rtg_pref "${CRP}" \
          --concat_act_pref "${CAP}" \
          --mo_rtg True \
          --seed "${SEED}" \
          --dataset "${DATASET}" \
          --model_type "${MODEL}" \
          --num_steps_per_iter "${STEP}" \
          --max_iters "${ITER}" \
          --num_eval_episodes 10 \
          --normalize_reward True \
          "${EXTRA_EVAL_ARGS[@]}"
      done
    done
  done
done