#!/bin/bash

FD_ROOT="../FairDICE/results"

OUT_DIR="fairdice_npz"
mkdir -p "${OUT_DIR}"

ENVS=(
  "MO-Hopper-v2"
  "MO-Walker2d-v2"
  "MO-Ant-v2"
  "MO-HalfCheetah-v2"
  "MO-Swimmer-v2"
  "MO-Hopper-v3"
)

DATASETS=(
  "expert_uniform"
  "amateur_uniform"
)

for ENV in "${ENVS[@]}"; do
  for DS in "${DATASETS[@]}"; do
    echo "============================================================"
    echo "Aggregating FairDICE for ENV=${ENV}, DATASET=${DS}"
    echo "  (all betas, all seeds)"
    echo "============================================================"

    OUT_FILE="${OUT_DIR}/fairdice_${ENV}_${DS}_allbetas.npz"

    python prepare_fd.py \
      --root "${FD_ROOT}" \
      --env "${ENV}" \
      --dataset "${DS}" \
      --beta_filter "beta1.0" \
      --out "${OUT_FILE}"
    echo
  done
done

echo "All FairDICE npz files are in ${OUT_DIR}/"