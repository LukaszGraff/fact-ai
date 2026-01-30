#!/bin/bash
#SBATCH --job-name=fd_fixed_grid
#SBATCH --partition=rome
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=02:00:00
#SBATCH --mem=6G
#SBATCH --array=0-79
#SBATCH --output=logs/fd_fixed_grid_%A_%a.out

set -euo pipefail

module purge
module load 2025
module load Anaconda3/2025.06-1
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate factai

cd /home/scur0132/claim2/fig3
mkdir -p logs

# Grid
SEEDS=(0 1 2 3 4)
BETAS=(0.001 0.01 0.1 1.0)
SIGMAS=(0.001 0.01 0.1 1.0)

N_SEEDS=${#SEEDS[@]}
N_BETAS=${#BETAS[@]}
N_SIGMAS=${#SIGMAS[@]}
N_TOTAL=$((N_SEEDS * N_BETAS * N_SIGMAS))

TASK_ID=${SLURM_ARRAY_TASK_ID}
if [ "$TASK_ID" -ge "$N_TOTAL" ]; then
  echo "TASK_ID $TASK_ID out of range (0..$((N_TOTAL-1)))"
  exit 1
fi

# Map TASK_ID -> (seed, beta, sigma)
seed_idx=$(( TASK_ID / (N_BETAS * N_SIGMAS) ))
rem=$(( TASK_ID % (N_BETAS * N_SIGMAS) ))
beta_idx=$(( rem / N_SIGMAS ))
sigma_idx=$(( rem % N_SIGMAS ))

SEED=${SEEDS[$seed_idx]}
BETA=${BETAS[$beta_idx]}
SIGMA=${SIGMAS[$sigma_idx]}

# Common root you will merge from
OUT_DIR="results/fixed/seed=${SEED}/beta=${BETA}/sigma=${SIGMA}"
mkdir -p "${OUT_DIR}"

MU_PATH="results/fig3_random_momdp/mu_star.npz"

echo "TASK_ID=${TASK_ID} => SEED=${SEED} BETA=${BETA} SIGMA=${SIGMA}"
echo "OUT_DIR=${OUT_DIR}"
echo "MU_PATH=${MU_PATH}"

srun python -u -m scripts.run_fairdice_fixed_perturbation \
  --seeds "${SEED}" \
  --betas "${BETA}" \
  --sigmas "${SIGMA}" \
  --train_steps 10000 \
  --eval_episodes 500 \
  --log_interval 200 \
  --eval_interval 200 \
  --out_dir "${OUT_DIR}" \
  --mu_path "${MU_PATH}"
