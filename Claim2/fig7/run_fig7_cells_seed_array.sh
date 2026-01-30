#!/bin/bash
#SBATCH --job-name=fig7_cells_s5
#SBATCH --partition=rome
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=08:00:00
#SBATCH --mem=16G
#SBATCH --array=0-2204
#SBATCH --output=logs/fig7_cells_s5_%A_%a.out

set -euo pipefail

module purge
module load 2025
module load Anaconda3/2025.06-1
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate factai

export PYTHONNOUSERSITE=1
unset PYTHONPATH

cd /home/scur0132/claim2/fig7_bartek
mkdir -p logs results/fig7_cells

TASK=${SLURM_ARRAY_TASK_ID}
SEED=$(( TASK / 441 ))
CELL=$(( TASK % 441 ))

OUT_DIR="results/fig7_cells/seed=${SEED}/cell=${CELL}"
mkdir -p "${OUT_DIR}"

echo "JOB=${SLURM_JOB_ID} TASK=${TASK} SEED=${SEED} CELL=${CELL}"
echo "OUT_DIR=${OUT_DIR}"

python -u utility-scripts/figure7_fourroom.py \
  --seed "${SEED}" \
  --mu_star_path ./results/mu_star/mu_star_avg.npz \
  --save_dir "${OUT_DIR}" \
  --grid_points 21 --grid_min -0.1 --grid_max 0.1 \
  --cell_index "${CELL}"
