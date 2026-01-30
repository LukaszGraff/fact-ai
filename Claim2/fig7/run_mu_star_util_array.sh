#!/bin/bash
#SBATCH --job-name=mu_star_util
#SBATCH --partition=rome
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=08:00:00
#SBATCH --mem=16G
#SBATCH --array=0-4
#SBATCH --output=logs/mu_star_util_%A_%a.out

set -euo pipefail

module purge
module load 2025
module load Anaconda3/2025.06-1
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate factai
export PYTHONNOUSERSITE=1
unset PYTHONPATH

# If you uploaded the whole repo:
WORKDIR="/home/scur0132/claim2/fig7_bartek"

cd "$WORKDIR"
mkdir -p logs results/mu_star

s=${SLURM_ARRAY_TASK_ID}

python -u utility-scripts/compute_mu_star.py \
  --seed "$s" \
  --save_dir "./results/mu_star"
