#!/bin/bash
#SBATCH --job-name=merge_fixed
#SBATCH --partition=rome
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=01:00:00
#SBATCH --mem=8G
#SBATCH --output=logs/merge_fixed_%j.out

set -euo pipefail

module purge
module load 2025
module load Anaconda3/2025.06-1
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate factai

cd /home/scur0132/claim2/fig3/fact_snellius
mkdir -p logs

python -u -m scripts.merge_fixed_results \
  --input_root results/fixed \
  --out_dir results/fixed_merged
