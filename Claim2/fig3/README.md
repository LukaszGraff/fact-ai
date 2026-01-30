# Fig3: Random MOMDP FairDICE Sweep

This folder contains the full pipeline used to produce Figure 3 on the random MOMDP setting. The steps below describe how to:
1) generate the random MOMDP dataset (auto-generated if missing),
2) train FairDICE to obtain \`mu_star\`,
3) run fixed-\`mu\` perturbations across (beta, sigma, seed),
4) aggregate results from distributed runs, and
5) create the final plot.

All commands below are written to be run from this directory:
`C:\Users\vladc\OneDrive\Escritorio\fact-ai\Claim2\fig3`

## Setup
Create the environment and install dependencies:
```
conda create -n factai python=3.10 -y
conda activate factai
pip install -r requirements.txt
```

## Step 1: Train FairDICE to get \`mu_star\`
This will (re)generate the random MOMDP dataset under \`data/random_momdp/\` if it does not exist, then train FairDICE and save the \`mu_star\` sweep.

Run locally:
```
python -m scripts.train_fairdice_mu_star \
  --seeds 0,1,2,3,4 \
  --betas 0.001,0.01,0.1,1.0 \
  --train_steps 10000 \
  --eval_episodes 500 \
  --log_interval 200 \
  --eval_interval 200 \
  --out_dir results/fig3_random_momdp
```

Expected outputs:
- \`results/fig3_random_momdp/mu_star.npz\`
- \`results/fig3_random_momdp/mu_star_meta.json\`
- \`results/fig3_random_momdp/final_nsw_mu_star.json\`

## Step 2: Run fixed-\`mu\` perturbations
This uses \`mu_star.npz\` and runs FairDICE with fixed \`mu\` under perturbations (sigma grid). It also writes a plot for each run.

### Option A: SLURM array (HPC)
Use the provided job file:
```
sbatch run_fixed_all_perturbations.sh
```

Note: The SLURM script uses a fixed \`cd /home/scur0132/claim2/fig3\`. Update it if your path differs.

### Option B: Local / single run
```
python -m scripts.run_fairdice_fixed_perturbation \
  --seeds 0,1,2,3,4 \
  --betas 0.001,0.01,0.1,1.0 \
  --sigmas 0.001,0.01,0.1,1.0 \
  --train_steps 10000 \
  --eval_episodes 500 \
  --log_interval 200 \
  --eval_interval 200 \
  --out_dir results/fig3_random_momdp \
  --mu_path results/fig3_random_momdp/mu_star.npz
```

Expected outputs (per run directory):
- \`fig3_random_momdp_results.npz\`
- \`fig3_random_momdp_meta.json\`
- \`final_nsw_fixed.json\`
- \`fig3_random_momdp.png\`

## Step 3: Aggregate results (if runs were distributed)
If you ran the grid as separate jobs (e.g., via SLURM array), merge them into a single results set:

```
python -m scripts.merge_fixed_results \
  --input_root results/fixed \
  --out_dir results/fixed_merged
```

Expected outputs:
- \`results/fixed_merged/fig3_random_momdp_results.npz\`
- \`results/fixed_merged/fig3_random_momdp_meta.json\`
- \`results/fixed_merged/final_nsw_fixed.json\`

If you also need to merge \`mu_star\` runs:
```
python -m scripts.merge_mu_star \
  --input_root results/fig3_random_momdp \
  --out_dir results/fig3_random_momdp_merged
```

## Step 4: Create the final plot
If you ran everything in a single run (Step 2, Option B), the plot is already saved as:
- \`results/fig3_random_momdp/fig3_random_momdp.png\`

If you merged distributed runs (Step 3), you can plot from the merged \`.npz\` file with the snippet below:

```
python - <<'PY'
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

npz_path = Path("results/fixed_merged/fig3_random_momdp_results.npz")
if not npz_path.exists():
    raise FileNotFoundError(npz_path)

with np.load(npz_path, allow_pickle=True) as data:
    betas = data["betas"].astype(float).tolist()
    sigmas = data["sigmas"].astype(float).tolist()
    nsw = data["nsw"]

mean_nsw = np.nanmean(nsw, axis=2)
if nsw.shape[2] > 1:
    stderr_nsw = np.nanstd(nsw, axis=2, ddof=1) / np.sqrt(nsw.shape[2])
else:
    stderr_nsw = np.zeros_like(mean_nsw)

sigmas_arr = np.array(sigmas, dtype=float)
positive_sigmas = sigmas_arr[sigmas_arr > 0]
min_pos = positive_sigmas.min() if positive_sigmas.size else 1.0
x_plot = sigmas_arr.copy()
x_plot[x_plot == 0] = min_pos * 0.1

fig, ax = plt.subplots(figsize=(6, 4))
for b_idx, beta in enumerate(betas):
    ax.errorbar(x_plot, mean_nsw[b_idx], yerr=stderr_nsw[b_idx], marker="o", capsize=3, label=f"beta={beta}")
ax.set_xscale("log")
ax.set_xlabel("sigma")
ax.set_ylabel("NSW")
ax.set_xticks(x_plot)
ax.set_xticklabels([str(s) for s in sigmas])
ax.legend()
ax.grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.5)
fig.tight_layout()

out_path = Path("results/fixed_merged/fig3_random_momdp.png")
fig.savefig(out_path, dpi=150)
print(f"Saved {out_path}")
PY
```

## Notes
- The dataset for this experiment is generated on the fly in \`scripts.generate_random_momdp_dataset\` and cached in \`data/random_momdp/\`.
- If \`mu_star.npz\` is missing, re-run Step 1.
- If the merge fails, verify each run folder contains \`fig3_random_momdp_results.npz\`.
