# FairDICE Extension: Full Experiment → Aggregation → Plot

This folder contains the full pipeline used for the Claim2 extension experiments. The steps below describe how to:
1) run the base FairDICE experiments,
2) compute the mean `mu` per environment,
3) generate perturbed `mu` tasks,
4) run the fixed-`mu` perturbation experiments,
5) aggregate NSW scores, and
6) create the final plot.

All commands below are written to be run from this directory:
`C:\Users\vladc\OneDrive\Escritorio\fact-ai\Claim2\extension`

## Setup
Create the environment and install dependencies:
```
conda env create -f environment.yml
conda activate fairdice
```

If you prefer pip:
```
pip install -r requirements.txt
```

## Data (manual download required)
The D4MORL dataset is **not** generated automatically by this code. You must download it and place it in the expected path.

`main.py` expects D4MORL data in `../data/<ENV>/<ENV>_50000_<quality>_<pref>.pkl`.
For example:
`../data/MO-Hopper-v2/MO-Hopper-v2_50000_expert_uniform.pkl`

If you already have the data, ensure it is placed in `Claim2/data/` (one level above this folder).

## Step 1: Run the base FairDICE experiments (produce `final_mu.txt`)
These runs create `results/<run_name>/final_mu.txt`, which is required for computing the average `mu`.

### Option A: SLURM array (HPC)
Use the provided job file:
```
sbatch run_fairdice_cpu_array.job
```

### Option B: Local / single run
```
python main.py ^
  --learner FairDICE ^
  --divergence SOFT_CHI ^
  --env_name MO-Hopper-v2 ^
  --quality expert ^
  --beta 0.1 ^
  --preference_dist uniform ^
  --eval_episodes 10 ^
  --batch_size 32 ^
  --hidden_dim 512 ^
  --num_layers 3 ^
  --total_train_steps 100000 ^
  --log_interval 1000 ^
  --normalize_reward True ^
  --seed 1
```

After this step, you should have:
`results/<run_name>/final_mu.txt`

## Step 2: Compute mean `mu` per environment
This scans all `results/*/final_mu.txt` and writes `results/mu_averages.txt`.
```
python compute_mu_avg.py
```

Expected output:
`results/mu_averages.txt`

## Step 3: Generate perturbed `mu` tasks
This uses `results/mu_averages.txt` to create a sweep of perturbation tasks.
```
python make_mu_perturb_tasks.py
```

Expected output:
`mu_perturb_tasks.txt`

If you need to change the perturbation stds or seeds, edit `STDS` and `SEEDS` in `make_mu_perturb_tasks.py`.

## Step 4: Run fixed-`mu` perturbation experiments
These runs use the generated tasks file and save into `results_mu_perturb/std_<STD>/`.

### Option A: SLURM array (HPC)
```
sbatch run_fairdice_fixed_perturb_array.job
```

### Option B: Local / single run
Pick a line from `mu_perturb_tasks.txt` and run:
```
python main.py ^
  --learner FairDICE_Fixed ^
  --divergence SOFT_CHI ^
  --env_name MO-Hopper-v2 ^
  --quality expert ^
  --beta 0.1 ^
  --preference_dist uniform ^
  --eval_episodes 10 ^
  --batch_size 128 ^
  --hidden_dim 768 ^
  --num_layers 3 ^
  --total_train_steps 100000 ^
  --log_interval 1000 ^
  --normalize_reward True ^
  --seed 1 ^
  --mu_init "0.1234,0.5678,0.9012" ^
  --save_path "results_mu_perturb/std_0.01"
```

Note: `aggregate_nsw_scores.py` also checks `results_mu_perturb_ant/` if it exists. If you want to separate out MO-Ant runs, save them there.

## Step 5: Aggregate NSW scores
This reads evaluation outputs from `results_mu_perturb/` (and `results_mu_perturb_ant/` if present) and writes a summary.
```
python aggregate_nsw_scores.py
```

Expected output:
`results/nsw_summary_mu_perturb_all.txt`

## Step 6: Create the plot
This uses the summary file and saves a PNG in `results/`.
```
python plot_nsw_mean_raw_by_std.py
```

Expected output:
`results/nsw_mean_raw_by_std.png`

## Notes
- Run names are auto-generated in `main.py` as:
  `YYYYMMDD_HHMMSS_<learner>_<env>_<quality>_<pref>_<div>_beta<beta>_seed<seed>`
- If `results/mu_averages.txt` is empty, confirm Step 1 produced `final_mu.txt` files.
- If the plot fails, make sure `results/nsw_summary_mu_perturb_all.txt` contains data.

## License
This project is licensed under the MIT License.
