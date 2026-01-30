# FairDICE (Reproduction)

This codebase reproduces serves to reproduce the FairDICE results for offline multi-objective RL, specifically results pertaining to the Four Room environment setting and Claim 4. It adds the MO-Four-Room environment, visualization scripts, goal-mix sweeps and other utilities used in the reproduction study.

## Contents

### Core training and evaluation
- `main.py`: Main training/evaluation for MuJoCo/D4MORL tasks, not used for the discrete Four Room setting.
- `main_fourroom.py`: Training/evaluation entrypoint for the FourRoom environment.
- `FairDICE.py`: FairDICE model, training step, and checkpoint I/O for continuous-control tasks.
- `evaluation.py`: Policy evaluation utilities (NSW, utility metrics, episode rollouts).
- `buffer.py`: Offline dataset buffer and sampling utilities.
- `utils.py`: Normalization and helper functions.

### Environments and data
- `environments/`: Reproduced and wrapped environments.
  - `MO-Four-Room.py`, `MO-Four-Room-gymnasium.py`, `MO-Nine-Room.py`: discrete X-Room environment definitions.
- `data/`: Offline datasets (generated FourRoom datasets).

### Job files (SLURM)
- `job-files/four-room-train.job`: Single FourRoom training run (FairDICE).
- `job-files/four-room-goalmix-sweep.job`: Goal-mix sweep for FourRoom.
- `job-files/four-room-sweep-and-vis.job`: Sweep + optional visualization pipeline.
- `job-files/four-room-visualize.job`: Visualize a trained FourRoom policy.
- `job-files/four-room-eval-welfare.job`: Evaluate NSW and aggregate results.

### Utility scripts
- `utility-scripts/train_fairdice_single_with_cc.py`: Run a single training job under CodeCarbon.
- `utility-scripts/evaluate_sweep_welfare.py`: Evaluate a sweep and write a CSV of metrics.
- `utility-scripts/merge_goalmix_nsw.py`: Merge sweep CSV with goal reach ratios.
- `utility-scripts/plot_goalmix_ternary.py`: Plot a ternary NSW vs reach ratios figure.
- `utility-scripts/compute_objective_reach.py`: Compute goal reach stats for a saved model.
- `utility-scripts/compute_dataset_welfare.py`: Compute NSW/USW/Jain for a dataset.
- `utility-scripts/generate_fourroom_data.py`: Generate FourRoom offline datasets.

### Results and logs
- `results/`: Checkpoints and sweep outputs.
- `logs/`: SLURM outputs and CodeCarbon logs.

## Setup

Create the environment:
```
conda env create -f environment.yml
conda activate fairdice
```

If on Snellius, load Anaconda before activating:
```
module purge
module load 2025
module load Anaconda3/2025.06-1
conda activate fairdice
```

## Example commands (run from `claim4/`)

`main_fourroom.py`:
```
python main_fourroom.py \
  --learner FairDICE \
  --divergence CHI \
  --env_name MO-FourRoom-v2 \
  --quality amateur \
  --beta 0.001 \
  --seed 1984 \
  --preference_dist uniform \
  --eval_episodes 10 \
  --batch_size 64 \
  --hidden_dim 256 \
  --num_layers 2 \
  --total_train_steps 100000 \
  --log_interval 1000 \
  --normalize_reward False \
  --max_seq_len 200 \
  --policy_lr 3e-4 \
  --nu_lr 3e-4 \
  --mu_lr 3e-4 \
  --gamma 0.99 \
  --save_model_mode last
```

`utility-scripts/visualize_fourroom.py` (single model):
```
python utility-scripts/visualize_fourroom.py \
  --model_path results/<run_dir>/model \
  --per_state_cap 1 \
  --max_steps 200 \
  --num_episodes 100 \
  --stochastic \
  --env_name MO-FourRoom-v2
```

`utility-scripts/visualize_fourroom.py` (aggregate across a sweep):
```
python utility-scripts/visualize_fourroom.py \
  --sweep_dir results/goalmix_sweep_YYYYMMDD_HHMMSS \
  --episodes_per_model 5 \
  --max_steps 200 \
  --env_name MO-FourRoom-v2
```

### Utility scripts
`utility-scripts/generate_fourroom_data.py`:
```
python utility-scripts/generate_fourroom_data.py \
  --env_name MO-FourRoom-v2 \
  --num_trajectories 300 \
  --behavior random
```

`utility-scripts/train_fairdice_single_with_cc.py`:
```
python utility-scripts/train_fairdice_single_with_cc.py \
  --cc project_name=FairDICE \
  --cc output_file=fairdice_runs_.csv
```

`utility-scripts/evaluate_sweep_welfare.py`:
```
python utility-scripts/evaluate_sweep_welfare.py \
  --sweep_dir results/goalmix_sweep_YYYYMMDD_HHMMSS \
  --episodes 100 \
  --group_by mix
```

`utility-scripts/compute_objective_reach.py`:
```
python utility-scripts/compute_objective_reach.py \
  --run_dir results/<run_dir> \
  --episodes 100
```

`utility-scripts/compute_dataset_welfare.py`:
```
python utility-scripts/compute_dataset_welfare.py \
  --data_path data/MO-FourRoom-v2/MO-FourRoom-v2_50000_amateur_uniform.pkl
```

`utility-scripts/merge_goalmix_nsw.py`:
```
python utility-scripts/merge_goalmix_nsw.py \
  --csv results/goalmix_sweep_YYYYMMDD_HHMMSS/reeval_welfare_episodes_100.csv \
  --data_dir data/MO-FourRoom-v2
```

`utility-scripts/plot_goalmix_ternary.py`:
```
python utility-scripts/plot_goalmix_ternary.py \
  --csv results/goalmix_sweep_YYYYMMDD_HHMMSS/reeval_welfare_episodes_100_with_reach.csv \
  --output results/goalmix_sweep_YYYYMMDD_HHMMSS/ternary_nsw.png
```

## Data

### FourRoom
A FourRoom dataset should be automatically generated by `main_fourroom.py` if missing. You can also generate explicitly (see the `generate_fourroom_data.py` example above).

## Notes on paths
- For SLURM jobs, the scripts assume `cd $HOME/fact-ai/claim4`, so use paths
  relative to that repo root.

## Reproducibility
- All default hyperparameters are documented in the job files and CLI commands above.
- Seeds: set `--seed` in `main_fourroom.py`.
