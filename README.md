# Reproducing FairDICE: Fairness-Driven Offline Multi-Objective Reinforcement Learning

This repository contains the codebase of Group 2 for the course 5204FACT6Y - Fairness, Accountability, Confidentiality and Transparency in AI. It is the reproduction of the following paper: https://openreview.net/forum?id=2jQJ7aNdT1. This README has the following structure:

- First, we describe the setup of the environment with the necessary libraries for running our scripts.
- Second, we provide the links for downloading the datasets used for training and evaluating the models both on the continuous and discrete domains.
- Third, we provide the source-code oif each experiment verifying the specific claims addressed in our paper in a specific folder (e.g. Claim 3). 


## Acknowledgements / Attribution

This repository is uses the following projects:

- **FairDICE**: https://github.com/ku-dmlab/FairDICE  
- **PEDA**: https://github.com/baitingzbt/PEDA  

We gratefully acknowledge the authors and contributors of these repositories for their work, ideas, and implementations that made this project possible.


## Setup
This project requires installing two separate environments. To create the FairDICE environment use:

  ```
  conda env create -f environment.yml
  ```

To create the PEDA environment use:
 ```
  sh install_peda.sh
```

## Data Download
For the continuous case, the D4MORL dataset, a benchmark suite designed for offline multi-objective reinforcement learning (MORL) was used. 

To download the data, run:
```
pip install gdown
gdown --folder https://drive.google.com/drive/folders/1wfd6BwAu-hNLC9uvsI1WPEOmPpLQVT9k?usp=sharing --output data
```
For the discrete case, only the Random-MOMDP dataset needs to be downloaded. We provide a dataset for the case in which episodes terminate upon reaching a goal state (in the "data_terminate" folder), and separately the dataset in which episodes go in for a fixed set horizon limit (in the "data" folder). The main plot presented in the for Claim 3 uses the data available in the "data" folder. The datasets can be accessed and downloaded from the following OneDrive directory (permission is granted for people within the UvA network).

- https://amsuni-my.sharepoint.com/:f:/r/personal/karoly_bodgal_student_uva_nl/Documents/5204FACT6Y%20-%20Fairness,%20Accountability,%20Confidentiality%20and%20Transparency%20in%20AI?csf=1&web=1&e=wWSxLH 

## Claim 1

From the repo root, start by switching into the experiment directory:

```bash
cd claim1
```

### 1) Run FairDICE (all envs × datasets × betas × seeds)

This script launches FairDICE training/evaluation runs across multiple MuJoCo multi-objective environments, both `expert` and `amateur` datasets, several `beta` values, and multiple seeds. 

```bash
bash FairDICE/run_all.sh
```

If you are working on Snellius, you can also use `sbatch FairDICE/run_all.job`.

### 2) Visualize FairDICE results (NSW vs. beta)

`FairDICE/visualize.py` searches your run directory recursively for folders that contain an `eval/` subdir with `normalized_returns_step_*.npy`, then aggregates Nash Social Welfare (NSW) across seeds and plots **Amateur vs Expert** for each beta. 

Minimal example:

```bash
python FairDICE/visualize.py \
  --root FairDICE/results \
  --out fairdice_results.png
```

Helpful options:

* `--dataset_suffix` (e.g., `uniform`) to require tags like `amateur_uniform` / `expert_uniform` in folder names 
* `--envs ...` to restrict to specific env names
* `--ncols` to control subplot layout 

### 3) Aggregate FairDICE outputs into `.npz` for baseline comparison

This prepares FairDICE results into a standardized `.npz` format (one per env/dataset) that the baseline plotting script can load later. By default it writes into `PEDA/fairdice_npz/`. 

```bash
bash PEDA/prepare_fd.sh
```

Notes:

* It expects FairDICE results under `../FairDICE/results` relative to `PEDA/`. 
* It currently passes `--beta_filter "beta1.0"`; adjust that if you want to aggregate a different beta set. 

### 4) Train + evaluate all baselines (BC / MODT / MORvS)

This script runs baseline training/eval across:

* models: `bc`, `dt`, `rvs`
* envs: Hopper/Swimmer/HalfCheetah/Walker2d/Ant (v2) + Hopper-v3
* datasets: `expert_uniform`, `amateur_uniform`
* seeds: 1–5 

```bash
bash PEDA/train_all.sh
```

If you are working on Snellius, you can also use `sbatch PEDA/train_all.job`.

### 5) Make comparison plots (baselines + FairDICE)

`PEDA/make_plots.py` loads baseline rollout logs under `--runs_root` (expects subdirs like `dt/`, `bc/`, `rvs/`) and optionally overlays FairDICE from `--fairdice_dir` (your `.npz` folder from step 3). 

Common usage patterns:

**Metric panels + Pareto panels (2-objective envs):**

```bash
python PEDA/make_plots.py \
  --runs_root PEDA/experiment_runs/uniform \
  --fairdice_dir PEDA/fairdice_npz \
  --out_dir plots \
  --panel
```

**3-objective Hopper-v3 figure:**

```bash
python PEDA/make_plots.py \
  --runs_root PEDA/experiment_runs/uniform \
  --fairdice_dir PEDA/fairdice_npz \
  --out_dir plots \
  --do_hopper_v3
```

**Smaller “metrics subset” grid:**

```bash
python PEDA/make_plots.py \
  --runs_root PEDA/experiment_runs/uniform \
  --fairdice_dir PEDA/fairdice_npz \
  --out_dir plots \
  --metrics_subset
```

(See the CLI flags in `make_plots.py` for the full set.) 

## Claim 2
See the readme files for [Figure 3](Claim2/fig3/README.md), [Figure 7](Claim2/fig7/README.md) and [Extension](Claim2/extension/README.md).

## Claim 3

This directory contains the necessary scripts for running the experiment for the valiadation of Claim 3 regarding the change in model behaviour with respect to different $alpha$ - $beta$ configurations. The directory contains the following scripts:

- environments\random_momdp.py: describes the Random-MOMDP environment with a fixed horizon for each episode.
- environments\random_momdp_terminate.py: describes the Random-MOMDP environment with each episode terminating upon reaching a goal state.
- main_experiment_3.py: the main file used for running the experiment veifying Claim 3. Appropriate flags are included to allow for command line running of the framework.
- evaluate_momdp.py: the discrete modification of the original evaluate.py for the MOMDP environment
- plotting_momdp.py & plotting_momdp2.py: two separate files used for the plotting of the results as presented in our paper. There exists two separate files, since the second version was specifically created to be run on a set of results from a parallelized jobs.
- FairDICE_momdp.py & FairDICE_momdp2.py: these files contain the FairDICE framework accomodated to the discrete case. They additionally include the alpha parameter as a hyperparameter, required for the experiment. For the results of the paper, the second version was used.

The hyperparameters used for the training are set as the default parameters. For running the experiment, running main_experiment_3.py with the correct data directory suffices, then the resulting .csv files can be used for the plotting.

## Claim 4
See the [Claim 4 readme](claim4/README.md).


## License
This project is licensed under the MIT License.