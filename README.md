# Reproducing FairDICE: Fairness-Driven Offline Multi-Objective Reinforcement Learning

This repository contains the codebase of Group 2 for the course 5204FACT6Y - Fairness, Accountability, Confidentiality and Transparency in AI. The directory has the following structure:

\begin{itemize}
\item First, we describe the setup of the environment with the necessary libraries for running our scripts.
\item Second, 
\end{itemize}


## Setup
  Use the offered Dockerfile for the setup and create conda environment using yml file.
  ```
  cd FairDICE
  conda env create -f environment.yml
  conda activate fairdice
  ```

## Data Download
This repository uses the D4MORL dataset, a benchmark suite designed for offline multi-objective reinforcement learning (MORL). The dataset was introduced in the following paper:

Zhu, Baiting, Meihua Dang, and Aditya Grover.
Scaling Pareto-Efficient Decision Making via Offline Multi-Objective RL.
The Eleventh International Conference on Learning Representations (ICLR), 2023.

D4MORL provides diverse multi-objective versions of standard MuJoCo locomotion tasks (e.g., Hopper, Walker2d, HalfCheetah), enabling the evaluation of Pareto-efficient and fairness-aware policies under offline constraints.

To download the data, run:
```
pip install gdown
gdown --folder https://drive.google.com/drive/folders/1wfd6BwAu-hNLC9uvsI1WPEOmPpLQVT9k?usp=sharing --output data
```

## Training
If you want to run all experiments
```
./run_all.sh
```
or if you want to run a single experiment
```
CUDA_VISIBLE_DEVICES=0 python main.py --learner FairDICE --divergence SOFT_CHI --env_name MO-Hopper-v2 --quality expert --beta 0.1 --preference_dist uniform --eval_episodes 10 --batch_size 256 --hidden_dim 768 --num_layers 3 --total_train_steps 100000 --log_interval 1000 --normalize_reward  True
```

## License
This project is licensed under the MIT License.
