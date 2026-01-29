import os
import gym
import pickle
import numpy as np
from collections import defaultdict
from utils import normalization, min_max_normalization
from buffer import Buffer
import argparse
from types import SimpleNamespace
from evaluate_momdp import evaluate_policy
import random
from tqdm import tqdm
import jax
from datetime import datetime
import jax.numpy as jnp
import csv

def log_final_metrics(save_dir, alpha, beta, seed, final_metrics):
    """Append final metrics for the current experiment to a CSV."""
    os.makedirs(save_dir, exist_ok=True)
    csv_path = os.path.join(save_dir, "experiment_results.csv")
    file_exists = os.path.exists(csv_path)

    with open(csv_path, mode="a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["alpha", "beta", "seed", "utilitarian", "jain", "nsw"])
        if not file_exists:
            writer.writeheader()  # write header if file is new
        writer.writerow({
            "alpha": alpha,
            "beta": beta,
            "seed": seed,
            **final_metrics
        })

def run_single_experiment(trajs, alpha, beta, seed, args, save_dir):
    """Run FairDICE training for a single alpha-beta-seed configuration."""
    config = SimpleNamespace(**vars(args))
    config.alpha = alpha
    config.beta = beta
    config.seed = seed

    # CREATE ENVIRONMENT & FIXED DIMENSIONS
    env = gym.make("RandomMOMDP-v1")
    config.is_discrete = True
    config.max_seq_len = 100
    config.normalize_state = True
    config.normalize_reward = False

    # State and action dimensions
    if hasattr(env.observation_space, 'shape') and len(env.observation_space.shape) > 0:
        config.state_dim = env.observation_space.shape[0]
    else:
        config.state_dim = 1

    if isinstance(env.action_space, gym.spaces.Discrete):
        config.action_dim = env.action_space.n
    else:
        config.action_dim = env.action_space.shape[0]

    config.reward_dim = getattr(env, 'obj_dim', 3)

    # State normalization (identity)
    config.state_mean = np.zeros(config.state_dim)
    config.state_std = np.ones(config.state_dim)
    config.ACTION_HIGH = np.array([float(env.action_space.n - 1)])
    config.ACTION_LOW = np.array([0.0])
    config.ACTION_SCALE = np.array([1.0])
    config.ACTION_BIAS = np.array([0.0])

    # Reward bounds
    reward_min, reward_max = None, None
    for traj in trajs:
        r = traj["raw_rewards"]
        r_min = r.min(axis=0)
        r_max = r.max(axis=0)
        if reward_min is None:
            reward_min, reward_max = r_min, r_max
        else:
            reward_min = np.minimum(reward_min, r_min)
            reward_max = np.maximum(reward_max, r_max)
    config.reward_min = reward_min
    config.reward_max = reward_max

    # Process trajectories
    for traj in trajs:
        traj["rewards"] = traj["raw_rewards"]
        traj["states"] = normalization(traj['observations'], config.state_mean, config.state_std)
        traj['next_states'] = normalization(traj['next_observations'], config.state_mean, config.state_std)
        traj['actions'] = traj['actions']
        traj["init_observations"] = np.tile(traj['observations'][0], (traj['observations'].shape[0], 1))
        traj["init_states"] = np.tile(traj['states'][0], (traj['states'].shape[0], 1))

    # Concatenate trajectories
    tmp = defaultdict(list)
    for traj in trajs:
        traj["terminals"] = traj["dones"]
        for key, value in traj.items():
            tmp[key].append(value)
    real_data = {key: np.concatenate(values, axis=0) for key, values in tmp.items()}
    real_data['masks'] = jnp.array(1.0 - real_data['terminals'].reshape(-1, 1).astype(np.float32))

    buffer = Buffer(real_data)

    # FairDICE hyperparameters
    config.hidden_dims = [256, 256]
    config.layer_norm = True
    config.divergence = "SOFT_CHI"
    config.tanh_squash_distribution = False
    config.nu_lr = 1e-4
    config.policy_lr = 1e-4
    config.mu_lr = 1e-4
    config.gradient_penalty_coeff = 1e-4
    config.temperature = 1.0
    config.num_layers = 2
    config.hidden_dim = 256

    from FairDICE_momdp import init_train_state, train_step, get_model, save_model

    random.seed(seed)
    np.random.seed(seed)
    key = jax.random.PRNGKey(seed)

    train_state = init_train_state(config)
    train_carry = (train_state, key)

    def train_body(carry, _):
        train_state, key = carry
        key, subkey = jax.random.split(key)
        data = buffer.sample(subkey, config.batch_size)
        train_state, update_info = train_step(config, train_state, data, subkey)
        return (train_state, key), (update_info)

    train_iterations = config.total_train_steps // config.log_interval
    for iter in tqdm(range(train_iterations), desc="Training FairDICE"):
        step = (iter + 1) * config.log_interval
        train_carry, update_info = jax.lax.scan(train_body, train_carry, length=config.log_interval)

        # Evaluate at each log interval (optional, or only at the last step)
        if iter == train_iterations - 1:  # Only evaluate at the end
            policy = get_model(train_carry[0].policy_state)[0]
            final_metrics = evaluate_policy(config, policy, env, save_dir="./tmp_eval",
                                            num_episodes=config.eval_episodes,
                                            max_steps=config.max_seq_len,
                                            t_env=step)
            print(f"Step {step}: nu_loss={update_info['nu_loss'][-1]:.4f}, Final metrics: {final_metrics}")

    # Evaluate policy after training
    policy = get_model(train_carry[0].policy_state)[0]
    final_metrics = evaluate_policy(config, policy, env, save_dir=save_dir,
                                    num_episodes=config.eval_episodes,
                                    max_steps=config.max_seq_len,
                                    t_env=None)
    # Log final metrics
    log_final_metrics(save_dir, alpha, beta, seed, final_metrics)

def main():
    parser = argparse.ArgumentParser(description="FairDICE RandomMOMDP Batch Experiments")
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--gamma", type=float, default=0.95)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--divergence", type=str, default="CHI", help="Divergence type (SOFT_CHI/CHI/KL)")
    parser.add_argument("--total_train_steps", type=int, default=1000)
    parser.add_argument("--log_interval", type=int, default=1000)
    parser.add_argument("--eval_episodes", type=int, default=100)
    parser.add_argument("--save_path", type=str, default='./results')
    parser.add_argument("--seed_start", type=int, default=1, help="First seed in the range")
    parser.add_argument("--seed_end", type=int, default=50, help="Last seed in the range (inclusive)")
    args = parser.parse_args()

    # Define hyperparameter grid
    alphas = [0.0, 0.5, 1.0, 1.25]
    betas = [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0, 5.0, 10.0, 50.0, 100.0]
    seeds = list(range(args.seed_start, args.seed_end + 1))

    for seed in seeds:
        # Load the dataset corresponding to this seed
        # Instead of using the home folder data path, use the compute node tmp folder
        # Use the compute node temporary folder
        #TMP_DATA = f"/tmp/{os.getenv('USER')}/fairdice_data/RandomMOMDP-v0"

        # Construct the full path to the pickle file for this seed
        #data_path = os.path.join(TMP_DATA, f"RandomMOMDP-v0_50000_expert_uniform_{seed}.pkl")
        data_path = f"./data_terminate/RandomMOMDP-v1/RandomMOMDP-v1_50000_expert_uniform_{seed}.pkl"
        print(f"Loading dataset: {data_path}")
        with open(data_path, "rb") as f:
            trajs = pickle.load(f)
        print(f"Loaded {len(trajs)} trajectories for seed {seed}")
        # Dynamic results folder per seed range
        save_dir = os.path.join(args.save_path, f"seeds_{args.seed_start}_{args.seed_end}")
        os.makedirs(save_dir, exist_ok=True)

        for alpha in alphas:
            for beta in betas:
                print(f"Running experiment: seed={seed}, alpha={alpha}, beta={beta}")
                run_single_experiment(trajs, alpha, beta, seed, args, save_dir)

if __name__ == "__main__":
    main()

