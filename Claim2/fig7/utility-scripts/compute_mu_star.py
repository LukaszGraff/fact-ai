import argparse
import json
import os
import sys
from types import SimpleNamespace

import numpy as np

# Gym expects np.bool8; NumPy 2.0 removed it. Provide compatibility alias.
if not hasattr(np, "bool8"):
    np.bool8 = np.bool_

import gym
import jax

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
sys.path.append(os.path.dirname(__file__))
import environments  # noqa: F401
from buffer import Buffer
from FairDICE import init_train_state, train_step, get_model, save_model
from utils import normalization, min_max_normalization


def _load_or_generate_data(env_name, quality, preference_dist):
    data_path = os.path.join(
        ".", "data", env_name, f"{env_name}_50000_{quality}_{preference_dist}.pkl"
    )
    if not os.path.exists(data_path):
        from generate_fourroom_data import generate_offline_data
        generate_offline_data(
            env_name=env_name,
            num_trajectories=300,
            quality=quality,
            preference_dist=preference_dist,
            max_steps=400,
            behavior="random",
        )
    import pickle
    with open(data_path, "rb") as f:
        trajs = pickle.load(f)
    return trajs, data_path


def _build_config_and_batch(trajs, env, args):
    config = SimpleNamespace(
        learner="FairDICE",
        gamma=args.gamma,
        beta=args.beta,
        divergence=args.divergence,
        gradient_penalty_coeff=args.gradient_penalty_coeff,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        temperature=args.temperature,
        layer_norm=args.layer_norm,
        nu_lr=args.nu_lr,
        policy_lr=args.policy_lr,
        mu_lr=args.mu_lr,
        batch_size=args.batch_size,
        quality=args.quality,
        preference_dist=args.preference_dist,
        max_seq_len=args.max_seq_len,
        normalize_reward=args.normalize_reward,
        env_name=args.env_name,
        total_train_steps=args.total_train_steps,
        log_interval=args.log_interval,
        eval_episodes=args.eval_episodes,
        wandb=False,
        save_path=args.save_dir,
        seed=args.seed,
        alpha=args.alpha,
        is_discrete=True,
        tanh_squash_distribution=False,
    )

    config.state_dim = env.observation_space.shape[0]
    config.action_dim = env.action_space.n
    config.reward_dim = env.num_objectives

    all_states = np.concatenate([traj["observations"] for traj in trajs], axis=0)
    config.state_mean = all_states.mean(axis=0)
    config.state_std = all_states.std(axis=0) + 1e-8

    config.ACTION_SCALE = 1.0
    config.ACTION_BIAS = 0.0

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

    for traj in trajs:
        if config.normalize_reward:
            traj["rewards"] = min_max_normalization(traj["raw_rewards"], reward_min, reward_max)
        else:
            traj["rewards"] = traj["raw_rewards"]
        traj["states"] = normalization(traj["observations"], config.state_mean, config.state_std)
        traj["next_states"] = normalization(traj["next_observations"], config.state_mean, config.state_std)
        traj["actions"] = traj["actions"].astype(np.float32)
        traj["init_observations"] = np.tile(traj["observations"][0], (traj["observations"].shape[0], 1))
        traj["init_states"] = np.tile(traj["states"][0], (traj["states"].shape[0], 1))
        if "terminals" not in traj:
            traj["terminals"] = np.zeros(len(traj["observations"]))
            traj["terminals"][-1] = 1.0

    tmp = {}
    for traj in trajs:
        for key, value in traj.items():
            tmp.setdefault(key, []).append(value)

    batch = {}
    for key, values in tmp.items():
        batch[key] = np.concatenate(values, axis=0)

    config.hidden_dims = [config.hidden_dim] * config.num_layers
    return config, batch


def _train(config, batch):
    key = jax.random.PRNGKey(config.seed)
    train_state = init_train_state(config)
    train_carry = (train_state, key)
    buffer = Buffer(batch)

    def train_body(carry, _):
        train_state, key = carry
        key, subkey = jax.random.split(key)
        data = buffer.sample(subkey, config.batch_size)
        train_state, update_info = train_step(config, train_state, data, subkey)
        return (train_state, key), update_info

    train_iterations = config.total_train_steps // config.log_interval
    for _ in range(train_iterations):
        train_carry, _ = jax.lax.scan(train_body, train_carry, length=config.log_interval)

    return train_carry[0]


def main():
    parser = argparse.ArgumentParser(description="Train FairDICE-NSW and save mu* for a seed.")
    parser.add_argument("--env_name", type=str, default="MO-FourRoom-v2")
    parser.add_argument("--quality", type=str, default="amateur", choices=["expert", "amateur"])
    parser.add_argument("--preference_dist", type=str, default="uniform", choices=["uniform", "wide", "narrow"])
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--beta", type=float, default=0.01)
    parser.add_argument("--divergence", type=str, default="CHI")
    parser.add_argument("--gradient_penalty_coeff", type=float, default=1e-4)
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--layer_norm", type=bool, default=True)
    parser.add_argument("--nu_lr", type=float, default=3e-4)
    parser.add_argument("--policy_lr", type=float, default=3e-4)
    parser.add_argument("--mu_lr", type=float, default=3e-4)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--max_seq_len", type=int, default=200)
    parser.add_argument("--normalize_reward", type=bool, default=False)
    parser.add_argument("--total_train_steps", type=int, default=100_000)
    parser.add_argument("--log_interval", type=int, default=1000)
    parser.add_argument("--eval_episodes", type=int, default=50)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--save_dir", type=str, default="./results/mu_star")
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)
    trajs, _ = _load_or_generate_data(args.env_name, args.quality, args.preference_dist)
    env = gym.make(args.env_name)
    config, batch = _build_config_and_batch(trajs, env, args)

    train_state = _train(config, batch)
    mu_network, _, _ = get_model(train_state.mu_state)
    mu_star = np.asarray(mu_network(), dtype=np.float32)

    seed_dir = os.path.join(args.save_dir, f"seed_{args.seed}")
    os.makedirs(seed_dir, exist_ok=True)
    save_model(train_state, os.path.join(seed_dir, "model"))
    with open(os.path.join(seed_dir, "mu_star.json"), "w") as f:
        json.dump(mu_star.tolist(), f, indent=2)
    np.savez(os.path.join(seed_dir, "mu_star.npz"), mu_star=mu_star)
    print(f"Saved mu* to {seed_dir}")


if __name__ == "__main__":
    main()
