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
from evaluation import evaluate_policy
from FairDICE import init_train_state, train_step, get_model, load_model, save_model
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


def _evaluate_metrics(config, policy, env, eval_dir):
    raw_returns, _ = evaluate_policy(
        config,
        policy,
        env,
        eval_dir,
        num_episodes=config.eval_episodes,
        max_steps=config.max_seq_len,
        t_env=None,
        key=jax.random.PRNGKey(config.seed),
    )
    returns = np.asarray(raw_returns)
    avg_returns = returns.mean(axis=0)
    eps = 1e-8
    nsw = float(np.sum(np.log(np.clip(avg_returns, eps, None))))
    util = float(np.sum(avg_returns))
    numerator = float(np.sum(avg_returns) ** 2)
    denom = float(len(avg_returns) * np.sum(avg_returns ** 2) + 1e-8)
    jain = numerator / denom
    return nsw, util, jain, avg_returns


def _load_mu_star(run_dir, config):
    model_dir = os.path.join(run_dir, "model")
    if not os.path.isdir(model_dir):
        raise FileNotFoundError(f"model dir not found: {model_dir}")
    train_state = load_model(os.path.abspath(model_dir), config)
    mu_network, _, _ = get_model(train_state.mu_state)
    mu_star = np.asarray(mu_network(), dtype=np.float32)
    return mu_star


def _load_mu_star_from_path(path):
    if path.endswith(".npz"):
        data = np.load(path)
        if "mu_star" not in data:
            raise KeyError(f"mu_star not found in {path}")
        return np.asarray(data["mu_star"], dtype=np.float32)
    with open(path, "r") as f:
        mu_star = json.load(f)
    return np.asarray(mu_star, dtype=np.float32)


def main():
    parser = argparse.ArgumentParser(description="Recreate Figure 7 for MO-Four-Room.")
    parser.add_argument("--env_name", type=str, default="MO-FourRoom-v2")
    parser.add_argument("--quality", type=str, default="amateur", choices=["expert", "amateur"])
    parser.add_argument("--preference_dist", type=str, default="uniform", choices=["uniform", "wide", "narrow"])
    parser.add_argument("--alpha", type=float, default=1.0, help="Use alpha=1.0 for NSW.")
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
    parser.add_argument("--save_dir", type=str, default="./results/fig7")
    parser.add_argument("--mu_star_run_dir", type=str, default="", help="Use existing run dir to load mu*.")
    parser.add_argument("--mu_star_path", type=str, default="", help="Path to mu* .json or .npz to use.")
    parser.add_argument("--grid_min", type=float, default=-0.1, help="Min perturbation for mu2/mu3.")
    parser.add_argument("--grid_max", type=float, default=0.1, help="Max perturbation for mu2/mu3.")
    parser.add_argument("--grid_points", type=int, default=21, help="Number of grid points per axis.")
    parser.add_argument("--output", type=str, default="fig7.png")
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)

    trajs, data_path = _load_or_generate_data(args.env_name, args.quality, args.preference_dist)
    env = gym.make(args.env_name)
    config, batch = _build_config_and_batch(trajs, env, args)

    if args.mu_star_path:
        mu_star = _load_mu_star_from_path(args.mu_star_path)
    elif args.mu_star_run_dir:
        mu_star = _load_mu_star(args.mu_star_run_dir, config)
    else:
        mu_star_state = _train(config, batch)
        mu_network, _, _ = get_model(mu_star_state.mu_state)
        mu_star = np.asarray(mu_network(), dtype=np.float32)
        mu_star_dir = os.path.join(args.save_dir, "mu_star_model")
        os.makedirs(mu_star_dir, exist_ok=True)
        save_model(mu_star_state, os.path.join(mu_star_dir, "model"))
        with open(os.path.join(mu_star_dir, "mu_star.json"), "w") as f:
            json.dump(mu_star.tolist(), f, indent=2)

    if mu_star.shape[0] != 3:
        raise ValueError(f"Expected 3 objectives for MO-FourRoom, got {mu_star.shape[0]}")

    deltas = np.linspace(args.grid_min, args.grid_max, args.grid_points)
    n = args.grid_points
    nsw_grid = np.zeros((n, n), dtype=np.float64)
    util_grid = np.zeros((n, n), dtype=np.float64)
    jain_grid = np.zeros((n, n), dtype=np.float64)
    obj_grid = np.zeros((n, n, 3), dtype=np.float64)

    for i, d2 in enumerate(deltas):
        for j, d3 in enumerate(deltas):
            fixed_mu = np.array([mu_star[0], mu_star[1] * (1 + d2), mu_star[2] * (1 + d3)], dtype=np.float32)
            run_dir = os.path.join(args.save_dir, f"mu2_{d2:+.3f}_mu3_{d3:+.3f}")
            os.makedirs(run_dir, exist_ok=True)
            cfg = SimpleNamespace(**vars(config))
            cfg.fixed_mu = fixed_mu
            cfg.seed = args.seed
            train_state = _train(cfg, batch)
            policy, _, _ = get_model(train_state.policy_state)
            nsw, util, jain, avg_returns = _evaluate_metrics(cfg, policy, env, run_dir)
            nsw_grid[i, j] = nsw
            util_grid[i, j] = util
            jain_grid[i, j] = jain
            obj_grid[i, j, :] = avg_returns

    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 3, figsize=(12, 8))
    panels = [
        ("Nash Social Welfare", nsw_grid),
        ("Utilitarian", util_grid),
        ("Jain's Fairness", jain_grid),
        ("Objective 1", obj_grid[:, :, 0]),
        ("Objective 2", obj_grid[:, :, 1]),
        ("Objective 3", obj_grid[:, :, 2]),
    ]
    extent = [args.grid_min * 100, args.grid_max * 100, args.grid_min * 100, args.grid_max * 100]
    for ax, (title, grid) in zip(axes.flat, panels):
        im = ax.imshow(grid, origin="lower", extent=extent, aspect="auto")
        ax.set_title(title)
        ax.set_xlabel("Perturbation on mu2 (%)")
        ax.set_ylabel("Perturbation on mu3 (%)")
        ax.plot(0, 0, "kx", markersize=8)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    fig.suptitle("Figure 7: FairDICE-fixed performance with perturbed μ in MO-Four-Room", fontsize=12)
    fig.tight_layout(rect=[0, 0.02, 1, 0.95])
    data_path = os.path.join(args.save_dir, "fig7_data.npz")
    np.savez(
        data_path,
        deltas=deltas,
        mu_star=mu_star,
        nsw_grid=nsw_grid,
        util_grid=util_grid,
        jain_grid=jain_grid,
        obj_grid=obj_grid,
    )
    fig_path = os.path.join(args.save_dir, args.output)
    fig.savefig(fig_path, dpi=200)
    print(f"Saved figure to {fig_path}")
    print(f"Saved data to {data_path}")


if __name__ == "__main__":
    main()
