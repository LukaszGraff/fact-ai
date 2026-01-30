import argparse
import json
import pickle
from pathlib import Path

import numpy as np

from environments.random_momdp import RandomMOMDPEnv


def _compute_q(env, v, gamma):
    next_states = env.transition_next_states
    probs = env.transition_probs
    goal_mask = np.isin(next_states, env.goal_states)
    rewards = goal_mask.astype(np.float32)
    v_next = v[next_states]
    q = np.sum(probs * (rewards + gamma * v_next), axis=-1)
    return q


def value_iteration_optimal_policy(env, gamma=0.95, tol=1e-10, max_iters=10000):
    v = np.zeros((env.n_states,), dtype=np.float32)
    for _ in range(max_iters):
        q = _compute_q(env, v, gamma)
        v_new = np.max(q, axis=1)
        v_new[env.goal_states] = 0.0
        if np.max(np.abs(v_new - v)) < tol:
            v = v_new
            break
        v = v_new
    q = _compute_q(env, v, gamma)
    pi_star = np.zeros((env.n_states, env.n_actions), dtype=np.float32)
    greedy_actions = np.argmax(q, axis=1)
    pi_star[np.arange(env.n_states), greedy_actions] = 1.0
    pi_star[env.goal_states] = 1.0 / env.n_actions
    return pi_star, v


def evaluate_policy(env, pi, gamma=0.95, tol=1e-10, max_iters=10000):
    v = np.zeros((env.n_states,), dtype=np.float32)
    for _ in range(max_iters):
        q = _compute_q(env, v, gamma)
        v_new = np.sum(pi * q, axis=1)
        v_new[env.goal_states] = 0.0
        if np.max(np.abs(v_new - v)) < tol:
            v = v_new
            break
        v = v_new
    return v[0]


def find_mixture_coefficient(env, pi_star, pi_unif, optimality=0.5, gamma=0.95):
    j_unif = evaluate_policy(env, pi_unif, gamma=gamma)
    j_star = evaluate_policy(env, pi_star, gamma=gamma)
    denom = j_star - j_unif
    if np.abs(denom) < 1e-8:
        return 0.5, j_unif, j_star
    low, high = 0.0, 1.0
    for _ in range(50):
        mid = 0.5 * (low + high)
        pi_mix = mid * pi_star + (1.0 - mid) * pi_unif
        j_mid = evaluate_policy(env, pi_mix, gamma=gamma)
        ratio = (j_mid - j_unif) / denom
        if ratio < optimality:
            low = mid
        else:
            high = mid
    c = 0.5 * (low + high)
    return c, j_unif, j_star


def generate_dataset(seed, num_traj=100, max_steps=200, optimality=0.5, out_dir="./data/random_momdp/"):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    data_path = out_dir / f"random_momdp_seed_{seed}.pkl"
    meta_path = out_dir / f"random_momdp_seed_{seed}_meta.json"
    if data_path.exists() and meta_path.exists():
        with open(data_path, "rb") as f:
            trajectories = pickle.load(f)
        with open(meta_path, "r", encoding="utf-8") as f:
            metadata = json.load(f)
        return trajectories, metadata

    env = RandomMOMDPEnv(seed=seed)
    pi_unif = np.full((env.n_states, env.n_actions), 1.0 / env.n_actions, dtype=np.float32)
    pi_star, _ = value_iteration_optimal_policy(env, gamma=env.gamma)
    c, j_unif, j_star = find_mixture_coefficient(env, pi_star, pi_unif, optimality=optimality, gamma=env.gamma)
    pi_mix = c * pi_star + (1.0 - c) * pi_unif

    rng = np.random.default_rng(seed + 12345)
    trajectories = []
    for _ in range(num_traj):
        obs = env.reset()
        observations = []
        actions = []
        next_observations = []
        raw_rewards = []
        terminals = []
        for _ in range(max_steps):
            state_idx = env.state
            action = rng.choice(env.n_actions, p=pi_mix[state_idx])
            next_obs, reward, done, _ = env.step(action)
            observations.append(obs)
            actions.append(action)
            next_observations.append(next_obs)
            raw_rewards.append(reward)
            terminals.append(done)
            obs = next_obs
            if done:
                break
        trajectories.append({
            "observations": np.asarray(observations, dtype=np.float32),
            "actions": np.asarray(actions, dtype=np.int64),
            "next_observations": np.asarray(next_observations, dtype=np.float32),
            "raw_rewards": np.asarray(raw_rewards, dtype=np.float32),
            "terminals": np.asarray(terminals, dtype=bool),
        })

    metadata = {
        "seed": seed,
        "gamma": env.gamma,
        "n_states": env.n_states,
        "n_actions": env.n_actions,
        "reward_dim": env.reward_dim,
        "goals": env.goal_states.tolist(),
        "optimality": optimality,
        "mixture_coefficient": float(c),
        "j_unif": float(j_unif),
        "j_star": float(j_star),
    }
    with open(data_path, "wb") as f:
        pickle.dump(trajectories, f)
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    return trajectories, metadata


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num_traj", type=int, default=100)
    parser.add_argument("--max_steps", type=int, default=200)
    parser.add_argument("--optimality", type=float, default=0.5)
    parser.add_argument("--out_dir", type=str, default="./data/random_momdp/")
    args = parser.parse_args()

    generate_dataset(
        seed=args.seed,
        num_traj=args.num_traj,
        max_steps=args.max_steps,
        optimality=args.optimality,
        out_dir=args.out_dir,
    )


if __name__ == "__main__":
    main()
