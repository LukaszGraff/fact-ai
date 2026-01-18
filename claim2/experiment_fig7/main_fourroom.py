import argparse
import os
import pickle
import random
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Iterable, Optional

try:
    import gymnasium as gym
except ModuleNotFoundError:
    import gym
import jax
import jax.numpy as jnp
import numpy as np
import wandb
from flax import nnx
from tqdm import tqdm

_here = Path(__file__).resolve()
sys.path.insert(0, str(_here.parent))
from FairDICE import (
    FDivergence,
    TrainState,
    f,
    f_derivative_inverse,
    get_model,
    init_train_state,
    save_model,
    train_step,
)
from buffer import Buffer
from evaluation import evaluate_policy
from fourroom_registration import ensure_fourroom_registered
from utils import min_max_normalization, normalization

def _pytree_l2_norm(pytree):
    leaves = jax.tree_util.tree_leaves(pytree)
    if not leaves:
        return 0.0
    total = 0.0
    for leaf in leaves:
        total += float(jnp.sum(jnp.square(leaf)))
    return float(np.sqrt(total))

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--learner",
        type=str,
        choices=["FairDICE", "FairDICEFixed"],
        default="FairDICE",
        help="Learner type",
    )
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor")
    parser.add_argument("--beta", type=float, default=0.001, help="beta hyperparameter")
    parser.add_argument(
        "--divergence",
        type=str,
        default="SOFT_CHI",
        help="Divergence type (SOFT_CHI/CHI/KL)",
    )
    parser.add_argument(
        "--gradient_penalty_coeff",
        type=float,
        default=1e-4,
        help="Gradient penalty coefficient",
    )
    parser.add_argument("--hidden_dim", type=int, default=256, help="Hidden dimension size")
    parser.add_argument("--num_layers", type=int, default=2, help="Number of layers in the network")
    parser.add_argument("--temperature", type=float, default=1.0, help="Temperature for the policy")
    parser.add_argument(
        "--tanh_squash_distribution",
        type=bool,
        default=False,
        help="Use tanh-squash distribution for actions if set",
    )
    parser.add_argument(
        "--layer_norm",
        type=bool,
        default=True,
        help="Use layer normalization if set",
    )
    parser.add_argument("--nu_lr", type=float, default=3e-4, help="Nu learning rate")
    parser.add_argument("--policy_lr", type=float, default=3e-4, help="Policy learning rate")
    parser.add_argument("--mu_lr", type=float, default=3e-4, help="Mu learning rate")
    parser.add_argument("--batch_size", type=int, default=256, help="Batch size for training")
    parser.add_argument(
        "--quality",
        type=str,
        choices=["expert", "amateur"],
        default="expert",
        help="Dataset quality",
    )
    parser.add_argument(
        "--preference_dist",
        type=str,
        choices=["uniform", "wide", "narrow"],
        default="uniform",
        help="Preference distribution",
    )
    parser.add_argument("--max_seq_len", type=int, default=200, help="Max sequence length in trajectories")
    parser.add_argument(
        "--normalize_reward",
        type=bool,
        default=False,
        help="Whether to normalize reward",
    )
    parser.add_argument("--env_name", type=str, default="MO-FourRoom-v2", help="Environment name")
    parser.add_argument(
        "--mode",
        type=str,
        default="train",
        choices=["train", "eval"],
        help="Running mode: 'train' or 'eval'",
    )
    parser.add_argument(
        "--load_path",
        type=str,
        default=None,
        help="Path to a saved model checkpoint (for eval mode).",
    )
    parser.add_argument("--total_train_steps", type=int, default=100_000, help="Total training steps")
    parser.add_argument("--log_interval", type=int, default=1000, help="Log interval")
    parser.add_argument("--eval_episodes", type=int, default=10, help="Evaluation episodes")
    parser.add_argument("--wandb", type=bool, default=False, help="Use wandb for logging")
    parser.add_argument("--save_path", type=str, default="./results", help="Path to save the model checkpoint")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument("--tag", type=str, default="", help="Tag for the experiment")
    parser.add_argument(
        "--mu_fixed_path",
        type=str,
        default=None,
        help="Path to a saved mu vector (.npy) for FairDICEFixed",
    )
    parser.add_argument(
        "--mu_fixed",
        type=str,
        default=None,
        help="Comma-separated mu vector for FairDICEFixed (e.g., '0.2,0.3,0.5')",
    )
    parser.add_argument(
        "--freeze_mu",
        type=bool,
        default=None,
        help="Freeze mu updates (defaults to True for FairDICEFixed)",
    )
    return parser


def parse_args(arg_list: Optional[Iterable[str]] = None):
    parser = build_parser()
    args, unknown = parser.parse_known_args(arg_list)
    return SimpleNamespace(**vars(args)), unknown


def make_config(**overrides) -> SimpleNamespace:
    config, _ = parse_args([])
    for key, value in overrides.items():
        setattr(config, key, value)
    return config


def _trajectory_batch(trajs, config):
    for traj in trajs:
        if config.normalize_reward:
            traj["rewards"] = min_max_normalization(traj["raw_rewards"], config.reward_min, config.reward_max)
        else:
            traj["rewards"] = traj["raw_rewards"]
        traj["states"] = normalization(traj["observations"], config.state_mean, config.state_std)
        traj["next_states"] = normalization(traj["next_observations"], config.state_mean, config.state_std)
        traj["actions"] = traj["actions"].astype(np.float32)
        traj["init_observations"] = np.tile(traj["observations"][0], (traj["observations"].shape[0], 1))
        traj["init_states"] = np.tile(traj["states"][0], (traj["states"].shape[0], 1))
        traj["terminals"] = np.zeros(len(traj["observations"]))
        traj["terminals"][-1] = 1.0

    tmp = defaultdict(list)
    for traj in trajs:
        for key, value in traj.items():
            tmp[key].append(value)

    batch = defaultdict(list)
    for key, values in tmp.items():
        batch[key] = np.concatenate(values, axis=0)
        print(f"{key}: {batch[key].shape}")
    return batch


def _reward_statistics(trajs):
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
    return reward_min, reward_max


def _ensure_dataset(config):
    data_path = f"./data/{config.env_name}/{config.env_name}_50000_{config.quality}_{config.preference_dist}.pkl"
    if not os.path.exists(data_path):
        print(f"Data not found at {data_path}. Generating data...")
        ensure_fourroom_registered()
        from generate_MO_Four_Room_data import generate_offline_data

        generate_offline_data(
            config.env_name,
            num_trajectories=300,
            quality=config.quality,
            preference_dist=config.preference_dist,
        )
    return data_path


def _resolve_mu_vector(config):
    vector = getattr(config, "mu_fixed_vector", None)
    if vector is None and config.mu_fixed_path:
        vector = np.load(config.mu_fixed_path)
    if vector is None and config.mu_fixed:
        try:
            vector = np.array([float(x.strip()) for x in config.mu_fixed.split(",")], dtype=np.float32)
        except ValueError as exc:
            raise ValueError(f"Invalid --mu_fixed value: {config.mu_fixed}") from exc
    if vector is not None:
        vector = np.array(vector, dtype=np.float32)
        if vector.shape[0] != config.reward_dim:
            raise ValueError(
                f"mu vector has shape {vector.shape}, expected ({config.reward_dim},)"
            )
        config.mu_fixed_vector = vector


def _set_mu_state(train_state: TrainState, mu_vector: np.ndarray) -> TrainState:
    mu_network, mu_optim, _ = get_model(train_state.mu_state)
    mu_network.mu = jnp.asarray(mu_vector, dtype=jnp.float32)
    mu_state = nnx.state((mu_network, mu_optim))
    mu_target = mu_state.filter(nnx.Param)
    return train_state._replace(
        mu_state=train_state.mu_state._replace(state=mu_state, target_params=mu_target)
    )


def _current_mu(train_state: TrainState) -> np.ndarray:
    mu_network, _, _ = get_model(train_state.mu_state)
    return np.array(mu_network())


def train_step_fixed_mu(config, train_state: TrainState, batch, key: jax.random.PRNGKey, fixed_mu):
    key, subkey = jax.random.split(key)
    step = train_state.step
    gamma = config.gamma
    beta = config.beta
    rewards = batch.rewards
    states = batch.states
    next_states = batch.next_states
    init_states = batch.init_states
    mask = batch.masks.astype(jnp.float32)

    policy, policy_optim, _ = get_model(train_state.policy_state)
    nu_network, nu_optim, _ = get_model(train_state.nu_state)

    f_divergence = FDivergence[config.divergence]
    eps = jax.random.uniform(subkey)
    mu = jnp.asarray(fixed_mu, dtype=jnp.float32)

    def nu_loss_fn(nu_network):
        nu = nu_network(states)
        next_nu = nu_network(next_states)
        init_nu = nu_network(init_states)
        k = 1.0 / (mu + 1e-8)
        weighted_rewards = (rewards @ mu).reshape(-1, 1)
        e = weighted_rewards + gamma * next_nu - nu
        w = jax.nn.relu(f_derivative_inverse(e / beta, f_divergence))
        loss_1 = (1 - gamma) * jnp.mean(init_nu)
        masked_term = mask * (w * e - beta * f(w, f_divergence))
        loss_2 = jnp.sum(masked_term) / (jnp.sum(mask) + 1e-8)
        loss_3 = jnp.sum(jnp.log(k) - mu * k)

        def nu_scalar(x):
            return jnp.squeeze(nu_network(x), -1)

        interpolated_observations = init_states * eps + next_states * (1 - eps)
        grad_fn = jax.vmap(jax.grad(nu_scalar), in_axes=0)
        nu_grad = grad_fn(interpolated_observations)
        grad_norm = jnp.linalg.norm(nu_grad, axis=1)
        grad_penalty = config.gradient_penalty_coeff * jnp.mean(jax.nn.relu(grad_norm - 5.0) ** 2)

        nu_loss = loss_1 + loss_2 + loss_3 + grad_penalty
        return nu_loss, (w, e, grad_penalty)

    (nu_loss, (w, e, grad_penalty)), nu_grads = nnx.value_and_grad(
        nu_loss_fn, argnums=0, has_aux=True
    )(nu_network)
    nu_optim.update(nu_network, nu_grads)
    nu_state_ = nnx.state((nu_network, nu_optim))

    is_discrete = getattr(config, "is_discrete", False)

    def policy_loss_fn(policy_module):
        output = policy_module(states)
        if is_discrete:
            logits, probs = output
            log_probs = jnp.log(probs + 1e-8)
            actions_int = batch.actions.astype(jnp.int32)
            log_probs = log_probs[jnp.arange(log_probs.shape[0]), actions_int].reshape(-1, 1)
        else:
            dist = output
            log_probs = dist.log_prob(batch.actions)
            if len(log_probs.shape) == 1:
                log_probs = log_probs.reshape(-1, 1)

        weighted_rewards = (rewards @ mu).reshape(-1, 1)
        nu_val = nu_network(states)
        next_nu = nu_network(next_states)
        e_val = weighted_rewards + gamma * next_nu - nu_val
        # No max-shift; avoids zeroing weights.
        stable_w_pre = jax.lax.stop_gradient(
            jax.nn.relu(f_derivative_inverse(e_val / beta, f_divergence))
        )
        stable_w_mean_before_norm = jnp.mean(stable_w_pre)
        stable_w_pct_nonzero = jnp.mean(stable_w_pre > 0)
        stable_w = stable_w_pre / (stable_w_mean_before_norm + 1e-8)
        policy_loss = -(mask * stable_w * log_probs).sum() / (jnp.sum(mask) + 1e-8)
        actor_stats = {
            "e_val_mean": jnp.mean(e_val),
            "e_val_std": jnp.std(e_val),
            "e_val_min": jnp.min(e_val),
            "e_val_max": jnp.max(e_val),
            "stable_w_mean": jnp.mean(stable_w),
            "stable_w_std": jnp.std(stable_w),
            "stable_w_min": jnp.min(stable_w),
            "stable_w_max": jnp.max(stable_w),
            "stable_w_pct_nonzero": stable_w_pct_nonzero,
            "stable_w_mean_before_norm": stable_w_mean_before_norm,
        }
        return policy_loss, (log_probs, actor_stats)

    (policy_loss, (_, actor_stats)), policy_grads = nnx.value_and_grad(
        policy_loss_fn, has_aux=True
    )(policy)
    policy_optim.update(policy, policy_grads)
    policy_state_ = nnx.state((policy, policy_optim))

    train_state = train_state._replace(
        policy_state=train_state.policy_state._replace(state=policy_state_),
        nu_state=train_state.nu_state._replace(state=nu_state_),
        step=step + 1,
    )

    update_info = {
        "policy_loss": policy_loss,
        "nu_loss": nu_loss,
        "mu": mu,
        "grad_penalty": grad_penalty,
    }
    update_info.update(actor_stats)
    return train_state, update_info


def run_training(config: SimpleNamespace):
    ensure_fourroom_registered()
    try:
        FDivergence[config.divergence]
    except KeyError as exc:
        raise ValueError(f"Unknown divergence: {config.divergence}") from exc
    print(f"[main_fourroom] Using divergence: {config.divergence}")
    print(
        "[config] "
        f"env={config.env_name} divergence={config.divergence} beta={config.beta} "
        f"gamma={config.gamma} deterministic_eval={getattr(config, 'deterministic_eval', True)} "
        f"eval_episodes={config.eval_episodes} "
        f"train_eval_episodes={getattr(config, 'train_eval_episodes', config.eval_episodes)} "
        f"log_interval={config.log_interval} total_train_steps={config.total_train_steps} "
        f"eval_mode={getattr(config, 'eval_mode', 'greedy')} "
        f"eval_seed={getattr(config, 'eval_seed', None)}"
    )
    data_path = _ensure_dataset(config)
    mtime = datetime.fromtimestamp(os.path.getmtime(data_path)).isoformat(timespec="seconds")
    print(f"[dataset] path={data_path} mtime={mtime}")

    with open(data_path, "rb") as f:
        trajs = pickle.load(f)
        print(f"Loaded {len(trajs)} trajectories from {data_path}")
    traj_lengths = [len(traj["observations"]) for traj in trajs]
    avg_len = float(np.mean(traj_lengths)) if traj_lengths else 0.0
    traj_goal_hits = 0
    per_goal_traj_hits = np.zeros(3, dtype=np.int32)
    for traj in trajs:
        rewards = np.asarray(traj["raw_rewards"])
        if (rewards.sum(axis=1) > 0).any():
            traj_goal_hits += 1
            per_goal_traj_hits += (rewards.sum(axis=0) > 0).astype(np.int32)
    print(
        "[dataset] "
        f"avg_traj_len={avg_len:.1f} "
        f"traj_goal_hits={traj_goal_hits} "
        f"per_goal_traj_hits={per_goal_traj_hits.tolist()}"
    )

    env = gym.make(config.env_name)
    config.state_dim = env.observation_space.shape[0]
    config.action_dim = env.action_space.n
    config.is_discrete = True
    if hasattr(env, "num_objectives"):
        config.reward_dim = env.num_objectives
    else:
        config.reward_dim = env.unwrapped.num_objectives

    all_states = np.concatenate([traj["observations"] for traj in trajs], axis=0)
    config.state_mean = all_states.mean(axis=0)
    config.state_std = all_states.std(axis=0) + 1e-8
    config.ACTION_SCALE = 1.0
    config.ACTION_BIAS = 0.0
    probe_states = []
    try:
        reset_out = env.reset(seed=0)
        start_state = reset_out[0] if isinstance(reset_out, tuple) else reset_out
        probe_states.append(("start", np.array(start_state)))
    except Exception:
        pass
    goals = getattr(env, "goals", None)
    if goals is None and hasattr(env, "unwrapped"):
        goals = getattr(env.unwrapped, "goals", None)
    if goals:
        for idx, goal in enumerate(goals):
            probe_states.append((f"goal{idx + 1}", np.array(goal)))

    reward_min, reward_max = _reward_statistics(trajs)
    config.reward_min = reward_min
    config.reward_max = reward_max

    batch = _trajectory_batch(trajs, config)
    config.hidden_dims = [config.hidden_dim] * config.num_layers

    time_stamp = datetime.today().strftime("%Y%m%d_%H%M%S")
    run_name = (
        f"{time_stamp}_{config.learner}_{config.env_name}_{config.quality}_"
        f"{config.preference_dist}_{config.divergence}_beta{config.beta}_seed{config.seed}"
    )

    save_dir = f"{config.save_path}/{run_name}"
    if not os.path.exists(save_dir):
        os.makedirs(os.path.join(save_dir, "eval"), exist_ok=True)

    random.seed(config.seed)
    np.random.seed(config.seed)
    key = jax.random.PRNGKey(config.seed)
    train_state = init_train_state(config)

    if config.freeze_mu is None:
        config.freeze_mu = config.learner == "FairDICEFixed"

    _resolve_mu_vector(config)

    if config.learner == "FairDICEFixed":
        if getattr(config, "mu_fixed_vector", None) is None:
            raise ValueError("FairDICEFixed requires --mu_fixed or --mu_fixed_path")
        train_state = _set_mu_state(train_state, config.mu_fixed_vector)
        fixed_mu = jnp.asarray(config.mu_fixed_vector, dtype=jnp.float32)

        def step_fn(cfg, state, data, rng):
            return train_step_fixed_mu(cfg, state, data, rng, fixed_mu)

        train_step_fn = step_fn
        print(f"[FairDICEFixed] Freezing mu at {config.mu_fixed_vector}")
    else:
        train_step_fn = train_step

    buffer = Buffer(batch)
    train_carry = (train_state, key)

    if config.wandb:
        wandb.init(project=f"exp_{config.tag}", name=run_name, config=vars(config))

    train_iterations = config.total_train_steps // config.log_interval
    if train_iterations <= 0:
        raise ValueError("log_interval must be <= total_train_steps and non-zero")

    best_score = None
    best_train_state = None
    best_metrics = None

    for iter_idx in tqdm(range(train_iterations), desc="Training ..."):
        train_state, key = train_carry

        def train_body(carry, _):
            state, rng = carry
            rng, subkey = jax.random.split(rng)
            data = buffer.sample(subkey, config.batch_size)
            state, update_info = train_step_fn(config, state, data, subkey)
            return (state, rng), update_info

        train_carry, update_info = jax.lax.scan(train_body, (train_state, key), None, length=config.log_interval)
        train_state, key = train_carry
        policy = get_model(train_state.policy_state)[0]

        step = (iter_idx + 1) * config.log_interval
        policy_params = train_state.policy_state.state.filter(nnx.Param)
        nu_params = train_state.nu_state.state.filter(nnx.Param)
        mu_vector = _current_mu(train_state)
        print(
            "[params] "
            f"policy_l2={_pytree_l2_norm(policy_params):.4f} "
            f"critic_l2={_pytree_l2_norm(nu_params):.4f} "
            f"mu={np.round(mu_vector, 4)}"
        )
        if probe_states:
            for label, probe_state in probe_states:
                probe_norm = normalization(probe_state, config.state_mean, config.state_std)
                logits, probs = policy(jnp.asarray(probe_norm))
                probs_np = np.asarray(probs)
                probs_flat = probs_np.reshape(-1)
                argmax_action = int(np.argmax(probs_flat))
                print(
                    f"[probe] {label} probs={np.round(probs_flat, 3).tolist()} "
                    f"argmax={argmax_action}"
                )
        if "stable_w_mean" in update_info:
            def _last_val(x):
                return float(np.asarray(x[-1]))
            print(
                "[actor stats] "
                f"e_val mean={_last_val(update_info['e_val_mean']):.4f} "
                f"std={_last_val(update_info['e_val_std']):.4f} "
                f"min={_last_val(update_info['e_val_min']):.4f} "
                f"max={_last_val(update_info['e_val_max']):.4f} | "
                f"stable_w mean={_last_val(update_info['stable_w_mean']):.4f} "
                f"std={_last_val(update_info['stable_w_std']):.4f} "
                f"min={_last_val(update_info['stable_w_min']):.4f} "
                f"max={_last_val(update_info['stable_w_max']):.4f} | "
                f"pct_nonzero={_last_val(update_info['stable_w_pct_nonzero']):.4f}"
            )
        if "policy_loss" in update_info:
            def _last_arr(x):
                return np.asarray(x[-1])
            policy_loss_val = _last_val(update_info["policy_loss"])
            nu_loss_val = _last_val(update_info["nu_loss"])
            grad_penalty_val = _last_val(update_info["grad_penalty"])
            mu_vec = _last_arr(update_info["mu"])
            mu_grad_vec = _last_arr(update_info.get("mu_grad", np.zeros_like(mu_vec)))
            sample_batch = buffer.sample(jax.random.PRNGKey(int(step)), min(config.batch_size, 256))
            sample_states = sample_batch.states
            logits, probs = policy(sample_states)
            mean_probs = np.asarray(probs).mean(axis=0)
            print(f"Policy Loss: {policy_loss_val:.6f}")
            print(f"  Nu Loss: {nu_loss_val:.6f}")
            print(f"  Mu: {np.round(mu_vec, 6)}")
            print(f"  Mu grad: {np.round(mu_grad_vec, 6)}")
            print(f"  Grad Penalty: {grad_penalty_val:.6f}")
            print(f"  Policy action probs: {np.round(mean_probs, 8)}")
        eval_mode = getattr(config, "eval_mode", "greedy")
        eval_seed = getattr(config, "eval_seed", None)
        if eval_mode == "both":
            print("[eval] mode=greedy")
            _, _, metrics_greedy = evaluate_policy(
                config,
                policy,
                env,
                os.path.join(save_dir, "eval"),
                num_episodes=config.eval_episodes,
                max_steps=config.max_seq_len,
                t_env=step,
                eval_mode="greedy",
                eval_seed=eval_seed,
            )
            print("[eval] mode=sample")
            _, _, metrics_sample = evaluate_policy(
                config,
                policy,
                env,
                os.path.join(save_dir, "eval"),
                num_episodes=config.eval_episodes,
                max_steps=config.max_seq_len,
                t_env=step,
                eval_mode="sample",
                eval_seed=eval_seed,
            )
            metrics = metrics_sample
            print(f"[mu] current={np.round(mu_vector, 6)}")
        else:
            _, _, metrics = evaluate_policy(
                config,
                policy,
                env,
                os.path.join(save_dir, "eval"),
                num_episodes=config.eval_episodes,
                max_steps=config.max_seq_len,
                t_env=step,
                eval_mode=eval_mode,
                eval_seed=eval_seed,
            )
            print(f"[mu] current={np.round(mu_vector, 6)}")

        candidate_score = (
            float(metrics["avg_discounted_normalized_nsw_score"]),
            float(metrics["goal_hit_rate"]),
            float(metrics["avg_raw_discounted_usw_score"]),
        )
        if best_score is None or candidate_score > best_score:
            best_score = candidate_score
            best_train_state = train_state
            best_metrics = metrics
            print(f"[best] mu={np.round(mu_vector, 6)} score={best_score}")

        if config.wandb:
            for k, value in update_info.items():
                if "loss" in k or "grad" in k or "debug" in k:
                    wandb.log({k: value[-1]}, step=step)
                else:
                    for i in range(config.reward_dim):
                        wandb.log({f"{k}_{i}": value[-1][i]}, step=step)

    if config.wandb:
        wandb.finish()

    final_state = best_train_state if best_train_state is not None else train_state
    final_policy = get_model(final_state.policy_state)[0]
    mu_vector = _current_mu(final_state)
    mu_save_path = os.path.join(save_dir, "mu_star.npy")
    np.save(mu_save_path, mu_vector)
    print(f"mu*: {mu_vector}")

    if config.save_path:
        save_model(final_state, os.path.abspath(os.path.join(save_dir, "model")))

    env.close()
    print(f"Training complete. Results saved to {save_dir}")

    return {
        "config": config,
        "train_state": train_state,
        "policy": final_policy,
        "mu_vector": mu_vector,
        "mu_path": mu_save_path,
        "save_dir": save_dir,
    }


def main():
    config, _ = parse_args()
    run_training(config)
    
if __name__ == "__main__":
    main()
