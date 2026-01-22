import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import jax
from evaluation import evaluate_policy

from buffer import Buffer
from utils import normalization, min_max_normalization
from environments.random_momdp import RandomMOMDPEnv
from scripts.generate_random_momdp_dataset import generate_dataset


def parse_int_list(s):
    return [int(x) for x in s.split(",") if x.strip() != ""]


def parse_float_list(s):
    return [float(x) for x in s.split(",") if x.strip() != ""]


def build_offline_batch(trajs, config):
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
        traj["init_states"] = np.tile(traj["states"][0], (traj["states"].shape[0], 1))

    batch = {}
    keys = ["states", "next_states", "actions", "rewards", "terminals", "init_states"]
    for key in keys:
        batch[key] = np.concatenate([traj[key] for traj in trajs], axis=0)
    return batch


def train_fairdice(train_step_fn, init_state_fn, config, batch, seed, log_interval=10000, label="", eval_interval=0, eval_fn=None):
    key = jax.random.PRNGKey(seed)
    train_state = init_state_fn(config)
    buffer = Buffer(batch, is_discrete=True)

    def train_body(carry, _):
        train_state, key = carry
        key, subkey = jax.random.split(key)
        data = buffer.sample(subkey, config.batch_size)
        train_state, update_info = train_step_fn(config, train_state, data, subkey)
        return (train_state, key), update_info

    remaining = config.total_train_steps
    print(f"  {label} init done; starting training for {config.total_train_steps} steps")
    first_chunk = True
    best_state = None
    best_metric = -np.inf
    while remaining > 0:
        step_chunk = min(log_interval, remaining)
        if first_chunk:
            print(f"  {label} compiling first chunk (size={step_chunk})...")
        (train_state, key), update_info = jax.lax.scan(
            train_body, (train_state, key), None, length=step_chunk
        )
        jax.block_until_ready(update_info["policy_loss"][-1])
        remaining -= step_chunk
        done_steps = config.total_train_steps - remaining
        last_update = {k: v[-1] for k, v in update_info.items()}
        nu_loss = float(np.asarray(last_update.get("nu_loss", 0.0)))
        policy_loss = float(np.asarray(last_update.get("policy_loss", 0.0)))
        print(f"  {label} progress: {done_steps}/{config.total_train_steps} (nu_loss={nu_loss:.4f}, policy_loss={policy_loss:.4f})")
        if config.debug_mu and "mu" in last_update:
            mu_val = np.asarray(last_update["mu"]).tolist()
            mu_delta = np.asarray(last_update.get("mu_delta", np.zeros_like(last_update["mu"]))).tolist()
            mu_grad = np.asarray(last_update.get("mu_grad", np.zeros_like(last_update["mu"]))).tolist()
            abs_mu_grad_mean = float(np.asarray(last_update.get("abs_mu_grad_mean", 0.0)))
            abs_mu_grad_max = float(np.asarray(last_update.get("abs_mu_grad_max", 0.0)))
            mu_grad_norm = float(np.asarray(last_update.get("mu_grad_norm", 0.0)))
            frac_w_pos = float(np.asarray(last_update.get("frac_w_pos", 0.0)))
            mean_w = float(np.asarray(last_update.get("mean_w", 0.0)))
            max_w = float(np.asarray(last_update.get("max_w", 0.0)))
            frac_w_gt_5 = float(np.asarray(last_update.get("frac_w_gt_5", 0.0)))
            frac_w_gt_20 = float(np.asarray(last_update.get("frac_w_gt_20", 0.0)))
            terminal_reward_mean = float(np.asarray(last_update.get("terminal_reward_mean", 0.0)))
            mean_e_over_beta = float(np.asarray(last_update.get("mean_e_over_beta", 0.0)))
            max_e_over_beta = float(np.asarray(last_update.get("max_e_over_beta", 0.0)))
            min_e_over_beta = float(np.asarray(last_update.get("min_e_over_beta", 0.0)))
            frac_e_over_beta_lt_neg10 = float(np.asarray(last_update.get("frac_e_over_beta_lt_neg10", 0.0)))
            frac_e_over_beta_gt_neg1 = float(np.asarray(last_update.get("frac_e_over_beta_gt_neg1", 0.0)))
            loss_1 = float(np.asarray(last_update.get("loss_1", 0.0)))
            loss_2 = float(np.asarray(last_update.get("loss_2", 0.0)))
            loss_3 = float(np.asarray(last_update.get("loss_3", 0.0)))
            grad_loss2_mu = np.asarray(last_update.get("grad_loss2_mu", np.zeros_like(last_update["mu"]))).tolist()
            grad_loss3_mu = np.asarray(last_update.get("grad_loss3_mu", np.zeros_like(last_update["mu"]))).tolist()
            grad_loss2_mu_norm = float(np.asarray(last_update.get("grad_loss2_mu_norm", 0.0)))
            grad_loss3_mu_norm = float(np.asarray(last_update.get("grad_loss3_mu_norm", 0.0)))
            print(f"  {label} mu={mu_val}")
            print(f"  {label} mu_delta={mu_delta}")
            print(f"  {label} mu_grad={mu_grad}")
            print(f"  {label} mu_grad_stats(mean_abs={abs_mu_grad_mean:.6f}, max_abs={abs_mu_grad_max:.6f}, l2={mu_grad_norm:.6f})")
            print(f"  {label} w_stats(frac_pos={frac_w_pos:.4f}, frac_gt_5={frac_w_gt_5:.4f}, frac_gt_20={frac_w_gt_20:.4f}, mean={mean_w:.6f}, max={max_w:.6f})")
            print(f"  {label} e_over_beta(mean={mean_e_over_beta:.6f}, max={max_e_over_beta:.6f}, min={min_e_over_beta:.6f})")
            print(f"  {label} e_over_beta_frac(lt_-10={frac_e_over_beta_lt_neg10:.4f}, gt_-1={frac_e_over_beta_gt_neg1:.4f})")
            print(f"  {label} terminal_reward_mean={terminal_reward_mean:.6f}")
            print(f"  {label} loss_terms(loss_1={loss_1:.6f}, loss_2={loss_2:.6f}, loss_3={loss_3:.6f})")
            print(f"  {label} grad_loss2_mu={grad_loss2_mu}")
            print(f"  {label} grad_loss3_mu={grad_loss3_mu}")
            print(f"  {label} grad_loss2_mu_norm={grad_loss2_mu_norm:.6f}, grad_loss3_mu_norm={grad_loss3_mu_norm:.6f}")
        if eval_interval and eval_fn and (done_steps % eval_interval == 0 or done_steps == config.total_train_steps):
            metric = eval_fn(train_state, done_steps)
            if metric > best_metric:
                best_metric = metric
                best_state = train_state
                print(f"  {label} new best NSW={best_metric:.4f} at step {done_steps}")
        first_chunk = False
    confirmation = f"  {label} done"
    print(confirmation)
    return best_state if best_state is not None else train_state


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=str, default="0")
    parser.add_argument("--betas", type=str, default="0.001,0.01,0.1,1.0")
    parser.add_argument("--train_steps", type=int, default=100000)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--hidden_dim", type=int, default=768)
    parser.add_argument("--num_layers", type=int, default=3)
    parser.add_argument("--divergence", type=str, default="SOFT_CHI")
    parser.add_argument("--max_steps", type=int, default=200)
    parser.add_argument("--out_dir", type=str, default="./results/fig3_random_momdp/")
    parser.add_argument("--log_interval", type=int, default=10000)
    parser.add_argument("--eval_interval", type=int, default=200, help="Evaluate every N steps (0 uses log_interval).")
    parser.add_argument("--eval_episodes", type=int, default=50)
    parser.add_argument("--debug_mu", action="store_true", help="Enable mu debug logging.")
    parser.add_argument("--mu_init_noise_std", type=float, default=0.0)
    args = parser.parse_args()

    seeds = parse_int_list(args.seeds)
    betas = parse_float_list(args.betas)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    mu_star = np.zeros((len(betas), len(seeds), 3), dtype=np.float32)

    if args.eval_interval <= 0:
        args.eval_interval = args.log_interval

    for seed_idx, seed in enumerate(seeds):
        print(f"=== seed {seed} ({seed_idx + 1}/{len(seeds)}) ===")
        env = RandomMOMDPEnv(seed=seed, max_steps=args.max_steps)
        trajs, _ = generate_dataset(
            seed=seed,
            num_traj=100,
            max_steps=args.max_steps,
            optimality=0.5,
            out_dir="./data/random_momdp/",
        )

        config = SimpleNamespace()
        config.is_discrete = True
        config.state_dim = env.n_states
        config.action_dim = env.n_actions
        config.reward_dim = env.reward_dim
        config.gamma = env.gamma
        config.hidden_dims = [args.hidden_dim] * args.num_layers
        config.total_train_steps = args.train_steps
        config.batch_size = args.batch_size
        config.divergence = args.divergence
        config.gradient_penalty_coeff = 1e-4
        config.temperature = 1.0
        config.tanh_squash_distribution = False
        config.layer_norm = True
        config.nu_lr = 3e-4
        config.policy_lr = 3e-4
        config.mu_lr = 3e-4
        config.normalize_reward = False
        config.state_mean = np.zeros((env.n_states,), dtype=np.float32)
        config.state_std = np.ones((env.n_states,), dtype=np.float32)
        config.debug_mu = args.debug_mu
        config.mu_init_noise_std = args.mu_init_noise_std

        batch = build_offline_batch(trajs, config)

        for beta_idx, beta in enumerate(betas):
            print(f"[seed {seed}] training FairDICE (beta={beta})")
            config.beta = beta
            config.seed = seed
            from FairDICE import init_train_state, train_step, get_model
            def eval_fn(train_state, step):
                policy = get_model(train_state.policy_state)[0]
                raw_returns, _ = evaluate_policy(
                    config,
                    policy,
                    env,
                    str(out_dir),
                    num_episodes=args.eval_episodes,
                    max_steps=args.max_steps,
                    t_env=None,
                )
                raw_returns = np.asarray(raw_returns)
                eps = 1e-8
                mean_returns = raw_returns.mean(axis=0)
                nsw_val = float(np.sum(np.log(np.clip(mean_returns, eps, None))))
                print(f"[seed {seed}] NSW@{step} (FairDICE, beta={beta})={nsw_val:.4f}")
                return nsw_val
            train_state = train_fairdice(
                train_step,
                init_train_state,
                config,
                batch,
                seed,
                log_interval=args.log_interval,
                label=f"FairDICE beta={beta}",
                eval_interval=args.eval_interval,
                eval_fn=eval_fn,
            )
            mu_net = get_model(train_state.mu_state)[0]
            mu_star[beta_idx, seed_idx] = np.clip(np.array(mu_net()), 1e-8, None)
            print(f"[seed {seed}] mu_star: {mu_star[beta_idx, seed_idx]}")

    mu_path = out_dir / "mu_star.npz"
    np.savez(
        mu_path,
        betas=np.array(betas, dtype=np.float32),
        seeds=np.array(seeds, dtype=np.int32),
        mu_star=mu_star,
    )
    meta_path = out_dir / "mu_star_meta.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "betas": betas,
                "seeds": seeds,
                "train_steps": args.train_steps,
                "batch_size": args.batch_size,
                "hidden_dim": args.hidden_dim,
                "num_layers": args.num_layers,
                "divergence": args.divergence,
                "max_steps": args.max_steps,
            },
            f,
            indent=2,
        )
    print(f"Saved mu_star: {mu_path}")


if __name__ == "__main__":
    main()
