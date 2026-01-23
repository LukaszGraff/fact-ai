import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import jax

from buffer import Buffer
from evaluation import evaluate_policy
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


def train_fairdice(train_step_fn, init_state_fn, config, batch, seed, log_interval=10000, label=""):
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
        first_chunk = False
    print(f"  {label} done")
    return train_state


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=str, default="0")
    parser.add_argument("--betas", type=str, default="0.001,0.01,0.1,1.0")
    parser.add_argument("--sigmas", type=str, default="0.001,0.01,0.1,1.0")
    parser.add_argument("--train_steps", type=int, default=100000)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--hidden_dim", type=int, default=768)
    parser.add_argument("--num_layers", type=int, default=3)
    parser.add_argument("--divergence", type=str, default="SOFT_CHI")
    parser.add_argument("--max_steps", type=int, default=200)
    parser.add_argument("--max_eval_steps", type=int, default=200)
    parser.add_argument("--eval_episodes", type=int, default=50)
    parser.add_argument("--out_dir", type=str, default="./results/fig3_random_momdp/")
    parser.add_argument("--mu_path", type=str, default="./results/fig3_random_momdp/mu_star.npz")
    parser.add_argument("--log_interval", type=int, default=10000)
    parser.add_argument("--eval_interval", type=int, default=200, help="Evaluate every N steps (0 uses log_interval).")
    parser.add_argument("--num_perturbations", type=int, default=3, help="Number of perturbations per (seed, beta, sigma).")
    args = parser.parse_args()

    seeds = parse_int_list(args.seeds)
    betas = parse_float_list(args.betas)
    sigmas = parse_float_list(args.sigmas)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    mu_data = np.load(args.mu_path, allow_pickle=True)
    mu_betas = mu_data["betas"].astype(np.float32).tolist()
    mu_seeds = mu_data["seeds"].astype(np.int32).tolist()
    mu_star = mu_data["mu_star"]

    def _has_beta(beta, candidates, atol=1e-8):
        return any(np.isclose(beta, cand, atol=atol) for cand in candidates)

    for beta in betas:
        if not _has_beta(beta, mu_betas):
            raise ValueError(f"beta={beta} not found in {args.mu_path}")
    for seed in seeds:
        if seed not in mu_seeds:
            raise ValueError(f"seed={seed} not found in {args.mu_path}")

    nsw = np.zeros((len(betas), len(sigmas), len(seeds)), dtype=np.float32)
    final_nsw_map = {}

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

        batch = build_offline_batch(trajs, config)

        for beta_idx, beta in enumerate(betas):
            config.beta = beta
            config.seed = seed
            mu_seed_idx = mu_seeds.index(seed)
            mu_beta_idx = int(np.argmin(np.abs(np.array(mu_betas) - beta)))
            mu_star_val = np.clip(mu_star[mu_beta_idx, mu_seed_idx], 1e-8, None)
            print(f"[seed {seed}] mu_star (beta={beta}): {mu_star_val}")

            for sigma_idx, sigma in enumerate(sigmas):
                print(f"[seed {seed}] training FairDICE-fixed (beta={beta}, sigma={sigma})")
                nsw_vals = []
                for pert_idx in range(args.num_perturbations):
                    seed_seq = np.random.SeedSequence(
                        [seed, int(round(beta * 1e6)), int(round(sigma * 1e3)), pert_idx]
                    )
                    rng = np.random.default_rng(seed_seq)
                    noise = rng.normal(0.0, sigma, size=mu_star_val.shape)
                    mu_pert = mu_star_val * (1.0 + noise)
                    mu_pert = np.clip(mu_pert, 1e-8, None)
                    mu_pert = mu_pert / (np.sum(mu_pert) + 1e-8)
                    print(f"[seed {seed}] fixed_mu (perturb {pert_idx + 1}/{args.num_perturbations}): {mu_pert}")
                    config.fixed_mu = mu_pert

                    from FairDICE_fixed import init_train_state as init_train_state_fixed
                    from FairDICE_fixed import train_step_fixed, get_model as get_model_fixed
                    train_state_fixed = train_fairdice(
                        train_step_fixed,
                        init_train_state_fixed,
                        config,
                        batch,
                        seed,
                        log_interval=args.log_interval,
                        label=f"FairDICE-fixed beta={beta} sigma={sigma} perturb={pert_idx + 1}",
                    )
                    policy = get_model_fixed(train_state_fixed.policy_state)[0]
                    raw_returns, _ = evaluate_policy(
                        config,
                        policy,
                        env,
                        str(out_dir),
                        num_episodes=args.eval_episodes,
                        max_steps=args.max_eval_steps,
                        t_env=None,
                    )
                    raw_returns = np.asarray(raw_returns)
                    eps = 1e-8
                    mean_returns = raw_returns.mean(axis=0)
                    nsw_val = float(np.sum(np.log(np.clip(mean_returns, eps, None))))
                    nsw_vals.append(nsw_val)
                    print(f"[seed {seed}] NSW@{config.total_train_steps} (Fixed, beta={beta}, sigma={sigma}, perturb={pert_idx + 1})={nsw_val:.4f}")
                    best_key = f"beta_{beta}_sigma_{sigma}"
                    if best_key not in final_nsw_map:
                        final_nsw_map[best_key] = []
                    final_nsw_map[best_key].append(nsw_val)
                nsw[beta_idx, sigma_idx, seed_idx] = float(np.mean(nsw_vals))
                print(f"[seed {seed}] NSW@{config.total_train_steps} mean over {args.num_perturbations} perturbations (Fixed, beta={beta}, sigma={sigma})={nsw[beta_idx, sigma_idx, seed_idx]:.4f}")

    results_path = out_dir / "fig3_random_momdp_results.npz"
    np.savez(
        results_path,
        betas=np.array(betas, dtype=np.float32),
        sigmas=np.array(sigmas, dtype=np.float32),
        seeds=np.array(seeds, dtype=np.int32),
        nsw=nsw,
    )

    mean_nsw = nsw.mean(axis=2)
    if len(seeds) > 1:
        stderr_nsw = nsw.std(axis=2, ddof=1) / np.sqrt(len(seeds))
    else:
        stderr_nsw = np.zeros_like(mean_nsw)
    sigmas_arr = np.array(sigmas, dtype=np.float32)
    positive_sigmas = sigmas_arr[sigmas_arr > 0]
    min_pos = positive_sigmas.min() if positive_sigmas.size > 0 else 1.0
    x_plot = sigmas_arr.copy()
    x_plot[x_plot == 0] = min_pos * 0.1

    fig, ax = plt.subplots(figsize=(6, 4))
    for beta_idx, beta in enumerate(betas):
        ax.errorbar(
            x_plot,
            mean_nsw[beta_idx],
            yerr=stderr_nsw[beta_idx],
            marker="o",
            capsize=3,
            label=f"beta={beta}",
        )
    ax.set_xscale("log")
    ax.set_xlabel("sigma")
    ax.set_ylabel("NSW")
    ax.set_xticks(x_plot)
    ax.set_xticklabels([str(s) for s in sigmas])
    ax.legend()
    ax.grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.5)
    fig.tight_layout()

    plot_path = out_dir / "fig3_random_momdp.png"
    fig.savefig(plot_path, dpi=150)

    meta_path = out_dir / "fig3_random_momdp_meta.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "betas": betas,
                "sigmas": sigmas,
                "seeds": seeds,
                "train_steps": args.train_steps,
                "batch_size": args.batch_size,
                "hidden_dim": args.hidden_dim,
                "num_layers": args.num_layers,
                "divergence": args.divergence,
                "eval_episodes": args.eval_episodes,
                "mu_path": args.mu_path,
                "num_perturbations": args.num_perturbations,
            },
            f,
            indent=2,
        )
    best_path = out_dir / "final_nsw_fixed.json"
    final_nsw_avg = {k: float(np.mean(v)) if v else None for k, v in final_nsw_map.items()}
    with open(best_path, "w", encoding="utf-8") as f:
        json.dump(final_nsw_avg, f, indent=2)
    print(f"Saved results: {results_path}")
    print(f"Saved plot: {plot_path}")


if __name__ == "__main__":
    main()
