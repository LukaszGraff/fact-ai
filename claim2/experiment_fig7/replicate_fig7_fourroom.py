import argparse
import json
import os
import random
import sys
from pathlib import Path

_here = Path(__file__).resolve()
sys.path.insert(0, str(_here.parent))

try:
    import gymnasium as gym
except ModuleNotFoundError:
    import gym
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from evaluation import evaluate_policy
from fourroom_registration import ensure_fourroom_registered
from generate_MO_Four_Room_data import generate_offline_data
from main_fourroom import make_config, run_training

EPS = 1e-8


def parse_args():
    parser = argparse.ArgumentParser(description="Replicate FairDICE Figure 7 (MO-Four-Room)")
    parser.add_argument("--env_name", type=str, default="MO-FourRoom-v2", help="Gym environment id")
    parser.add_argument("--quality", type=str, default="expert", choices=["expert", "amateur"], help="Dataset quality")
    parser.add_argument(
        "--preference_dist",
        type=str,
        default="uniform",
        choices=["uniform", "wide", "narrow"],
        help="Preference distribution tag",
    )
    parser.add_argument("--beta", type=float, default=0.01, help="FairDICE beta (finite-domain setup)")
    parser.add_argument("--divergence", type=str, default="SOFT_CHI", choices=["CHI", "SOFT_CHI", "KL"], help="Divergence")
    parser.add_argument("--total_train_steps", type=int, default=100_000, help="Training steps per run")
    parser.add_argument("--log_interval", type=int, default=10_000, help="Logging frequency (divides total_train_steps)")
    parser.add_argument("--batch_size", type=int, default=256, help="Batch size")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument(
        "--seeds",
        type=str,
        default="0,1,2,3,4",
        help="Comma-separated seeds for FairDICEFixed averaging",
    )
    parser.add_argument(
        "--avg_over_seeds",
        type=bool,
        default=True,
        help="Average FairDICEFixed runs over multiple seeds",
    )
    parser.add_argument("--train_eval_episodes", type=int, default=50, help="Episodes for in-training evals")
    parser.add_argument("--eval_episodes", type=int, default=50, help="Episodes for final metrics")
    parser.add_argument(
        "--deterministic_eval",
        type=bool,
        default=True,
        help="Use greedy actions for discrete-policy evaluation",
    )
    parser.add_argument(
        "--eval_mode",
        type=str,
        default="sample",
        choices=["greedy", "sample", "both"],
        help="Evaluation action mode",
    )
    parser.add_argument(
        "--eval_seed",
        type=int,
        default=0,
        help="Random seed for stochastic evaluation",
    )
    parser.add_argument(
        "--mu_star_path",
        type=str,
        default=None,
        help="Path to a saved mu_star.npy (skip FairDICE training)",
    )
    parser.add_argument(
        "--grid",
        type=str,
        default="-0.10,-0.06,-0.02,0.02,0.06,0.10",
        help="Comma-separated perturbations for mu2/mu3 (percent in decimal form)",
    )
    parser.add_argument("--save_root", type=str, default="./fig7_runs", help="Directory for individual training runs")
    parser.add_argument("--output_dir", type=str, default="./fig7_results", help="Directory for aggregated outputs")
    parser.add_argument("--tag", type=str, default="fig7", help="Tag for naming runs")
    parser.add_argument(
        "--regen_data",
        action="store_true",
        help="Regenerate offline data even if the dataset file already exists",
    )
    return parser.parse_args()


def ensure_dataset(env_name: str, quality: str, preference: str, regen: bool = False):
    data_path = Path("data") / env_name / f"{env_name}_50000_{quality}_{preference}.pkl"
    if data_path.exists() and not regen:
        return str(data_path)
    ensure_fourroom_registered()
    if data_path.exists():
        data_path.unlink()
    print(f"Dataset missing ({data_path}). Generating with random policy...")
    generate_offline_data(env_name, num_trajectories=300, quality=quality, preference_dist=preference)
    return str(data_path)


def welfare_metrics(avg_returns: np.ndarray):
    avg_returns = np.asarray(avg_returns, dtype=np.float64)
    utilitarian = float(np.sum(avg_returns))
    jain = float((utilitarian ** 2) / (avg_returns.size * np.sum(avg_returns ** 2) + 1e-12))
    nsw = float(np.sum(np.log(np.clip(avg_returns, EPS, None))))
    return nsw, utilitarian, jain


def build_metrics(avg_returns: np.ndarray):
    nsw, utilitarian, jain = welfare_metrics(avg_returns)
    metrics = {
        "nsw": nsw,
        "usw": utilitarian,
        "jain": jain,
    }
    for idx, val in enumerate(avg_returns):
        metrics[f"obj{idx + 1}"] = float(val)
    return metrics


def build_overrides(args, learner="FairDICE"):
    return {
        "env_name": args.env_name,
        "quality": args.quality,
        "preference_dist": args.preference_dist,
        "beta": args.beta,
        "divergence": args.divergence,
        "total_train_steps": args.total_train_steps,
        "log_interval": args.log_interval,
        "batch_size": args.batch_size,
        "seed": args.seed,
        "tag": args.tag,
        "learner": learner,
        "eval_episodes": args.train_eval_episodes,
        "train_eval_episodes": args.train_eval_episodes,
        "deterministic_eval": args.deterministic_eval,
        "eval_mode": args.eval_mode,
        "eval_seed": args.eval_seed,
        "save_path": args.save_root,
        "wandb": False,
    }


def run_fixed_training(args, mu_vector, d2, d3, seed, base_overrides):
    overrides = dict(base_overrides)
    overrides.update(
        {
            "learner": "FairDICEFixed",
            "tag": f"{args.tag}_fixed_d2_{int(d2 * 100)}_d3_{int(d3 * 100)}",
            "seed": seed,
        }
    )
    config = make_config(**overrides)
    assert config.divergence == args.divergence, (
        f"Config divergence mismatch: {config.divergence} != {args.divergence}"
    )
    config.mu_fixed_vector = np.array(mu_vector, dtype=np.float32)
    config.freeze_mu = True
    return run_training(config)


def main():
    args = parse_args()
    print(f"[replicate_fig7_fourroom] Using divergence: {args.divergence}")
    print(f"[replicate_fig7_fourroom] Eval mode: {args.eval_mode}")
    print(f"[replicate_fig7_fourroom] Eval seed: {args.eval_seed}")
    base_seed = args.seed
    mu_star_seed = base_seed if args.mu_star_path is None else "from_file"
    seeds = [int(x) for x in args.seeds.split(",") if x.strip() != ""]
    if not args.avg_over_seeds:
        seeds = [base_seed]
    if len(seeds) < 1:
        raise ValueError("Expected at least one seed for averaging.")
    print(f"[replicate_fig7_fourroom] dataset_seed={base_seed}, mu_star_seed={mu_star_seed}")
    print(f"[replicate_fig7_fourroom] avg_over_seeds={args.avg_over_seeds}, seeds={seeds}")
    ensure_fourroom_registered()
    os.makedirs(args.save_root, exist_ok=True)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    random.seed(base_seed)
    np.random.seed(base_seed)
    ensure_dataset(args.env_name, args.quality, args.preference_dist, regen=args.regen_data)

    grid_values = np.array([float(x) for x in args.grid.split(",")], dtype=np.float32)
    if np.any(np.abs(grid_values) > 1.0):
        grid_values = grid_values / 100.0
    if grid_values.size != 6:
        print("Warning: expected 6 perturbation values; proceeding with provided list.")

    base_overrides = build_overrides(args, learner="FairDICE")
    base_config = make_config(**base_overrides)
    assert base_config.divergence == args.divergence, (
        f"Config divergence mismatch: {base_config.divergence} != {args.divergence}"
    )
    env = gym.make(base_config.env_name)
    if hasattr(env, "num_objectives"):
        base_config.reward_dim = env.num_objectives
    else:
        base_config.reward_dim = env.unwrapped.num_objectives

    if args.mu_star_path:
        print(f"[replicate_fig7_fourroom] Loading mu_star from {args.mu_star_path}")
        mu_star = np.load(args.mu_star_path)
        base_result = None
        base_avg_returns = None
        base_metrics = None
    else:
        base_result = run_training(base_config)
        mu_star = base_result["mu_vector"]
        base_raw_returns = None
        eval_mode = args.eval_mode
        eval_seed = args.eval_seed
        if eval_mode == "both":
            print("[eval] mode=greedy")
            base_raw_returns, _, _ = evaluate_policy(
                base_config,
                base_result["policy"],
                env,
                os.path.join(base_result["save_dir"], "eval"),
                num_episodes=args.eval_episodes,
                max_steps=base_config.max_seq_len,
                eval_mode="greedy",
                eval_seed=eval_seed,
            )
            print("[eval] mode=sample")
            evaluate_policy(
                base_config,
                base_result["policy"],
                env,
                os.path.join(base_result["save_dir"], "eval"),
                num_episodes=args.eval_episodes,
                max_steps=base_config.max_seq_len,
                eval_mode="sample",
                eval_seed=eval_seed,
            )
        else:
            base_raw_returns, _, _ = evaluate_policy(
                base_config,
                base_result["policy"],
                env,
                os.path.join(base_result["save_dir"], "eval"),
                num_episodes=args.eval_episodes,
                max_steps=base_config.max_seq_len,
                eval_mode=eval_mode,
                eval_seed=eval_seed,
            )
        base_avg_returns = np.mean(base_raw_returns, axis=0)
        base_metrics = welfare_metrics(base_avg_returns)

    np.save(output_dir / "mu_star.npy", mu_star)

    env.close()
    eval_mode = args.eval_mode
    eval_seed = args.eval_seed

    grid_shape = (len(grid_values), len(grid_values))
    metric_keys = ["nsw", "usw", "jain"] + [f"obj{i+1}" for i in range(base_config.reward_dim)]
    grids_mean = {key: np.zeros(grid_shape, dtype=np.float32) for key in metric_keys}
    grids_std = {key: np.zeros(grid_shape, dtype=np.float32) for key in metric_keys}

    progress = tqdm(total=len(grid_values) ** 2, desc="Perturbation grid")
    for y_idx, d3 in enumerate(grid_values):
        for x_idx, d2 in enumerate(grid_values):
            mu_vec = mu_star.copy()
            mu_vec[1] = mu_vec[1] * (1.0 + d2)
            mu_vec[2] = mu_vec[2] * (1.0 + d3)
            print(
                f"[mu_fixed] d2={d2:+.2f} d3={d3:+.2f} "
                f"mu={np.round(mu_vec, 6)}"
            )
            run_metrics = []
            for seed in seeds:
                try:
                    fixed_result = run_fixed_training(args, mu_vec, d2, d3, seed, base_overrides)
                    env = gym.make(base_config.env_name)
                    if eval_mode == "both":
                        print("[eval] mode=greedy")
                        raw_returns, _, _ = evaluate_policy(
                            fixed_result["config"],
                            fixed_result["policy"],
                            env,
                            os.path.join(fixed_result["save_dir"], "eval"),
                            num_episodes=args.eval_episodes,
                            max_steps=fixed_result["config"].max_seq_len,
                            eval_mode="greedy",
                            eval_seed=eval_seed,
                        )
                        print("[eval] mode=sample")
                        evaluate_policy(
                            fixed_result["config"],
                            fixed_result["policy"],
                            env,
                            os.path.join(fixed_result["save_dir"], "eval"),
                            num_episodes=args.eval_episodes,
                            max_steps=fixed_result["config"].max_seq_len,
                            eval_mode="sample",
                            eval_seed=eval_seed,
                        )
                    else:
                        raw_returns, _, _ = evaluate_policy(
                            fixed_result["config"],
                            fixed_result["policy"],
                            env,
                            os.path.join(fixed_result["save_dir"], "eval"),
                            num_episodes=args.eval_episodes,
                            max_steps=fixed_result["config"].max_seq_len,
                            eval_mode=eval_mode,
                            eval_seed=eval_seed,
                        )
                    env.close()
                except Exception as exc:
                    raise RuntimeError(
                        f"Seed run failed for d2={d2}, d3={d3}, seed={seed}"
                    ) from exc
                avg_returns = np.mean(raw_returns, axis=0)
                run_metrics.append(build_metrics(avg_returns))
            for key in metric_keys:
                values = np.array([m[key] for m in run_metrics], dtype=np.float32)
                grids_mean[key][y_idx, x_idx] = float(np.mean(values))
                grids_std[key][y_idx, x_idx] = float(np.std(values))
            print(
                f"[grid] d2={d2:+.2f}, d3={d3:+.2f}, "
                f"nsw={grids_mean['nsw'][y_idx, x_idx]:.4f}"
                f"+/-{grids_std['nsw'][y_idx, x_idx]:.4f}"
            )
            progress.set_postfix({"d2%": int(d2 * 100), "d3%": int(d3 * 100)})
            progress.update(1)
    progress.close()

    percent_labels = [f"{int(val * 100):+d}" for val in grid_values]
    fig, axes = plt.subplots(2, 3, figsize=(14, 9), constrained_layout=True)
    metric_maps = [
        ("Nash Social Welfare", grids_mean["nsw"]),
        ("Utilitarian Sum", grids_mean["usw"]),
        ("Jain Index", grids_mean["jain"]),
    ]
    for idx in range(base_config.reward_dim):
        metric_maps.append((f"Return R{idx + 1}", grids_mean[f"obj{idx + 1}"]))

    for ax, (title, data) in zip(axes.flat, metric_maps):
        im = ax.imshow(data, origin="lower", cmap="viridis")
        ax.set_xticks(range(len(grid_values)))
        ax.set_xticklabels(percent_labels)
        ax.set_yticks(range(len(grid_values)))
        ax.set_yticklabels(percent_labels)
        ax.set_xlabel("mu2 perturb (%)")
        ax.set_ylabel("mu3 perturb (%)")
        ax.set_title(title)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)

    seed_suffix = f"_seedavg{len(seeds)}"
    fig_path_png = output_dir / f"figure7_replication{seed_suffix}.png"
    fig_path_pdf = output_dir / f"figure7_replication{seed_suffix}.pdf"
    fig.savefig(fig_path_png, dpi=300)
    fig.savefig(fig_path_pdf)
    plt.close(fig)

    npz_path = output_dir / f"results_grid{seed_suffix}.npz"
    npz_payload = {
        "d2_values": grid_values,
        "d3_values": grid_values,
        "mu_star": mu_star,
        "seeds": np.array(seeds, dtype=np.int32),
        "base_seed": np.int32(base_seed),
    }
    for key in metric_keys:
        npz_payload[f"{key}_mean"] = grids_mean[key]
        npz_payload[f"{key}_std"] = grids_std[key]
    np.savez(npz_path, **npz_payload)

    metadata = {
        "env_name": args.env_name,
        "quality": args.quality,
        "preference_dist": args.preference_dist,
        "beta": args.beta,
        "divergence": args.divergence,
        "total_train_steps": args.total_train_steps,
        "log_interval": args.log_interval,
        "seed": args.seed,
        "base_seed": base_seed,
        "seeds": seeds,
        "avg_over_seeds": args.avg_over_seeds,
        "grid": grid_values.tolist(),
        "mu_star": mu_star.tolist(),
        "base_avg_returns": base_avg_returns.tolist() if base_avg_returns is not None else None,
        "base_metrics": {
            "nsw": base_metrics[0],
            "utilitarian": base_metrics[1],
            "jain": base_metrics[2],
        } if base_metrics is not None else None,
        "figure_png": str(fig_path_png),
        "figure_pdf": str(fig_path_pdf),
        "results_npz": str(npz_path),
    }

    with open(output_dir / "figure7_metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print("\nReplication complete:")
    print(f"  mu*: {mu_star}")
    print(f"  Figure saved to {fig_path_png}")
    print(f"  Data grid saved to {npz_path}")


if __name__ == "__main__":
    main()
