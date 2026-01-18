import argparse
import json
import os
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
    parser.add_argument("--divergence", type=str, default="CHI", choices=["CHI", "SOFT_CHI", "KL"], help="Divergence")
    parser.add_argument("--total_train_steps", type=int, default=100_000, help="Training steps per run")
    parser.add_argument("--log_interval", type=int, default=10_000, help="Logging frequency (divides total_train_steps)")
    parser.add_argument("--batch_size", type=int, default=256, help="Batch size")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument("--train_eval_episodes", type=int, default=10, help="Episodes for in-training evals")
    parser.add_argument("--eval_episodes", type=int, default=50, help="Episodes for final metrics")
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
        "save_path": args.save_root,
        "wandb": False,
    }


def run_fixed_training(args, mu_vector, d2, d3, base_overrides):
    overrides = dict(base_overrides)
    overrides.update(
        {
            "learner": "FairDICEFixed",
            "tag": f"{args.tag}_fixed_d2_{int(d2 * 100)}_d3_{int(d3 * 100)}",
        }
    )
    config = make_config(**overrides)
    config.mu_fixed_vector = np.array(mu_vector, dtype=np.float32)
    config.freeze_mu = True
    return run_training(config)


def main():
    args = parse_args()
    ensure_fourroom_registered()
    os.makedirs(args.save_root, exist_ok=True)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    ensure_dataset(args.env_name, args.quality, args.preference_dist, regen=args.regen_data)

    grid_values = np.array([float(x) for x in args.grid.split(",")], dtype=np.float32)
    if np.any(np.abs(grid_values) > 1.0):
        grid_values = grid_values / 100.0
    if grid_values.size != 6:
        print("Warning: expected 6 perturbation values; proceeding with provided list.")

    base_overrides = build_overrides(args, learner="FairDICE")
    base_config = make_config(**base_overrides)
    base_result = run_training(base_config)
    mu_star = base_result["mu_vector"]
    np.save(output_dir / "mu_star.npy", mu_star)

    env = gym.make(base_config.env_name)
    base_raw_returns, _ = evaluate_policy(
        base_config,
        base_result["policy"],
        env,
        os.path.join(base_result["save_dir"], "eval"),
        num_episodes=args.eval_episodes,
        max_steps=base_config.max_seq_len,
    )
    env.close()
    base_avg_returns = np.mean(base_raw_returns, axis=0)
    base_metrics = welfare_metrics(base_avg_returns)

    grid_shape = (len(grid_values), len(grid_values))
    nsw_grid = np.zeros(grid_shape, dtype=np.float32)
    util_grid = np.zeros_like(nsw_grid)
    jain_grid = np.zeros_like(nsw_grid)
    returns_grid = np.zeros(grid_shape + (base_config.reward_dim,), dtype=np.float32)

    progress = tqdm(total=len(grid_values) ** 2, desc="Perturbation grid")
    for y_idx, d3 in enumerate(grid_values):
        for x_idx, d2 in enumerate(grid_values):
            mu_vec = mu_star.copy()
            mu_vec[1] = mu_vec[1] * (1.0 + d2)
            mu_vec[2] = mu_vec[2] * (1.0 + d3)
            fixed_result = run_fixed_training(args, mu_vec, d2, d3, base_overrides)
            env = gym.make(base_config.env_name)
            raw_returns, _ = evaluate_policy(
                fixed_result["config"],
                fixed_result["policy"],
                env,
                os.path.join(fixed_result["save_dir"], "eval"),
                num_episodes=args.eval_episodes,
                max_steps=fixed_result["config"].max_seq_len,
            )
            env.close()
            avg_returns = np.mean(raw_returns, axis=0)
            returns_grid[y_idx, x_idx] = avg_returns
            nsw, utilitarian, jain = welfare_metrics(avg_returns)
            nsw_grid[y_idx, x_idx] = nsw
            util_grid[y_idx, x_idx] = utilitarian
            jain_grid[y_idx, x_idx] = jain
            progress.set_postfix({"d2%": int(d2 * 100), "d3%": int(d3 * 100)})
            progress.update(1)
    progress.close()

    percent_labels = [f"{int(val * 100):+d}" for val in grid_values]
    fig, axes = plt.subplots(2, 3, figsize=(14, 9), constrained_layout=True)
    metric_maps = [
        ("Nash Social Welfare", nsw_grid),
        ("Utilitarian Sum", util_grid),
        ("Jain Index", jain_grid),
        ("Return R1", returns_grid[..., 0]),
        ("Return R2", returns_grid[..., 1]),
        ("Return R3", returns_grid[..., 2]),
    ]

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

    fig_path_png = output_dir / "figure7_replication.png"
    fig_path_pdf = output_dir / "figure7_replication.pdf"
    fig.savefig(fig_path_png, dpi=300)
    fig.savefig(fig_path_pdf)
    plt.close(fig)

    np.savez(
        output_dir / "results_grid.npz",
        deltas=grid_values,
        mu_star=mu_star,
        nsw=nsw_grid,
        utilitarian=util_grid,
        jain=jain_grid,
        r1=returns_grid[..., 0],
        r2=returns_grid[..., 1],
        r3=returns_grid[..., 2],
    )

    metadata = {
        "env_name": args.env_name,
        "quality": args.quality,
        "preference_dist": args.preference_dist,
        "beta": args.beta,
        "divergence": args.divergence,
        "total_train_steps": args.total_train_steps,
        "log_interval": args.log_interval,
        "seed": args.seed,
        "grid": grid_values.tolist(),
        "mu_star": mu_star.tolist(),
        "base_avg_returns": base_avg_returns.tolist(),
        "base_metrics": {
            "nsw": base_metrics[0],
            "utilitarian": base_metrics[1],
            "jain": base_metrics[2],
        },
        "figure_png": str(fig_path_png),
        "figure_pdf": str(fig_path_pdf),
    }

    with open(output_dir / "figure7_metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print("\nReplication complete:")
    print(f"  mu*: {mu_star}")
    print(f"  Figure saved to {fig_path_png}")
    print(f"  Data grid saved to {output_dir / 'results_grid.npz'}")


if __name__ == "__main__":
    main()
