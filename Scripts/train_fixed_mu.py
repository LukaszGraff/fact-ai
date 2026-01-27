import argparse
import json
import math
import os
from datetime import datetime
from typing import Dict, List, Tuple

import numpy as np
from pathlib import Path
import sys

try:
    import gymnasium as gym
except ModuleNotFoundError:
    import gym

_here = Path(__file__).resolve()
sys.path.insert(0, str(_here.parent.parent))

from evaluation import evaluate_policy
from fourroom_registration import ensure_fourroom_registered
from generate_MO_Four_Room_data import generate_offline_data
from main_fourroom import make_config, run_training


EPS = 1e-8


def _json_friendly(value):
    if isinstance(value, (int, float, str, bool)) or value is None:
        return value
    if hasattr(value, "tolist"):
        return value.tolist()
    return str(value)


def _welfare_metrics(avg_returns: np.ndarray) -> Dict[str, float]:
    avg_returns = np.asarray(avg_returns, dtype=np.float64)
    utilitarian = float(np.sum(avg_returns))
    jain = float((utilitarian ** 2) / (avg_returns.size * np.sum(avg_returns ** 2) + 1e-12))
    nsw = float(np.sum(np.log(np.clip(avg_returns, EPS, None))))
    return {"nsw": nsw, "usw": utilitarian, "jain": jain}


def _build_overrides(args) -> Dict[str, object]:
    return {
        "learner": "FairDICEFixed",
        "env_name": args.env_name,
        "quality": args.quality,
        "preference_dist": args.preference_dist,
        "beta": args.beta,
        "divergence": args.divergence,
        "total_train_steps": args.total_train_steps,
        "log_interval": args.log_interval,
        "max_seq_len": args.max_seq_len,
        "batch_size": args.batch_size,
        "mu_lr": args.mu_lr,
        "policy_lr": args.policy_lr,
        "nu_lr": args.nu_lr,
        "seed": args.seed,
        "eval_episodes": args.eval_episodes,
        "eval_mode": args.eval_mode,
        "eval_seed": args.eval_seed,
        "data_path": args.data_path,
        "data_seed": args.seed,
        "save_path": args.save_root,
        "tag": args.tag,
        "wandb": False,
    }


def _load_mu_star(path: str) -> np.ndarray:
    mu_star = np.load(path)
    return np.asarray(mu_star, dtype=np.float32)


def _perturb_mu(mu_star: np.ndarray, d2: float, d3: float) -> np.ndarray:
    mu_vec = mu_star.copy()
    if mu_vec.shape[0] < 3:
        raise ValueError("Expected mu_star to have at least 3 objectives.")
    mu_vec[1] = mu_vec[1] * (1.0 + d2)
    mu_vec[2] = mu_vec[2] * (1.0 + d3)
    return mu_vec


def run_single(args) -> str:
    data_dir = os.path.join("data", args.env_name)
    os.makedirs(data_dir, exist_ok=True)
    args.data_path = os.path.join(
        data_dir,
        f"{args.env_name}_50000_{args.quality}_{args.preference_dist}_seed{args.seed}.pkl",
    )
    generate_offline_data(
        args.env_name,
        quality=args.quality,
        preference_dist=args.preference_dist,
        max_steps=args.max_seq_len,
        seed=args.seed,
        save_path=args.data_path,
    )
    mu_star = _load_mu_star(args.mu_star_path)
    mu_vec = _perturb_mu(mu_star, args.d2, args.d3)

    overrides = _build_overrides(args)
    config = make_config(**overrides)
    config.mu_fixed_vector = mu_vec
    config.freeze_mu = True
    config.fixed_mu = mu_vec

    result = run_training(config)

    ensure_fourroom_registered()
    env = gym.make(config.env_name)
    _, _, metrics = evaluate_policy(
        config,
        result["policy"],
        env,
        os.path.join(result["save_dir"], "eval"),
        num_episodes=args.eval_episodes,
        max_steps=config.max_seq_len,
        eval_mode=args.eval_mode,
        eval_seed=args.eval_seed,
    )
    env.close()

    avg_returns = np.asarray(metrics["avg_raw_returns"], dtype=np.float32)
    welfare = _welfare_metrics(avg_returns)

    summary = {
        "timestamp": datetime.utcnow().isoformat(timespec="seconds"),
        "seed": args.seed,
        "d2": float(args.d2),
        "d3": float(args.d3),
        "mu_star_path": args.mu_star_path,
        "mu_fixed": _json_friendly(mu_vec),
        "save_dir": result["save_dir"],
        "avg_raw_returns": _json_friendly(avg_returns),
        "metrics": {k: _json_friendly(v) for k, v in metrics.items()},
        "welfare": welfare,
    }
    summary_path = os.path.join(result["save_dir"], "fixed_run.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"Saved fixed-mu summary to {summary_path}")
    return summary_path


def _collect_runs(runs_dir: str) -> List[Dict[str, object]]:
    runs = []
    for root, _, files in os.walk(runs_dir):
        if "fixed_run.json" in files:
            path = os.path.join(root, "fixed_run.json")
            with open(path, "r", encoding="utf-8") as f:
                runs.append(json.load(f))
    return runs


def aggregate_runs(args) -> str:
    runs = _collect_runs(args.runs_dir)
    if not runs:
        raise ValueError(f"No fixed_run.json files found in {args.runs_dir}")

    d2_values = sorted({float(r["d2"]) for r in runs})
    d3_values = sorted({float(r["d3"]) for r in runs})
    d2_index = {val: idx for idx, val in enumerate(d2_values)}
    d3_index = {val: idx for idx, val in enumerate(d3_values)}

    grid_shape = (len(d3_values), len(d2_values))
    metric_keys = ["nsw", "usw", "jain"]
    num_objectives = len(runs[0]["avg_raw_returns"])
    metric_keys += [f"obj{i+1}" for i in range(num_objectives)]

    accum: Dict[str, List[List[List[float]]]] = {
        key: [[[] for _ in d2_values] for _ in d3_values] for key in metric_keys
    }

    for run in runs:
        d2 = float(run["d2"])
        d3 = float(run["d3"])
        y = d3_index[d3]
        x = d2_index[d2]
        avg_returns = np.asarray(run["avg_raw_returns"], dtype=np.float64)
        welfare = _welfare_metrics(avg_returns)
        accum["nsw"][y][x].append(welfare["nsw"])
        accum["usw"][y][x].append(welfare["usw"])
        accum["jain"][y][x].append(welfare["jain"])
        for i in range(num_objectives):
            accum[f"obj{i+1}"][y][x].append(float(avg_returns[i]))

    grids_mean = {}
    grids_std = {}
    for key in metric_keys:
        mean_grid = np.zeros(grid_shape, dtype=np.float32)
        std_grid = np.zeros(grid_shape, dtype=np.float32)
        for y in range(len(d3_values)):
            for x in range(len(d2_values)):
                vals = np.asarray(accum[key][y][x], dtype=np.float32)
                if vals.size == 0:
                    mean = float("nan")
                    std = float("nan")
                else:
                    mean = float(np.mean(vals))
                    std = float(np.std(vals))
                mean_grid[y, x] = mean
                std_grid[y, x] = std
        grids_mean[key] = mean_grid
        grids_std[key] = std_grid

    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)
    seed_count = len({int(r["seed"]) for r in runs})
    suffix = f"_seedavg{seed_count}"
    npz_path = os.path.join(output_dir, f"results_grid{suffix}.npz")
    payload = {
        "d2_values": np.asarray(d2_values, dtype=np.float32),
        "d3_values": np.asarray(d3_values, dtype=np.float32),
        "seeds": np.asarray(sorted({int(r["seed"]) for r in runs}), dtype=np.int32),
    }
    for key in metric_keys:
        payload[f"{key}_mean"] = grids_mean[key]
        payload[f"{key}_std"] = grids_std[key]
    np.savez(npz_path, **payload)

    summary = {
        "timestamp": datetime.utcnow().isoformat(timespec="seconds"),
        "runs_dir": args.runs_dir,
        "output_dir": output_dir,
        "d2_values": d2_values,
        "d3_values": d3_values,
        "seeds": sorted({int(r["seed"]) for r in runs}),
        "npz_path": npz_path,
    }
    summary_path = os.path.join(output_dir, f"grid_summary{suffix}.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"Saved aggregate grid to {npz_path}")
    print(f"Saved grid summary to {summary_path}")
    return summary_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run FairDICE-fixed training or aggregate runs.")
    sub = parser.add_subparsers(dest="command", required=True)

    run_parser = sub.add_parser("run", help="Run one fixed-mu training job.")
    run_parser.add_argument("--mu_star_path", type=str, required=True)
    run_parser.add_argument("--d2", type=float, required=True)
    run_parser.add_argument("--d3", type=float, required=True)
    run_parser.add_argument("--seed", type=int, default=0)
    run_parser.add_argument("--env_name", type=str, default="MO-FourRoom-v2")
    run_parser.add_argument("--quality", type=str, default="expert", choices=["expert", "amateur"])
    run_parser.add_argument("--preference_dist", type=str, default="uniform", choices=["uniform", "wide", "narrow"])
    run_parser.add_argument("--beta", type=float, default=0.001)
    run_parser.add_argument("--divergence", type=str, default="SOFT_CHI", choices=["SOFT_CHI", "CHI", "KL"])
    run_parser.add_argument("--total_train_steps", type=int, default=100_000)
    run_parser.add_argument("--log_interval", type=int, default=1000)
    run_parser.add_argument("--max_seq_len", type=int, default=500)
    run_parser.add_argument("--batch_size", type=int, default=256)
    run_parser.add_argument("--mu_lr", type=float, default=1e-4)
    run_parser.add_argument("--policy_lr", type=float, default=3e-4)
    run_parser.add_argument("--nu_lr", type=float, default=3e-4)
    run_parser.add_argument("--eval_episodes", type=int, default=10)
    run_parser.add_argument("--eval_mode", type=str, default="sample", choices=["greedy", "sample", "both"])
    run_parser.add_argument("--eval_seed", type=int, default=0)
    run_parser.add_argument("--save_root", type=str, default="./fixed_runs")
    run_parser.add_argument("--tag", type=str, default="fixed")

    agg_parser = sub.add_parser("aggregate", help="Aggregate fixed-mu runs into a grid.")
    agg_parser.add_argument("--runs_dir", type=str, default="./fixed_runs")
    agg_parser.add_argument("--output_dir", type=str, default="./fig7_results")

    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()
    if args.command == "run":
        run_single(args)
    else:
        aggregate_runs(args)


if __name__ == "__main__":
    main()
