import argparse
import json
import os
from datetime import datetime

import numpy as np
from pathlib import Path
import sys

_here = Path(__file__).resolve()
sys.path.insert(0, str(_here.parent.parent))

from generate_MO_Four_Room_data import generate_offline_data
from main_fourroom import make_config, run_training


def _json_friendly(value):
    if isinstance(value, (int, float, str, bool)) or value is None:
        return value
    if hasattr(value, "tolist"):
        return value.tolist()
    return str(value)


def run_single(args) -> str:
    data_dir = os.path.join("data", args.env_name)
    os.makedirs(data_dir, exist_ok=True)
    data_path = os.path.join(
        data_dir,
        f"{args.env_name}_50000_{args.quality}_{args.preference_dist}_seed{args.seed}.pkl",
    )
    generate_offline_data(
        args.env_name,
        quality=args.quality,
        preference_dist=args.preference_dist,
        max_steps=args.max_seq_len,
        seed=args.seed,
        save_path=data_path,
    )
    overrides = {
        "learner": "FairDICE",
        "env_name": args.env_name,
        "quality": args.quality,
        "preference_dist": args.preference_dist,
        "beta": args.beta,
        "divergence": args.divergence,
        "total_train_steps": args.total_train_steps,
        "log_interval": args.log_interval,
        "max_seq_len": args.max_seq_len,
        "batch_size": args.batch_size,
        "seed": args.seed,
        "mu_lr": args.mu_lr,
        "policy_lr": args.policy_lr,
        "nu_lr": args.nu_lr,
        "eval_episodes": args.eval_episodes,
        "eval_mode": args.eval_mode,
        "eval_seed": args.eval_seed,
        "data_path": data_path,
        "data_seed": args.seed,
        "save_path": args.save_root,
        "tag": args.tag,
        "wandb": False,
        "debug_train": args.debug_train,
    }
    config = make_config(**overrides)
    result = run_training(config)

    summary = {
        "timestamp": datetime.utcnow().isoformat(timespec="seconds"),
        "seed": args.seed,
        "mu_star_path": result["mu_path"],
        "mu_star": _json_friendly(result["mu_vector"]),
        "save_dir": result["save_dir"],
        "config": {k: _json_friendly(v) for k, v in vars(result["config"]).items()},
    }
    summary_path = os.path.join(result["save_dir"], "mu_star_run.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"Saved mu_star summary to {summary_path}")
    return summary_path


def _collect_runs(runs_dir: str):
    runs = []
    for root, _, files in os.walk(runs_dir):
        if "mu_star_run.json" in files:
            path = os.path.join(root, "mu_star_run.json")
            with open(path, "r", encoding="utf-8") as f:
                runs.append(json.load(f))
    return runs


def aggregate_runs(args) -> str:
    runs = _collect_runs(args.runs_dir)
    if not runs:
        raise ValueError(f"No mu_star_run.json files found in {args.runs_dir}")

    mu_list = [np.asarray(r["mu_star"], dtype=np.float32) for r in runs]
    mu_stack = np.stack(mu_list, axis=0)
    mu_avg = np.mean(mu_stack, axis=0)
    seeds = sorted({int(r["seed"]) for r in runs})

    os.makedirs(args.output_dir, exist_ok=True)
    mu_path = os.path.join(args.output_dir, "mu_star_avg.npy")
    np.save(mu_path, mu_avg)

    summary = {
        "timestamp": datetime.utcnow().isoformat(timespec="seconds"),
        "runs_dir": args.runs_dir,
        "output_dir": args.output_dir,
        "seeds": seeds,
        "mu_star_avg_path": mu_path,
        "mu_star_avg": _json_friendly(mu_avg),
    }
    summary_path = os.path.join(args.output_dir, "mu_star_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"Saved mu_star average to {mu_path}")
    print(f"Saved mu_star summary to {summary_path}")
    return summary_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train FairDICE and save mu_star.")
    sub = parser.add_subparsers(dest="command", required=True)

    run_parser = sub.add_parser("run", help="Train one mu_star model.")
    run_parser.add_argument("--env_name", type=str, default="MO-FourRoom-v2")
    run_parser.add_argument("--quality", type=str, default="expert", choices=["expert", "amateur"])
    run_parser.add_argument("--preference_dist", type=str, default="uniform", choices=["uniform", "wide", "narrow"])
    run_parser.add_argument("--beta", type=float, default=0.001)
    run_parser.add_argument("--divergence", type=str, default="SOFT_CHI", choices=["SOFT_CHI", "CHI", "KL"])
    run_parser.add_argument("--total_train_steps", type=int, default=100_000)
    run_parser.add_argument("--log_interval", type=int, default=1000)
    run_parser.add_argument("--max_seq_len", type=int, default=500)
    run_parser.add_argument("--batch_size", type=int, default=256)
    run_parser.add_argument("--seed", type=int, default=0)
    run_parser.add_argument("--eval_episodes", type=int, default=10)
    run_parser.add_argument("--eval_mode", type=str, default="sample", choices=["greedy", "sample", "both"])
    run_parser.add_argument("--eval_seed", type=int, default=0)
    run_parser.add_argument("--mu_lr", type=float, default=1e-4)
    run_parser.add_argument("--policy_lr", type=float, default=3e-4)
    run_parser.add_argument("--nu_lr", type=float, default=3e-4)
    run_parser.add_argument("--save_root", type=str, default="./mu_star_runs")
    run_parser.add_argument("--tag", type=str, default="mu_star")
    run_parser.add_argument("--debug_train", type=bool, default=False)

    agg_parser = sub.add_parser("aggregate", help="Average mu_star across runs.")
    agg_parser.add_argument("--runs_dir", type=str, default="./mu_star_runs")
    agg_parser.add_argument("--output_dir", type=str, default="./mu_star_runs")

    return parser


if __name__ == "__main__":
    parser = build_parser()
    args = parser.parse_args()
    if args.command == "run":
        run_single(args)
    else:
        aggregate_runs(args)
