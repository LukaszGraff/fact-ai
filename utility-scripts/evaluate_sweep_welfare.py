import argparse
import csv
import json
import re
from collections import defaultdict
from types import SimpleNamespace

import os
import sys
# for relative imports
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from FairDICE import get_model, load_model

import gym
import jax
import numpy as np

import environments
from evaluation import evaluate_policy


RUN_RE = re.compile(r"^(?:.*_)?seed_(\d+)_beta_([^_]+)_h(\d+)_bs(\d+)$")


def _load_config(run_dir: str) -> SimpleNamespace:
    config_path = os.path.join(run_dir, "config.json")
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Missing config.json in {run_dir}")
    with open(config_path, "r") as f:
        cfg = json.load(f)

    for key in ("state_mean", "state_std", "reward_min", "reward_max"):
        if key in cfg and cfg[key] is not None:
            cfg[key] = np.array(cfg[key], dtype=np.float32)

    cfg.setdefault("wandb", False)
    return SimpleNamespace(**cfg)


def _is_fourroom(env_name: str) -> bool:
    name = env_name.lower()
    return "fourroom" in name or "four-room" in name


def _nsw_score(returns: np.ndarray, eps: float, fourroom: bool) -> float:
    if fourroom:
        avg = returns.mean(axis=0)
        return float(np.sum(np.log(avg + eps)))
    return float(np.mean(np.sum(np.log(returns + eps), axis=1)))


def _usw_score(returns: np.ndarray) -> float:
    return float(np.mean(np.sum(returns, axis=1)))


def _jain_index(returns: np.ndarray) -> float:
    avg = returns.mean(axis=0)
    numerator = float(np.sum(avg) ** 2)
    denom = float(len(avg) * np.sum(avg ** 2) + 1e-8)
    return numerator / denom


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Re-evaluate each sweep model and aggregate welfare per hyperparameter setting."
    )
    parser.add_argument("--sweep_dir", required=True, help="Sweep directory with seed_* runs.")
    parser.add_argument("--episodes", type=int, default=30, help="Episodes per model.")
    parser.add_argument("--max_steps", type=int, default=200, help="Max steps per episode.")
    parser.add_argument(
        "--save_step",
        type=int,
        default=None,
        help="Eval step tag used for saved npy files (defaults to config.total_train_steps).",
    )
    parser.add_argument("--eps", type=float, default=0.001, help="Epsilon for NSW log.")
    parser.add_argument(
        "--output_csv",
        default="",
        help="Optional CSV output path (defaults to sweep dir).",
    )
    parser.add_argument(
        "--group_by",
        choices=["hparams", "mix", "mix+hparams"],
        default="hparams",
        help="How to group runs before aggregation.",
    )
    args = parser.parse_args()

    sweep_dir = args.sweep_dir
    per_setting = defaultdict(list)
    missing = []

    for entry in sorted(os.listdir(sweep_dir)):
        run_dir = os.path.join(sweep_dir, entry)
        if not os.path.isdir(run_dir):
            continue
        m = RUN_RE.match(entry)
        if not m:
            continue

        seed, beta, hdim, bsize = m.groups()
        mix_tag = None
        if "_seed_" in entry:
            mix_tag = entry.split("_seed_")[0]
        model_dir = os.path.join(run_dir, "model")
        if not os.path.isdir(model_dir):
            missing.append(entry)
            continue

        config = _load_config(run_dir)
        config.wandb = False
        env = gym.make(config.env_name)
        fourroom = _is_fourroom(config.env_name)

        eval_dir = os.path.join(run_dir, "eval")
        os.makedirs(eval_dir, exist_ok=True)

        save_step = args.save_step if args.save_step is not None else config.total_train_steps
        train_state = load_model(os.path.abspath(model_dir), config)
        policy, _, _ = get_model(train_state.policy_state)

        raw_returns, _ = evaluate_policy(
            config,
            policy,
            env,
            eval_dir,
            num_episodes=args.episodes,
            max_steps=args.max_steps,
            t_env=save_step,
            key=jax.random.PRNGKey(int(seed)),
        )
        raw_returns = np.asarray(raw_returns)

        nsw = _nsw_score(raw_returns, args.eps, fourroom)
        usw = _usw_score(raw_returns)
        jain = _jain_index(raw_returns)

        if args.group_by == "hparams":
            key = (beta, hdim, bsize)
        elif args.group_by == "mix":
            key = (mix_tag,)
        else:
            key = (mix_tag, beta, hdim, bsize)
        per_setting[key].append(
            {
                "seed": int(seed),
                "nsw": nsw,
                "usw": usw,
                "jain": jain,
            }
        )

    rows = []
    def _sort_key(item):
        key = item[0]
        if args.group_by == "hparams":
            return (float(key[0]), int(key[1]), int(key[2]))
        if args.group_by == "mix":
            return (str(key[0]),)
        return (str(key[0]), float(key[1]), int(key[2]), int(key[3]))

    for key, vals in sorted(per_setting.items(), key=_sort_key):
        if args.group_by == "hparams":
            beta, hdim, bsize = key
        elif args.group_by == "mix":
            (mix_tag,) = key
            beta = hdim = bsize = ""
        else:
            mix_tag, beta, hdim, bsize = key

        nsw_vals = [v["nsw"] for v in vals]
        usw_vals = [v["usw"] for v in vals]
        jain_vals = [v["jain"] for v in vals]
        row = {
            "beta": beta,
            "hidden_dim": hdim,
            "batch_size": bsize,
            "num_seeds": len(vals),
            "nsw_mean": float(np.mean(nsw_vals)),
            "nsw_std": float(np.std(nsw_vals)),
            "usw_mean": float(np.mean(usw_vals)),
            "usw_std": float(np.std(usw_vals)),
            "jain_mean": float(np.mean(jain_vals)),
            "jain_std": float(np.std(jain_vals)),
        }
        if args.group_by in ("mix", "mix+hparams"):
            row["mix_tag"] = mix_tag
        rows.append(row)

    output_csv = args.output_csv or os.path.join(
        sweep_dir, f"reeval_welfare_episodes_{args.episodes}.csv"
    )
    with open(output_csv, "w", newline="") as f:
        fieldnames = [
            "beta",
            "hidden_dim",
            "batch_size",
            "num_seeds",
            "nsw_mean",
            "nsw_std",
            "usw_mean",
            "usw_std",
            "jain_mean",
            "jain_std",
        ]
        if args.group_by in ("mix", "mix+hparams"):
            fieldnames = ["mix_tag"] + fieldnames
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {len(rows)} settings to {output_csv}")
    if missing:
        print(f"Missing model dirs: {len(missing)} (first 10: {missing[:10]})")


if __name__ == "__main__":
    main()
