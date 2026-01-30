#!/usr/bin/env python3
"""
NSW vs beta for multiple envs.
For each beta: plot Amateur and Expert side-by-side with ±1 SE error bars.
Per-subplot y-scale starting from 0. One shared legend at the bottom.

Assumes run_dir/eval/normalized_returns_step_*.npy exists.

Folder name should contain:
  - env like MO-Hopper-v2, MO-Hopper-v3, MO-Walker2d-v2, etc.
  - beta<value>
  - seed<int>
  - contains 'amateur' or 'expert' (e.g., amateur_uniform, expert_uniform)
"""

import argparse
import math
import os
import re
from glob import glob
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt


ENV_RE = re.compile(r"(MO-[A-Za-z0-9]+(?:-3obj)?-v[23]|MO-[A-Za-z0-9]+(?:-3obj)?-v2|MO-[A-Za-z0-9]+(?:-3obj)?)")
BETA_RE = re.compile(r"beta([0-9]*\.?[0-9]+(?:e-?\d+)?)", re.IGNORECASE)
SEED_RE = re.compile(r"seed(\d+)", re.IGNORECASE)


def parse_dataset(name: str, dataset_suffix: str = "") -> Optional[str]:
    lname = name.lower()
    if dataset_suffix:
        suf = "_" + dataset_suffix.lower()
        if ("amateur" + suf) in lname:
            return "amateur"
        if ("expert" + suf) in lname:
            return "expert"
        return None
    if "amateur" in lname:
        return "amateur"
    if "expert" in lname:
        return "expert"
    return None


def find_latest_normalized_returns(eval_dir: str) -> Optional[str]:
    cands = glob(os.path.join(eval_dir, "normalized_returns_step_*.npy"))
    if not cands:
        return None

    def step_of(p: str) -> int:
        m = re.search(r"step_(\d+)\.npy$", os.path.basename(p))
        return int(m.group(1)) if m else -1

    return max(cands, key=step_of)


def safe_log(x: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    return np.log(np.maximum(x, eps))


def beta_label(b: float) -> str:
    if b != 0 and (abs(b) < 1e-3 or abs(b) >= 1e3):
        return f"{b:.0e}".replace("e-0", "e-").replace("e+0", "e+")
    if b < 0.01:
        return f"{b:.5f}".rstrip("0").rstrip(".")
    if b < 1.0:
        return f"{b:.3f}".rstrip("0").rstrip(".")
    return f"{b:.1f}"


@dataclass
class RunResult:
    env: str
    dataset: str
    beta: float
    seed: int
    nsw_mean: float


def parse_run_dir(run_dir: str, dataset_suffix: str = "") -> Optional[Tuple[str, str, float, int]]:
    name = os.path.basename(run_dir.rstrip("/"))

    m = ENV_RE.search(name)
    if not m:
        return None
    env = m.group(1)

    dataset = parse_dataset(name, dataset_suffix=dataset_suffix)
    if dataset is None:
        return None

    m = BETA_RE.search(name)
    if not m:
        return None
    try:
        beta = float(m.group(1))
    except ValueError:
        return None

    m = SEED_RE.search(name)
    if not m:
        return None
    seed = int(m.group(1))

    return env, dataset, beta, seed


def load_run(run_dir: str, dataset_suffix: str = "") -> Optional[RunResult]:
    parsed = parse_run_dir(run_dir, dataset_suffix=dataset_suffix)
    if parsed is None:
        return None
    env, dataset, beta, seed = parsed

    npy_path = find_latest_normalized_returns(os.path.join(run_dir, "eval"))
    if npy_path is None:
        return None

    norm_returns = np.load(npy_path)
    if norm_returns.ndim != 2 or norm_returns.shape[0] < 1:
        return None

    nsw_episode = safe_log(norm_returns).sum(axis=1)
    nsw_mean = float(nsw_episode.mean())

    return RunResult(env=env, dataset=dataset, beta=beta, seed=seed, nsw_mean=nsw_mean)


@dataclass
class AggPoint:
    mean: float
    se: float
    n: int


def agg_seed_means(seed_means: List[float]) -> AggPoint:
    v = np.array(seed_means, dtype=float)
    n = len(v)
    if n == 0:
        return AggPoint(mean=float("nan"), se=float("nan"), n=0)
    mean = float(v.mean())
    se = float(v.std(ddof=1) / math.sqrt(n)) if n > 1 else 0.0
    return AggPoint(mean=mean, se=se, n=n)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="Root directory containing run folders (searched recursively).")
    ap.add_argument("--out", default="fairdice_results.png", help="Output image path.")
    ap.add_argument("--dataset_suffix", default="", help="Optional suffix like 'uniform' to require expert_uniform/amateur_uniform.")
    ap.add_argument("--ncols", type=int, default=3, help="Subplot columns (3 for 6 envs -> 2x3).")
    ap.add_argument("--envs", nargs="*", default=None,
                    help="Optional explicit env list to plot (exact names as in folder names).")
    args = ap.parse_args()

    candidates = [
        d for d in glob(os.path.join(args.root, "**"), recursive=True)
        if os.path.isdir(d) and os.path.isdir(os.path.join(d, "eval"))
    ]

    runs: List[RunResult] = []
    for d in candidates:
        rr = load_run(d, dataset_suffix=args.dataset_suffix)
        if rr is not None:
            runs.append(rr)

    if not runs:
        raise SystemExit("No runs found. Check --root, naming (env/beta/seed), dataset tags, and eval/*.npy files.")

    grouped: Dict[str, Dict[str, Dict[float, List[float]]]] = {"amateur": {}, "expert": {}}
    for r in runs:
        grouped[r.dataset].setdefault(r.env, {}).setdefault(r.beta, []).append(r.nsw_mean)

    all_envs = sorted(set(grouped["amateur"].keys()) | set(grouped["expert"].keys()))
    if args.envs:
        envs = args.envs
    else:
        envs = [e for e in all_envs if e in grouped["amateur"] and e in grouped["expert"]]
        if not envs:
            envs = all_envs

    betas = sorted(set(b for ds in grouped.values() for e in ds.values() for b in e.keys()))

    agg: Dict[str, Dict[str, Dict[float, AggPoint]]] = {"amateur": {}, "expert": {}}
    for ds in ["amateur", "expert"]:
        for env in envs:
            for beta in betas:
                seed_means = grouped.get(ds, {}).get(env, {}).get(beta, [])
                if seed_means:
                    agg[ds].setdefault(env, {})[beta] = agg_seed_means(seed_means)

    COLOR_AMATEUR = "#4C72B0"
    COLOR_EXPERT  = "#DD8452"

    n = len(envs)
    ncols = args.ncols
    nrows = int(math.ceil(n / ncols))
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(4.8 * ncols, 3.8 * nrows), squeeze=False)

    legend_handles = None
    legend_labels = None

    for idx, env in enumerate(envs):
        r = idx // ncols
        c = idx % ncols
        ax = axes[r][c]

        present = [b for b in betas if (b in agg["amateur"].get(env, {}) and b in agg["expert"].get(env, {}))]
        if not present:
            ax.set_title(env + " (no paired data)")
            ax.axis("off")
            continue

        x = np.arange(len(present))
        width = 0.40

        am_means = [agg["amateur"][env][b].mean for b in present]
        am_se    = [agg["amateur"][env][b].se   for b in present]
        ex_means = [agg["expert"][env][b].mean  for b in present]
        ex_se    = [agg["expert"][env][b].se    for b in present]

        h1 = ax.bar(x - width/2, am_means, width=width, yerr=am_se, capsize=3,
                    color=COLOR_AMATEUR, label="Amateur")
        h2 = ax.bar(x + width/2, ex_means, width=width, yerr=ex_se, capsize=3,
                    color=COLOR_EXPERT, label="Expert")

        ax.set_title(env)
        ax.set_xticks(x)
        ax.set_xticklabels([beta_label(b) for b in present], rotation=45, ha="right")
        ax.set_xlabel("Beta")
        ax.set_ylabel("NSW Score")
        ax.set_ylim(bottom=0)  # per-subplot scale, start from 0
        ax.grid(True, axis="y", alpha=0.3)

        if legend_handles is None:
            legend_handles, legend_labels = ax.get_legend_handles_labels()

    for j in range(n, nrows * ncols):
        axes[j // ncols][j % ncols].axis("off")

    fig.legend(legend_handles, legend_labels, loc="lower center", ncol=2, frameon=True,
               bbox_to_anchor=(0.5, -0.01))

    fig.tight_layout(rect=[0, 0.05, 1, 0.95])

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    fig.savefig(args.out, dpi=200, bbox_inches="tight")
    print(f"[saved] {args.out}")


if __name__ == "__main__":
    main()
