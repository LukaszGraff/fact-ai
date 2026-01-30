import argparse
import glob
import os
import re
import numpy as np


def _step_from_fname(path: str) -> int:
    """Extract integer step from '*_step_<int>.npy'. Returns -1 if missing."""
    base = os.path.basename(path)
    m = re.search(r"_step_?(\d+)", base)
    return int(m.group(1)) if m else -1


def _latest_step_file(eval_dir: str, pattern: str) -> str:
    files = glob.glob(os.path.join(eval_dir, pattern))
    if not files:
        raise RuntimeError(f"No files matching {os.path.join(eval_dir, pattern)}")
    return max(files, key=_step_from_fname)


def load_fairdice_returns_from_run(run_dir: str):
    """
    Load per-episode per-objective returns from one FairDICE run.

    Expected layout:
      run_dir/
        eval/
          normalized_returns_step_<STEP>.npy   # shape (n_episodes, n_obj)
          raw_returns_step_<STEP>.npy          # shape (n_episodes, n_obj)

    Returns:
      (raw_returns, normalized_returns, step)
    """
    eval_dir = os.path.join(run_dir, "eval")
    if not os.path.isdir(eval_dir):
        raise RuntimeError(f"No eval/ directory in {run_dir}")

    raw_path = _latest_step_file(eval_dir, "raw_returns_step_*.npy")
    norm_path = _latest_step_file(eval_dir, "normalized_returns_step_*.npy")

    raw = np.asarray(np.load(raw_path), dtype=float)
    norm = np.asarray(np.load(norm_path), dtype=float)
    step = max(_step_from_fname(raw_path), _step_from_fname(norm_path))

    if raw.ndim != 2 or norm.ndim != 2:
        raise ValueError(
            f"Expected 2D arrays (episodes,obj). Got raw{raw.shape}, norm{norm.shape} in {run_dir}."
        )
    if raw.shape[1] != norm.shape[1]:
        raise ValueError(
            f"Objective dim mismatch: raw{raw.shape} vs norm{norm.shape} in {run_dir}."
        )
    return raw, norm, step


def main():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--root",
        required=True,
        help="Root directory containing FairDICE result folders (e.g. ../FairDICE/results)",
    )
    p.add_argument("--env", required=True, help="Env name, e.g. MO-Hopper-v2")
    p.add_argument(
        "--dataset",
        required=True,
        help="Dataset string in folder names, e.g. expert_uniform or amateur_uniform",
    )
    p.add_argument(
        "--beta_filter",
        default=None,
        help=(
            "Optional substring to select only runs with a given beta, "
            "e.g. 'beta1.0' or 'beta0.001'. If omitted, all betas are used."
        ),
    )
    p.add_argument(
        "--out",
        required=True,
        help=(
            "Output .npz path (consumed by make_plots.py). "
            "This script stores BOTH raw and normalized returns."
        ),
    )
    args = p.parse_args()

    pattern = os.path.join(args.root, f"*FairDICE_{args.env}_{args.dataset}_*")
    cand_dirs = sorted(d for d in glob.glob(pattern) if os.path.isdir(d))
    if args.beta_filter is not None:
        cand_dirs = [d for d in cand_dirs if args.beta_filter in d]

    if not cand_dirs:
        raise RuntimeError(
            f"No FairDICE run dirs found matching {pattern} (beta_filter={args.beta_filter})"
        )

    print("Found FairDICE runs:")
    for d in cand_dirs:
        print("  ", d)

    raws, norms, steps = [], [], []
    for d in cand_dirs:
        raw, norm, step = load_fairdice_returns_from_run(d)
        print(f"  loaded raw{raw.shape} norm{norm.shape} step={step} from {d}")
        raws.append(raw)
        norms.append(norm)
        steps.append(step)

    # returns_* have shape (n_runs, n_episodes, n_obj)
    returns_raw = np.stack(raws, axis=0)
    returns_norm = np.stack(norms, axis=0)

    # Per-run means: (n_runs, n_obj)
    mean_raw_per_run = returns_raw.mean(axis=1)
    mean_norm_per_run = returns_norm.mean(axis=1)

    # Aggregate mean across runs: (n_obj,)
    mean_raw = mean_raw_per_run.mean(axis=0)
    mean_norm = mean_norm_per_run.mean(axis=0)

    step = int(max(steps)) if steps else -1

    np.savez(
        args.out,
        env=args.env,
        dataset=args.dataset,
        step=step,
        returns_raw=returns_raw,
        returns_norm=returns_norm,
        mean_raw_per_run=mean_raw_per_run,
        mean_norm_per_run=mean_norm_per_run,
        mean_raw=mean_raw,
        mean_norm=mean_norm,
    )

    print(f"\nSaved FairDICE aggregated returns to {args.out}")
    print("  returns_raw:", returns_raw.shape)
    print("  returns_norm:", returns_norm.shape)
    print("  mean_raw:", mean_raw.shape)
    print("  mean_norm:", mean_norm.shape)


if __name__ == "__main__":
    main()
