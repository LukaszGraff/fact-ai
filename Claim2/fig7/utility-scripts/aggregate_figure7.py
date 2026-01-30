import argparse
import os
from typing import List

import numpy as np
import matplotlib.pyplot as plt


def _find_npz_files(parent_dir: str) -> List[str]:
    npz_files = []
    for entry in sorted(os.listdir(parent_dir)):
        run_dir = os.path.join(parent_dir, entry)
        if not os.path.isdir(run_dir):
            continue
        data_path = os.path.join(run_dir, "fig7_data.npz")
        if os.path.exists(data_path):
            npz_files.append(data_path)
    return npz_files


def _load_grids(paths: List[str]):
    grids = []
    deltas = None
    mu_star = None
    for path in paths:
        data = np.load(path)
        if deltas is None:
            deltas = data["deltas"]
        if mu_star is None:
            mu_star = data["mu_star"]
        grids.append(
            {
                "nsw": data["nsw_grid"],
                "util": data["util_grid"],
                "jain": data["jain_grid"],
                "obj": data["obj_grid"],
            }
        )
    return deltas, mu_star, grids


def _mean_grid(grids, key):
    arr = np.stack([g[key] for g in grids], axis=0)
    return arr.mean(axis=0)


def _mean_obj_grid(grids):
    arr = np.stack([g["obj"] for g in grids], axis=0)
    return arr.mean(axis=0)


def main():
    parser = argparse.ArgumentParser(description="Aggregate Figure 7 grids across seeds.")
    parser.add_argument("--parent_dir", type=str, required=True, help="Directory containing per-seed subfolders.")
    parser.add_argument("--output", type=str, default="fig7_avg.png")
    args = parser.parse_args()

    npz_files = _find_npz_files(args.parent_dir)
    if not npz_files:
        raise FileNotFoundError(f"No fig7_data.npz found under {args.parent_dir}")

    deltas, mu_star, grids = _load_grids(npz_files)
    nsw_grid = _mean_grid(grids, "nsw")
    util_grid = _mean_grid(grids, "util")
    jain_grid = _mean_grid(grids, "jain")
    obj_grid = _mean_obj_grid(grids)

    fig, axes = plt.subplots(2, 3, figsize=(12, 8))
    panels = [
        ("Nash Social Welfare", nsw_grid),
        ("Utilitarian", util_grid),
        ("Jain's Fairness", jain_grid),
        ("Objective 1", obj_grid[:, :, 0]),
        ("Objective 2", obj_grid[:, :, 1]),
        ("Objective 3", obj_grid[:, :, 2]),
    ]
    extent = [deltas.min() * 100, deltas.max() * 100, deltas.min() * 100, deltas.max() * 100]
    for ax, (title, grid) in zip(axes.flat, panels):
        im = ax.imshow(grid, origin="lower", extent=extent, aspect="auto")
        ax.set_title(title)
        ax.set_xlabel("Perturbation on mu2 (%)")
        ax.set_ylabel("Perturbation on mu3 (%)")
        ax.plot(0, 0, "kx", markersize=8)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    fig.suptitle("Figure 7 (Averaged): FairDICE-fixed performance with perturbed μ", fontsize=12)
    fig.tight_layout(rect=[0, 0.02, 1, 0.95])
    out_path = os.path.join(args.parent_dir, args.output)
    fig.savefig(out_path, dpi=200)
    print(f"Saved averaged figure to {out_path}")


if __name__ == "__main__":
    main()
