import argparse
import os
from typing import Dict, List, Tuple

import numpy as np


def _grid_deltas(grid_min: float, grid_max: float, grid_points: int) -> np.ndarray:
    return np.linspace(grid_min, grid_max, grid_points)


def _find_cell_outputs(root_dir: str, cell_output: str) -> List[str]:
    matches: List[str] = []
    for dirpath, _dirnames, filenames in os.walk(root_dir):
        if cell_output in filenames:
            matches.append(os.path.join(dirpath, cell_output))
    return sorted(matches)


def _load_cell(path: str) -> Dict[str, np.ndarray]:
    data = np.load(path, allow_pickle=False)
    nsw = float(np.asarray(data["nsw"]))
    util = float(np.asarray(data["util"]))
    jain = float(np.asarray(data["jain"]))
    avg_returns = np.asarray(data["avg_returns"], dtype=np.float64)
    mu_star = np.asarray(data["mu_star"], dtype=np.float32)
    i = int(np.asarray(data["i"]))
    j = int(np.asarray(data["j"]))
    seed = int(np.asarray(data["seed"]))
    grid_min = float(np.asarray(data["grid_min"]))
    grid_max = float(np.asarray(data["grid_max"]))
    grid_points = int(np.asarray(data["grid_points"]))
    return {
        "nsw": np.asarray(nsw),
        "util": np.asarray(util),
        "jain": np.asarray(jain),
        "avg_returns": avg_returns,
        "mu_star": mu_star,
        "i": np.asarray(i),
        "j": np.asarray(j),
        "seed": np.asarray(seed),
        "grid_min": np.asarray(grid_min),
        "grid_max": np.asarray(grid_max),
        "grid_points": np.asarray(grid_points),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Aggregate per-cell Figure 7 results into full grids."
    )
    parser.add_argument(
        "--root_dir",
        type=str,
        default="./results/fig7_cells",
        help="Root directory to search recursively for cell outputs.",
    )
    parser.add_argument("--grid_min", type=float, default=-0.1)
    parser.add_argument("--grid_max", type=float, default=0.1)
    parser.add_argument("--grid_points", type=int, default=21)
    parser.add_argument(
        "--seed",
        type=int,
        default=-1,
        help="If set >= 0, only aggregate this seed.",
    )
    parser.add_argument(
        "--include_seeds",
        type=str,
        default="",
        help="Comma-separated list of seeds to include (overrides --seed if set).",
    )
    parser.add_argument(
        "--cell_output",
        type=str,
        default="cell_result.npz",
        help="Filename produced by figure7_fourroom.py single-cell runs.",
    )
    parser.add_argument(
        "--output_npz",
        type=str,
        default="fig7_data_aggregated.npz",
        help="Output NPZ filename inside root_dir.",
    )
    parser.add_argument(
        "--output_fig",
        type=str,
        default="fig7_aggregated.png",
        help="Output figure filename inside root_dir.",
    )
    args = parser.parse_args()

    os.makedirs(args.root_dir, exist_ok=True)

    cell_paths = _find_cell_outputs(args.root_dir, args.cell_output)
    if not cell_paths:
        raise FileNotFoundError(
            f"No '{args.cell_output}' files found under {args.root_dir}."
        )

    include_seeds = None
    if args.include_seeds.strip():
        include_seeds = {int(s.strip()) for s in args.include_seeds.split(",") if s.strip() != ""}

    deltas = _grid_deltas(args.grid_min, args.grid_max, args.grid_points)
    n = args.grid_points

    def empty_grids() -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        return (
            np.full((n, n), np.nan, dtype=np.float64),
            np.full((n, n), np.nan, dtype=np.float64),
            np.full((n, n), np.nan, dtype=np.float64),
            np.full((n, n, 3), np.nan, dtype=np.float64),
        )

    mu_star_ref = None
    per_seed: Dict[int, Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = {}
    per_seed_counts: Dict[int, int] = {}

    for cell_path in cell_paths:
        cell = _load_cell(cell_path)
        seed = int(cell["seed"])
        if include_seeds is not None:
            if seed not in include_seeds:
                continue
        elif args.seed >= 0 and seed != args.seed:
            continue

        cell_grid_min = float(cell["grid_min"])
        cell_grid_max = float(cell["grid_max"])
        cell_grid_points = int(cell["grid_points"])
        if (
            not np.isclose(cell_grid_min, args.grid_min)
            or not np.isclose(cell_grid_max, args.grid_max)
            or cell_grid_points != args.grid_points
        ):
            raise ValueError(
                "Grid settings mismatch between aggregator args and cell outputs: "
                f"cell({cell_grid_min}, {cell_grid_max}, {cell_grid_points}) vs "
                f"args({args.grid_min}, {args.grid_max}, {args.grid_points})."
            )

        i = int(cell["i"])
        j = int(cell["j"])

        if seed not in per_seed:
            per_seed[seed] = empty_grids()
            per_seed_counts[seed] = 0

        nsw_grid_s, util_grid_s, jain_grid_s, obj_grid_s = per_seed[seed]
        nsw_grid_s[i, j] = float(cell["nsw"])
        util_grid_s[i, j] = float(cell["util"])
        jain_grid_s[i, j] = float(cell["jain"])
        obj_grid_s[i, j, :] = np.asarray(cell["avg_returns"], dtype=np.float64)
        per_seed_counts[seed] += 1

        if mu_star_ref is None:
            mu_star_ref = np.asarray(cell["mu_star"], dtype=np.float32)

    if not per_seed:
        if include_seeds is not None:
            seed_msg = f" for seeds={sorted(include_seeds)}"
        else:
            seed_msg = f" for seed={args.seed}" if args.seed >= 0 else ""
        raise FileNotFoundError(
            f"No matching cell outputs found under {args.root_dir}{seed_msg}."
        )

    total = n * n
    seeds_sorted = sorted(per_seed.keys())
    for seed in seeds_sorted:
        count = per_seed_counts.get(seed, 0)
        print(f"Seed {seed}: found {count}/{total} cells.")

    # Stack across seeds and aggregate with nanmean so missing cells don't crash.
    nsw_stack = np.stack([per_seed[s][0] for s in seeds_sorted], axis=0)
    util_stack = np.stack([per_seed[s][1] for s in seeds_sorted], axis=0)
    jain_stack = np.stack([per_seed[s][2] for s in seeds_sorted], axis=0)
    obj_stack = np.stack([per_seed[s][3] for s in seeds_sorted], axis=0)

    nsw_grid = np.nanmean(nsw_stack, axis=0)
    util_grid = np.nanmean(util_stack, axis=0)
    jain_grid = np.nanmean(jain_stack, axis=0)
    obj_grid = np.nanmean(obj_stack, axis=0)

    if mu_star_ref is None:
        raise FileNotFoundError(
            f"No cell outputs found under {args.root_dir} with name {args.cell_output}."
        )

    npz_path = os.path.join(args.root_dir, args.output_npz)
    np.savez(
        npz_path,
        deltas=deltas,
        mu_star=mu_star_ref,
        nsw_grid=nsw_grid,
        util_grid=util_grid,
        jain_grid=jain_grid,
        obj_grid=obj_grid,
        seeds=np.asarray(seeds_sorted, dtype=np.int32),
    )
    print(f"Saved aggregated data to {npz_path}")

    # Plot, mirroring the original layout.
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 3, figsize=(12, 8))
    panels = [
        ("Nash Social Welfare", nsw_grid),
        ("Utilitarian", util_grid),
        ("Jain's Fairness", jain_grid),
        ("Objective 1", obj_grid[:, :, 0]),
        ("Objective 2", obj_grid[:, :, 1]),
        ("Objective 3", obj_grid[:, :, 2]),
    ]
    extent = [
        args.grid_min * 100,
        args.grid_max * 100,
        args.grid_min * 100,
        args.grid_max * 100,
    ]
    for ax, (title, grid) in zip(axes.flat, panels):
        im = ax.imshow(grid, origin="lower", extent=extent, aspect="auto")
        ax.set_title(title)
        ax.set_xlabel("Perturbation on mu2 (%)")
        ax.set_ylabel("Perturbation on mu3 (%)")
        ax.plot(0, 0, "kx", markersize=8)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    fig.suptitle(
        "Figure 7 (Aggregated): FairDICE-fixed performance with perturbed μ in MO-Four-Room",
        fontsize=12,
    )
    fig.tight_layout(rect=[0, 0.02, 1, 0.95])

    fig_path = os.path.join(args.root_dir, args.output_fig)
    fig.savefig(fig_path, dpi=200)
    print(f"Saved aggregated figure to {fig_path}")


if __name__ == "__main__":
    main()
