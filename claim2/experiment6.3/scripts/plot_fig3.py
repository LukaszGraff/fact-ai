import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot Figure 3 with error bands")
    parser.add_argument("--in_csv", type=str, default="outputs/fig3_agg.csv")
    parser.add_argument("--out_png", type=str, default="outputs/fig3.png")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    in_path = Path(args.in_csv)
    out_path = Path(args.out_png)
    ensure_parent(out_path)

    df = pd.read_csv(in_path)
    required_cols = {"beta", "sigma", "mean_nsw", "sem_nsw"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns in {in_path}: {missing}")

    betas = sorted(df["beta"].unique())
    min_positive_sigma = df.loc[df["sigma"] > 0, "sigma"].min()

    plt.figure()
    for beta in betas:
        sub = df[df["beta"] == beta].sort_values("sigma")
        sigma_vals = sub["sigma"].to_numpy(dtype=float)
        x_plot = sigma_vals.copy()
        if min_positive_sigma is not None:
            x_plot[x_plot == 0.0] = min_positive_sigma * 0.5
        y = sub["mean_nsw"].to_numpy(dtype=float)
        sem = sub["sem_nsw"].fillna(0.0).to_numpy(dtype=float)
        plt.fill_between(x_plot, y - sem, y + sem, alpha=0.2)
        plt.plot(x_plot, y, label=f"beta={beta}")

    plt.xscale("log")
    plt.xlabel("Perturbation on mu* (sigma)")
    plt.ylabel("sum log NSW")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
