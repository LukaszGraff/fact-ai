import argparse
import os
import re
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


ENV_HEADER_RE = re.compile(r"^(MO-[^:]+):")
LINE_RE = re.compile(
    r"^\s*std=(?P<std>[0-9.]+):\s+mean_raw_nsw=(?P<mean>[0-9.Ee+-]+)\s+std_raw_nsw=(?P<std_raw_nsw>[0-9.Ee+-]+)\s+n=(?P<n>\d+)"
)


def parse_summary(path):
    data = defaultdict(list)
    current_env = None
    with open(path, "r") as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line:
                continue
            env_match = ENV_HEADER_RE.match(line)
            if env_match:
                current_env = env_match.group(1)
                continue
            if current_env is None:
                continue
            match = LINE_RE.match(line)
            if not match:
                continue
            std_val = float(match.group("std"))
            mean_val = float(match.group("mean"))
            std_raw_val = float(match.group("std_raw_nsw"))
            n_val = int(match.group("n"))
            data[current_env].append((std_val, mean_val, std_raw_val, n_val))
    return data


def plot_summary(data, output_path, title="Mean Raw NSW vs. Perturbation Std"):
    plt.figure(figsize=(12, 8))
    for env, entries in sorted(data.items()):
        entries_sorted = sorted(entries, key=lambda x: x[0])
        stds = [e[0] for e in entries_sorted]
        means = [e[1] for e in entries_sorted]
        std_devs = [e[2] for e in entries_sorted]
        plt.errorbar(stds, means, yerr=std_devs, marker="o", linewidth=2, capsize=4, label=env)

    plt.xscale("log")
    plt.xlabel("Std (sigma)")
    plt.ylabel("Mean Raw NSW")
    plt.title(title)
    plt.grid(True, which="both", linestyle="--", alpha=0.4)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--summary",
        default=os.path.join(os.path.dirname(__file__), "results", "nsw_summary_mu_perturb_all.txt"),
        help="Path to nsw summary file",
    )
    parser.add_argument(
        "--output",
        default=os.path.join(os.path.dirname(__file__), "results", "nsw_mean_raw_by_std.png"),
        help="Output PNG path",
    )
    parser.add_argument("--title", default="Mean Raw NSW vs. Perturbation Std")
    args = parser.parse_args()

    data = parse_summary(args.summary)
    if not data:
        raise ValueError(f"No data parsed from {args.summary}")

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    plot_summary(data, args.output, title=args.title)
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
