import argparse
import json
import os

import numpy as np


def _find_mu_star_files(parent_dir):
    paths = []
    for entry in sorted(os.listdir(parent_dir)):
        seed_dir = os.path.join(parent_dir, entry)
        if not os.path.isdir(seed_dir):
            continue
        json_path = os.path.join(seed_dir, "mu_star.json")
        if os.path.exists(json_path):
            paths.append(json_path)
    return paths


def main():
    parser = argparse.ArgumentParser(description="Aggregate mu* across seeds.")
    parser.add_argument("--parent_dir", type=str, required=True, help="Directory containing seed_* subfolders.")
    parser.add_argument("--output", type=str, default="mu_star_avg.json")
    args = parser.parse_args()

    mu_paths = _find_mu_star_files(args.parent_dir)
    if not mu_paths:
        raise FileNotFoundError(f"No mu_star.json found under {args.parent_dir}")

    mus = []
    for path in mu_paths:
        with open(path, "r") as f:
            mus.append(np.asarray(json.load(f), dtype=np.float32))
    mu_avg = np.mean(np.stack(mus, axis=0), axis=0)
    out_path = os.path.join(args.parent_dir, args.output)
    with open(out_path, "w") as f:
        json.dump(mu_avg.tolist(), f, indent=2)
    np.savez(os.path.join(args.parent_dir, "mu_star_avg.npz"), mu_star=mu_avg)
    print(f"Saved averaged mu* to {out_path}")


if __name__ == "__main__":
    main()
