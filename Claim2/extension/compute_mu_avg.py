import os
import glob
import numpy as np

RESULTS_DIR = os.path.join(os.getcwd(), "results")
OUTPUT_PATH = os.path.join(RESULTS_DIR, "mu_averages.txt")

# run_name format:
# YYYYMMDD_HHMMSS_<learner>_<env>_<quality>_<pref>_<div>_beta<beta>_seed<seed>

def parse_env_from_run_name(run_name: str) -> str:
    parts = run_name.split("_")
    if len(parts) < 8:
        raise ValueError(f"Unexpected run name format: {run_name}")
    return parts[3]


def load_mu(mu_path: str) -> np.ndarray:
    with open(mu_path, "r") as f:
        line = f.readline().strip()
    if not line:
        raise ValueError(f"Empty mu file: {mu_path}")
    return np.array([float(x) for x in line.split(",")], dtype=float)


def main():
    if not os.path.isdir(RESULTS_DIR):
        print(f"results dir not found: {RESULTS_DIR}")
        return

    mu_files = glob.glob(os.path.join(RESULTS_DIR, "*", "final_mu.txt"))
    if not mu_files:
        print("No final_mu.txt files found under results/")
        return

    env_to_mus = {}
    for mu_file in mu_files:
        run_dir = os.path.basename(os.path.dirname(mu_file))
        env = parse_env_from_run_name(run_dir)
        mu = load_mu(mu_file)
        env_to_mus.setdefault(env, []).append(mu)

    lines = []
    for env, mus in sorted(env_to_mus.items()):
        arr = np.stack(mus, axis=0)
        mean_mu = arr.mean(axis=0)
        lines.append(f"{env}: mean_mu={mean_mu.tolist()} n={arr.shape[0]}")

    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        f.write("\n".join(lines) + "\n")

    print(f"Wrote {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
