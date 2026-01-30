import os
import re
import numpy as np
import hashlib

BASE_DIR = os.getcwd()
MU_AVG_PATH = os.path.join(BASE_DIR, "results", "mu_averages.txt")
TASKS_PATH = os.path.join(BASE_DIR, "mu_perturb_tasks.txt")

# Configure here if needed
STDS = [1e-3, 1e-2, 1e-1, 1e0]
SEEDS = [1, 2, 3, 4, 5]
EXCLUDE_ENVS = {"MO-Ant-v2"}

pattern = re.compile(r"^(?P<env>[^:]+):\s*mean_mu=\[(?P<mu>[^\]]+)\]")


def parse_mu_averages(path: str):
    env_to_mu = {}
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            m = pattern.match(line)
            if not m:
                continue
            env = m.group("env").strip()
            mu_vals = [float(x.strip()) for x in m.group("mu").split(",")]
            env_to_mu[env] = np.array(mu_vals, dtype=float)
    return env_to_mu


def noise_seed(env: str, std: float, seed: int) -> int:
    # Stable deterministic seed per (env, std, seed)
    key = f"{env}|{std}|{seed}".encode("utf-8")
    digest = hashlib.md5(key).hexdigest()
    return int(digest[:8], 16)


def main():
    if not os.path.exists(MU_AVG_PATH):
        print(f"mu_averages.txt not found: {MU_AVG_PATH}")
        return

    env_to_mu = parse_mu_averages(MU_AVG_PATH)
    if not env_to_mu:
        print("No mu averages found.")
        return

    lines = []
    for env, mean_mu in sorted(env_to_mu.items()):
        if env in EXCLUDE_ENVS:
            continue
        for std in STDS:
            for seed in SEEDS:
                rng = np.random.default_rng(noise_seed(env, std, seed))
                noise = rng.normal(loc=0.0, scale=std, size=mean_mu.shape)
                mu_init = mean_mu * (1.0 + noise)
                mu_str = ",".join([f"{x:.8f}" for x in mu_init.tolist()])
                lines.append(f"{env}\t{std}\t{seed}\t{mu_str}")

    with open(TASKS_PATH, "w") as f:
        f.write("env\tstd\tseed\tmu_init\n")
        for line in lines:
            f.write(line + "\n")

    print(f"Wrote {TASKS_PATH} with {len(lines)} tasks")


if __name__ == "__main__":
    main()
