import os
import re
import glob
import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(BASE_DIR, "results")
PERTURB_DIRS = [
    os.path.join(BASE_DIR, "results_mu_perturb"),
    os.path.join(BASE_DIR, "results_mu_perturb_ant"),
]
OUTPUT_PATH = os.path.join(RESULTS_DIR, "nsw_summary_mu_perturb_all.txt")

env_token_re = re.compile(r"^MO-")


def avg_raw_nsw_from_eval(eval_dir: str):
    # Use the max step raw_returns file
    files = glob.glob(os.path.join(eval_dir, "raw_returns_step_*.npy"))
    if not files:
        return None
    # pick max step
    def step_of(path):
        m = re.search(r"raw_returns_step_(\d+)\.npy", os.path.basename(path))
        return int(m.group(1)) if m else -1
    path = max(files, key=step_of)
    raw_returns = np.load(path)
    # raw_returns shape: (episodes, reward_dim)
    mask = np.all(raw_returns > 0, axis=1)
    if not np.any(mask):
        return None
    safe_returns = raw_returns[mask]
    return float(np.mean(np.sum(np.log(safe_returns), axis=1)))


def parse_run_name(name: str):
    parts = name.split("_")
    if len(parts) < 8:
        return None
    # find env token
    env_idx = None
    for i, p in enumerate(parts):
        if env_token_re.match(p):
            env_idx = i
            break
    if env_idx is None or env_idx < 2:
        return None
    learner = "_".join(parts[2:env_idx])
    env = parts[env_idx]
    if env_idx + 2 >= len(parts):
        return None
    quality = parts[env_idx + 1]
    pref = parts[env_idx + 2]
    # divergence tokens are between pref and beta token
    beta_idx = None
    seed_idx = None
    for i, p in enumerate(parts):
        if p.startswith("beta"):
            beta_idx = i
        if p.startswith("seed"):
            seed_idx = i
    if beta_idx is None or seed_idx is None or beta_idx <= env_idx + 2:
        return None
    divergence = "_".join(parts[env_idx + 3:beta_idx])
    beta = parts[beta_idx].replace("beta", "")
    seed = parts[seed_idx].replace("seed", "")
    if not seed.isdigit():
        return None
    return {
        "learner": learner,
        "env": env,
        "quality": quality,
        "pref": pref,
        "divergence": divergence,
        "beta": beta,
        "seed": int(seed),
    }


def collect_runs(base_dir: str):
    runs = []
    for run_dir in glob.glob(os.path.join(base_dir, "*")):
        if not os.path.isdir(run_dir):
            continue
        name = os.path.basename(run_dir)
        meta = parse_run_name(name)
        if not meta:
            continue
        eval_dir = os.path.join(run_dir, "eval")
        score = avg_raw_nsw_from_eval(eval_dir)
        if score is None:
            continue
        runs.append({
            "env": meta["env"],
            "learner": meta["learner"],
            "seed": meta["seed"],
            "score": score,
        })
    return runs


def collect_perturbed():
    runs = []
    for base_dir in PERTURB_DIRS:
        for std_dir in glob.glob(os.path.join(base_dir, "std_*")):
            if not os.path.isdir(std_dir):
                continue
            std = os.path.basename(std_dir).replace("std_", "")
            for run_dir in glob.glob(os.path.join(std_dir, "*")):
                if not os.path.isdir(run_dir):
                    continue
                name = os.path.basename(run_dir)
                meta = parse_run_name(name)
                if not meta:
                    continue
                eval_dir = os.path.join(run_dir, "eval")
                score = avg_raw_nsw_from_eval(eval_dir)
                if score is None:
                    continue
                runs.append({
                    "env": meta["env"],
                    "std": std,
                    "seed": meta["seed"],
                    "score": score,
                })
    return runs


def main():
    perturb_runs = collect_perturbed()

    # Aggregate perturbed scores by env+std
    perturb_by_env = {}
    for r in perturb_runs:
        key = (r["env"], r["std"])
        perturb_by_env.setdefault(key, []).append(r["score"])

    lines = []
    envs = sorted({e for e, _ in perturb_by_env.keys()})
    for env in envs:
        lines.append(f"{env}:")
        # stds
        stds = sorted({std for (e, std) in perturb_by_env.keys() if e == env}, key=lambda s: float(s))
        for std in stds:
            scores = perturb_by_env.get((env, std), [])
            if scores:
                mean_score = float(np.mean(scores))
                if len(scores) > 1:
                    std_score = float(np.std(scores, ddof=1))
                else:
                    std_score = 0.0
                lines.append(
                    f"  std={std}: mean_raw_nsw={mean_score:.6f} "
                    f"std_raw_nsw={std_score:.6f} n={len(scores)}"
                )
            else:
                lines.append(f"  std={std}: mean_raw_nsw=NA std_raw_nsw=NA n=0")

    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        f.write("\n".join(lines) + "\n")

    print(f"Wrote {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
