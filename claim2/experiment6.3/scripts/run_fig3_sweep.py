import sys
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

# Ensure src package (one level up) is importable when running as a script
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.random_momdp import RandomMOMDP
from src.dataset import make_behavior_policy_half_optimal, collect_offline_dataset
from src.occupancy_solvers import empirical_d_dataset, solve_fairdice_nsw, solve_fairdice_fixed
from src.metrics import nsw, mu_from_nsw_opt, perturb_mu

def main():
    env_seed = 0
    rollout_seed = 10000

    print(f"[setup] Initializing RandomMOMDP with seed={env_seed}")
    env = RandomMOMDP(seed=env_seed)
    print("[setup] Building half-optimal behavior policy")
    pi_b, stats = make_behavior_policy_half_optimal(env)
    print(f"[setup] Collecting offline dataset with rollout_seed={rollout_seed}")
    dataset = collect_offline_dataset(env, pi_b, n_traj=100, seed=rollout_seed)

    print("[setup] Computing empirical occupancy measure")
    dD = empirical_d_dataset(dataset, env.n_states, env.n_actions, env.gamma)

    # In Figure 3, curves correspond to a few beta choices (consistent with paper-style sweeps)
    betas = [0.001, 0.01, 0.1, 1.0]
    # x-axis: perturbation magnitude (log scale)
    sigmas = np.logspace(-3, 0, 13)  # 1e-3 ... 1e0

    n_noise = 20  # average multiple perturb draws per sigma
    rng = np.random.default_rng(2026)

    print(f"[sweep] Running for betas={betas}")
    print(f"[sweep] Sigmas range: {sigmas[0]:.3g} to {sigmas[-1]:.3g} with {len(sigmas)} points")

    rows = []
    for beta in betas:
        print(f"[beta] Solving FairDICE-NSW for beta={beta}")
        # solve FairDICE-NSW to get R* then µ*
        _, R_star = solve_fairdice_nsw(env, dD, beta=beta)
        mu_star = mu_from_nsw_opt(R_star)

        # IMPORTANT: baseline should be FairDICE-fixed with mu_star
        _, R_fixed_star = solve_fairdice_fixed(env, dD, beta=beta, mu=mu_star)
        base = nsw(R_fixed_star)
        rows.append(dict(beta=beta, sigma=0.0, nsw=base))


        for sigma in tqdm(sigmas, desc=f"beta={beta}"):
            print(f"    [sigma] beta={beta} sigma={float(sigma):.3g} averaging over {n_noise} noises")
            vals = []
            for _ in range(n_noise):
                mu_t = perturb_mu(mu_star, sigma=float(sigma), rng=rng)
                _, R_lin = solve_fairdice_fixed(env, dD, beta=beta, mu=mu_t)
                vals.append(nsw(R_lin))
            rows.append(dict(beta=beta, sigma=float(sigma), nsw=float(np.mean(vals))))

    df = pd.DataFrame(rows)
    df.to_csv("outputs/fig3_raw.csv", index=False)
    print("[results] Sample of aggregated dataframe:")
    print(df.head())
    print("[results] Saved outputs/fig3_raw.csv")

if __name__ == "__main__":
    main()
