import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Ensure src package (one level up) is importable when running as a script
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.random_momdp import RandomMOMDP
from src.dataset import make_behavior_policy_half_optimal, collect_offline_dataset
from src.occupancy_solvers import (
    empirical_d_dataset,
    solve_fairdice_nsw,
    solve_fairdice_fixed,
    build_transition_tensor,
    expected_obj_rewards,
)
from src.metrics import nsw_log_sum, perturb_mu


def parse_sigmas(arg: str) -> np.ndarray:
    arg = arg.strip()
    if arg.startswith("logspace"):
        parts = arg.split(":")
        if len(parts) != 4:
            raise ValueError("logspace format must be logspace:start:stop:num")
        start, stop, num = float(parts[1]), float(parts[2]), int(parts[3])
        return np.logspace(start, stop, num)
    vals = [float(x) for x in arg.split(",") if x.strip()]
    if not vals:
        raise ValueError("sigmas specification produced no values")
    return np.array(vals, dtype=float)


def parse_betas(arg: str) -> list:
    vals = [float(x) for x in arg.split(",") if x.strip()]
    if not vals:
        raise ValueError("betas specification produced no values")
    return vals


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def make_rng_seed(env_seed: int, beta: float, offset: int) -> int:
    base = 2026 + 1000 * env_seed + int(round(beta * 1_000_000))
    return base + offset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Figure 3 sweep with multi-seed aggregation")
    parser.add_argument("--seed_start", type=int, default=0)
    parser.add_argument("--seed_end", type=int, default=99)
    parser.add_argument("--n_noise", type=int, default=100)
    parser.add_argument("--sigmas", type=str, default="logspace:-3:0:13")
    parser.add_argument("--betas", type=str, default="0.001,0.01,0.1,1.0")
    parser.add_argument("--out_csv", type=str, default="outputs/fig3_raw.csv")
    parser.add_argument("--out_csv_agg", type=str, default="outputs/fig3_agg.csv")
    parser.add_argument("--rollout_seed_base", type=int, default=10000)
    parser.add_argument("--verbose", action="store_true", help="Enable verbose solver logging")
    return parser.parse_args()


def main():
    args = parse_args()
    betas = parse_betas(args.betas)
    sigmas = parse_sigmas(args.sigmas)
    out_csv = Path(args.out_csv)
    out_csv_agg = Path(args.out_csv_agg)
    ensure_parent(out_csv)
    ensure_parent(out_csv_agg)

    print(
        f"[args] env seeds {args.seed_start}..{args.seed_end}, betas={betas}, "
        f"sigmas={sigmas}, n_noise={args.n_noise}"
    )

    rows = []
    for env_seed in range(args.seed_start, args.seed_end + 1):
        print(f"[seed] Starting env_seed={env_seed}")
        env = RandomMOMDP(seed=env_seed)
        pi_b, _ = make_behavior_policy_half_optimal(env)
        dataset_seed = args.rollout_seed_base + env_seed
        dataset = collect_offline_dataset(env, pi_b, n_traj=100, seed=dataset_seed)
        dD = empirical_d_dataset(dataset, env.n_states, env.n_actions, env.gamma)
        P = build_transition_tensor(env)
        Rsa = expected_obj_rewards(env, P)

        for beta in betas:
            print(f"[seed/beta] env_seed={env_seed} beta={beta}")
            res = solve_fairdice_nsw(env, dD, beta=beta, P=P, Rsa=Rsa, verbose=args.verbose)
            d_star, R_star, mu_star = res
            if R_star is None or mu_star is None:
                print(
                    f"[warn] Skipping env_seed={env_seed}, beta={beta} due to NSW solver failure"
                )
                continue
            mu_sum = float(mu_star.sum())
            if mu_sum <= 0:
                print(
                    f"[warn] Invalid mu_star for env_seed={env_seed}, beta={beta}; skipping"
                )
                continue
            mu_star = mu_star / mu_sum

            fixed_res = solve_fairdice_fixed(env, dD, beta=beta, mu=mu_star, P=P, Rsa=Rsa)
            if fixed_res[1] is None:
                print(
                    f"[warn] Skipping baseline for env_seed={env_seed}, beta={beta}"
                )
                continue
            _, R_fixed_star = fixed_res
            base = nsw_log_sum(R_fixed_star)
            rows.append(
                dict(env_seed=env_seed, beta=beta, sigma=0.0, nsw_mean_seed=float(base))
            )

            for sigma_idx, sigma in enumerate(sigmas):
                print(
                    "    [sigma] env_seed=%s beta=%s sigma=%.4g averaging %d draws"
                    % (env_seed, beta, sigma, args.n_noise)
                )
                vals = []
                rng_seed = make_rng_seed(env_seed, beta, sigma_idx)
                rng = np.random.default_rng(rng_seed)
                for _ in range(args.n_noise):
                    mu_t = perturb_mu(mu_star, sigma=float(sigma), rng=rng)
                    solved = solve_fairdice_fixed(
                        env,
                        dD,
                        beta=beta,
                        mu=mu_t,
                        P=P,
                        Rsa=Rsa,
                    )
                    if solved[1] is None:
                        continue
                    _, R_lin = solved
                    vals.append(nsw_log_sum(R_lin))
                if not vals:
                    print(
                        f"[warn] No successful solves for env_seed={env_seed}, beta={beta}, sigma={sigma}"
                    )
                    continue
                rows.append(
                    dict(
                        env_seed=env_seed,
                        beta=beta,
                        sigma=float(sigma),
                        nsw_mean_seed=float(np.mean(vals)),
                    )
                )

    df = pd.DataFrame(rows)
    if df.empty:
        print("[error] No data collected; exiting without writing CSVs")
        return

    df.to_csv(out_csv, index=False)
    print(f"[results] Wrote per-seed results to {out_csv}")

    df_agg = (
        df.groupby(["beta", "sigma"], as_index=False)
        .agg(
            mean_nsw=("nsw_mean_seed", "mean"),
            std_nsw=("nsw_mean_seed", "std"),
            n_seeds=("nsw_mean_seed", "count"),
        )
    )
    df_agg["std_nsw"] = df_agg["std_nsw"].fillna(0.0)
    denom = df_agg["n_seeds"].astype(float).clip(lower=1.0)
    df_agg["sem_nsw"] = df_agg["std_nsw"] / np.sqrt(denom)

    df_agg.to_csv(out_csv_agg, index=False)
    print(f"[results] Wrote aggregated results to {out_csv_agg}")
    print("[results] Aggregated head:")
    print(df_agg.head())


if __name__ == "__main__":
    main()
