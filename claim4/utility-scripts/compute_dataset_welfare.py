import argparse
import pickle

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute NSW/USW/Jain on a FourRoom dataset.")
    parser.add_argument("--data_path", required=True, help="Path to dataset .pkl")
    parser.add_argument("--eps", type=float, default=0.001, help="Epsilon for log in NSW.")
    args = parser.parse_args()

    with open(args.data_path, "rb") as f:
        trajs = pickle.load(f)

    returns = []
    for traj in trajs:
        rewards = traj["raw_rewards"]
        returns.append(np.sum(rewards, axis=0))

    returns = np.asarray(returns)
    mean_returns = returns.mean(axis=0)

    nsw = float(np.sum(np.log(mean_returns + args.eps)))
    usw = float(np.mean(np.sum(returns, axis=1)))
    numerator = float(np.sum(mean_returns) ** 2)
    denom = float(len(mean_returns) * np.sum(mean_returns ** 2) + 1e-8)
    jain = numerator / denom

    print(f"dataset: {args.data_path}")
    print(f"num_trajectories: {len(returns)}")
    print(f"mean_returns: {mean_returns}")
    print(f"NSW_of_mean: {nsw}")
    print(f"USW_mean_sum: {usw}")
    print(f"Jain_index: {jain}")


if __name__ == "__main__":
    main()
