import argparse
import csv
import os
import pickle
from typing import Dict, List, Optional, Tuple

import numpy as np


def _normalize_mix_tag(mix_tag: str) -> str:
    # Some runs use a double prefix like "mix_mix_0p1_0p1_0p8".
    while mix_tag.startswith("mix_mix_"):
        mix_tag = mix_tag[len("mix_") :]
    return mix_tag


def _parse_mix_tag(mix_tag: str) -> Optional[Tuple[float, float, float]]:
    mix_tag = _normalize_mix_tag(mix_tag)
    if not mix_tag.startswith("mix_"):
        return None
    parts = mix_tag[len("mix_") :].split("_")
    if len(parts) != 3:
        return None
    try:
        vals = [float(p.replace("p", ".")) for p in parts]
    except ValueError:
        return None
    return tuple(vals)


def _goal_reach_stats(pkl_path: str) -> Dict[str, float]:
    with open(pkl_path, "rb") as f:
        trajectories = pickle.load(f)

    reward_dim = None
    counts = None
    no_goal = 0
    for traj in trajectories:
        raw_rewards = np.asarray(traj["raw_rewards"])
        if raw_rewards.size == 0:
            no_goal += 1
            continue
        if reward_dim is None:
            reward_dim = raw_rewards.shape[1]
            counts = np.zeros(reward_dim, dtype=int)
        nz = np.where((raw_rewards != 0).any(axis=1))[0]
        if nz.size == 0:
            no_goal += 1
            continue
        last = raw_rewards[nz[-1]]
        max_val = last.max()
        if max_val <= 0:
            no_goal += 1
            continue
        winners = np.where(last == max_val)[0]
        counts[winners] += 1

    total = len(trajectories)
    stats = {
        "traj_count": total,
        "no_goal_frac": (no_goal / total) if total else 0.0,
    }
    if counts is not None:
        for i, c in enumerate(counts):
            stats[f"reach_obj{i}_frac"] = c / total if total else 0.0
    return stats


def _load_csv(path: str) -> Tuple[List[str], List[Dict[str, str]]]:
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        return reader.fieldnames or [], rows


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Merge goal-mix dataset reach stats with NSW CSV results."
    )
    parser.add_argument("--csv", required=True, help="Input CSV from evaluate_sweep_welfare.py.")
    parser.add_argument(
        "--data_dir",
        default="./data/MO-FourRoom-v2",
        help="Directory containing goal-mix datasets.",
    )
    parser.add_argument("--env_name", default="MO-FourRoom-v2", help="Environment name.")
    parser.add_argument("--quality", default="amateur", help="Dataset quality tag.")
    parser.add_argument(
        "--output_csv",
        default="",
        help="Output CSV path (defaults to input CSV with _with_reach suffix).",
    )
    args = parser.parse_args()

    fieldnames, rows = _load_csv(args.csv)
    if "mix_tag" not in fieldnames:
        raise SystemExit("CSV missing mix_tag column. Re-run evaluate_sweep_welfare.py with --group_by mix.")

    out_rows = []
    missing = []
    for row in rows:
        mix_tag = _normalize_mix_tag(row["mix_tag"])
        data_path = os.path.join(
            args.data_dir,
            f"{args.env_name}_50000_{args.quality}_{mix_tag}.pkl",
        )
        if not os.path.exists(data_path):
            missing.append(mix_tag)
            continue

        stats = _goal_reach_stats(data_path)
        mix_vals = _parse_mix_tag(mix_tag)
        if mix_vals is not None:
            row["mix_target_0"] = f"{mix_vals[0]:.6f}"
            row["mix_target_1"] = f"{mix_vals[1]:.6f}"
            row["mix_target_2"] = f"{mix_vals[2]:.6f}"

        for k, v in stats.items():
            row[k] = f"{v:.6f}" if isinstance(v, float) else str(v)
        out_rows.append(row)

    output_csv = args.output_csv
    if not output_csv:
        base, ext = os.path.splitext(args.csv)
        output_csv = f"{base}_with_reach{ext or '.csv'}"

    out_fieldnames = list(fieldnames)
    for extra in ("mix_target_0", "mix_target_1", "mix_target_2", "traj_count", "no_goal_frac"):
        if extra not in out_fieldnames:
            out_fieldnames.append(extra)
    # add any reach_obj columns dynamically
    for row in out_rows:
        for k in row.keys():
            if k.startswith("reach_obj") and k not in out_fieldnames:
                out_fieldnames.append(k)

    with open(output_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=out_fieldnames)
        writer.writeheader()
        writer.writerows(out_rows)

    print(f"Wrote merged CSV to {output_csv}")
    if missing:
        print(f"Missing datasets for mix tags: {sorted(set(missing))}")


if __name__ == "__main__":
    main()
