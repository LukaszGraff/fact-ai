import argparse
import json
from pathlib import Path

import numpy as np


def _load_json(path):
    if not path.exists():
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _fmt_beta(beta):
    return f"{beta:.12g}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dirs", nargs="*")
    parser.add_argument("--input_root", type=str, default=None)
    parser.add_argument("--pattern", type=str, default="**/mu_star.npz")
    parser.add_argument("--out_dir", required=True)
    args = parser.parse_args()

    input_dirs = []
    if args.input_dirs:
        input_dirs.extend([Path(p) for p in args.input_dirs])
    if args.input_root:
        root = Path(args.input_root)
        matches = sorted(root.glob(args.pattern))
        input_dirs.extend([m.parent for m in matches])
    if not input_dirs:
        raise ValueError("Provide --input_dirs and/or --input_root")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    betas = []
    seeds = []
    mu_entries = {}
    nsw_map = {}
    meta_ref = None

    for d in input_dirs:
        mu_path = d / "mu_star.npz"
        if not mu_path.exists():
            raise FileNotFoundError(f"Missing mu_star.npz in {d}")
        mu_data = np.load(mu_path, allow_pickle=True)
        mu_betas = mu_data["betas"].astype(np.float32).tolist()
        mu_seeds = mu_data["seeds"].astype(np.int32).tolist()
        mu_star = mu_data["mu_star"]

        for b_idx, beta in enumerate(mu_betas):
            beta_key = _fmt_beta(beta)
            if beta not in betas:
                betas.append(beta)
            for s_idx, seed in enumerate(mu_seeds):
                if seed not in seeds:
                    seeds.append(seed)
                key = (beta_key, seed)
                if key in mu_entries:
                    raise ValueError(f"Duplicate entry for beta={beta_key}, seed={seed} from {d}")
                mu_entries[key] = mu_star[b_idx, s_idx]

        nsw_path = d / "final_nsw_mu_star.json"
        nsw_json = _load_json(nsw_path)
        if nsw_json:
            for key, val in nsw_json.items():
                if val is None:
                    continue
                if key not in nsw_map:
                    nsw_map[key] = []
                nsw_map[key].append(float(val))

        meta_path = d / "mu_star_meta.json"
        meta_json = _load_json(meta_path)
        if meta_json:
            if meta_ref is None:
                meta_ref = meta_json
            else:
                # Ensure configs match across runs.
                for k, v in meta_ref.items():
                    if k in ("betas", "seeds"):
                        continue
                    if meta_json.get(k) != v:
                        raise ValueError(f"Mismatch in meta field '{k}' between runs.")

    betas = sorted(betas)
    seeds = sorted(seeds)
    reward_dim = len(next(iter(mu_entries.values())))
    mu_star_out = np.full((len(betas), len(seeds), reward_dim), np.nan, dtype=np.float32)

    for b_idx, beta in enumerate(betas):
        beta_key = _fmt_beta(beta)
        for s_idx, seed in enumerate(seeds):
            key = (beta_key, seed)
            if key in mu_entries:
                mu_star_out[b_idx, s_idx] = mu_entries[key]

    mu_path = out_dir / "mu_star.npz"
    np.savez(
        mu_path,
        betas=np.array(betas, dtype=np.float32),
        seeds=np.array(seeds, dtype=np.int32),
        mu_star=mu_star_out,
    )

    meta_out = dict(meta_ref) if meta_ref else {}
    meta_out["betas"] = betas
    meta_out["seeds"] = seeds
    meta_path = out_dir / "mu_star_meta.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta_out, f, indent=2)

    final_nsw_avg = {k: float(np.mean(v)) if v else None for k, v in nsw_map.items()}
    best_path = out_dir / "final_nsw_mu_star.json"
    with open(best_path, "w", encoding="utf-8") as f:
        json.dump(final_nsw_avg, f, indent=2)

    print(f"Saved mu_star: {mu_path}")


if __name__ == "__main__":
    main()
