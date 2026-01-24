import argparse
import json
from pathlib import Path

import numpy as np


def _load_json(path):
    if not path.exists():
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _fmt_float(val):
    return f"{val:.12g}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dirs", nargs="*")
    parser.add_argument("--input_root", type=str, default=None)
    parser.add_argument("--pattern", type=str, default="**/fig3_random_momdp_results.npz")
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
    sigmas = []
    seeds = []
    nsw_entries = {}
    meta_ref = None
    nsw_map = {}

    for d in input_dirs:
        results_path = d / "fig3_random_momdp_results.npz"
        if not results_path.exists():
            raise FileNotFoundError(f"Missing fig3_random_momdp_results.npz in {d}")
        data = np.load(results_path, allow_pickle=True)
        r_betas = data["betas"].astype(np.float32).tolist()
        r_sigmas = data["sigmas"].astype(np.float32).tolist()
        r_seeds = data["seeds"].astype(np.int32).tolist()
        nsw = data["nsw"]

        for b_idx, beta in enumerate(r_betas):
            if beta not in betas:
                betas.append(beta)
            for s_idx, sigma in enumerate(r_sigmas):
                if sigma not in sigmas:
                    sigmas.append(sigma)
            for seed in r_seeds:
                if seed not in seeds:
                    seeds.append(seed)
                for s_idx, sigma in enumerate(r_sigmas):
                    key = (_fmt_float(beta), _fmt_float(sigma), seed)
                    if key in nsw_entries:
                        raise ValueError(
                            f"Duplicate entry for beta={beta}, sigma={sigma}, seed={seed} from {d}"
                        )
                    nsw_entries[key] = nsw[b_idx, s_idx, r_seeds.index(seed)]

        nsw_path = d / "final_nsw_fixed.json"
        nsw_json = _load_json(nsw_path)
        if nsw_json:
            for key, val in nsw_json.items():
                if val is None:
                    continue
                if key not in nsw_map:
                    nsw_map[key] = []
                nsw_map[key].append(float(val))

        meta_path = d / "fig3_random_momdp_meta.json"
        meta_json = _load_json(meta_path)
        if meta_json:
            if meta_ref is None:
                meta_ref = meta_json
            else:
                for k, v in meta_ref.items():
                    if k in ("betas", "sigmas", "seeds"):
                        continue
                    if meta_json.get(k) != v:
                        raise ValueError(f"Mismatch in meta field '{k}' between runs.")

    betas = sorted(betas)
    sigmas = sorted(sigmas)
    seeds = sorted(seeds)
    nsw_out = np.full((len(betas), len(sigmas), len(seeds)), np.nan, dtype=np.float32)

    for b_idx, beta in enumerate(betas):
        for s_idx, sigma in enumerate(sigmas):
            for seed_idx, seed in enumerate(seeds):
                key = (_fmt_float(beta), _fmt_float(sigma), seed)
                if key in nsw_entries:
                    nsw_out[b_idx, s_idx, seed_idx] = nsw_entries[key]

    results_path = out_dir / "fig3_random_momdp_results.npz"
    np.savez(
        results_path,
        betas=np.array(betas, dtype=np.float32),
        sigmas=np.array(sigmas, dtype=np.float32),
        seeds=np.array(seeds, dtype=np.int32),
        nsw=nsw_out,
    )

    meta_out = dict(meta_ref) if meta_ref else {}
    meta_out["betas"] = betas
    meta_out["sigmas"] = sigmas
    meta_out["seeds"] = seeds
    meta_path = out_dir / "fig3_random_momdp_meta.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta_out, f, indent=2)

    final_nsw_avg = {k: float(np.mean(v)) if v else None for k, v in nsw_map.items()}
    best_path = out_dir / "final_nsw_fixed.json"
    with open(best_path, "w", encoding="utf-8") as f:
        json.dump(final_nsw_avg, f, indent=2)

    print(f"Saved results: {results_path}")


if __name__ == "__main__":
    main()
