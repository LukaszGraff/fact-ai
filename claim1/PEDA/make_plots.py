import argparse
import glob
import os
import re
import pickle
import math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from modt.utils import undominated_indices


# Fairness metrics

def nsw(returns, eps=1e-8):
    r = np.asarray(returns, dtype=float)
    invalid = ~np.isfinite(r) | ((r + eps) <= 0)
    r = np.where(invalid, np.nan, r)
    return np.nansum(np.log(r + eps), axis=-1)

def utilitarian_sum(returns):
    r = np.asarray(returns, dtype=float)
    return np.nansum(r, axis=-1)

def maxmin_utility(returns):
    returns = np.asarray(returns, dtype=float)
    return np.min(returns, axis=-1)

def jain_index(returns, eps=1e-8):
    r = np.asarray(returns, dtype=float)
    s = np.sum(r, axis=-1)
    s2 = np.sum(r ** 2, axis=-1)
    d = r.shape[-1]
    return (s ** 2) / (d * s2 + eps)

def gini_coefficient(returns, eps=1e-8):
    r = np.asarray(returns, dtype=float)
    sorted_r = np.sort(r, axis=-1)
    n = r.shape[-1]
    if n == 0:
        return np.zeros_like(sorted_r[..., 0])
    idx = np.arange(1, n + 1)
    num = np.sum(idx * sorted_r, axis=-1)
    den = np.sum(sorted_r, axis=-1) + eps
    g = (2.0 * num) / (n * den) - (n + 1.0) / n
    return g


# Plot styling

COLOR_MAP = {
    "bc":  "#1f77b4",  # blue
    "dt":  "#2ca02c",  # green
    "rvs": "#ff7f0e",  # orange
}
MARKER_MAP = {"bc": "o", "dt": "s", "rvs": "^"}
LABEL_MAP  = {"bc": "BC(P)", "dt": "MODT(P)", "rvs": "MORvS(P)"}

FAIRDICE_COLOR = "#d62728"
FAIRDICE_LS = "--"

RAW3D_VIEW = dict(elev=18, azim=-65)
SIMPLEX_VIEW = dict(elev=14, azim=55)


# Helpers: load rollouts

def _latest_rollout_pkl(logs_dir: str):
    cands = glob.glob(os.path.join(logs_dir, "step=*rollout.pkl"))
    if not cands:
        return None
    def step_num(p):
        m = re.search(r"step=(\d+)_rollout\.pkl", os.path.basename(p))
        return int(m.group(1)) if m else -1
    return max(cands, key=step_num)

def _load_rollout_dict(pkl_path: str):
    with open(pkl_path, "rb") as f:
        return pickle.load(f)

def _prefs_from_rollout(rollout_logs):
    prefs = np.asarray(rollout_logs["target_prefs"], dtype=float)
    s = np.sum(prefs, axis=1, keepdims=True)
    s = np.where(s == 0, 1.0, s)
    return prefs / s

def group_model_results_from_rollouts(runs_root, env, dataset, models=("dt", "bc", "rvs")):
    """
    Expects: runs_root/<model>/<env>/<dataset>/.../logs/step=*_rollout.pkl

    Returns: dict model -> list[(run_logs_dir, dict_like)]
    dict_like has:
      - pref_set:  (num_prefs, n_obj)
      - mean_raw:  (num_prefs, n_obj)  (raw objective sums; rollout_original_raw_r)
      - mean_norm: (num_prefs, n_obj)  (normalized objective sums; rollout_unweighted_raw_r)
    """
    out = {}
    for m in models:
        pattern = os.path.join(runs_root, m, env, dataset, "**", "logs")
        logs_dirs = glob.glob(pattern, recursive=True)
        for logsdir in logs_dirs:
            pkl = _latest_rollout_pkl(logsdir)
            if pkl is None:
                continue
            roll = _load_rollout_dict(pkl)
            pref_set = _prefs_from_rollout(roll)

            mean_raw = np.asarray(roll.get("rollout_original_raw_r"), dtype=float)
            mean_norm = np.asarray(roll.get("rollout_unweighted_raw_r", mean_raw), dtype=float)

            out.setdefault(m, []).append(
                (
                    logsdir,
                    {
                        "pref_set": pref_set,
                        "mean_raw": mean_raw,
                        "mean_norm": mean_norm,
                        "model_type": m,
                    },
                )
            )
    return out


def load_fairdice_from_dir(fairdice_dir, env, dataset):
    """Load FairDICE aggregation produced by prepare_fd.py."""
    if fairdice_dir is None:
        return None

    pat = os.path.join(fairdice_dir, f"fairdice_{env}_{dataset}_*.npz")
    cands = sorted(glob.glob(pat))
    if not cands:
        return None

    fname = cands[0]
    fd = np.load(fname, allow_pickle=True)

    out = {"_file": fname}

    if "returns_raw" in fd and "returns_norm" in fd:
        out["returns_raw"] = np.asarray(fd["returns_raw"], dtype=float)
        out["returns_norm"] = np.asarray(fd["returns_norm"], dtype=float)
        out["mean_raw"] = np.asarray(fd.get("mean_raw", out["returns_raw"].mean(axis=(0, 1))), dtype=float)
        out["mean_norm"] = np.asarray(fd.get("mean_norm", out["returns_norm"].mean(axis=(0, 1))), dtype=float)
        return out

    return None


def _reduce_metric_over_seeds_prefs(arr_seeds_prefs_obj, metric_fn):
    """arr shape is either (seeds,prefs,obj) or (seeds,prefs,eps,obj)."""
    vals = metric_fn(arr_seeds_prefs_obj)
    if vals.ndim == 3:
        vals = np.nanmean(vals, axis=-1)  # mean over episodes
    return np.nanmean(vals, axis=0)


def _reduce_fairdice_metric(fd_dict, metric_fn, which="norm"):
    """Return a scalar metric for the FairDICE policy (macro avg over episodes + runs if available)."""
    if fd_dict is None:
        return None

    if which == "raw":
        if "returns_raw" in fd_dict:
            vals = metric_fn(fd_dict["returns_raw"])  # (runs,eps)
            return float(np.nanmean(vals))
        if "mean_raw" in fd_dict:
            return float(np.nanmean(metric_fn(fd_dict["mean_raw"])))
        return None

    if "returns_norm" in fd_dict:
        vals = metric_fn(fd_dict["returns_norm"])  # (runs,eps)
        return float(np.nanmean(vals))
    if "mean_norm" in fd_dict:
        return float(np.nanmean(metric_fn(fd_dict["mean_norm"])))
    if "mean_raw" in fd_dict:
        return float(np.nanmean(metric_fn(fd_dict["mean_raw"])))
    return None


# 3D simplex helpers

def _simplex_xy_from_w(w):
    w = np.asarray(w, dtype=float)
    v1 = np.array([1.0, 0.0])
    v2 = np.array([0.5, math.sqrt(3) / 2.0])
    v3 = np.array([0.0, 0.0])
    return w[..., 0:1] * v1 + w[..., 1:2] * v2 + w[..., 2:3] * v3  # (...,2)


def _try_add_3d_pareto_surface(ax, pts, color, alpha=0.12):
    pts = np.asarray(pts, dtype=float)
    if pts.shape[0] < 4:
        return

    pf_idx = undominated_indices(pts, tolerance=0.0)
    front = pts[pf_idx]
    if front.shape[0] < 4:
        return

    try:
        from scipy.spatial import ConvexHull
        hull = ConvexHull(front)
        tris = hull.simplices
        polys = [front[t] for t in tris]
        coll = Poly3DCollection(polys, facecolors=color, edgecolors="none", alpha=alpha)
        ax.add_collection3d(coll)
    except Exception:
        return


def _set_equal_3d_aspect(ax, xs, ys, zs):
    xs = np.asarray(xs); ys = np.asarray(ys); zs = np.asarray(zs)
    xmid = (xs.max() + xs.min()) / 2.0
    ymid = (ys.max() + ys.min()) / 2.0
    zmid = (zs.max() + zs.min()) / 2.0

    r = max(xs.max()-xs.min(), ys.max()-ys.min(), zs.max()-zs.min()) / 2.0
    ax.set_xlim(xmid - r, xmid + r)
    ax.set_ylim(ymid - r, ymid + r)
    ax.set_zlim(zmid - r, zmid + r)

    try:
        ax.set_box_aspect((1, 1, 1))
    except Exception:
        pass


def _plot_hopper3obj_raw_returns(ax, model_to_results, fairdice_dict, title):
    all_pts = []
    per_method_pts = {}

    for m, res_list in model_to_results.items():
        arr = np.stack([d["mean_raw"] for _, d in res_list], axis=0)  # (seeds,prefs,3)
        pts = np.mean(arr, axis=0)  # [speed, height, energy]
        per_method_pts[m] = pts
        all_pts.append(pts)

    if not all_pts:
        ax.set_visible(False)
        return

    all_pts = np.concatenate(all_pts, axis=0)

    for m, pts in per_method_pts.items():
        color = COLOR_MAP.get(m, None)
        marker = MARKER_MAP.get(m, "o")
        ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], color=color, marker=marker, alpha=0.55, s=18)
        _try_add_3d_pareto_surface(ax, pts, color=color, alpha=0.10)

    if fairdice_dict is not None:
        fd_mean_raw = fairdice_dict.get("mean_raw")
        if fd_mean_raw is None and "returns_raw" in fairdice_dict:
            fd_mean_raw = np.nanmean(fairdice_dict["returns_raw"], axis=(0, 1))
        if fd_mean_raw is not None:
            ax.scatter(
                fd_mean_raw[0], fd_mean_raw[1], fd_mean_raw[2],
                s=70, marker="o", color=FAIRDICE_COLOR, edgecolors="black",
                linewidths=1.0, zorder=50
            )
            txt = f"({fd_mean_raw[0]:.1f}, {fd_mean_raw[1]:.1f}, {fd_mean_raw[2]:.1f})"
            ax.text2D(
                0.02, 0.90, txt, transform=ax.transAxes,
                bbox=dict(boxstyle="round", facecolor="white", edgecolor=FAIRDICE_COLOR, linewidth=1.2),
                color=FAIRDICE_COLOR, fontsize=9
            )

    ax.set_title(title)
    ax.set_xlabel("Speed")
    ax.set_ylabel("Height")
    ax.set_zlabel("Energy")

    _set_equal_3d_aspect(ax, all_pts[:, 0], all_pts[:, 1], all_pts[:, 2])
    ax.view_init(**RAW3D_VIEW)


def _plot_hopper3obj_simplex_nsw(ax, model_to_results, fairdice_dict, title):
    any_res = next(iter(model_to_results.values()))[0][1]
    prefs = np.asarray(any_res["pref_set"], dtype=float)
    s = np.sum(prefs, axis=1, keepdims=True)
    s = np.where(s == 0, 1.0, s)
    prefs = prefs / s

    xy = _simplex_xy_from_w(prefs)
    x = xy[:, 0]
    y = xy[:, 1]

    all_z = []
    z_by_model = {}
    for m, res_list in model_to_results.items():
        arr = np.stack([d["mean_norm"] for _, d in res_list], axis=0)
        z = np.nanmean(nsw(arr), axis=0)
        z_by_model[m] = z
        all_z.append(z)

    z_all = np.concatenate([np.asarray(z) for z in all_z], axis=0)
    zmin = float(np.nanmin(z_all))
    zmax = float(np.nanmax(z_all))
    if not (np.isfinite(zmin) and np.isfinite(zmax) and zmax > zmin):
        zmin, zmax = 0.0, 1.0

    tri_xy = np.array([
        [0.0, 0.0],
        [1.0, 0.0],
        [0.5, math.sqrt(3)/2.0],
    ])
    base = np.c_[tri_xy, np.full(3, zmin)]
    base_poly = Poly3DCollection([base], facecolors="lightgray", alpha=0.20,
                                 edgecolors="gray", linewidths=0.6)
    ax.add_collection3d(base_poly)

    for m, z in z_by_model.items():
        ax.scatter(
            x, y, z,
            color=COLOR_MAP.get(m, None),
            marker=MARKER_MAP.get(m, "o"),
            alpha=0.75,
            s=18,
        )

    fd_mean = _reduce_fairdice_metric(fairdice_dict, nsw, which="norm")
    if fd_mean is not None and np.isfinite(fd_mean):
        plane = np.c_[tri_xy, np.full(3, fd_mean)]
        plane_poly = Poly3DCollection([plane], facecolors=FAIRDICE_COLOR, alpha=0.18, edgecolors="none")
        ax.add_collection3d(plane_poly)

        ax.text2D(
            0.82, 0.88, f"{fd_mean:.2f}", transform=ax.transAxes,
            bbox=dict(boxstyle="round", facecolor="white", edgecolor=FAIRDICE_COLOR, linewidth=1.2),
            color=FAIRDICE_COLOR, fontsize=9
        )

    eps_txt = 0.02 * (zmax - zmin)
    ax.text(1.0, 0.0, zmin + eps_txt, "w1 = 1", fontsize=9)
    ax.text(0.5, math.sqrt(3)/2.0, zmin + eps_txt, "w2 = 1", fontsize=9)
    ax.text(0.0, 0.0, zmin + eps_txt, "w3 = 1", fontsize=9)

    ax.set_title(title)
    ax.set_xlabel("Preference Weight Simplex")
    ax.set_ylabel("")
    ax.set_zlabel("NSW Score")

    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, math.sqrt(3)/2.0 + 0.05)
    ax.set_zlim(zmin - 0.2, zmax + 0.2)

    ax.view_init(**SIMPLEX_VIEW)
    ax.set_xticks([])
    ax.set_yticks([])

    try:
        ax.set_box_aspect((1, 1, 0.8))
    except Exception:
        pass


def build_hopper3obj_figure(runs_root, fairdice_dir, out_dir, env="MO-Hopper-v3"):
    os.makedirs(out_dir, exist_ok=True)
    datasets = [("expert_uniform", "Expert"), ("amateur_uniform", "Amateur")]

    fig = plt.figure(figsize=(15.5, 4.8))
    axes = [fig.add_subplot(1, 4, i + 1, projection="3d") for i in range(4)]

    for j, (ds, ds_title) in enumerate(datasets):
        model_to_results = group_model_results_from_rollouts(runs_root, env, ds)
        fairdice_dict = load_fairdice_from_dir(fairdice_dir, env, ds)
        if not model_to_results:
            axes[j].set_visible(False)
            continue
        _plot_hopper3obj_raw_returns(axes[j], model_to_results, fairdice_dict, title=ds_title)

    for j, (ds, ds_title) in enumerate(datasets):
        model_to_results = group_model_results_from_rollouts(runs_root, env, ds)
        fairdice_dict = load_fairdice_from_dir(fairdice_dir, env, ds)
        if not model_to_results:
            axes[2 + j].set_visible(False)
            continue
        _plot_hopper3obj_simplex_nsw(axes[2 + j], model_to_results, fairdice_dict, title=ds_title)

    fig.suptitle("MO-Hopper-3obj", y=0.98)

    handles = [
        Line2D([0], [0], marker=MARKER_MAP["bc"], linestyle="", color=COLOR_MAP["bc"], label="BC(P)"),
        Line2D([0], [0], marker=MARKER_MAP["dt"], linestyle="", color=COLOR_MAP["dt"], label="MODT(P)"),
        Line2D([0], [0], marker=MARKER_MAP["rvs"], linestyle="", color=COLOR_MAP["rvs"], label="MORvS(P)"),
        Line2D([0], [0], marker="o", linestyle="", color=FAIRDICE_COLOR,
               markeredgecolor="black", label="FairDICE"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=4, bbox_to_anchor=(0.5, -0.02))
    fig.subplots_adjust(left=0.02, right=0.99, top=0.92, bottom=0.18, wspace=0.3)

    out_path = os.path.join(out_dir, "hopper3obj_fig.png")
    plt.savefig(out_path, dpi=200, bbox_inches="tight", pad_inches=0.30)
    plt.close()
    print(f"Saved {out_path}")


# Preference axis helpers

def _pref_x_and_sort(pref_set):
    w1 = np.asarray(pref_set)[:, 0]
    idx = np.argsort(w1)[::-1]  # 1 -> 0
    x = np.arange(len(w1))
    return x, idx

def _paper_ticks(n):
    desired = [0, 10, 20, 30]
    return [t for t in desired if t < n]


# Metric panels (2-objective envs)

def plot_metric_panel_cell(ax, model_to_results, fairdice_dict, metric_fn):
    if not model_to_results:
        ax.set_visible(False)
        return

    any_res = next(iter(model_to_results.values()))[0][1]
    pref_set = any_res["pref_set"]
    x, idx = _pref_x_and_sort(pref_set)
    n = len(x)

    for m, res_list in model_to_results.items():
        arr = np.stack([d["mean_norm"] for _, d in res_list], axis=0)
        mean_m = _reduce_metric_over_seeds_prefs(arr, metric_fn)
        color = COLOR_MAP.get(m, None)
        marker = MARKER_MAP.get(m, "o")

        ax.plot(x, mean_m[idx], color=color, linewidth=2, marker=marker, markersize=3)

    fd_mean = _reduce_fairdice_metric(fairdice_dict, metric_fn, which="norm")
    if fd_mean is not None:
        ax.axhline(fd_mean, linestyle=FAIRDICE_LS, color=FAIRDICE_COLOR, linewidth=2)

    ticks = _paper_ticks(n)
    if ticks:
        ax.set_xticks(ticks)


def build_full_metric_panel(runs_root, fairdice_dir, out_dir, envs, datasets, metric_fn, metric_ylabel, fname_tag):
    os.makedirs(out_dir, exist_ok=True)

    fig, axes = plt.subplots(nrows=2, ncols=len(envs), figsize=(14, 5), sharex=True)

    for col, env in enumerate(envs):
        for row, dataset in enumerate(datasets):
            ax = axes[row, col]
            model_to_results = group_model_results_from_rollouts(runs_root, env, dataset)
            fairdice_dict = load_fairdice_from_dir(fairdice_dir, env, dataset)

            plot_metric_panel_cell(ax, model_to_results, fairdice_dict, metric_fn)

            if row == 0:
                ax.set_title(env.replace("-v2", "").replace("-v3", ""))

            if col == 0:
                ax.set_ylabel(metric_ylabel + "\n" + ("Expert" if dataset == "expert_uniform" else "Amateur"))
            else:
                ax.set_ylabel("")

    fig.text(0.5, 0.06, "Preference Weight ([1.0, 0.0] \u2192 [0.0, 1.0])", ha="center")

    handles = [
        Line2D([0], [0], color=COLOR_MAP["bc"], marker=MARKER_MAP["bc"], linewidth=2, label="BC(P)"),
        Line2D([0], [0], color=COLOR_MAP["dt"], marker=MARKER_MAP["dt"], linewidth=2, label="MODT(P)"),
        Line2D([0], [0], color=COLOR_MAP["rvs"], marker=MARKER_MAP["rvs"], linewidth=2, label="MORvS(P)"),
        Line2D([0], [0], color=FAIRDICE_COLOR, linestyle=FAIRDICE_LS, linewidth=2, label="FairDICE"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=4, bbox_to_anchor=(0.5, -0.05))

    fig.subplots_adjust(bottom=0.18, left=0.06, right=0.99, top=0.90, wspace=0.3)

    out_path = os.path.join(out_dir, f"{fname_tag}_panel_all_envs.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight", pad_inches=0.05)
    plt.close()
    print(f"Saved {fname_tag} panel to {out_path}")


def plot_metric_curve_single_ax(ax, model_to_results, fairdice_dict, metric_fn):
    if not model_to_results:
        ax.set_visible(False)
        return

    any_res = next(iter(model_to_results.values()))[0][1]
    pref_set = any_res["pref_set"]
    x, idx = _pref_x_and_sort(pref_set)
    n = len(x)

    for m, res_list in model_to_results.items():
        arr = np.stack([d["mean_norm"] for _, d in res_list], axis=0)
        vals = metric_fn(arr)
        mean_curve = np.nanmean(vals, axis=0)

        ax.plot(
            x, mean_curve[idx],
            color=COLOR_MAP.get(m, None),
            linewidth=2,
            marker=MARKER_MAP.get(m, "o"),
            markersize=3,
        )

    fd_mean = _reduce_fairdice_metric(fairdice_dict, metric_fn, which="norm")
    if fd_mean is not None:
        ax.axhline(fd_mean, linestyle=FAIRDICE_LS, color=FAIRDICE_COLOR, linewidth=2)

    ticks = _paper_ticks(n)
    if ticks:
        ax.set_xticks(ticks)


def build_2x5_env_metric_grid(runs_root, fairdice_dir, out_dir, dataset="amateur_uniform"):
    os.makedirs(out_dir, exist_ok=True)

    envs = ["MO-Hopper-v2", "MO-Swimmer-v2"]
    metrics = [
        ("NSW", nsw),
        ("USW", utilitarian_sum),
        ("ESW", maxmin_utility),
        ("Jain's Index", jain_index),
        ("Gini coefficient", gini_coefficient),
    ]

    fig, axes = plt.subplots(
        nrows=len(envs), ncols=len(metrics),
        figsize=(17.0, 5.5),
        sharex=True
    )

    for r, env in enumerate(envs):
        for c, (mname, mfn) in enumerate(metrics):
            ax = axes[r, c]
            model_to_results = group_model_results_from_rollouts(runs_root, env, dataset)
            fairdice_dict = load_fairdice_from_dir(fairdice_dir, env, dataset)
            plot_metric_curve_single_ax(ax, model_to_results, fairdice_dict, mfn)

            if r == 0:
                ax.set_title(mname)

            if c == 0:
                ax.set_ylabel(env.replace("-v2", "").replace("-v3", ""))
            else:
                ax.set_ylabel("")

    fig.text(0.5, 0.06, "Preference Weight ([1.0, 0.0] \u2192 [0.0, 1.0])", ha="center")

    handles = [
        Line2D([0], [0], color=COLOR_MAP["bc"], marker=MARKER_MAP["bc"], linewidth=2, label="BC(P)"),
        Line2D([0], [0], color=COLOR_MAP["dt"], marker=MARKER_MAP["dt"], linewidth=2, label="MODT(P)"),
        Line2D([0], [0], color=COLOR_MAP["rvs"], marker=MARKER_MAP["rvs"], linewidth=2, label="MORvS(P)"),
        Line2D([0], [0], color=FAIRDICE_COLOR, linestyle=FAIRDICE_LS, linewidth=2, label="FairDICE"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=4, bbox_to_anchor=(0.5, -0.02))

    fig.subplots_adjust(bottom=0.18, left=0.07, right=0.995, top=0.90, wspace=0.25, hspace=0.25)

    out_path = os.path.join(out_dir, f"grid2x5_hopper_swimmer_{dataset}.png")
    plt.savefig(out_path, dpi=200, bbox_inches="tight", pad_inches=0.05)
    plt.close()
    print(f"Saved 2x5 grid to {out_path}")


# Pareto panels (2-objective envs) — RAW returns

PARETO_XLABELS = {
    "MO-Hopper-v2": "Speed vs. Height",
    "MO-Walker2d-v2": "Speed vs. Energy",
    "MO-Ant-v2": "Horizontal vs. Vertical Speed",
    "MO-HalfCheetah-v2": "Speed vs. Energy",
    "MO-Swimmer-v2": "Speed vs. Energy",
}

def plot_pareto_panel_cell(ax, model_to_results, fairdice_dict, env):
    all_points = []
    for _, res_list in model_to_results.items():
        for _, d in res_list:
            all_points.append(d["mean_raw"])
    if not all_points:
        ax.set_visible(False)
        return

    all_points = np.concatenate(all_points, axis=0)
    if all_points.shape[1] != 2:
        ax.set_visible(False)
        return

    x_min = min(0.0, float(all_points[:, 0].min()))
    y_min = min(0.0, float(all_points[:, 1].min()))

    for m, res_list in model_to_results.items():
        pts = np.concatenate([d["mean_raw"] for _, d in res_list], axis=0)
        color = COLOR_MAP.get(m, None)
        marker = MARKER_MAP.get(m, "o")

        pf_idx = undominated_indices(pts, tolerance=0.0)
        front = pts[pf_idx]
        front = front[np.argsort(front[:, 0])]

        if front.shape[0] > 0:
            x_f = front[:, 0]
            y_f = front[:, 1]
            x_poly = np.concatenate([[x_min], x_f])
            y_poly = np.concatenate([[y_f[0]], y_f])

            ax.fill_between(x_poly, y_poly, y_min, color=color, alpha=0.08)
            ax.plot(front[:, 0], front[:, 1], linewidth=2, color=color)

        ax.scatter(pts[:, 0], pts[:, 1], alpha=0.25, color=color, marker=marker)

    if fairdice_dict is not None:
        fd = fairdice_dict.get("mean_raw")
        if fd is None and "returns_raw" in fairdice_dict:
            fd = np.nanmean(fairdice_dict["returns_raw"], axis=(0, 1))
        if fd is not None:
            ax.scatter(
                fd[0], fd[1], marker="o", s=60, color=FAIRDICE_COLOR,
                edgecolors="black", linewidths=1.0, zorder=10
            )


def build_full_pareto_panel(runs_root, fairdice_dir, out_dir, envs, datasets):
    os.makedirs(out_dir, exist_ok=True)

    fig, axes = plt.subplots(nrows=2, ncols=len(envs), figsize=(14, 6), sharex=False)

    for col, env in enumerate(envs):
        for row, dataset in enumerate(datasets):
            ax = axes[row, col]
            model_to_results = group_model_results_from_rollouts(runs_root, env, dataset)
            fairdice_dict = load_fairdice_from_dir(fairdice_dir, env, dataset)
            plot_pareto_panel_cell(ax, model_to_results, fairdice_dict, env)

            if row == 0:
                ax.set_title(env.replace("-v2", "").replace("-v3", ""))

            if col == 0:
                ax.set_ylabel("Expert" if dataset == "expert_uniform" else "Amateur")
            else:
                ax.set_ylabel("")

            if row == 1:
                ax.set_xlabel(PARETO_XLABELS.get(env, "Obj0 vs Obj1"))

    handles = [
        Line2D([0], [0], marker=MARKER_MAP["bc"], linestyle="", color=COLOR_MAP["bc"], label="BC(P)"),
        Line2D([0], [0], marker=MARKER_MAP["dt"], linestyle="", color=COLOR_MAP["dt"], label="MODT(P)"),
        Line2D([0], [0], marker=MARKER_MAP["rvs"], linestyle="", color=COLOR_MAP["rvs"], label="MORvS(P)"),
        Line2D([0], [0], marker="o", linestyle="", color=FAIRDICE_COLOR, markeredgecolor="black", label="FairDICE"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=4, bbox_to_anchor=(0.5, 0.0))

    fig.subplots_adjust(bottom=0.16, left=0.06, right=0.99, top=0.92, wspace=0.35)

    out_path = os.path.join(out_dir, "pareto_panel_all_envs.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight", pad_inches=0.05)
    plt.close()
    print(f"Saved full Pareto panel to {out_path}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--runs_root", type=str, required=True,
                   help="Root directory that contains dt/bc/rvs subdirs with experiment.py outputs.")
    p.add_argument("--out_dir", type=str, default="plots_rollout")
    p.add_argument("--fairdice_dir", type=str, default=None)
    p.add_argument("--panel", action="store_true",
                   help="Build metric panels + Pareto panels (2-objective envs).")
    p.add_argument("--do_hopper_v3", action="store_true",
                   help="Create MO-Hopper-v3 (3-objective) plot.")
    p.add_argument("--metrics_subset", action="store_true",
                   help="Single grid: rows=Hopper/Swimmer, cols=NSW/USW/ESW/Jain/Gini (amateur).")
    args = p.parse_args()

    envs_2obj = [
        "MO-Hopper-v2",
        "MO-Walker2d-v2",
        "MO-Ant-v2",
        "MO-HalfCheetah-v2",
        "MO-Swimmer-v2",
    ]
    datasets = ["expert_uniform", "amateur_uniform"]

    if args.panel:
        build_full_metric_panel(args.runs_root, args.fairdice_dir, args.out_dir, envs_2obj, datasets,
                                metric_fn=nsw, metric_ylabel="Avg Nash Social Welfare", fname_tag="nsw")
        build_full_metric_panel(args.runs_root, args.fairdice_dir, args.out_dir, envs_2obj, datasets,
                                metric_fn=maxmin_utility, metric_ylabel="Avg Max-min Utility", fname_tag="maxmin")
        build_full_metric_panel(args.runs_root, args.fairdice_dir, args.out_dir, envs_2obj, datasets,
                                metric_fn=jain_index, metric_ylabel="Avg Jain Index", fname_tag="jain")
        build_full_metric_panel(args.runs_root, args.fairdice_dir, args.out_dir, envs_2obj, datasets,
                                metric_fn=gini_coefficient, metric_ylabel="Avg Gini (lower=better)", fname_tag="gini")
        build_full_pareto_panel(args.runs_root, args.fairdice_dir, args.out_dir, envs_2obj, datasets)

    if args.do_hopper_v3:
        build_hopper3obj_figure(args.runs_root, args.fairdice_dir, args.out_dir, env="MO-Hopper-v3")

    if args.metrics_subset:
        build_2x5_env_metric_grid(args.runs_root, args.fairdice_dir, args.out_dir, dataset="amateur_uniform")


if __name__ == "__main__":
    main()
