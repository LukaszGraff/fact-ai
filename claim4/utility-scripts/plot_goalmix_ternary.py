import argparse
import csv
import math
from typing import List, Tuple

import matplotlib.pyplot as plt


def _load_points(csv_path: str) -> Tuple[List[Tuple[float, float, float, float]], List[str]]:
    points = []
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                a = float(row["reach_obj0_frac"])
                b = float(row["reach_obj1_frac"])
                c = float(row["reach_obj2_frac"])
                nsw = float(row["nsw_mean"])
            except (KeyError, ValueError):
                continue
            points.append((a, b, c, nsw))
    return points, reader.fieldnames or []


def _to_xy(a: float, b: float, c: float) -> Tuple[float, float]:
    # Barycentric -> 2D coordinates (triangle corners: a=1 left, b=1 right, c=1 top)
    x = 0.5 * (2.0 * b + c)
    y = (math.sqrt(3) / 2.0) * c
    return x, y


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot ternary heatmap of NSW over reach ratios.")
    parser.add_argument("--csv", required=True, help="Merged CSV with reach_obj*_frac and nsw_mean.")
    parser.add_argument(
        "--output",
        default="ternary_nsw.png",
        help="Output image path.",
    )
    parser.add_argument(
        "--title",
        default="NSW vs Reach Ratios (Ternary)",
        help="Plot title.",
    )
    parser.add_argument(
        "--point_size",
        type=float,
        default=120.0,
        help="Scatter point size.",
    )
    args = parser.parse_args()

    points, _ = _load_points(args.csv)
    if not points:
        raise SystemExit("No valid points found in CSV.")

    xs = []
    ys = []
    nsws = []
    for a, b, c, nsw in points:
        total = a + b + c
        if total <= 0:
            continue
        # Normalize because some episodes do not terminate with a goal.
        a_n = a / total
        b_n = b / total
        c_n = c / total
        x, y = _to_xy(a_n, b_n, c_n)
        xs.append(x)
        ys.append(y)
        nsws.append(nsw)

    fig, ax = plt.subplots(figsize=(7, 6))

    # Triangle boundary (a=1, b=1, c=1)
    tri_x = [0.0, 1.0, 0.5, 0.0]
    tri_y = [0.0, 0.0, math.sqrt(3) / 2.0, 0.0]
    ax.plot(tri_x, tri_y, color="black", linewidth=1.5)

    # Gridlines and tick labels
    ticks = [0.2, 0.4, 0.6, 0.8]
    h = math.sqrt(3) / 2.0
    for t in ticks:
        # Gridlines
        # c = t (parallel to base)
        x1, y1 = _to_xy(1.0 - t, 0.0, t)
        x2, y2 = _to_xy(0.0, 1.0 - t, t)
        ax.plot([x1, x2], [y1, y2], color="black", alpha=0.15, linestyle="--", linewidth=1.0)
        # a = t (parallel to right edge)
        x3, y3 = _to_xy(t, 1.0 - t, 0.0)
        x4, y4 = _to_xy(t, 0.0, 1.0 - t)
        ax.plot([x3, x4], [y3, y4], color="black", alpha=0.15, linestyle="--", linewidth=1.0)
        # b = t (parallel to left edge)
        x5, y5 = _to_xy(1.0 - t, t, 0.0)
        x6, y6 = _to_xy(0.0, t, 1.0 - t)
        ax.plot([x5, x6], [y5, y6], color="black", alpha=0.15, linestyle="--", linewidth=1.0)

        # Tick labels on edges
        # Obj0 values along base (c=0), a=t at (a=t, b=1-t)
        xb0, yb0 = _to_xy(t, 1.0 - t, 0.0)
        ax.text(xb0, yb0 - 0.04, f"{t:.1f}", ha="center", va="top", fontsize=8)
        # Obj1 values along right edge (a=0), b=t at (a=0, b=t, c=1-t)
        xb1, yb1 = _to_xy(0.0, t, 1.0 - t)
        ax.text(xb1 + 0.03, yb1, f"{t:.1f}", ha="left", va="center", fontsize=8)
        # Obj2 values along left edge (b=0), c=t at (a=1-t, b=0, c=t)
        xb2, yb2 = _to_xy(1.0 - t, 0.0, t)
        ax.text(xb2 - 0.03, yb2, f"{t:.1f}", ha="right", va="center", fontsize=8)

    sc = ax.scatter(xs, ys, c=nsws, cmap="viridis", s=args.point_size, edgecolors="black")
    cbar = plt.colorbar(sc, ax=ax)
    cbar.set_label("NSW (mean)")

    ax.set_title(args.title)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_aspect("equal", "box")
    ax.set_frame_on(False)
    for spine in ax.spines.values():
        spine.set_visible(False)

    # Corner labels (pad outward so they don't overlap the boundary)
    label_pad = 0.035
    ax.text(-label_pad, -label_pad * 0.55, "Obj0", ha="right", va="top")
    ax.text(1.0 + label_pad, -label_pad * 0.55, "Obj1", ha="left", va="top")
    ax.text(0.5, math.sqrt(3) / 2.0 + label_pad, "Obj2", ha="center", va="bottom")

    frame_pad = 0.06
    ax.set_xlim(-frame_pad, 1.0 + frame_pad)
    ax.set_ylim(-frame_pad * 0.7, math.sqrt(3) / 2.0 + frame_pad)

    plt.tight_layout()
    fig.savefig(args.output, dpi=200)
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
