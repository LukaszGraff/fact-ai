import os
import glob
import pandas as pd
import matplotlib.pyplot as plt

# Base results directory (the one you copied from the cluster)
base_results_dir = r"C:\Users\bodga\OneDrive\Documents\Documents\University of Amsterdam 2025-2026\Period 3\5204FACT6Y - Fairness, Accountability, Confidentiality and Transparency in AI\results"

# Find all experiment_results.csv files recursively
csv_files = glob.glob(
    os.path.join(base_results_dir, "**", "experiment_results.csv"),
    recursive=True
)

print(f"Found {len(csv_files)} experiment_results.csv files")

# Load and concatenate
dfs = []
for path in csv_files:
    df_part = pd.read_csv(path)
    df_part["source_file"] = os.path.relpath(path, base_results_dir)  # optional but useful
    dfs.append(df_part)

df = pd.concat(dfs, ignore_index=True)

print(f"Total rows loaded: {len(df)}")


grouped = (
    df
    .groupby(["alpha", "beta"])
    .agg(
        utilitarian_mean=("utilitarian", "mean"),
        utilitarian_std=("utilitarian", "std"),
        jain_mean=("jain", "mean"),
        jain_std=("jain", "std"),
        nsw_mean=("nsw", "mean"),
        nsw_std=("nsw", "std"),
    )
    .reset_index()
)
print(df.groupby("alpha")["seed"].nunique())
print(df.groupby(["alpha", "beta"]).size().describe())
print(grouped[grouped["alpha"].isin([0.5, 1.0])])


# Combined plot
# =====================
fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharex=True)

metrics = [
    ("nsw", "Nash Social Welfare"),
    ("utilitarian", "Utilitarian "),
    ("jain", "Jain's Fairness"),
]

for ax, (metric, ylabel) in zip(axes, metrics):
    for alpha in sorted(grouped["alpha"].unique()):
        sub = grouped[grouped["alpha"] == alpha].sort_values("beta")

        x = sub["beta"].values
        mean = sub[f"{metric}_mean"].values
        std = sub[f"{metric}_std"].values

        ax.plot(x, mean, marker="o", label=f"$\\alpha={alpha}$")
        ax.fill_between(x, mean - std, mean + std, alpha=0.2)
    linestyles = {
        0.0: "-",
        0.5: "--",
        1.0: ":",
        1.25: "-."
    }
    ax.set_xscale("log")
    ax.set_xlabel("Beta")
    ax.set_ylabel(ylabel)
    ax.set_title(f"{ylabel}")
    ax.grid(True, which="both", linestyle="--", alpha=0.4)


# Single legend for all subplots
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc="upper center", ncol=len(labels), frameon=False)

plt.tight_layout(rect=[0, 0, 1, 0.9])
plt.show()