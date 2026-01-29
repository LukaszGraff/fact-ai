import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# =====================
# Load data
# =====================
csv_path = r"C:\Users\bodga\OneDrive\Documents\Documents\University of Amsterdam 2025-2026\Period 3\5204FACT6Y - Fairness, Accountability, Confidentiality and Transparency in AI\fact-ai\results\seeds_1_50\experiment_results.csv"  # adjust if needed
df = pd.read_csv(csv_path)

# =====================
# Aggregate over seeds
# =====================
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

# =====================
# Plot helper
# =====================
def plot_metric(metric_name, ylabel):
    plt.figure(figsize=(7, 5))

    for alpha in sorted(grouped["alpha"].unique()):
        sub = grouped[grouped["alpha"] == alpha].sort_values("beta")

        x = sub["beta"].values
        mean = sub[f"{metric_name}_mean"].values
        std = sub[f"{metric_name}_std"].values

        plt.plot(x, mean, marker="o", label=f"alpha = {alpha}")
        plt.fill_between(x, mean - std, mean + std, alpha=0.2)

    plt.xscale("log")
    plt.xlabel("Beta")
    plt.ylabel(ylabel)
    plt.title(f"{ylabel} vs Beta")
    plt.legend()
    plt.grid(True, which="both", linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.show()

# =====================
# Generate plots
# =====================
plot_metric("utilitarian", "Utilitarian Welfare")
plot_metric("jain", "Jain Index")
plot_metric("nsw", "Nash Social Welfare")

