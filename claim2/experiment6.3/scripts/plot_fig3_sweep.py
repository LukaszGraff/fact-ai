import pandas as pd
import matplotlib.pyplot as plt

def main():
    df = pd.read_csv("outputs/fig3_raw.csv")
    betas = sorted(df["beta"].unique())

    plt.figure()
    for beta in betas:
        sub = df[df["beta"] == beta].sort_values("sigma")
        x = sub["sigma"].replace(0.0, sub["sigma"][sub["sigma"] > 0].min() * 0.5)
        plt.plot(x, sub["nsw"], label=f"beta={beta}")

    plt.xscale("log")
    plt.xlabel("Perturbation on mu* (sigma)")
    plt.ylabel("NSW")
    plt.legend()
    plt.tight_layout()
    plt.savefig("outputs/fig3.png", dpi=200)
    print("Saved outputs/fig3.png")

if __name__ == "__main__":
    main()
