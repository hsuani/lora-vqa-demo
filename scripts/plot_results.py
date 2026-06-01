"""
Generate rank/alpha sensitivity plots from a sweep results CSV.

  python scripts/plot_results.py                                   # full sweep (5000)
  python scripts/plot_results.py --csv results/sweep_results.csv   # fast sweep (2000)

Outputs to results/figures/: acc_vs_rank.png, acc_vs_params.png
"""
import argparse
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--csv", default="results/sweep_results_full.csv",
                   help="sweep results CSV to plot")
    p.add_argument("--outdir", default="results/figures")
    return p.parse_args()


def main():
    args = parse_args()
    sns.set_theme(style="whitegrid", palette="muted")
    out = Path(args.outdir)
    out.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.csv)
    df["eval_acc"] = df["eval_acc"].astype(float)
    df["alpha_ratio"] = df["lora_alpha"] / df["r"]
    df["trainable_M"] = df["trainable_params"] / 1e6

    # --- Plot 1: acc vs rank, one line per alpha/r ratio ---
    fig, ax = plt.subplots(figsize=(7, 4))
    for ratio, grp in df.groupby("alpha_ratio"):
        grp = grp.sort_values("r")
        ax.plot(grp["r"], grp["eval_acc"], marker="o", linewidth=2,
                label=f"alpha/r = {ratio:.0f}x")
    ax.set_xscale("log", base=2)
    ax.set_xticks(sorted(df["r"].unique()))
    ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())
    ax.set_xlabel("LoRA rank (r)  [log2 scale]")
    ax.set_ylabel("Eval exact-match accuracy")
    ax.set_title("Qwen2-VL-2B · VQAv2 — Accuracy vs Rank")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / "acc_vs_rank.png", dpi=150)
    print(f"Saved {out/'acc_vs_rank.png'}")

    # --- Plot 2: trainable params vs acc ---
    fig, ax = plt.subplots(figsize=(6.5, 4))
    sc = ax.scatter(df["trainable_M"], df["eval_acc"], c=df["r"],
                    cmap="viridis", s=120, zorder=3, edgecolor="white")
    for _, row in df.iterrows():
        ax.annotate(row["run_name"], (row["trainable_M"], row["eval_acc"]),
                    fontsize=7, xytext=(5, 4), textcoords="offset points")
    plt.colorbar(sc, ax=ax, label="rank (r)")
    ax.set_xlabel("Trainable params (M)")
    ax.set_ylabel("Eval exact-match accuracy")
    ax.set_title("Accuracy vs Trainable Parameters")
    fig.tight_layout()
    fig.savefig(out / "acc_vs_params.png", dpi=150)
    print(f"Saved {out/'acc_vs_params.png'}")


if __name__ == "__main__":
    main()
