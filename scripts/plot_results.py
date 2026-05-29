"""
Generate rank/alpha sensitivity plots from results/sweep_results.csv.
Outputs: results/figures/acc_vs_rank.png, acc_vs_alpha_ratio.png
"""
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

sns.set_theme(style="whitegrid", palette="muted")
OUT = Path("results/figures")
OUT.mkdir(parents=True, exist_ok=True)

df = pd.read_csv("results/sweep_results.csv")
df["eval_acc"] = df["eval_acc"].astype(float)
df["alpha_ratio"] = df["lora_alpha"] / df["r"]
df["trainable_M"] = df["trainable_params"] / 1e6

# --- Plot 1: acc vs rank (grouped by alpha_ratio) ---
fig, ax = plt.subplots(figsize=(7, 4))
for ratio, grp in df.groupby("alpha_ratio"):
    grp = grp.sort_values("r")
    ax.plot(grp["r"], grp["eval_acc"], marker="o", label=f"alpha/r={ratio:.0f}x")
ax.set_xlabel("LoRA rank (r)")
ax.set_ylabel("Eval exact-match accuracy")
ax.set_title("Qwen2-VL-2B · VQAv2 — Accuracy vs Rank")
ax.legend()
fig.tight_layout()
fig.savefig(OUT / "acc_vs_rank.png", dpi=150)
print("Saved acc_vs_rank.png")

# --- Plot 2: trainable params vs acc (scatter) ---
fig, ax = plt.subplots(figsize=(6, 4))
sc = ax.scatter(df["trainable_M"], df["eval_acc"], c=df["r"],
                cmap="viridis", s=100, zorder=3)
for _, row in df.iterrows():
    ax.annotate(row["run_name"], (row["trainable_M"], row["eval_acc"]),
                fontsize=7, xytext=(4, 4), textcoords="offset points")
plt.colorbar(sc, ax=ax, label="rank (r)")
ax.set_xlabel("Trainable params (M)")
ax.set_ylabel("Eval exact-match accuracy")
ax.set_title("Accuracy vs Trainable Params")
fig.tight_layout()
fig.savefig(OUT / "acc_vs_params.png", dpi=150)
print("Saved acc_vs_params.png")

plt.show()
