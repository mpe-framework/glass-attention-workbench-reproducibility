"""
make_fig06_dose_response.py
Fig 6 — Rank-1 W_V repair dose-response curve

Source: findings/METHODS_REPORT_S061.md (sealed, May 2026)
Data:   paper/data/fig06_dose_response.csv
Output: paper/figures/fig06_rank1_repair_dose_response.{pdf,png}

CLAIM BOUNDARY:
- n=30 fail group (substituted-walk, has_around, SEED=42)
- Success-group specificity: n=2 (S-062) — not sufficient for broad specificity claim
- 16/30 at alpha=1.0 is a sampled count, NOT a population-level repair rate
- Broader specificity (walk around, run around, look around) not yet tested

Sealed values only — no experimental reruns.
"""

import csv
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT    = os.path.join(os.path.dirname(__file__), "..")
DATA    = os.path.join(ROOT, "data", "fig06_dose_response.csv")
OUT_PDF = os.path.join(ROOT, "figures", "fig06_rank1_repair_dose_response.pdf")
OUT_PNG = os.path.join(ROOT, "figures", "fig06_rank1_repair_dose_response.png")

arm_styles = {
    "L5H5_only": {"color": "#2980b9", "marker": "o", "label": "L5H5 only"},
    "L4H6_only": {"color": "#e67e22", "marker": "s", "label": "L4H6 only"},
    "joint":     {"color": "#c0392b", "marker": "^", "label": "Joint L4H6 + L5H5"},
}

data = {arm: {"alpha": [], "delta": [], "flips": []} for arm in arm_styles}
with open(DATA) as f:
    for r in csv.DictReader(f):
        arm = r["arm"]
        data[arm]["alpha"].append(float(r["alpha"]))
        data[arm]["delta"].append(float(r["delta_margin"]))
        data[arm]["flips"].append(int(r["flips"]))

fig, axes = plt.subplots(1, 2, figsize=(9, 4.2))

# Panel A — Δmargin vs α
ax0 = axes[0]
for arm, st in arm_styles.items():
    d = data[arm]
    ax0.plot(d["alpha"], d["delta"], color=st["color"], marker=st["marker"],
             label=st["label"], lw=1.8, markersize=7, zorder=4)

ax0.axhline(0, color="black", lw=0.8, zorder=3)
ax0.set_xlabel("Correction strength α")
ax0.set_ylabel("Mean Δm_fail  (logit_jump − logit_walk change)")
ax0.set_title("(A) Mean fail-margin improvement vs α\n(n=30 fail group)", fontsize=9)
ax0.legend(fontsize=8.5)
ax0.yaxis.grid(True, lw=0.4, color="#eeeeee", zorder=0)
ax0.set_axisbelow(True)
ax0.set_xticks([0.25, 0.50, 0.75, 1.00])

# Panel B — flips vs α
ax1 = axes[1]
for arm, st in arm_styles.items():
    d = data[arm]
    ax1.plot(d["alpha"], d["flips"], color=st["color"], marker=st["marker"],
             label=st["label"], lw=1.8, markersize=7, zorder=4)

ax1.axhline(0, color="black", lw=0.8, zorder=3)
ax1.set_xlabel("Correction strength α")
ax1.set_ylabel("Sampled failures flipped  (of 30)")
ax1.set_title("(B) Failures flipped vs α\n(n=30 fail group; not a population rate)", fontsize=9)
ax1.legend(fontsize=8.5)
ax1.yaxis.grid(True, lw=0.4, color="#eeeeee", zorder=0)
ax1.set_axisbelow(True)
ax1.set_xticks([0.25, 0.50, 0.75, 1.00])
ax1.set_ylim(-1, 20)

# annotate joint α=1.0
ax1.annotate("16/30 flipped\n(joint, α=1.0)",
             xy=(1.00, 16), xytext=(0.72, 17.5),
             fontsize=7.5, color="#c0392b",
             arrowprops=dict(arrowstyle="->", color="#c0392b", lw=0.8))

fig.suptitle(
    "Rank-1 W_V repair dose-response — T5-small SCAN add_prim_jump\n"
    "Source: S-061 (METHODS_REPORT_S061.md). "
    "Success specificity: S-062 (n=2). Broader specificity not yet tested.",
    fontsize=8.5, y=1.01
)

plt.tight_layout()
plt.savefig(OUT_PDF, bbox_inches="tight", dpi=300)
plt.savefig(OUT_PNG, bbox_inches="tight", dpi=150)
print(f"Saved: {OUT_PDF}")
print(f"Saved: {OUT_PNG}")
