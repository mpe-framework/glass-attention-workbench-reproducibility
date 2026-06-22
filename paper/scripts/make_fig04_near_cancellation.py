"""
make_fig04_near_cancellation.py
Fig 4 — Near-cancellation bar chart

Source: findings/METHODS_REPORT_S061.md and findings/METHODS_REPORT_G057.md
        (S-059b post-LayerNorm corrected values, sealed May 2026)
Data:   paper/data/fig04_near_cancellation.csv  ← single source of truth
Output: paper/figures/fig04_near_cancellation.{pdf,png}

IMPORTANT CALIBRATION NOTE (enforced by FIGURE_SOURCE_AUDIT.md):
- S-059 V0.1.0 pre-LayerNorm reconstruction was ~32,193x — INVALID for mechanism claims.
- S-059b post-LayerNorm corrected reconstruction is ~28.5x (values in CSV are corrected).
- These values illustrate near-cancellation structure, not calibrated final-logit magnitudes.
- Do NOT use S-059 V0.1.0 figures anywhere in the paper.

Sealed values only — no experimental reruns. Do not hardcode values here; read from CSV.
"""

import csv
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT    = os.path.join(os.path.dirname(__file__), "..")
DATA    = os.path.join(ROOT, "data", "fig04_near_cancellation.csv")
OUT_PDF = os.path.join(ROOT, "figures", "fig04_near_cancellation.pdf")
OUT_PNG = os.path.join(ROOT, "figures", "fig04_near_cancellation.png")

# --- load from CSV (single source of truth) ---
components = []
values     = []
with open(DATA) as f:
    for row in csv.DictReader(f):
        components.append(row["label"])
        values.append(float(row["logit_contribution"]))

colors = ["#c0392b", "#2980b9", "#7f8c8d"]

fig, ax = plt.subplots(figsize=(5.5, 4.2))

bars = ax.bar(components, values, color=colors, width=0.52, zorder=3,
              edgecolor="white", linewidth=0.8)

# zero line
ax.axhline(0, color="black", lw=0.9, zorder=4)

# value labels
for bar, val in zip(bars, values):
    ypos = val + (4 if val > 0 else -7)
    ax.text(bar.get_x() + bar.get_width() / 2, ypos,
            f"{val:+.2f}", ha="center", va="bottom" if val > 0 else "top",
            fontsize=10, fontweight="bold")

ax.set_ylabel("Logit contribution  (I_JUMP − I_WALK direction)\nS-059b post-LayerNorm corrected units")
ax.set_ylim(-190, 185)
ax.yaxis.grid(True, lw=0.4, color="#dddddd", zorder=0)
ax.set_axisbelow(True)

# near-cancellation bracket annotation
ax.annotate("", xy=(-0.35, values[0]), xytext=(-0.35, values[1]),
            arrowprops=dict(arrowstyle="<->", color="#555555", lw=1.2))
net_label = f"near-\ncancellation\n({values[2]:+.1f} net)"
ax.text(-0.48, 0, net_label, ha="right", va="center",
        fontsize=7.5, color="#555555")

ax.set_title(
    "Near-cancellation structure at the action slot\n"
    "T5-small fail group (n=30 substituted-walk examples, SEED=42)\n"
    "Source: S-059b (post-LN corrected); raw pre-LN S-059 values not used",
    fontsize=8.5
)

# calibration note
note = (
    "Note: these are logit-decomposition units after LayerNorm correction.\n"
    "They illustrate near-cancellation structure, not calibrated final-logit magnitudes."
)
fig.text(0.5, -0.04, note, ha="center", fontsize=7, style="italic",
         wrap=True, color="#555555")

plt.tight_layout()
plt.savefig(OUT_PDF, bbox_inches="tight", dpi=300)
plt.savefig(OUT_PNG, bbox_inches="tight", dpi=150)
print(f"Saved: {OUT_PDF}")
print(f"Saved: {OUT_PNG}")
