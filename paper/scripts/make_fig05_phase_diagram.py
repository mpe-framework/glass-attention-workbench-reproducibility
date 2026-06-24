"""
make_fig05_phase_diagram.py
Fig 5 — Born filter three-regime phase diagram

Source: findings/METHODS_REPORT_G054.md (sealed, May 2026)
Data:   paper/data/fig05_phase_diagram.csv
Output: paper/figures/fig05_three_regime_phase_diagram.{pdf,png}

CLAIM BOUNDARY: controlled G-track geometry only — NOT T5-small.
The 867x amplification applies to FIN_WEIGHT=0.3 in the controlled setting.
Do not extrapolate to production models.

Sealed values only — no experimental reruns.
"""

import csv
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

ROOT    = os.path.join(os.path.dirname(__file__), "..")
DATA    = os.path.join(ROOT, "data", "fig05_phase_diagram.csv")
OUT_PDF = os.path.join(ROOT, "figures", "fig05_three_regime_phase_diagram.pdf")
OUT_PNG = os.path.join(ROOT, "figures", "fig05_three_regime_phase_diagram.png")

rows = []
with open(DATA) as f:
    for r in csv.DictReader(f):
        rows.append({
            "fw":     float(r["fin_weight"]),
            "ratio":  float(r["peak_ratio"]),
            "regime": r["regime"],
        })

regime_colors = {
    "slow-collapse":  "#2980b9",
    "sharp-collapse": "#c0392b",
    "no-collapse":    "#27ae60",
}
regime_labels = {
    "slow-collapse":  "Slow-collapse (~2×)",
    "sharp-collapse": "Sharp-collapse (867×)",
    "no-collapse":    "No-collapse (<1×)",
}

fig, ax = plt.subplots(figsize=(6.5, 4.4))

# plot in log scale for y
for r in rows:
    c = regime_colors[r["regime"]]
    ax.scatter(r["fw"], r["ratio"], color=c, s=80, zorder=5,
               edgecolors="white", linewidths=0.6)

# connect with line (broken by the extreme point)
fws    = [r["fw"]    for r in rows]
ratios = [r["ratio"] for r in rows]
ax.plot(fws, ratios, color="#aaaaaa", lw=0.9, zorder=3, ls="-")

# annotate the 867x point
ax.annotate(
    "867×\n(FIN_WEIGHT=0.3\nphase boundary)",
    xy=(0.3, 867.45), xytext=(0.55, 400),
    fontsize=7.5, color="#c0392b",
    arrowprops=dict(arrowstyle="->", color="#c0392b", lw=0.9),
    ha="left",
)

# phase boundary marker
ax.axvline(0.4, color="#888888", lw=0.9, ls="--", zorder=2)
ax.text(0.41, 5, "phase\nboundary\n(~0.4–0.5)", fontsize=7.5, color="#888888")

ax.set_yscale("log")
ax.set_xlabel("FIN_WEIGHT (bank token FIN embedding weight)")
ax.set_ylabel("Peak Born filter ratio (log scale)")
ax.yaxis.grid(True, lw=0.4, color="#eeeeee", zorder=0)
ax.set_axisbelow(True)

patches = [mpatches.Patch(color=c, label=regime_labels[r])
           for r, c in regime_colors.items()]
ax.legend(handles=patches, fontsize=8, loc="upper right")

ax.set_title(
    "Born filter three-regime phase diagram (G-054)\n"
    "Controlled G-track geometry — NOT T5-small production model\n"
    "Source: findings/METHODS_REPORT_G054.md",
    fontsize=8.5
)

plt.tight_layout()
plt.savefig(OUT_PDF, bbox_inches="tight", dpi=300)
plt.savefig(OUT_PNG, bbox_inches="tight", dpi=150)
print(f"Saved: {OUT_PDF}")
print(f"Saved: {OUT_PNG}")
