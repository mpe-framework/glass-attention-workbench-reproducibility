"""
make_fig01_failure_rates.py
Fig 1 — Failure rate by command type (S-047 per-structural-feature breakdown)

Source: findings/METHODS_REPORT_S047.md (sealed, May 2026)
        findings/METHODS_REPORT_S043.md (overall 56.8% rate)
Data:   paper/data/fig01_failure_rates.csv
Output: paper/figures/fig01_failure_rate_by_command_type.{pdf,png}

Sealed values only — no experimental reruns.
"""

import csv
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

ROOT = os.path.join(os.path.dirname(__file__), "..")
DATA = os.path.join(ROOT, "data", "fig01_failure_rates.csv")
OUT_PDF = os.path.join(ROOT, "figures", "fig01_failure_rate_by_command_type.pdf")
OUT_PNG = os.path.join(ROOT, "figures", "fig01_failure_rate_by_command_type.png")

# --- load data ---
rows = []
with open(DATA) as f:
    for r in csv.DictReader(f):
        rows.append({
            "feature":  r["feature"],
            "present":  float(r["fail_rate_present"]),
            "absent":   float(r["fail_rate_absent"]),
            "sig":      r["significant"] == "True",
        })

# sort by lift (present - absent) descending
rows.sort(key=lambda r: r["present"] - r["absent"], reverse=True)

features  = [r["feature"] for r in rows]
present   = [r["present"] * 100 for r in rows]
absent    = [r["absent"]  * 100 for r in rows]
sig_mask  = [r["sig"]     for r in rows]

# --- overall rate (S-043) ---
OVERALL = 56.8

# --- plot ---
fig, axes = plt.subplots(1, 2, figsize=(9, 4.2),
                         gridspec_kw={"width_ratios": [1, 2.6]})

# Panel A — overall rate
ax0 = axes[0]
ax0.bar(["All compound\njump commands"], [OVERALL],
        color="#c0392b", width=0.45, zorder=3)
ax0.axhline(50, color="gray", lw=0.8, ls="--", zorder=2)
ax0.set_ylim(0, 100)
ax0.set_ylabel("Failure rate (%)")
ax0.set_title("(A) Overall failure rate\n(S-043, n=7706)", fontsize=9)
ax0.tick_params(axis="x", labelsize=8)
ax0.yaxis.grid(True, lw=0.4, color="#dddddd", zorder=0)
ax0.set_axisbelow(True)
ax0.text(0, OVERALL + 1.5, f"{OVERALL}%", ha="center", fontsize=9, fontweight="bold")

# Panel B — per-feature grouped bars
ax1 = axes[1]
x   = np.arange(len(features))
w   = 0.38

bars_present = ax1.bar(x - w/2, present, w, label="Feature present",
                       color="#2980b9", zorder=3)
bars_absent  = ax1.bar(x + w/2, absent,  w, label="Feature absent",
                       color="#aed6f1", zorder=3)

# highlight statistically significant pairs
for i, (sig, p_val, a_val) in enumerate(zip(sig_mask, present, absent)):
    if sig:
        ax1.bar(x[i] - w/2, p_val, w, color="#c0392b", zorder=4)
        ax1.bar(x[i] + w/2, a_val, w, color="#f1948a", zorder=4)

ax1.axhline(OVERALL, color="gray", lw=0.9, ls="--", zorder=2,
            label=f"Overall ({OVERALL}%)")
ax1.set_xticks(x)
ax1.set_xticklabels(features, rotation=35, ha="right", fontsize=8)
ax1.set_ylim(0, 100)
ax1.set_ylabel("Failure rate (%)")
ax1.set_title("(B) Failure rate by structural feature\n(S-047, n=200 examples)", fontsize=9)
ax1.yaxis.grid(True, lw=0.4, color="#dddddd", zorder=0)
ax1.set_axisbelow(True)

sig_patch = mpatches.Patch(color="#c0392b", label="Feature present (p<0.05)")
nsig_patch = mpatches.Patch(color="#2980b9", label="Feature present (n.s.)")
abs_patch  = mpatches.Patch(color="#aed6f1", label="Feature absent")
dash_patch = mpatches.Patch(color="gray",    label=f"Overall rate ({OVERALL}%)",
                             linestyle="--", fill=False)
ax1.legend(handles=[sig_patch, nsig_patch, abs_patch],
           fontsize=7.5, loc="upper right")

fig.suptitle(
    "Failure rate by command type — T5-small on SCAN add_prim_jump\n"
    "Source: S-043 (overall rate), S-047 (per-feature). "
    "Red = statistically significant (chi² p<0.05).",
    fontsize=8.5, y=1.01
)

plt.tight_layout()
plt.savefig(OUT_PDF, bbox_inches="tight", dpi=300)
plt.savefig(OUT_PNG, bbox_inches="tight", dpi=150)
print(f"Saved: {OUT_PDF}")
print(f"Saved: {OUT_PNG}")
