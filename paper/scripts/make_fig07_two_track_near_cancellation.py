"""
make_fig07_two_track_near_cancellation.py
Fig 7 — Two-track near-cancellation convergence

Source (Panel A): S-059b post-LayerNorm corrected values (sealed)
                  Cited in METHODS_REPORT_S061.md and METHODS_REPORT_G057.md
Source (Panel B): G-057 gamma sweep — METHODS_REPORT_G057.md / FINDINGS.md F-G057 (sealed)
Data:   paper/data/fig07_two_track_near_cancellation.csv
Output: paper/figures/fig07_two_track_near_cancellation.{pdf,png}

CLAIM BOUNDARY (enforced):
- Panel A: Production near-cancellation values are S-059b corrected (post-LayerNorm).
  These are logit-decomposition units after LayerNorm correction.
- Panel B: G-057 gamma sweep shows reconstruction fraction diverging as gamma→1.
  This is an analogy / convergence demonstration, NOT a calibrated toy-to-production
  logit prediction.
- Do NOT mention R²=1.0.
- Do NOT claim toy parameters reproduce production logits.
- Do NOT call the gamma sweep a calibrated toy-to-production fit.
- Do NOT say G-057 proves the T5 mechanism.
- gamma=1.0 point stored as the large measured proxy at exact cancellation (value read from CSV).
- gamma=1.1 and gamma=2.0 are shown as inversion regime, not on the log-scale line.

Sealed values only — no experimental reruns. Do not hardcode values here; read from CSV.
"""

import csv
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT    = os.path.join(os.path.dirname(__file__), "..")
DATA    = os.path.join(ROOT, "data", "fig07_two_track_near_cancellation.csv")
OUT_PDF = os.path.join(ROOT, "figures", "fig07_two_track_near_cancellation.pdf")
OUT_PNG = os.path.join(ROOT, "figures", "fig07_two_track_near_cancellation.png")

# --- load data ---
prod_components, prod_values = [], []
toy_gamma, toy_recon, toy_ablation = [], [], []

with open(DATA) as f:
    for r in csv.DictReader(f):
        if r["track"] == "production":
            prod_components.append(r["component_or_gamma"])
            prod_values.append(float(r["logit_contribution"]))
        elif r["track"] == "toy":
            gamma = float(r["component_or_gamma"])
            recon = float(r["recon_fraction"]) if r["recon_fraction"] else None
            abl   = float(r["ablation_pct"])   if r["ablation_pct"]   else None
            toy_gamma.append(gamma)
            toy_recon.append(recon)
            toy_ablation.append(abl)

fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.4))

# --- Panel A: Production S-track near-cancellation (compact bar) ---
ax0 = axes[0]
prod_labels  = ["Cross-attention\n(L4H6+L5H5)", "FFN correction\n(L6H2/L6H6)", "Net margin"]
prod_colors  = ["#c0392b", "#2980b9", "#7f8c8d"]

bars = ax0.bar(prod_labels, prod_values, color=prod_colors, width=0.52, zorder=3,
               edgecolor="white", linewidth=0.8)
ax0.axhline(0, color="black", lw=0.9, zorder=4)

for bar, val in zip(bars, prod_values):
    ypos = val + (5 if val > 0 else -8)
    ax0.text(bar.get_x() + bar.get_width() / 2, ypos,
             f"{val:+.2f}", ha="center", va="bottom" if val > 0 else "top",
             fontsize=9.5, fontweight="bold")

ax0.annotate("", xy=(-0.38, prod_values[0]), xytext=(-0.38, prod_values[1]),
             arrowprops=dict(arrowstyle="<->", color="#555555", lw=1.2))
ax0.text(-0.52, 0, f"near-\ncancellation\n({prod_values[2]:+.1f} net)",
         ha="right", va="center", fontsize=7.5, color="#555555")

ax0.set_ylabel("Logit contribution (I_JUMP − I_WALK direction)\nS-059b post-LN corrected units")
ax0.set_ylim(-195, 190)
ax0.yaxis.grid(True, lw=0.4, color="#eeeeee", zorder=0)
ax0.set_axisbelow(True)
ax0.set_title("(A) Production S-track anchor (S-059b)\n"
              "Near-cancellation: large opposing contributions, small residual",
              fontsize=8.8)

# --- Panel B: G-057 toy gamma sweep (log scale) ---
ax1 = axes[1]

# Plot only gamma <= 1.0 (ascending near-cancellation regime) on log scale
plot_gamma = [g for g, r in zip(toy_gamma, toy_recon) if r is not None and r > 0 and g <= 1.0]
plot_recon = [r for g, r in zip(toy_gamma, toy_recon) if r is not None and r > 0 and g <= 1.0]

ax1.plot(plot_gamma, plot_recon, color="#2980b9", marker="o",
         lw=2.0, markersize=7, zorder=4, label="recon_fraction (γ ≤ 1.0)")

ax1.set_yscale("log")

# annotate gamma=1.0 as exact cancellation
gamma_1_recon = [r for g, r in zip(toy_gamma, toy_recon) if g == 1.0 and r is not None][0]
gamma_1_label = f"γ=1.0: exact cancellation\n(∞; {gamma_1_recon:,.0f}× measured)"
ax1.annotate(gamma_1_label,
             xy=(1.00, gamma_1_recon), xytext=(0.65, 1000),
             fontsize=7.5, color="#c0392b",
             arrowprops=dict(arrowstyle="->", color="#c0392b", lw=0.9))

# annotate inversion regime (gamma=2.0) as text box, not plotted on log line
abl_2 = [a for g, a in zip(toy_gamma, toy_ablation) if g == 2.0][0]
ax1.text(0.98, 0.08,
         f"γ=2.0 (inversion regime):\n"
         f"{abl_2:.1f}% ablation change\n"
         f"(G-056 FIN-inversion reproduced\nvia FFN over-correction path)",
         ha="right", va="bottom", transform=ax1.transAxes,
         fontsize=7, color="#7f8c8d",
         bbox=dict(boxstyle="round,pad=0.3", facecolor="#f9f9f9",
                   edgecolor="#cccccc", lw=0.7))

ax1.set_xlabel("FFN correction parameter γ")
ax1.set_ylabel("Reconstruction fraction (log scale)\n"
               "= head_signal / observed_margin(γ)")
ax1.set_title("(B) Toy G-track: G-057 γ sweep\n"
              "Near-cancellation regime: recon_fraction → ∞ as γ → 1",
              fontsize=8.8)
ax1.yaxis.grid(True, lw=0.4, color="#eeeeee", zorder=0)
ax1.set_axisbelow(True)
ax1.legend(fontsize=8)
ax1.set_xlim(-0.05, 1.1)

fig.suptitle(
    "Two-track near-cancellation convergence\n"
    "Production measurement (S-059b) shows near-cancellation in T5-small. "
    "Controlled toy geometry (G-057) independently\ndemonstrates that "
    "near-cancellation is a real regime as γ → 1.0. "
    "Analogy/convergence claim only — not a calibrated toy-to-production logit prediction.",
    fontsize=8.0, y=1.03
)

plt.tight_layout()
plt.savefig(OUT_PDF, bbox_inches="tight", dpi=300)
plt.savefig(OUT_PNG, bbox_inches="tight", dpi=150)
print(f"Saved: {OUT_PDF}")
print(f"Saved: {OUT_PNG}")
