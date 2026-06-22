"""
make_fig02_decoder_routing_summary.py
Fig 2 — Decoder routing summary by layer

Source: S-049_DECODER_XATTN_V0.1.0 (sealed)
        Drive folder 1lgFKvr-67lgqGHykau2A6FOlNyMThpHQ / 049_results.json
Data:   paper/data/fig02_decoder_routing_summary.csv
Output: paper/figures/fig02_decoder_routing_summary.{pdf,png}

CLAIM BOUNDARY:
- Summary statistics only. Full per-example attention matrices were not saved.
- Do NOT call this a heat map. Do NOT imply raw attention matrices exist.
- h1: mean attention to 'around' encoder position is NOT significantly different
  between fail and success (p=0.9208) — routing is statistically identical.
- h4: per-layer profile shows max divergence at layer 4 (stored as max_div_layer=4
  in sealed JSON). This is the only layer-level divergence claim made.
- This figure supports the claim: routing is not the primary discriminator in this comparison.
  The value-projection claim is supported jointly by Fig 3, Fig 4, and Fig 6.

Sealed values only — no experimental reruns. Do not hardcode values here; read from CSV.
"""

import csv
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT    = os.path.join(os.path.dirname(__file__), "..")
DATA    = os.path.join(ROOT, "data", "fig02_decoder_routing_summary.csv")
OUT_PDF = os.path.join(ROOT, "figures", "fig02_decoder_routing_summary.pdf")
OUT_PNG = os.path.join(ROOT, "figures", "fig02_decoder_routing_summary.png")

layers_fail, layers_succ = [], []
weights_fail, weights_succ = [], []
summary = {}

with open(DATA) as f:
    for r in csv.DictReader(f):
        if r["type"] == "per_layer":
            layer = int(r["layer"])
            val   = float(r["value"])
            if r["group_or_metric"] == "fail":
                layers_fail.append(layer)
                weights_fail.append(val)
            else:
                layers_succ.append(layer)
                weights_succ.append(val)
        elif r["type"] == "summary":
            summary[r["group_or_metric"]] = float(r["value"])

max_div_layer = int(summary["max_div_layer"])
fail_mean     = summary["fail_mean_xattn"]
success_mean  = summary["success_mean_xattn"]
h1_p          = summary["h1_p"]

fig, axes = plt.subplots(1, 2, figsize=(9.5, 4.0))

# --- Panel A: per-layer attention weight profile ---
ax0 = axes[0]
layer_labels = [f"L{l}" for l in layers_fail]

ax0.plot(layer_labels, weights_fail, color="#c0392b", marker="o",
         lw=2.0, markersize=7, label="Fail group (n=50)", zorder=4)
ax0.plot(layer_labels, weights_succ, color="#2980b9", marker="s",
         lw=2.0, markersize=7, linestyle="--", label="Success group (n=25)", zorder=4)

ax0.axvline(f"L{max_div_layer}", color="#888888", lw=1.0, ls=":", zorder=2)
ax0.text(f"L{max_div_layer}", 0.16,
         f"peak divergence\n(h4, L{max_div_layer})", ha="center", fontsize=7,
         color="#888888", va="bottom")

ax0.set_ylabel("Mean cross-attention weight\n(to 'around' encoder position)")
ax0.set_title("(A) Per-layer attention weight profile\n"
              "Fail vs. success — similar across all layers",
              fontsize=9)
ax0.legend(fontsize=8.5)
ax0.yaxis.grid(True, lw=0.4, color="#eeeeee", zorder=0)
ax0.set_axisbelow(True)
ax0.set_ylim(0, 0.20)

# --- Panel B: h1 aggregate comparison ---
ax1 = axes[1]
groups   = ["Fail group\n(n=50)", "Success group\n(n=25)"]
means    = [fail_mean, success_mean]
clrs     = ["#c0392b", "#2980b9"]
bars = ax1.bar(groups, means, color=clrs, width=0.45, zorder=3,
               edgecolor="white", linewidth=0.8)
for bar, val in zip(bars, means):
    ax1.text(bar.get_x() + bar.get_width() / 2, val + 0.002,
             f"{val:.4f}", ha="center", va="bottom", fontsize=10, fontweight="bold")

ax1.set_ylabel("Mean attention to 'around' token")
ax1.set_title(f"(B) h1: mean attention comparison\n"
              f"p = {h1_p:.4f} (NS) — routing statistically identical",
              fontsize=9)
ax1.yaxis.grid(True, lw=0.4, color="#eeeeee", zorder=0)
ax1.set_axisbelow(True)
ax1.set_ylim(0, 0.14)

ax1.text(0.5, 0.96,
         f"h1 p = {h1_p:.4f} (not significant)\n"
         "K1: routing profiles similar — not the primary discriminator",
         ha="center", va="top", transform=ax1.transAxes,
         fontsize=7.5, color="#555555",
         bbox=dict(boxstyle="round,pad=0.3", facecolor="#f9f9f9",
                   edgecolor="#cccccc", lw=0.7))

fig.suptitle(
    "Decoder routing summary by layer — T5-small SCAN add_prim_jump\n"
    "Source: S-049 (S-049_DECODER_XATTN_V0.1.0). Summary statistics only; "
    "raw attention matrices not available.\n"
    "Routing profiles nearly identical in fail and success cases "
    "(K1: routing is not the primary discriminator in this comparison).",
    fontsize=8.2, y=1.02
)

plt.tight_layout()
plt.savefig(OUT_PDF, bbox_inches="tight", dpi=300)
plt.savefig(OUT_PNG, bbox_inches="tight", dpi=150)
print(f"Saved: {OUT_PDF}")
print(f"Saved: {OUT_PNG}")
