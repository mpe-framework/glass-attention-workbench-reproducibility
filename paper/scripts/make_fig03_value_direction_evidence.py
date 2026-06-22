"""
make_fig03_value_direction_evidence.py
Fig 3 — Layer and head value-direction evidence

Source: S-051_LEVER_ARM_DISSECTION_V0.1.0 (sealed)
        Drive folder 1lgFKvr-67lgqGHykau2A6FOlNyMThpHQ / 051_results.json
Data:   paper/data/fig03_value_direction_evidence.csv
Output: paper/figures/fig03_value_direction_evidence.{pdf,png}

CLAIM BOUNDARY:
- Option B: layer-wise cosine similarity of residual stream to I_WALK and I_JUMP
  output embeddings at action slot (n_fail=227 slots, n_success=144 slots).
- Option C: top-5 heads by walk_fail cosine — these are value-direction fingerprints,
  NOT causal attribution. Do NOT call this a 6×8 head heat map.
- Do NOT claim L4H6 and L5H5 are the only causal heads.
- Causal evidence (what W_V intervention does) belongs to S-061, reported in Fig 6.
- Key repair heads L4H6 and L5H5 are flagged here as 'is_key_head' from S-061 context
  only; the causal link is established in §7, not from this figure.

Sealed values only — no experimental reruns. Do not hardcode values here; read from CSV.
"""

import csv
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT    = os.path.join(os.path.dirname(__file__), "..")
DATA    = os.path.join(ROOT, "data", "fig03_value_direction_evidence.csv")
OUT_PDF = os.path.join(ROOT, "figures", "fig03_value_direction_evidence.pdf")
OUT_PNG = os.path.join(ROOT, "figures", "fig03_value_direction_evidence.png")

layer_labels = []
walk_f, walk_s, jump_f, jump_s = [], [], [], []

head_labels, head_wf = [], []
head_key = []

with open(DATA) as f:
    for r in csv.DictReader(f):
        if r["panel"] == "layer_cosine":
            layer_labels.append(r["x_label"])
            walk_f.append(float(r["walk_fail"]))
            walk_s.append(float(r["walk_success"]))
            jump_f.append(float(r["jump_fail"]))
            jump_s.append(float(r["jump_success"]))
        elif r["panel"] == "top5_heads":
            head_labels.append(r["x_label"])
            head_wf.append(float(r["walk_fail"]))
            head_key.append(r["is_key_head"].strip() == "1")

fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.4))

# --- Panel A: layer-wise cosine series ---
ax0 = axes[0]
ax0.plot(layer_labels, walk_f, color="#c0392b", marker="o",
         lw=2.0, markersize=6, label="I_WALK cos (fail)", zorder=4)
ax0.plot(layer_labels, walk_s, color="#e07070", marker="o",
         lw=1.5, markersize=5, linestyle="--", label="I_WALK cos (success)", zorder=4)
ax0.plot(layer_labels, jump_f, color="#1a5276", marker="s",
         lw=2.0, markersize=6, label="I_JUMP cos (fail)", zorder=4)
ax0.plot(layer_labels, jump_s, color="#5dade2", marker="s",
         lw=1.5, markersize=5, linestyle="--", label="I_JUMP cos (success)", zorder=4)

ax0.axhline(0, color="black", lw=0.7, zorder=3)
ax0.set_ylabel("Cosine similarity to output embedding")
ax0.set_title("(A) Residual stream alignment by decoder position\n"
              "I_WALK cosine (fail) peaks at L5 — "
              "higher than success and than I_JUMP (fail)",
              fontsize=8.8)
ax0.legend(fontsize=7.5, loc="upper left")
ax0.yaxis.grid(True, lw=0.4, color="#eeeeee", zorder=0)
ax0.set_axisbelow(True)

# annotate L5 peak
peak_idx = walk_f.index(max(walk_f))
ax0.annotate(f"walk_fail\npeak\n{walk_f[peak_idx]:.3f}",
             xy=(layer_labels[peak_idx], walk_f[peak_idx]),
             xytext=(peak_idx - 1.2, 0.19),
             fontsize=7, color="#c0392b",
             arrowprops=dict(arrowstyle="->", color="#c0392b", lw=0.8))

# --- Panel B: top-5 head walk_fail bar chart ---
ax1 = axes[1]
n = len(head_labels)
x = np.arange(n)
bar_colors = ["#c0392b" if k else "#7f8c8d" for k in head_key]

bars = ax1.bar(x, head_wf, color=bar_colors, zorder=3,
               edgecolor="white", linewidth=0.8)
for bar, val, k in zip(bars, head_wf, head_key):
    ypos = val + 0.003
    ax1.text(bar.get_x() + bar.get_width() / 2, ypos,
             f"{val:.4f}", ha="center", va="bottom",
             fontsize=8.5, fontweight="bold" if k else "normal")

ax1.set_xticks(x)
ax1.set_xticklabels(head_labels, fontsize=10)
ax1.set_ylabel("Cosine similarity to I_WALK embedding\n(fail group, action slot)")
ax1.set_title("(B) Top-5 heads: I_WALK value-direction evidence\n"
              "Red = repair heads from S-061 (L4H6, L5H5). "
              "Causal role: Fig 6 / §7.",
              fontsize=8.8)
ax1.yaxis.grid(True, lw=0.4, color="#eeeeee", zorder=0)
ax1.set_axisbelow(True)
ax1.set_ylim(0, 0.185)

ax1.text(0.97, 0.97,
         "n_fail=227 slots, n_success=144 slots\n"
         "Gini=0.450 — concentrated head contribution\n"
         "These are value-direction fingerprints,\nnot causal attribution (see S-061)",
         ha="right", va="top", transform=ax1.transAxes,
         fontsize=7, color="#555555",
         bbox=dict(boxstyle="round,pad=0.3", facecolor="#f9f9f9",
                   edgecolor="#cccccc", lw=0.7))

fig.suptitle(
    "Layer and head value-direction evidence — T5-small SCAN add_prim_jump\n"
    "Source: S-051 (S-051_LEVER_ARM_DISSECTION_V0.1.0). "
    "Top-5 heads by walk_fail cosine (option_c). "
    "Layer series from option_b. Not a 6×8 head heat map.",
    fontsize=8.2, y=1.02
)

plt.tight_layout()
plt.savefig(OUT_PDF, bbox_inches="tight", dpi=300)
plt.savefig(OUT_PNG, bbox_inches="tight", dpi=150)
print(f"Saved: {OUT_PDF}")
print(f"Saved: {OUT_PNG}")
