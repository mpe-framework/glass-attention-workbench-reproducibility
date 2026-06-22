#!/usr/bin/env python3
"""
S-047_BIT_FAILURE_DECOMPOSITION.py — Per-Bit Failure Decomposition
Applied Categorical Physics Workbench
Troy Teno | May 2026 | Open Access

Pre-registered: workbench/proposals/S-047_BIT_FAILURE_DECOMPOSITION_PROPOSAL.md
DO NOT modify hypotheses after seeing results.

─────────────────────────────────────────────────────────────────────
NO GPU REQUIRED. Runs on CPU in < 1 minute.
Uses artifacts from S-046 — no T5 re-encoding needed.

REQUIRES:
  046_encoder_reps.npy        — (200, 512) frozen layer 5 mean-pool
  046_structural_bits.json    — ground truth bits + correct/fail labels
  043_family_ab_results.json  — original family_b_jump sample

QUESTION: S-046 found AUC=0.88 for a linear failure classifier on the
raw 512-dim encoder representations. Which of the 8 structural bits
drive that signal? The answer determines the S-048 intervention target.

H1: At least one bit's probe direction achieves AUC ≥ 0.70 as failure predictor
H2: Failure rate increases monotonically with structural complexity (Spearman r > 0)
H3: has_after shows the highest per-bit failure lift of all 8 bits
H4: has_after=1 AND has_around=1 commands fail at ≥ 80%
─────────────────────────────────────────────────────────────────────
"""

import numpy as np
import json
import os
import sys
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
from scipy.stats import chi2_contingency, spearmanr

SCRIPT_ID   = "S-047_BIT_FAILURE_DECOMP_V0.1.0"
REPS_FILE   = "046_encoder_reps.npy"
BITS_FILE   = "046_structural_bits.json"
FAMILY_FILE = "043_family_ab_results.json"

# ── Google Drive persistence ───────────────────────────────────────────────────
DRIVE_DIR = None
try:
    from google.colab import drive as _drive
    _drive.mount("/content/drive", force_remount=False)
    DRIVE_DIR = "/content/drive/MyDrive/workbench_artifacts"
    os.makedirs(DRIVE_DIR, exist_ok=True)
    import shutil as _shutil
    for _fname in (REPS_FILE, BITS_FILE, FAMILY_FILE):
        _src = os.path.join(DRIVE_DIR, _fname)
        if not os.path.exists(_fname) and os.path.exists(_src):
            _shutil.copy(_src, _fname)
            print(f"  Restored from Drive: {_fname}")
    print(f"  Drive mounted. Artifacts will be saved to: {DRIVE_DIR}")
except Exception:
    print("  Drive not available. Saving to local /content/ only.")

def save_to_drive(*filenames):
    if DRIVE_DIR is None:
        return
    import shutil
    for fname in filenames:
        if os.path.exists(fname):
            shutil.copy(fname, os.path.join(DRIVE_DIR, fname))

BIT_NAMES = [
    "has_direction", "is_left", "has_around", "has_opposite",
    "has_twice", "has_thrice", "has_and", "has_after",
]
IDX = {name: i for i, name in enumerate(BIT_NAMES)}

print(f"\n{'='*70}")
print(f"  {SCRIPT_ID}")
print(f"  Per-Bit Failure Decomposition")
print(f"  Which structural bits drive the S-046 AUC=0.88 failure signal?")
print(f"{'='*70}")

# ── Prerequisites ─────────────────────────────────────────────────────────────
for f in (REPS_FILE, BITS_FILE, FAMILY_FILE):
    if not os.path.exists(f):
        sys.exit(f"\nERROR: {f!r} not found.\nRun S-046 first to generate this artifact.")

# ── Load ────────────────────────────────────────────────────────────────────────
X = np.load(REPS_FILE)                          # (200, 512)
with open(BITS_FILE) as f:
    bits_data = json.load(f)
with open(FAMILY_FILE) as f:
    _ = json.load(f)    # sanity load only

examples      = bits_data["examples"]
bits_matrix   = np.array([ex["bits"] for ex in examples])      # (200, 8)
correct_labels = np.array([int(ex["correct"]) for ex in examples])  # (200,)
N = len(correct_labels)
n_fail = int((correct_labels == 0).sum())
n_succ = int((correct_labels == 1).sum())

print(f"\n  Loaded {N} examples  ({n_fail} failures, {n_succ} successes)")
print(f"  Overall failure rate: {100*n_fail/N:.1f}%")

scaler   = StandardScaler()
X_scaled = scaler.fit_transform(X)
cv       = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# ══════════════════════════════════════════════════════════════════════════════
print(f"\n{'='*70}")
print(f"  PHASE 1 — PER-BIT FAILURE RATES")
print(f"{'='*70}")
print(f"\n  For each bit: failure rate when bit=0 vs bit=1, chi-square test.")
print(f"\n  {'Bit':<20} {'N(=0)':>6} {'Fail%0':>7} {'N(=1)':>6} {'Fail%1':>7} "
      f"{'Lift':>7} {'chi2 p':>9}")
print(f"  {'-'*70}")

per_bit_stats = {}
for i, name in enumerate(BIT_NAMES):
    y_bit  = bits_matrix[:, i]
    mask0  = (y_bit == 0)
    mask1  = (y_bit == 1)
    n0, n1 = int(mask0.sum()), int(mask1.sum())
    fail0  = float((1 - correct_labels[mask0]).mean()) if n0 > 0 else float("nan")
    fail1  = float((1 - correct_labels[mask1]).mean()) if n1 > 0 else float("nan")
    lift   = fail1 - fail0 if not (np.isnan(fail0) or np.isnan(fail1)) else float("nan")

    # Chi-square 2×2: bit value vs outcome
    if n0 > 0 and n1 > 0:
        table = np.array([
            [int((1 - correct_labels[mask0]).sum()), int(correct_labels[mask0].sum())],
            [int((1 - correct_labels[mask1]).sum()), int(correct_labels[mask1].sum())],
        ])
        # Avoid chi2 if any cell is zero (use continuity correction)
        chi2_val, p_chi2, _, _ = chi2_contingency(table, correction=True)
        p_chi2 = float(p_chi2)
    else:
        p_chi2 = float("nan")

    per_bit_stats[name] = {
        "n_bit0": n0, "n_bit1": n1,
        "fail_rate_bit0": fail0, "fail_rate_bit1": fail1,
        "lift": float(lift), "chi2_p": p_chi2,
    }
    print(f"  {name:<20} {n0:>6}  {100*fail0:>6.1f}%  {n1:>6}  {100*fail1:>6.1f}%  "
          f"{lift:>+6.3f}  {p_chi2:>9.4f}")

# ══════════════════════════════════════════════════════════════════════════════
print(f"\n{'='*70}")
print(f"  PHASE 2 — PER-PROBE-DIRECTION FAILURE AUC")
print(f"{'='*70}")
print(f"\n  Fit probe for each bit, project X onto probe weight vector,")
print(f"  measure AUC of that scalar as a predictor of T5 failure.")
print(f"  High AUC means failure lives in the direction that encodes this bit.")
print(f"\n  {'Bit':<20} {'Direction AUC':>14}  (symmetrized)")
print(f"  {'-'*40}")

per_bit_auc = {}
for i, name in enumerate(BIT_NAMES):
    y_bit = bits_matrix[:, i]
    if y_bit.sum() == 0 or y_bit.sum() == N:
        per_bit_auc[name] = {"direction_auc": float("nan"), "skip": True}
        print(f"  {name:<20}  (constant — skipped)")
        continue

    probe = LogisticRegression(C=1.0, max_iter=1000, random_state=42)
    probe.fit(X_scaled, y_bit)

    # Project each example onto the probe weight direction
    scores = probe.decision_function(X_scaled)   # positive → bit=1 predicted

    # AUC of probe score as a failure predictor
    auc = roc_auc_score(1 - correct_labels, scores)
    auc = max(auc, 1 - auc)    # symmetrize (direction may flip)

    per_bit_auc[name] = {"direction_auc": float(auc), "skip": False}
    print(f"  {name:<20}  {auc:>14.3f}")

# ══════════════════════════════════════════════════════════════════════════════
print(f"\n{'='*70}")
print(f"  PHASE 3 — COMPLEXITY SCORE ANALYSIS")
print(f"{'='*70}")

complexity = bits_matrix.sum(axis=1).astype(int)   # (N,)
r, p_r = spearmanr(complexity, 1 - correct_labels)

print(f"\n  Spearman r(complexity_score, failure): r={r:.3f}  p={p_r:.4f}")
print(f"\n  Failure rate by number of active bits:")
print(f"  {'Score':>6} {'N':>5} {'Failures':>9} {'Fail%':>7}")
print(f"  {'-'*32}")

complexity_table = {}
for score in sorted(set(complexity.tolist())):
    mask  = (complexity == score)
    n     = int(mask.sum())
    nfail = int((1 - correct_labels[mask]).sum())
    rate  = float(nfail / n) if n > 0 else float("nan")
    complexity_table[score] = {"n": n, "n_fail": nfail, "fail_rate": rate}
    print(f"  {score:>6} {n:>5} {nfail:>9}  {100*rate:>6.1f}%")

h2_pass = bool(r > 0)
scores_sorted = sorted(complexity_table.keys())
# Check strict monotone (ignoring ties in fail_rate)
monotone = all(
    complexity_table[scores_sorted[j]]["fail_rate"]
    <= complexity_table[scores_sorted[j+1]]["fail_rate"]
    for j in range(len(scores_sorted) - 1)
    if complexity_table[scores_sorted[j]]["n"] > 0
    and complexity_table[scores_sorted[j+1]]["n"] > 0
)
print(f"\n  Strictly monotone: {'Yes' if monotone else 'No'}")
print(f"  H2: {'✓ CONFIRMED' if h2_pass else '✗ FAILED'}  (Spearman r={r:.3f} > 0)")

# ══════════════════════════════════════════════════════════════════════════════
print(f"\n{'='*70}")
print(f"  PHASE 4 — KEY BIT COMBINATIONS")
print(f"{'='*70}")
print(f"\n  Failure rates for key structural combinations.")
print(f"\n  {'Combination':<36} {'N':>5} {'Fail':>6} {'Fail%':>7}")
print(f"  {'-'*60}")

combinations = [
    ("has_after=1 + has_around=1",
        lambda b: b[IDX["has_after"]]==1 and b[IDX["has_around"]]==1),
    ("has_after=1 + has_opposite=1",
        lambda b: b[IDX["has_after"]]==1 and b[IDX["has_opposite"]]==1),
    ("has_after=1  (any around)",
        lambda b: b[IDX["has_after"]]==1),
    ("has_around=1 (no after)",
        lambda b: b[IDX["has_around"]]==1 and b[IDX["has_after"]]==0),
    ("has_opposite=1 (no after)",
        lambda b: b[IDX["has_opposite"]]==1 and b[IDX["has_after"]]==0),
    ("has_and=1   (no after)",
        lambda b: b[IDX["has_and"]]==1 and b[IDX["has_after"]]==0),
    ("has_thrice=1 (no after)",
        lambda b: b[IDX["has_thrice"]]==1 and b[IDX["has_after"]]==0),
    ("has_twice=1  (no after)",
        lambda b: b[IDX["has_twice"]]==1 and b[IDX["has_after"]]==0),
    ("has_after=0  (no reversal)",
        lambda b: b[IDX["has_after"]]==0),
]

combo_results = {}
for label, cond in combinations:
    mask  = np.array([cond(bits_matrix[j]) for j in range(N)])
    n     = int(mask.sum())
    if n == 0:
        combo_results[label] = {"n": 0, "n_fail": 0, "fail_rate": float("nan")}
        print(f"  {label:<36} {n:>5}   (none)")
        continue
    nfail = int((1 - correct_labels[mask]).sum())
    rate  = float(nfail / n)
    combo_results[label] = {"n": n, "n_fail": nfail, "fail_rate": rate}
    print(f"  {label:<36} {n:>5} {nfail:>6}  {100*rate:>6.1f}%")

# ══════════════════════════════════════════════════════════════════════════════
# Hypothesis verdicts
lifts = {name: per_bit_stats[name]["lift"] for name in BIT_NAMES}
h3_best_bit = max(lifts, key=lifts.get)
h3_pass = bool(h3_best_bit == "has_after")

aucs = {name: per_bit_auc[name]["direction_auc"]
        for name in BIT_NAMES
        if not per_bit_auc.get(name, {}).get("skip", False)}
h1_max  = max(aucs.values()) if aucs else 0.0
h1_best = max(aucs, key=aucs.get) if aucs else "none"
h1_pass = bool(h1_max >= 0.70)

h4_rate = combo_results.get("has_after=1 + has_around=1", {}).get("fail_rate", 0.0)
h4_n    = combo_results.get("has_after=1 + has_around=1", {}).get("n", 0)
h4_pass = bool(h4_rate >= 0.80 and h4_n > 0)

# ══════════════════════════════════════════════════════════════════════════════
print(f"\n{'='*70}")
print(f"  PHASE 5 — RANKED SUMMARY")
print(f"{'='*70}")

# Sort bits by failure lift
print(f"\n  Bits ranked by failure lift (bit=1 fail% − bit=0 fail%):")
print(f"  {'Bit':<20} {'Lift':>7}  {'Direction AUC':>14}")
print(f"  {'-'*46}")
for name in sorted(BIT_NAMES, key=lambda n: lifts[n], reverse=True):
    l = lifts[name]
    a = per_bit_auc.get(name, {}).get("direction_auc", float("nan"))
    print(f"  {name:<20} {l:>+7.3f}  {a:>14.3f}")

# ══════════════════════════════════════════════════════════════════════════════
# Save
output = {
    "N": N, "n_failures": n_fail, "n_successes": n_succ,
    "per_bit_stats":    per_bit_stats,
    "per_bit_auc":      per_bit_auc,
    "complexity_analysis": {
        "spearman_r": float(r), "spearman_p": float(p_r),
        "monotone": bool(monotone),
        "table": {str(k): v for k, v in complexity_table.items()},
    },
    "combination_results": combo_results,
    "h1": {"pass": bool(h1_pass), "best_bit": h1_best, "best_auc": float(h1_max)},
    "h2": {"pass": bool(h2_pass), "spearman_r": float(r), "monotone": bool(monotone)},
    "h3": {"pass": bool(h3_pass), "best_lift_bit": h3_best_bit,
           "best_lift": float(lifts[h3_best_bit])},
    "h4": {"pass": bool(h4_pass), "fail_rate": float(h4_rate), "n": int(h4_n)},
}
with open("047_bit_failure_decomp.json", "w") as f:
    json.dump(output, f, indent=2)
save_to_drive("047_bit_failure_decomp.json", "047_verdict.txt")

verdict_lines = [
    f"VERDICT — {SCRIPT_ID}", "",
    f"N={N}  ({n_fail} failures, {n_succ} successes)", "",
    f"H1: {'CONFIRMED' if h1_pass else 'FAILED'}  "
        f"best probe direction AUC = {h1_max:.3f} ({h1_best})",
    f"H2: {'CONFIRMED' if h2_pass else 'FAILED'}  "
        f"complexity Spearman r = {r:.3f}  p = {p_r:.4f}",
    f"H3: {'CONFIRMED' if h3_pass else 'FAILED'}  "
        f"highest failure lift = {h3_best_bit} ({lifts[h3_best_bit]:+.3f})",
    f"H4: {'CONFIRMED' if h4_pass else 'FAILED'}  "
        f"has_after+has_around fail rate = {100*h4_rate:.1f}% (N={h4_n})",
    "",
    f"Saved: 047_bit_failure_decomp.json",
]
with open("047_verdict.txt", "w") as f:
    f.write("\n".join(verdict_lines) + "\n")

# ══════════════════════════════════════════════════════════════════════════════
print(f"\n{'='*70}")
print(f"  VERDICT SUMMARY — {SCRIPT_ID}")
print(f"{'='*70}")
print(f"\n  {'✓' if h1_pass else '✗'} H1  Best probe direction AUC = {h1_max:.3f}  ({h1_best})")
print(f"  {'✓' if h2_pass else '✗'} H2  Complexity Spearman r = {r:.3f}  p = {p_r:.4f}")
print(f"  {'✓' if h3_pass else '✗'} H3  Highest failure lift: {h3_best_bit} "
      f"({lifts[h3_best_bit]:+.3f})")
print(f"  {'✓' if h4_pass else '✗'} H4  has_after+has_around fail rate = "
      f"{100*h4_rate:.1f}%  (N={h4_n})")
print()

n_pass = sum([h1_pass, h2_pass, h3_pass, h4_pass])
print(f"  {n_pass}/4 hypotheses confirmed.")
print()
if h3_pass and h1_pass:
    print(f"  has_after drives both the failure lift and the probe-direction AUC.")
    print(f"  S-048 target: auxiliary loss on has_after"
          + (" + has_around" if h4_pass else "") + " during T5 fine-tuning.")
elif h2_pass and not h3_pass:
    print(f"  Complexity is predictive but {h3_best_bit} (not has_after) dominates.")
    print(f"  S-048 target: auxiliary loss on {h3_best_bit}.")
elif not h2_pass and not h1_pass:
    print(f"  K1 fires: failure is not explained by the 8 structural bits individually.")
    print(f"  The AUC=0.88 comes from geometry beyond the bit set. Investigate.")
else:
    print(f"  Partial confirmation. Review per-bit ranked table above for S-048 target.")

print(f"\n  Saved: 047_bit_failure_decomp.json  047_verdict.txt")
print(f"\n{'='*70}")
print(f"  END {SCRIPT_ID}")
print(f"{'='*70}\n")
