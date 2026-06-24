#!/usr/bin/env python3
"""
S-045_PER_TOKEN_BC.py — Per-Token BC Defect at Jump Position
Applied Categorical Physics Workbench
Troy Teno | May 2026 | Open Access

Pre-registered: workbench/proposals/S-045_PER_TOKEN_BC_PROPOSAL.md
Validated at toy scale by G-047 (morphism-type distinction finding).
DO NOT modify hypotheses after seeing results.

─────────────────────────────────────────────────────────────────────
SETUP (run once in Colab):
  pip install torch transformers sentencepiece numpy scipy -q

REQUIRES (both must be present in working directory or Drive):
  043_t5_scan_checkpoint/    — fine-tuned T5 weights from S-043
  043_family_ab_results.json — per-example BC data from S-043/S-044

If the checkpoint was lost (Colab session expiry), re-run S-043 Phase 3
(training only, ~30-60 min on T4). Save to Drive so this doesn't repeat:
  !cp -r 043_t5_scan_checkpoint /content/drive/MyDrive/

Runtime: ~5-15 min on T4 GPU (encoder forward passes only, no generation)
─────────────────────────────────────────────────────────────────────

WHAT THIS SCRIPT ASKS:
  S-044 showed that mean-pool BC defect does not separate failures from
  successes within the jump-compound group (H1 null, p=0.9999, d=−0.410).
  The direction was reversed: successes had higher mean-pool distance.
  Diagnosis: mean-pool is confounded by jump-token density — commands
  with more jump tokens relative to total length score higher, but those
  tend to be shorter commands T5 gets right.

  G-047 (geometry track) confirmed the confound mechanism at toy scale
  and identified the fix: cross-sequence per-token measurement at the
  jump token position eliminates the count confound. The S-track already
  uses cross-sequence (separate forward passes for "jump X" and "walk X").
  The only change: extract the hidden state at the jump/walk position(s)
  instead of mean-pooling over the whole sequence.

  H1: per-token defect at layer 5 is higher for failures than successes.
  H2/H3: per-token defect is NOT correlated with command length or n_jumps.
  H4: monotone per-token layer profile for failures (layers 1→5).
"""

import numpy as np
import json
import os
import sys
from scipy import stats

SCRIPT_ID     = "S-045_PER_TOKEN_BC_V0.1.0"
N_LAYERS      = 6
PEAK_LAYER    = 4          # 0-indexed; layer 5 (1-indexed), peak from S-043
MAX_INPUT_LEN = 50
CHECKPOINT_DIR = "043_t5_scan_checkpoint"
DATA_FILE      = "043_family_ab_results.json"

print(f"\n{'='*70}")
print(f"  {SCRIPT_ID}")
print(f"  Per-Token BC Defect at Jump Position: Failures vs Successes")
print(f"  Cross-sequence measurement — corrected instrument for S-044 H1")
print(f"{'='*70}")

# ── Prerequisites ─────────────────────────────────────────────────────────────
if not os.path.exists(DATA_FILE):
    sys.exit(
        f"\nERROR: {DATA_FILE!r} not found.\n"
        f"Run S-043 first, or restore from Drive:\n"
        f"  !cp -r /content/drive/MyDrive/043_family_ab_results.json ."
    )

if not os.path.exists(CHECKPOINT_DIR):
    sys.exit(
        f"\nERROR: Checkpoint {CHECKPOINT_DIR!r} not found.\n"
        f"Re-run S-043 Phase 3 (training, ~30-60 min), or restore:\n"
        f"  !cp -r /content/drive/MyDrive/043_t5_scan_checkpoint ."
    )

# ── Load S-043/S-044 artifacts ────────────────────────────────────────────────
print(f"\n  Loading {DATA_FILE!r}...")
with open(DATA_FILE) as f:
    data = json.load(f)

family_b_jump = data["family_b_jump"]
N_total   = len(family_b_jump)
n_fail    = sum(1 for r in family_b_jump if not r["correct"])
n_success = sum(1 for r in family_b_jump if r["correct"])

print(f"  {N_total} jump-compound examples: {n_fail} failures, {n_success} successes")

if n_fail < 5 or n_success < 5:
    sys.exit(f"\nInsufficient data: need ≥5 per group.")

# ── Load T5 ───────────────────────────────────────────────────────────────────
print(f"\n  Loading T5 from {CHECKPOINT_DIR!r}...")
try:
    import torch
    from transformers import T5ForConditionalGeneration, T5Tokenizer
except ImportError as e:
    sys.exit(f"\nMissing library: {e}\nRun: pip install torch transformers sentencepiece")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"  Device: {device}")

tokenizer = T5Tokenizer.from_pretrained(CHECKPOINT_DIR)
model     = T5ForConditionalGeneration.from_pretrained(CHECKPOINT_DIR).to(device)
model.eval()
print(f"  Loaded. Encoder: {N_LAYERS} layers, d=512")

# ── Identify jump/walk token IDs ──────────────────────────────────────────────
# Encode known-good examples to find the token IDs reliably.
# "jump" and "walk" are common words; each should be a single SentencePiece token.
_jids = tokenizer.encode("jump twice", add_special_tokens=False)
_wids = tokenizer.encode("walk twice", add_special_tokens=False)
jump_id = _jids[0]
walk_id = _wids[0]

print(f"\n  Token IDs:")
print(f"    'jump' → id={jump_id}  decoded={tokenizer.decode([jump_id])!r}")
print(f"    'walk' → id={walk_id}  decoded={tokenizer.decode([walk_id])!r}")

if len(_jids) < 1 or len(_wids) < 1:
    sys.exit("ERROR: 'jump' or 'walk' tokenization failed.")

# ── Helpers ───────────────────────────────────────────────────────────────────
def get_hidden_states(text):
    """Run T5 encoder. Returns (input_ids list, list of N_LAYERS arrays (seq_len, 512))."""
    toks = tokenizer(
        text, return_tensors="pt",
        max_length=MAX_INPUT_LEN, truncation=True
    ).to(device)
    with torch.no_grad():
        enc = model.encoder(
            input_ids=toks["input_ids"],
            attention_mask=toks["attention_mask"],
            output_hidden_states=True,
            return_dict=True,
        )
    ids    = toks["input_ids"][0].tolist()
    hidden = [h[0].cpu().numpy() for h in enc.hidden_states[1:]]
    return ids, hidden


def per_token_bc_dist(cmd_jump, cmd_walk):
    """
    Cross-sequence per-token BC defect.

    Runs cmd_jump and cmd_walk as separate forward passes.
    Finds jump positions in cmd_jump tokens, walk positions in cmd_walk tokens.
    Returns mean L2 distance across those positions per layer.

    Returns: (dists, n_jump_tokens, n_total_tokens) or None on alignment failure.
    """
    ids_j, h_j = get_hidden_states(cmd_jump)
    ids_w, h_w = get_hidden_states(cmd_walk)

    jump_positions = [i for i, tid in enumerate(ids_j) if tid == jump_id]
    walk_positions = [i for i, tid in enumerate(ids_w) if tid == walk_id]

    # Need at least one jump position and matching counts
    if not jump_positions:
        return None
    if len(jump_positions) != len(walk_positions):
        return None
    # Positions should be identical (only token content changes, not structure)
    if jump_positions != walk_positions:
        return None

    dists = np.zeros(N_LAYERS)
    for l in range(N_LAYERS):
        layer_dists = [
            float(np.linalg.norm(h_j[l][jp] - h_w[l][wp]))
            for jp, wp in zip(jump_positions, walk_positions)
        ]
        dists[l] = np.mean(layer_dists)

    return dists, len(jump_positions), len(ids_j)

# ── Phase 1: Compute per-token BC defect ─────────────────────────────────────
print(f"\n{'='*70}")
print(f"  PHASE 1 — COMPUTE PER-TOKEN BC DEFECT ({N_total} examples)")
print(f"{'='*70}")
print(f"\n  Encoder forward passes only — no generation. "
      f"Expect ~5-15 min on T4.")

results   = []
n_skipped = 0

for idx, r in enumerate(family_b_jump):
    if idx % 50 == 0:
        print(f"    {idx}/{N_total}...")

    out = per_token_bc_dist(r["cmd_jump"], r["cmd_walk"])

    if out is None:
        n_skipped += 1
        continue

    pt_dists, n_jumps, n_tokens = out

    results.append({
        "cmd_jump":          r["cmd_jump"],
        "cmd_walk":          r["cmd_walk"],
        "correct":           r["correct"],
        "n_jumps":           n_jumps,
        "n_tokens":          n_tokens,
        "pt_dist":           pt_dists.tolist(),
        "pt_dist_peak":      float(pt_dists[PEAK_LAYER]),
        "mean_pool_dist_s044": r["dist"][PEAK_LAYER],  # S-044 comparison
    })

print(f"\n  Done. Processed: {len(results)}/{N_total}  Skipped: {n_skipped}")
if n_skipped > 10:
    print(f"  WARNING: {n_skipped} skipped — check tokenizer alignment.")

# Early save before any analysis that could crash
with open("045_per_token_results.json", "w") as f:
    json.dump(results, f, indent=2)
print(f"  Early save: 045_per_token_results.json")

# ── Split by outcome ──────────────────────────────────────────────────────────
fail_r    = [r for r in results if not r["correct"]]
success_r = [r for r in results if r["correct"]]
n_f, n_s  = len(fail_r), len(success_r)

fail_pt    = np.array([r["pt_dist_peak"]        for r in fail_r])
success_pt = np.array([r["pt_dist_peak"]        for r in success_r])
fail_mp    = np.array([r["mean_pool_dist_s044"] for r in fail_r])
success_mp = np.array([r["mean_pool_dist_s044"] for r in success_r])

# ── Per-layer profile ─────────────────────────────────────────────────────────
print(f"\n{'='*70}")
print(f"  PHASE 2 — PER-LAYER PROFILE")
print(f"{'='*70}")
print(f"\n  {'Layer':>7} {'Fail mean':>12} {'Succ mean':>12} {'Diff':>10} {'Ratio':>8}")
print(f"  {'-'*55}")

for l in range(N_LAYERS):
    fm = np.mean([r["pt_dist"][l] for r in fail_r])
    sm = np.mean([r["pt_dist"][l] for r in success_r])
    marker = " ◄ peak" if l == PEAK_LAYER else ""
    print(f"  Layer {l+1:>2}  {fm:>12.4f}  {sm:>12.4f}  {fm-sm:>10.4f}  "
          f"{fm/(sm+1e-12):>8.4f}{marker}")

# ── H1: per-token failures > successes at peak layer ─────────────────────────
print(f"\n{'='*70}")
print(f"  H1: PER-TOKEN BC DEFECT AT LAYER {PEAK_LAYER+1} — FAILURES vs SUCCESSES")
print(f"{'='*70}")

print(f"\n  Failures  mean = {fail_pt.mean():.4f}  (n={n_f})")
print(f"  Successes mean = {success_pt.mean():.4f}  (n={n_s})")
print(f"  S-044 reference: fail mean-pool={fail_mp.mean():.2f}  "
      f"succ mean-pool={success_mp.mean():.2f}")

stat1, p1 = stats.mannwhitneyu(fail_pt, success_pt, alternative="greater")
h1_pass   = p1 < 0.05

print(f"\n  Mann-Whitney U = {stat1:.1f}   p = {p1:.4f}")
print(f"  H1: {'✓ CONFIRMED' if h1_pass else '✗ FAILED'}  "
      f"({'per-token defect is graded within failure category' if h1_pass else 'no within-group difference'})")

if h1_pass:
    print(f"\n  → Per-token BC defect at jump position separates failures from")
    print(f"    successes. The encoder's representation of the unseen primitive")
    print(f"    is graded — mean-pool washed out the signal. The information")
    print(f"    was present at the token level all along.")
else:
    print(f"\n  → Per-token BC defect does not separate failures from successes.")
    print(f"    The encoder marks all jump-compound examples uniformly even at")
    print(f"    the token level. K1 fires: decoder determines failure on")
    print(f"    decoder-side grounds, not encoder representational quality.")

# ── H2/H3: Confound checks ────────────────────────────────────────────────────
print(f"\n{'='*70}")
print(f"  H2/H3: CONFOUND CHECKS (per-token vs mean-pool)")
print(f"{'='*70}")

all_pt   = np.array([r["pt_dist_peak"]        for r in results])
all_mp   = np.array([r["mean_pool_dist_s044"] for r in results])
all_len  = np.array([r["n_tokens"]            for r in results])
all_njmp = np.array([r["n_jumps"]             for r in results])

r_pt_len,  _ = stats.spearmanr(all_pt, all_len)
r_pt_njmp, _ = stats.spearmanr(all_pt, all_njmp)
r_mp_len,  _ = stats.spearmanr(all_mp, all_len)
r_mp_njmp, _ = stats.spearmanr(all_mp, all_njmp)

print(f"\n  Spearman r with per-token defect (S-045):")
print(f"    vs command length (n_tokens): r = {r_pt_len:+.3f}")
print(f"    vs n_jump_tokens:             r = {r_pt_njmp:+.3f}")
print(f"\n  Spearman r with mean-pool defect (S-044, reference):")
print(f"    vs command length (n_tokens): r = {r_mp_len:+.3f}")
print(f"    vs n_jump_tokens:             r = {r_mp_njmp:+.3f}")

h2_pass = abs(r_pt_len)  < 0.2
h3_pass = abs(r_pt_njmp) < 0.2

print(f"\n  H2 (length confound eliminated): {'✓ CONFIRMED' if h2_pass else '✗ FAILED'}  "
      f"|r| = {abs(r_pt_len):.3f} ({'< 0.2' if h2_pass else '>= 0.2'})")
print(f"  H3 (count confound eliminated):  {'✓ CONFIRMED' if h3_pass else '✗ FAILED'}  "
      f"|r| = {abs(r_pt_njmp):.3f} ({'< 0.2' if h3_pass else '>= 0.2'})")

# ── H4: Monotone layer profile ────────────────────────────────────────────────
print(f"\n{'='*70}")
print(f"  H4: MONOTONE LAYER PROFILE (per-token, failures, layers 1-5)")
print(f"{'='*70}")

fail_means = np.array([
    np.mean([r["pt_dist"][l] for r in fail_r])
    for l in range(N_LAYERS - 1)   # exclude layer 6 (T5 collapse)
])
diffs   = np.diff(fail_means)
h4_pass = bool(np.all(diffs > 0))

print(f"\n  Layer 1-5 means (failures): " + "  ".join(f"{v:.2f}" for v in fail_means))
print(f"  Differences:                " + "  ".join(f"{v:+.2f}" for v in diffs))
print(f"  H4: {'✓ CONFIRMED' if h4_pass else '✗ FAILED'}  "
      f"({'monotone' if h4_pass else 'non-monotone'})")

# ── Effect size ───────────────────────────────────────────────────────────────
print(f"\n{'='*70}")
print(f"  EFFECT SIZE — Per-Token vs Mean-Pool Comparison at Layer {PEAK_LAYER+1}")
print(f"{'='*70}")

def cohens_d(a, b):
    n_a, n_b = len(a), len(b)
    if n_a + n_b < 3:
        return 0.0
    pooled = np.sqrt(
        (a.std()**2 * (n_a - 1) + b.std()**2 * (n_b - 1)) / (n_a + n_b - 2)
    )
    return (a.mean() - b.mean()) / (pooled + 1e-12)

d_pt = cohens_d(fail_pt, success_pt)
d_mp = cohens_d(fail_mp, success_mp)

def effect_label(d):
    a = abs(d)
    if a >= 0.8: return "large (≥ 0.8)"
    if a >= 0.5: return "medium (0.5-0.8)"
    if a >= 0.2: return "small (0.2-0.5)"
    return "negligible (< 0.2)"

print(f"\n  Per-token (S-045): d = {d_pt:+.3f}  [{effect_label(d_pt)}]")
print(f"  Mean-pool (S-044): d = {d_mp:+.3f}  [{effect_label(d_mp)}]  (reference: −0.410)")
if d_pt > d_mp:
    print(f"  Per-token is a stronger positive signal than mean-pool.")
elif d_pt < 0 and d_mp < 0:
    print(f"  Both directions are reversed (successes > failures).")

# ── Sample examples ───────────────────────────────────────────────────────────
print(f"\n{'='*70}")
print(f"  HIGH vs LOW PER-TOKEN DEFECT EXAMPLES (layer {PEAK_LAYER+1})")
print(f"{'='*70}")

sorted_fail = sorted(fail_r,    key=lambda r: r["pt_dist_peak"], reverse=True)
sorted_succ = sorted(success_r, key=lambda r: r["pt_dist_peak"], reverse=True)

print(f"\n  Top-3 failures (highest per-token defect, model wrong):")
for r in sorted_fail[:3]:
    print(f"    pt={r['pt_dist_peak']:.2f}  mp={r['mean_pool_dist_s044']:.2f}"
          f"  len={r['n_tokens']}  nj={r['n_jumps']}  {r['cmd_jump']}")

print(f"\n  Top-3 successes (highest per-token defect, model right):")
for r in sorted_succ[:3]:
    print(f"    pt={r['pt_dist_peak']:.2f}  mp={r['mean_pool_dist_s044']:.2f}"
          f"  len={r['n_tokens']}  nj={r['n_jumps']}  {r['cmd_jump']}")

print(f"\n  Bottom-3 failures (lowest per-token defect, model wrong):")
for r in sorted_fail[-3:]:
    print(f"    pt={r['pt_dist_peak']:.2f}  mp={r['mean_pool_dist_s044']:.2f}"
          f"  len={r['n_tokens']}  nj={r['n_jumps']}  {r['cmd_jump']}")

# ── Save verdict ──────────────────────────────────────────────────────────────
verdict = [
    f"VERDICT — {SCRIPT_ID}",
    f"",
    f"N failures:  {n_f}",
    f"N successes: {n_s}",
    f"N skipped:   {n_skipped}",
    f"",
    f"H1: {'✓ CONFIRMED' if h1_pass else '✗ FAILED'}  "
        f"p={p1:.4f}  fail={fail_pt.mean():.4f}  succ={success_pt.mean():.4f}",
    f"H2: {'✓ CONFIRMED' if h2_pass else '✗ FAILED'}  "
        f"length confound r={r_pt_len:.3f}  (mean-pool ref: {r_mp_len:.3f})",
    f"H3: {'✓ CONFIRMED' if h3_pass else '✗ FAILED'}  "
        f"count confound  r={r_pt_njmp:.3f}  (mean-pool ref: {r_mp_njmp:.3f})",
    f"H4: {'✓ CONFIRMED' if h4_pass else '✗ FAILED'}  "
        f"{'monotone' if h4_pass else 'non-monotone'} layer profile",
    f"",
    f"Cohen's d (per-token S-045): {d_pt:+.3f}",
    f"Cohen's d (mean-pool S-044): {d_mp:+.3f}",
]
with open("045_verdict.txt", "w") as f:
    f.write("\n".join(verdict) + "\n")
print(f"\n  Saved: 045_per_token_results.json  045_verdict.txt")

# ── VERDICT ───────────────────────────────────────────────────────────────────
print(f"\n{'='*70}")
print(f"  VERDICT SUMMARY — {SCRIPT_ID}")
print(f"{'='*70}")

print(f"\n  {'✓' if h1_pass else '✗'} H1  Per-token defect separates failures from successes  (p={p1:.4f})")
print(f"  {'✓' if h2_pass else '✗'} H2  Length confound eliminated  (r={r_pt_len:+.3f})")
print(f"  {'✓' if h3_pass else '✗'} H3  Count confound eliminated   (r={r_pt_njmp:+.3f})")
print(f"  {'✓' if h4_pass else '✗'} H4  Monotone layer profile 1→5")
print(f"\n  Per-token Cohen's d = {d_pt:+.3f}  (mean-pool S-044 reference: {d_mp:+.3f})")
print()

if h1_pass and h2_pass and h3_pass:
    print("  BC defect is graded at the per-token level within the failing")
    print("  category. Mean-pool (S-044) washed out the signal by diluting")
    print("  jump positions over the whole command sequence.")
    print()
    print("  The encoder's representation of the unseen primitive at the")
    print("  jump token position predicts per-example decoder failure.")
    print("  This is the strongest S-track result to date.")
    print()
    print("  Next: S-046 — causal bit probes on frozen T5 encoder layer 5.")
elif h1_pass and not (h2_pass and h3_pass):
    print("  H1 confirmed but confound check failed. The separation is real")
    print("  but may be partially driven by a residual length/count artifact.")
    print("  Investigate before treating as a clean result.")
elif not h1_pass and h2_pass and h3_pass:
    print("  K1 fires: per-token defect does not grade difficulty within the")
    print("  failing category, and confounds are confirmed eliminated.")
    print("  The encoder marks all jump-compound examples uniformly at the")
    print("  token level. Decoder failure is determined on decoder-side grounds.")
    print()
    print("  [Sargsyan attribution removed in public export — see RETIREMENTS_AND_METHOD_LESSONS.md]")
    print()
    print("  Next: S-046 — causal bit probes on frozen T5 encoder layer 5.")
    print("  If probe accuracy is high, the encoder contains the right bits")
    print("  but the decoder fails to compose them. in this condition.")
else:
    print("  Mixed result. Check confound correlations and n_skipped above.")
    print("  If K2 fired, investigate tokenization alignment before S-046.")

print(f"\n{'='*70}")
print(f"  END {SCRIPT_ID}")
print(f"{'='*70}\n")
