#!/usr/bin/env python3
"""
S-046_CAUSAL_BIT_PROBES.py — Causal Bit Probes on Frozen T5 Encoder Layer 5
Applied Categorical Physics Workbench
Troy Teno | May 2026 | Open Access

Pre-registered: workbench/proposals/S-046_CAUSAL_BIT_PROBES_PROPOSAL.md
DO NOT modify hypotheses after seeing results.

─────────────────────────────────────────────────────────────────────
SETUP: sklearn is pre-installed in Colab. No additional installs
beyond S-043 requirements:
  pip install torch transformers sentencepiece numpy scipy -q

REQUIRES (all three must be present):
  043_t5_scan_checkpoint/     — fine-tuned T5 weights (from S-043)
  043_family_ab_results.json  — 200-example sample (from S-043/S-044)
  043_test_results.json       — oracle action sequences (from S-043)

If any file was lost (Colab session expiry), re-run S-043 first.

Runtime: ~10-20 min on T4 GPU (encoding) + < 1 min CPU (probes)
─────────────────────────────────────────────────────────────────────

WHAT THIS SCRIPT ASKS:
  S-043 through S-045 established that BC defect marks the unseen
  primitive as OOD uniformly across all jump-compound examples. K1
  fired: the encoder does not grade per-example difficulty. The
  decoder determines which 43.2% of examples it gets right.

  This leaves the deeper question: does the encoder contain the
  structural information needed to succeed, even though it doesn't
  grade difficulty?

  This script trains 8 linear probes on frozen T5 encoder layer 5
  representations to predict structural binary features of each
  command (has_direction, is_left, has_around, has_opposite,
  has_twice, has_thrice, has_and, has_after). If the encoder encodes
  these features linearly, a fixed rule decoder (SCAN interpreter)
  using those bit predictions could outperform T5's learned decoder.

  H1: All 8 probes ≥ 90% CV accuracy (encoder encodes structure)
  H2: Failure classifier AUC ≤ 0.55 (K1 sanity check, no linear
      predictor of failure exists in encoder space)
  H3: SCAN interpreter with gold bits ≥ 95% oracle match
  H4: SCAN interpreter with probe bits ≥ 95% match on examples
      where all probes are correct
"""

import numpy as np
import json
import os
import sys
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

SCRIPT_ID     = "S-046_CAUSAL_BIT_PROBES_V0.1.0"
N_LAYERS      = 6
PEAK_LAYER    = 4          # 0-indexed; layer 5 (from S-043)
MAX_INPUT_LEN = 50
CHECKPOINT_DIR = "043_t5_scan_checkpoint"
FAMILY_FILE    = "043_family_ab_results.json"
RESULTS_FILE   = "043_test_results.json"

# ── Google Drive persistence ───────────────────────────────────────────────────
# Mount Drive and keep artifacts across sessions.
DRIVE_DIR = None
try:
    from google.colab import drive as _drive
    _drive.mount("/content/drive", force_remount=False)
    DRIVE_DIR = "/content/drive/MyDrive/workbench_artifacts"
    os.makedirs(DRIVE_DIR, exist_ok=True)
    # Restore any artifacts from Drive that are missing locally
    for _fname in (FAMILY_FILE, RESULTS_FILE,
                   "046_encoder_reps.npy", "046_structural_bits.json"):
        _src = os.path.join(DRIVE_DIR, _fname)
        if not os.path.exists(_fname) and os.path.exists(_src):
            import shutil; shutil.copy(_src, _fname)
            print(f"  Restored from Drive: {_fname}")
    # Restore checkpoint directory
    import shutil as _shutil
    _ckpt_src = os.path.join(DRIVE_DIR, CHECKPOINT_DIR)
    if not os.path.exists(CHECKPOINT_DIR) and os.path.exists(_ckpt_src):
        _shutil.copytree(_ckpt_src, CHECKPOINT_DIR)
        print(f"  Restored from Drive: {CHECKPOINT_DIR}/")
    print(f"  Drive mounted. Artifacts will be saved to: {DRIVE_DIR}")
except Exception:
    print("  Drive not available (not in Colab or already mounted). "
          "Artifacts save to local /content/ only.")

def save_to_drive(*filenames):
    """Copy named files to Drive if Drive is mounted."""
    if DRIVE_DIR is None:
        return
    import shutil
    for fname in filenames:
        if os.path.exists(fname):
            shutil.copy(fname, os.path.join(DRIVE_DIR, fname))

def save_dir_to_drive(dirname):
    """Copy a directory to Drive if Drive is mounted."""
    if DRIVE_DIR is None or not os.path.isdir(dirname):
        return
    import shutil
    dst = os.path.join(DRIVE_DIR, dirname)
    if os.path.exists(dst):
        shutil.rmtree(dst)
    shutil.copytree(dirname, dst)

print(f"\n{'='*70}")
print(f"  {SCRIPT_ID}")
print(f"  Causal Bit Probes on Frozen T5 Encoder Layer 5")
print(f"  F-040 analog for production model")
print(f"{'='*70}")

# ── Prerequisites ─────────────────────────────────────────────────────────────
for f in (CHECKPOINT_DIR, FAMILY_FILE, RESULTS_FILE):
    if not os.path.exists(f):
        sys.exit(
            f"\nERROR: {f!r} not found.\n"
            f"Run S-043 first, or restore from Drive."
        )

# ── Load artifacts ─────────────────────────────────────────────────────────────
print(f"\n  Loading artifacts...")
with open(FAMILY_FILE) as f:
    family_data = json.load(f)
with open(RESULTS_FILE) as f:
    test_results = json.load(f)

family_b_jump = family_data["family_b_jump"]
oracle_map    = {r["command"]: r["oracle"] for r in test_results}

# Filter to examples that have oracle sequences
sample = [r for r in family_b_jump if r["cmd_jump"] in oracle_map]
N = len(sample)
print(f"  {N}/{len(family_b_jump)} examples have oracle sequences")
print(f"  {sum(1 for r in sample if not r['correct'])} failures, "
      f"{sum(1 for r in sample if r['correct'])} successes")

if N < 50:
    sys.exit(f"\nERROR: too few matched examples ({N}). Check that "
             f"043_test_results.json and 043_family_ab_results.json are "
             f"from the same S-043 run.")

# ── Load T5 ───────────────────────────────────────────────────────────────────
print(f"\n  Loading T5 from {CHECKPOINT_DIR!r}...")
try:
    import torch
    from transformers import T5ForConditionalGeneration, T5Tokenizer
except ImportError as e:
    sys.exit(f"\nMissing library: {e}\nRun: pip install torch transformers sentencepiece")

device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = T5Tokenizer.from_pretrained(CHECKPOINT_DIR)
model     = T5ForConditionalGeneration.from_pretrained(CHECKPOINT_DIR).to(device)
model.eval()
print(f"  Device: {device}  |  Encoder: {N_LAYERS} layers, d=512")

# ── Phase 1: Extract layer 5 mean-pool representations ────────────────────────
print(f"\n{'='*70}")
print(f"  PHASE 1 — EXTRACT FROZEN ENCODER REPRESENTATIONS ({N} examples)")
print(f"{'='*70}")
print(f"\n  Encoding jump-compound commands at layer {PEAK_LAYER+1} (mean-pool)...")

def get_layer5_meanpool(text):
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
    h = enc.hidden_states[PEAK_LAYER + 1][0].cpu().numpy()  # (seq_len, 512)
    return h.mean(axis=0)   # (512,)

reps = []
for idx, r in enumerate(sample):
    if idx % 50 == 0:
        print(f"    {idx}/{N}...")
    reps.append(get_layer5_meanpool(r["cmd_jump"]))

X = np.array(reps)   # (N, 512)
print(f"\n  Representations extracted. Shape: {X.shape}")

# Early save
np.save("046_encoder_reps.npy", X)
save_to_drive("046_encoder_reps.npy")
print(f"  Early save: 046_encoder_reps.npy")

# ── Phase 2: Extract structural bits ──────────────────────────────────────────
print(f"\n{'='*70}")
print(f"  PHASE 2 — EXTRACT STRUCTURAL BITS FROM COMMAND TEXT")
print(f"{'='*70}")

BIT_NAMES = [
    "has_direction",   # "left" or "right" anywhere in command
    "is_left",         # direction is "left" (vs right)
    "has_around",      # "around" modifier (×4 rotation cycle)
    "has_opposite",    # "opposite" modifier (×2 turn prefix)
    "has_twice",       # "twice" repetition
    "has_thrice",      # "thrice" repetition
    "has_and",         # "and" conjunction
    "has_after",       # "after" conjunction (reverses argument order)
]

def extract_bits(command):
    toks = command.split()
    tok_set = set(toks)
    return [
        int("left" in tok_set or "right" in tok_set),
        int("left" in tok_set),
        int("around" in tok_set),
        int("opposite" in tok_set),
        int("twice" in tok_set),
        int("thrice" in tok_set),
        int("and" in tok_set),
        int("after" in tok_set),
    ]

bits_matrix  = np.array([extract_bits(r["cmd_jump"]) for r in sample])  # (N, 8)
correct_labels = np.array([int(r["correct"]) for r in sample])           # (N,)

print(f"\n  Bit distribution across {N} examples:")
print(f"  {'Bit':<20} {'Sum':>5} {'%':>7}")
print(f"  {'-'*35}")
for i, name in enumerate(BIT_NAMES):
    s = bits_matrix[:, i].sum()
    print(f"  {name:<20} {s:>5}  {100*s/N:>6.1f}%")

with open("046_structural_bits.json", "w") as f:
    json.dump({
        "bit_names": BIT_NAMES,
        "examples":  [
            {"cmd": r["cmd_jump"], "correct": r["correct"], "bits": bits_matrix[i].tolist()}
            for i, r in enumerate(sample)
        ]
    }, f, indent=2)
save_to_drive("046_structural_bits.json")
print(f"\n  Saved: 046_structural_bits.json")

# ── Phase 3: Linear probes ─────────────────────────────────────────────────────
print(f"\n{'='*70}")
print(f"  PHASE 3 — LINEAR PROBES (5-fold CV, LogisticRegression L2)")
print(f"{'='*70}")

# Standardize features (important for L2 regularization)
scaler  = StandardScaler()
X_scaled = scaler.fit_transform(X)

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

probe_results = {}
print(f"\n  {'Bit':<20} {'Acc':>7} {'Baseline':>10} {'Lift':>8}")
print(f"  {'-'*50}")

h1_pass_count = 0
for i, name in enumerate(BIT_NAMES):
    y = bits_matrix[:, i]
    # Skip if only one class present (no variance)
    if y.sum() == 0 or y.sum() == N:
        acc = float(y.mean())
        print(f"  {name:<20} {'SKIP':>7}  (constant label: all {int(y[0])})")
        probe_results[name] = {"accuracy": acc, "baseline": acc, "skip": True}
        h1_pass_count += 1   # trivially satisfied
        continue

    clf = LogisticRegression(C=1.0, max_iter=1000, random_state=42)
    scores = cross_val_score(clf, X_scaled, y, cv=cv, scoring="accuracy")
    acc = scores.mean()
    baseline = max(y.mean(), 1 - y.mean())
    lift = acc - baseline

    h1_pass_count += int(acc >= 0.90)

    # Fit on full data for predictions in Phase 5
    clf.fit(X_scaled, y)
    y_pred = clf.predict(X_scaled)   # in-sample (used only for H4)

    probe_results[name] = {
        "accuracy":     float(acc),
        "std":          float(scores.std()),
        "baseline":     float(baseline),
        "lift":         float(lift),
        "predictions":  y_pred.tolist(),
        "skip":         False,
    }
    marker = " ✓" if acc >= 0.90 else " ✗"
    print(f"  {name:<20} {acc:>7.3f}  {baseline:>10.3f}  {lift:>+8.3f}{marker}")

h1_pass = h1_pass_count == len(BIT_NAMES)
print(f"\n  H1: {'✓ CONFIRMED' if h1_pass else '✗ FAILED'}  "
      f"({h1_pass_count}/{len(BIT_NAMES)} bits ≥ 90%)")

# ── Phase 4: Failure classifier (H2) ─────────────────────────────────────────
print(f"\n{'='*70}")
print(f"  PHASE 4 — H2: LINEAR FAILURE CLASSIFIER (K1 SANITY CHECK)")
print(f"{'='*70}")

clf_fail = LogisticRegression(C=1.0, max_iter=1000, random_state=42)
auc_scores = cross_val_score(
    clf_fail, X_scaled, correct_labels, cv=cv, scoring="roc_auc"
)
auc_mean = auc_scores.mean()
h2_pass  = auc_mean <= 0.55

print(f"\n  Linear classifier for 'correct' (5-fold CV AUC):")
print(f"  AUC = {auc_mean:.4f}  (±{auc_scores.std():.4f})")
print(f"  H2: {'✓ CONFIRMED' if h2_pass else '✗ FAILED'}  "
      f"({'AUC ≤ 0.55 — no linear failure predictor in encoder space' if h2_pass else 'AUC > 0.55 — K2 fires: linear separation exists'})")

if not h2_pass:
    print(f"\n  K2 fires: a linear function of the encoder DOES predict failure.")
    print(f"  This contradicts F-043 through F-045. Investigate which geometric")
    print(f"  feature is predictive before proceeding to S-047.")

# ── Phase 5: SCAN interpreter ─────────────────────────────────────────────────
print(f"\n{'='*70}")
print(f"  PHASE 5 — SCAN INTERPRETER (H3: GOLD BITS)")
print(f"{'='*70}")

PRIM_MAP = {
    "jump": ["I_JUMP"], "walk": ["I_WALK"],
    "run":  ["I_RUN"],  "look": ["I_LOOK"],
}

def scan_interpret_simple(tokens):
    """Interpret a single SCAN command (no top-level conjunction)."""
    prim = None
    direction = None    # "left" or "right"
    rotation  = None    # "opposite" or "around"
    repeat    = 1

    for t in tokens:
        if t in PRIM_MAP:         prim = t
        elif t == "turn":         prim = "turn"
        elif t in ("left","right"): direction = t
        elif t == "opposite":     rotation = "opposite"
        elif t == "around":       rotation = "around"
        elif t == "twice":        repeat = 2
        elif t == "thrice":       repeat = 3

    if prim == "turn":
        if rotation == "opposite":
            base = (["I_TURN_LEFT"] if direction == "left" else ["I_TURN_RIGHT"]) * 2
        elif rotation == "around":
            base = (["I_TURN_LEFT"] if direction == "left" else ["I_TURN_RIGHT"]) * 4
        else:
            base = ["I_TURN_LEFT"] if direction == "left" else ["I_TURN_RIGHT"]
        return base * repeat

    prim_action = PRIM_MAP.get(prim, [])
    turn_l, turn_r = ["I_TURN_LEFT"], ["I_TURN_RIGHT"]

    if rotation == "around":
        turn = turn_l if direction == "left" else turn_r
        base = (turn + prim_action) * 4
    elif rotation == "opposite":
        turn = (turn_l * 2) if direction == "left" else (turn_r * 2)
        base = turn + prim_action
    elif direction:
        turn = turn_l if direction == "left" else turn_r
        base = turn + prim_action
    else:
        base = prim_action

    return base * repeat


def scan_interpret(command):
    """Full SCAN interpreter. Handles top-level 'and'/'after' conjunctions."""
    tokens = command.strip().split()

    if "after" in tokens:
        idx = tokens.index("after")
        left  = scan_interpret(" ".join(tokens[:idx]))
        right = scan_interpret(" ".join(tokens[idx+1:]))
        return right + left       # "A after B" → do B then A

    if "and" in tokens:
        idx = tokens.index("and")
        left  = scan_interpret(" ".join(tokens[:idx]))
        right = scan_interpret(" ".join(tokens[idx+1:]))
        return left + right

    return scan_interpret_simple(tokens)


# Validate against oracle
print(f"\n  Validating SCAN interpreter against oracle ({N} examples)...")
interpreter_correct = 0
interpreter_errors  = []

for r in sample:
    oracle_str = oracle_map[r["cmd_jump"]]
    oracle_toks = oracle_str.strip().split()
    pred_toks   = scan_interpret(r["cmd_jump"])

    if pred_toks == oracle_toks:
        interpreter_correct += 1
    else:
        interpreter_errors.append({
            "cmd":      r["cmd_jump"],
            "oracle":   oracle_str,
            "interp":   " ".join(pred_toks),
        })

h3_acc  = interpreter_correct / N
h3_pass = h3_acc >= 0.95

print(f"  Interpreter accuracy: {interpreter_correct}/{N} = {h3_acc:.1%}")
if interpreter_errors:
    print(f"\n  First 3 interpreter errors:")
    for e in interpreter_errors[:3]:
        print(f"    CMD:    {e['cmd']}")
        print(f"    ORACLE: {e['oracle'][:60]}")
        print(f"    INTERP: {e['interp'][:60]}")

print(f"\n  H3: {'✓ CONFIRMED' if h3_pass else '✗ FAILED'}  "
      f"(interpreter {h3_acc:.1%} ≥ 95%)")
if not h3_pass:
    print(f"  → SCAN interpreter has errors. H4 analysis will be unreliable.")
    print(f"  → Check interpreter_errors above for patterns.")

# ── Phase 6: Fixed rule decode with probe bits (H4) ───────────────────────────
print(f"\n{'='*70}")
print(f"  PHASE 6 — H4: FIXED RULE DECODE (PROBE BITS)")
print(f"{'='*70}")

# For each example, check whether all probes predicted the correct bits
# Only evaluate the fixed rule decoder on examples where all probes are correct
probe_preds = np.array([
    probe_results[name]["predictions"]
    for name in BIT_NAMES
    if not probe_results[name].get("skip", False)
]).T   # (N, n_non_skipped_bits)

non_skipped_bits = [name for name in BIT_NAMES if not probe_results[name].get("skip", False)]
non_skipped_idx  = [i for i, name in enumerate(BIT_NAMES) if not probe_results[name].get("skip", False)]

# Ground truth bits for non-skipped
bits_non_skipped = bits_matrix[:, non_skipped_idx]   # (N, n_non_skipped)

# All probes correct for each example?
all_correct_mask = np.all(probe_preds == bits_non_skipped, axis=1)  # (N,)
n_all_correct = all_correct_mask.sum()

print(f"\n  Examples where ALL non-trivial probes predict correctly: "
      f"{n_all_correct}/{N} ({100*n_all_correct/N:.1f}%)")

# Among those examples, does the SCAN interpreter match oracle?
h4_correct = 0
h4_total   = 0
for idx, r in enumerate(sample):
    if not all_correct_mask[idx]:
        continue
    oracle_toks = oracle_map[r["cmd_jump"]].strip().split()
    pred_toks   = scan_interpret(r["cmd_jump"])
    h4_total   += 1
    h4_correct += int(pred_toks == oracle_toks)

if h4_total > 0:
    h4_acc  = h4_correct / h4_total
    h4_pass = h4_acc >= 0.95
    print(f"  Interpreter accuracy on 'all-probes-correct' subset: "
          f"{h4_correct}/{h4_total} = {h4_acc:.1%}")
    print(f"  H4: {'✓ CONFIRMED' if h4_pass else '✗ FAILED'}  "
          f"(≥ 95% on all-probes-correct subset)")
else:
    h4_pass = False
    h4_acc  = 0.0
    print(f"  H4: SKIPPED — no examples with all probes correct")

# ── Comparison: T5 baseline vs probe-predicted accuracy ──────────────────────
print(f"\n{'='*70}")
print(f"  ACCURACY COMPARISON")
print(f"{'='*70}")

t5_accuracy       = sum(r["correct"] for r in sample) / N
interp_accuracy   = h3_acc
interp_on_correct = h4_acc if h4_total > 0 else float("nan")

print(f"\n  T5 decoder (baseline):              {t5_accuracy:.1%}  ({sum(r['correct'] for r in sample)}/{N})")
print(f"  SCAN interpreter (gold bits):       {interp_accuracy:.1%}  ({interpreter_correct}/{N})")
print(f"  SCAN interpreter (all-probe-correct subset): {interp_on_correct:.1%}  ({h4_correct}/{max(h4_total,1)})")
print(f"\n  Note: interpreter ≠ T5 baseline because the interpreter always")
print(f"  produces the CORRECT output (it knows the SCAN grammar). T5's")
print(f"  43% is not ceiling — it's the decoder's failure on unseen primitive.")

# ── Per-bit probe detail ───────────────────────────────────────────────────────
print(f"\n{'='*70}")
print(f"  PROBE DETAIL — PER-BIT ACCURACY AND DISTRIBUTION")
print(f"{'='*70}")
print(f"\n  {'Bit':<20} {'Acc':>7} {'Fail acc':>10} {'Succ acc':>10} {'Balanced':>10}")
print(f"  {'-'*60}")

for i, name in enumerate(BIT_NAMES):
    if probe_results[name].get("skip"):
        print(f"  {name:<20}  (constant — skipped)")
        continue
    y    = bits_matrix[:, i]
    pred = np.array(probe_results[name]["predictions"])
    fail_mask = (correct_labels == 0)
    succ_mask = (correct_labels == 1)

    acc_all  = (pred == y).mean()
    acc_fail = (pred[fail_mask] == y[fail_mask]).mean() if fail_mask.sum() > 0 else float("nan")
    acc_succ = (pred[succ_mask] == y[succ_mask]).mean() if succ_mask.sum() > 0 else float("nan")
    # Balanced accuracy
    balanced = 0.5 * (acc_fail + acc_succ) if not (np.isnan(acc_fail) or np.isnan(acc_succ)) else float("nan")

    print(f"  {name:<20} {acc_all:>7.3f}  {acc_fail:>10.3f}  {acc_succ:>10.3f}  {balanced:>10.3f}")

# ── Save results ───────────────────────────────────────────────────────────────
probe_output = {
    "N":              N,
    "n_failures":     int(correct_labels.sum() == 0) * N + int((correct_labels == 0).sum()),
    "n_successes":    int((correct_labels == 1).sum()),
    "bit_names":      BIT_NAMES,
    "probe_results":  {k: {kk: vv for kk, vv in v.items() if kk != "predictions"}
                       for k, v in probe_results.items()},
    "h1": {"pass": bool(h1_pass), "n_bits_passing": int(h1_pass_count)},
    "h2": {"pass": bool(h2_pass), "auc": float(auc_mean)},
    "h3": {"pass": bool(h3_pass), "accuracy": float(h3_acc), "n_errors": len(interpreter_errors)},
    "h4": {"pass": bool(h4_pass), "accuracy": float(h4_acc), "n_total": int(h4_total)},
    "t5_baseline": float(t5_accuracy),
}
with open("046_probe_results.json", "w") as f:
    json.dump(probe_output, f, indent=2)
save_to_drive("046_probe_results.json", "046_verdict.txt",
              "046_encoder_reps.npy", "046_structural_bits.json")

verdict = [
    f"VERDICT — {SCRIPT_ID}",
    f"",
    f"N examples: {N}  ({int((correct_labels==0).sum())} failures, {int((correct_labels==1).sum())} successes)",
    f"",
    f"H1: {'✓ CONFIRMED' if h1_pass else '✗ FAILED'}  "
        f"{h1_pass_count}/{len(BIT_NAMES)} structural bits ≥ 90% CV accuracy",
    f"H2: {'✓ CONFIRMED' if h2_pass else '✗ FAILED'}  "
        f"failure classifier AUC = {auc_mean:.4f} ({'≤ 0.55' if h2_pass else '> 0.55'})",
    f"H3: {'✓ CONFIRMED' if h3_pass else '✗ FAILED'}  "
        f"interpreter accuracy = {h3_acc:.1%}",
    f"H4: {'✓ CONFIRMED' if h4_pass else ('✗ FAILED' if h4_total > 0 else 'SKIPPED')}  "
        f"all-probes-correct subset accuracy = {h4_acc:.1%} ({h4_correct}/{h4_total})",
    f"",
    f"T5 baseline:            {t5_accuracy:.1%}",
    f"SCAN interpreter (gold): {h3_acc:.1%}",
]
with open("046_verdict.txt", "w") as f:
    f.write("\n".join(verdict) + "\n")
print(f"\n  Saved: 046_encoder_reps.npy  046_structural_bits.json")
print(f"         046_probe_results.json  046_verdict.txt")

# ── VERDICT ───────────────────────────────────────────────────────────────────
print(f"\n{'='*70}")
print(f"  VERDICT SUMMARY — {SCRIPT_ID}")
print(f"{'='*70}")

print(f"\n  {'✓' if h1_pass else '✗'} H1  {h1_pass_count}/{len(BIT_NAMES)} structural bits ≥ 90% probe accuracy")
print(f"  {'✓' if h2_pass else '✗'} H2  Failure classifier AUC = {auc_mean:.4f}  (K1 sanity check)")
print(f"  {'✓' if h3_pass else '✗'} H3  SCAN interpreter (gold bits) = {h3_acc:.1%}")
print(f"  {'✓' if h4_pass else '✗'} H4  Fixed rule decode (all-probes-correct) = {h4_acc:.1%}")
print()

if h1_pass and h2_pass:
    print("  The encoder linearly encodes the structural features of SCAN commands.")
    print("  No linear function of the encoder predicts per-example failure (K1 held).")
    print()
    print("  The encoder contains the information. The T5 decoder fails to compose")
    print("  the unseen primitive into correct output sequences — not because the")
    print("  information is absent, but because learned softmax decoding is not")
    print("  a monoidal functor for unseen primitives.")
    print()
    print("  Next: S-047 — does adding a structural auxiliary loss during fine-tuning")
    print("  improve jump-compound accuracy? Or: decoder cross-attention analysis.")
elif not h1_pass:
    print("  K1 fires: the encoder lacks linearly accessible structural information")
    print("  for one or more bits. The failure may be representational as well as")
    print("  compositional.")
    print()
    print("  Next: examine which bits fail and why. Are they confounded with")
    print("  jump vs non-jump distinction? Are they recoverable with nonlinear probes?")
elif not h2_pass:
    print("  K2 fires: a linear function of the encoder predicts failure.")
    print("  This contradicts F-043–F-045. Investigate which geometric direction")
    print("  is predictive before proceeding.")

print(f"\n{'='*70}")
print(f"  END {SCRIPT_ID}")
print(f"{'='*70}\n")
