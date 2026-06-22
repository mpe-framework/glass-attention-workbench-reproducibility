#!/usr/bin/env python3
"""
S-049_DECODER_CROSS_ATTENTION.py — Decoder Cross-Attention Pattern Analysis
Applied Categorical Physics Workbench
Troy Teno | May 2026 | Open Access

Pre-registered hypotheses:
  workbench/proposals/S-049_DECODER_CROSS_ATTENTION_PROPOSAL.md
DO NOT modify hypotheses after seeing results.

─────────────────────────────────────────────────────────────────────
SETUP (run once in Colab or terminal):
  pip install torch transformers datasets numpy scipy sentencepiece scikit-learn -q

TO RUN IN GOOGLE COLAB:
  1. Mount Drive (S-043 checkpoint required)
  2. Cell 1: !pip install torch transformers datasets numpy scipy sentencepiece scikit-learn -q
  3. Cell 2: paste this entire script
  4. CPU runtime is fine (~15-30 min); T4 is faster

WHAT THIS SCRIPT DOES:
  Loads the S-043 checkpoint (no new training). Runs three groups of examples:
    Group A: has_around=1 + FAIL (N≈50)
    Group B: has_around=1 + SUCCESS (N≈25)
    Group C: has_around=0 + SUCCESS (N≈50)
  Captures decoder cross-attention weights at each generation step.
  Measures attention to the "around" encoder position (Groups A/B).
  Tests four pre-registered hypotheses about routing vs value failure.

REQUIRED ARTIFACT:
  043_t5_scan_checkpoint/ — from S-043. Must be on Drive or local.
─────────────────────────────────────────────────────────────────────
"""

import numpy as np
import json
import os
import random
from collections import defaultdict

SCRIPT_ID = "S-049_DECODER_XATTN_V0.1.0"
SEED = 42
np.random.seed(SEED)
random.seed(SEED)

# ── Configuration ─────────────────────────────────────────────────────────────────────────────
CHECKPOINT_DIR = "043_t5_scan_checkpoint"
MAX_INPUT_LEN  = 50
MAX_TARGET_LEN = 100
N_GROUP_A      = 50    # has_around=1 + fail
N_GROUP_B      = 25    # has_around=1 + success (rare — take all if fewer)
N_GROUP_C      = 50    # has_around=0 + success

# ── Google Drive persistence ───────────────────────────────────────────────────────────────────────────
DRIVE_DIR = None
try:
    from google.colab import drive as _drive
    _drive.mount("/content/drive", force_remount=False)
    DRIVE_DIR = "/content/drive/MyDrive/workbench_artifacts"
    os.makedirs(DRIVE_DIR, exist_ok=True)
    import shutil as _shutil
    _ckpt_src = os.path.join(DRIVE_DIR, CHECKPOINT_DIR)
    if not os.path.exists(CHECKPOINT_DIR) and os.path.exists(_ckpt_src):
        _shutil.copytree(_ckpt_src, CHECKPOINT_DIR)
        print(f"  Restored from Drive: {CHECKPOINT_DIR}/")
    print(f"  Drive mounted. Artifacts → {DRIVE_DIR}")
except Exception:
    print("  Drive not available.")

def save_to_drive(*filenames):
    if DRIVE_DIR is None:
        return
    import shutil
    for fname in filenames:
        if os.path.exists(fname):
            shutil.copy(fname, os.path.join(DRIVE_DIR, fname))
            print(f"  Saved to Drive: {fname}")

print(f"\n{'='*70}")
print(f"  {SCRIPT_ID}")
print(f"  Decoder cross-attention pattern analysis — T5-small S-043 ckpt")
print(f"  Groups: A=has_around+fail N={N_GROUP_A}, B=has_around+success N≤{N_GROUP_B},"
      f" C=no_around+success N={N_GROUP_C}")
print(f"{'='*70}")

# ── Imports ─────────────────────────────────────────────────────────────────────────────────
print("\nImporting libraries...")
try:
    import torch
    from transformers import T5ForConditionalGeneration, T5Tokenizer
    from scipy import stats as scipy_stats
except ImportError as e:
    raise SystemExit(f"\nMissing library: {e}\n"
                     "Run: pip install torch transformers sentencepiece scipy\n")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"  Device: {device}")

torch.manual_seed(SEED)

# ── Phase 1: Load checkpoint ─────────────────────────────────────────────────────────────────────────────
print("\n" + "="*70)
print("  PHASE 1 — LOAD S-043 CHECKPOINT")
print("="*70)

if not os.path.exists(CHECKPOINT_DIR):
    raise SystemExit(
        f"\n  Checkpoint not found: {CHECKPOINT_DIR}\n"
        "  Run S-043 first, or restore from Drive.\n"
    )

tokenizer = T5Tokenizer.from_pretrained(CHECKPOINT_DIR)
model     = T5ForConditionalGeneration.from_pretrained(CHECKPOINT_DIR).to(device)
model.eval()
print(f"  Loaded S-043 checkpoint from {CHECKPOINT_DIR!r}.")

# ── Phase 2: Load SCAN ───────────────────────────────────────────────────────────────────────────────
print("\n" + "="*70)
print("  PHASE 2 — LOAD SCAN add_prim_jump")
print("="*70)

train_raw = None
test_raw  = None

try:
    from datasets import load_dataset
    for _cfg in ("addprim_jump", "add_prim_jump"):
        try:
            _ds = load_dataset("scan", _cfg, trust_remote_code=True)
            train_raw = list(_ds["train"])
            test_raw  = list(_ds["test"])
            break
        except Exception:
            pass
except ImportError:
    pass

if train_raw is None:
    print("  Downloading from GitHub...")
    import urllib.request
    _BASE = ("https://raw.githubusercontent.com/"
             "brendenlake/SCAN/master/add_prim_split/")

    def _load_scan_txt(url):
        with urllib.request.urlopen(url) as f:
            lines = f.read().decode("utf-8").strip().split("\n")
        out = []
        for line in lines:
            line = line.strip()
            if not line:
                continue
            cmd_part, act_part = line.split(" OUT: ", 1)
            out.append({"commands": cmd_part.replace("IN: ", "").strip(),
                        "actions":  act_part.strip()})
        return out

    train_raw = _load_scan_txt(_BASE + "tasks_train_addprim_jump.txt")
    test_raw  = _load_scan_txt(_BASE + "tasks_test_addprim_jump.txt")
    print(f"  Downloaded. Train: {len(train_raw)}, Test: {len(test_raw)}")

def is_jump_compound(ex):
    tokens = ex["commands"].split()
    return "jump" in tokens and len(tokens) > 1

test_jc = [ex for ex in test_raw if is_jump_compound(ex)]
print(f"  Jump-compound test examples: {len(test_jc)}")

# ── Phase 3: Classify examples ─────────────────────────────────────────────────────────────────────────────
print("\n" + "="*70)
print("  PHASE 3 — CLASSIFY EXAMPLES (run + sort into groups)")
print("="*70)

def has_around(cmd):
    return "around" in cmd.lower().split()

def around_token_pos(cmd, tokenizer):
    """Return the index of the 'around' token in the tokenized encoder sequence."""
    ids = tokenizer(cmd, truncation=True, max_length=MAX_INPUT_LEN)["input_ids"]
    tokens = tokenizer.convert_ids_to_tokens(ids)
    # T5 tokenizer may produce '▁around' (with space prefix)
    for i, tok in enumerate(tokens):
        if "around" in tok.lower():
            return i
    return None

print(f"\n  Running inference on all {len(test_jc)} jump-compound examples...")
print(f"  (Needed to sort into fail/success groups before sampling)")

group_a = []   # has_around=1, fail
group_b = []   # has_around=1, success
group_c = []   # has_around=0, success

EVAL_BATCH = 32
with torch.no_grad():
    for i in range(0, len(test_jc), EVAL_BATCH):
        batch = test_jc[i : i + EVAL_BATCH]
        inp = tokenizer(
            [ex["commands"] for ex in batch],
            padding=True, truncation=True,
            max_length=MAX_INPUT_LEN, return_tensors="pt"
        ).to(device)
        gen_ids = model.generate(input_ids=inp["input_ids"],
                                 attention_mask=inp["attention_mask"],
                                 max_length=MAX_TARGET_LEN)
        preds = tokenizer.batch_decode(gen_ids, skip_special_tokens=True)
        for ex, pred in zip(batch, preds):
            correct = (pred.strip() == ex["actions"].strip())
            ex["predicted"] = pred.strip()
            ex["correct"]   = correct
            if has_around(ex["commands"]):
                if correct:
                    group_b.append(ex)
                else:
                    group_a.append(ex)
            else:
                if correct:
                    group_c.append(ex)

        if (i // EVAL_BATCH) % 20 == 0:
            print(f"    {i+len(batch)}/{len(test_jc)} evaluated...")

print(f"\n  Group A (has_around=1 + fail):    {len(group_a)}")
print(f"  Group B (has_around=1 + success): {len(group_b)}")
print(f"  Group C (has_around=0 + success): {len(group_c)}")

# Sample to target sizes
random.seed(SEED)
group_a = random.sample(group_a, min(N_GROUP_A, len(group_a)))
group_b = group_b[:N_GROUP_B]   # take all if fewer than N_GROUP_B
group_c = random.sample(group_c, min(N_GROUP_C, len(group_c)))
print(f"\n  Sampled: A={len(group_a)}, B={len(group_b)}, C={len(group_c)}")

if len(group_b) < 5:
    print(f"\n  WARNING: Group B only has {len(group_b)} examples.")
    print(f"  H1/H2 measurements will have low statistical power.")

# ── Phase 3b: Failure mode analysis (Troy's 180° hypothesis) ─────────────────────────────────────────────────────
print("\n" + "="*70)
print("  PHASE 3b — FAILURE MODE ANALYSIS")
print("  Q: Does T5 emit 2 repetitions (180°) instead of 4 (360°)?")
print("="*70)
print("""
  Natural English 'around' = 180° flip (most common meaning).
  SCAN 'around' = 360° = 4× quarter-turns = (TURN JUMP)×4.
  SCAN 'opposite' = 180° = 2× quarter-turns — the natural-language 'around'.

  Hypothesis: T5's pretrained weights encode 'around' as 180°.
  Fine-tuning on walk/run/look-around partially corrects this.
  Under OOD stress (jump), T5 regresses toward the pretrained prior.
  Signature: predicted output has ~2 I_JUMP tokens instead of 4.
""")

# Collect ALL group_a examples (before sampling) for this analysis
# group_a has already been sampled to N_GROUP_A; use all of them
sample_fail = group_a   # up to 50 examples

def count_action_tokens(action_str, token):
    """Count occurrences of a specific action token in an action string."""
    return action_str.split().count(token)

def detect_cycle_pattern(action_str, turn_tok, jump_tok):
    """
    Count clean (TURN JUMP) repetitions at the start of the action sequence.
    Ignores leading tokens that are not part of the cycle (e.g. prefix from 'after').
    Returns the count of (TURN JUMP) pairs found anywhere in the sequence.
    """
    tokens = action_str.split()
    count = 0
    i = 0
    while i < len(tokens) - 1:
        if tokens[i] == turn_tok and tokens[i+1] == jump_tok:
            count += 1
            i += 2
        else:
            i += 1
    return count

# For has_around examples, determine the direction token from the command
def infer_dir_token(cmd):
    tokens = cmd.lower().split()
    if "left" in tokens:
        return "I_TURN_LEFT"
    if "right" in tokens:
        return "I_TURN_RIGHT"
    return None

failure_categories = defaultdict(int)
sample_print = []   # for display

print(f"  Analyzing {len(sample_fail)} has_around=1 failing examples...\n")
print(f"  {'CMD':<45}  {'GOLD_J':>6}  {'PRED_J':>6}  {'CATEGORY'}")
print(f"  {'─'*45}  {'─'*6}  {'─'*6}  {'─'*20}")

for ex in sample_fail:
    cmd  = ex["commands"]
    gold = ex["actions"]
    pred = ex["predicted"]

    n_gold_jump = count_action_tokens(gold, "I_JUMP")
    n_pred_jump = count_action_tokens(pred, "I_JUMP")
    n_gold_walk = count_action_tokens(gold, "I_WALK")
    n_pred_walk = count_action_tokens(pred, "I_WALK")

    dir_tok = infer_dir_token(cmd)
    gold_cycles = detect_cycle_pattern(gold, dir_tok, "I_JUMP") if dir_tok else 0

    # Categorize
    if n_pred_jump == 0:
        if n_pred_walk > 0:
            cat = "substituted_walk"    # generated WALK instead of JUMP
        else:
            cat = "no_jump"             # no JUMP at all
    elif n_pred_jump == 2 and n_gold_jump == 4:
        cat = "half_count_2of4"        # 180° hypothesis — exactly half
    elif n_pred_jump == 1 and n_gold_jump == 4:
        cat = "quarter_count_1of4"     # only 1 cycle
    elif n_pred_jump == 3 and n_gold_jump == 4:
        cat = "three_quarter_3of4"     # 3 cycles
    elif 0 < n_pred_jump < n_gold_jump:
        cat = f"short_count_{n_pred_jump}of{n_gold_jump}"
    elif n_pred_jump > n_gold_jump:
        cat = "over_count"             # too many jumps
    else:
        cat = "wrong_structure"        # right count, wrong structure

    failure_categories[cat] += 1

    # Abbreviated command for display
    cmd_short = cmd if len(cmd) <= 43 else cmd[:40] + "..."
    print(f"  {cmd_short:<45}  {n_gold_jump:>6}  {n_pred_jump:>6}  {cat}")

    sample_print.append({
        "cmd": cmd, "gold": gold, "predicted": pred,
        "n_gold_jump": n_gold_jump, "n_pred_jump": n_pred_jump,
        "category": cat,
    })

# Summary table
print(f"\n  {'─'*60}")
print(f"  FAILURE MODE SUMMARY (N={len(sample_fail)})")
print(f"  {'─'*60}")
total_fail = len(sample_fail)
for cat, cnt in sorted(failure_categories.items(), key=lambda x: -x[1]):
    print(f"  {cat:<35}  {cnt:>4}  ({100*cnt/total_fail:.1f}%)")

# Test the 180° hypothesis:
# Is "half_count_2of4" significantly overrepresented vs chance?
half_count  = failure_categories.get("half_count_2of4", 0)
no_jump     = failure_categories.get("no_jump", 0)
sub_walk    = failure_categories.get("substituted_walk", 0)
short_other = total_fail - half_count - no_jump - sub_walk

h0_180deg = bool(half_count / total_fail > 0.25)  # >25% of failures match 180° pattern

print(f"\n  180° hypothesis support (half_count_2of4 > 25% of failures): "
      f"{'YES' if h0_180deg else 'NO'} ({100*half_count/total_fail:.1f}%)")
print(f"  Zero-jump failures (complete breakdown): {no_jump} ({100*no_jump/total_fail:.1f}%)")
print(f"  Walk substitution (wrong primitive):     {sub_walk} ({100*sub_walk/total_fail:.1f}%)")

# Show a few full gold vs predicted examples
print(f"\n  DETAILED EXAMPLES (first 5):")
for ex in sample_print[:5]:
    print(f"  CMD:  {ex['cmd']}")
    print(f"  GOLD: {ex['gold'][:120]}")
    print(f"  PRED: {ex['predicted'][:120]}")
    print(f"  ({ex['category']})")
    print()

# ── Phase 4: Capture cross-attention ──────────────────────────────────────────────────────────────────────────────
print("\n" + "="*70)
print("  PHASE 4 — CAPTURE DECODER CROSS-ATTENTION")
print("="*70)
print(f"\n  Running model.generate() with output_attentions=True for all groups...")
print(f"  This is slow (one example at a time). Please wait.\n")

def capture_cross_attention(examples, group_name, compute_around_pos=True):
    """
    Returns list of dicts with:
      - 'cmd': command text
      - 'n_decoder_steps': length of generated sequence
      - 'around_pos': encoder position of 'around' token (or None)
      - 'cross_attn_to_around': [n_steps] float array — mean cross-attn to around pos
        across all layers and heads at each decoder step. None if no around token.
      - 'cross_attn_entropy': [n_steps] float array — mean entropy per decoder step
      - 'cross_attn_by_layer': [n_layers, n_steps] — per-layer mean cross-attn to around
    """
    results = []
    n_layers = model.config.num_decoder_layers   # 6 for T5-small
    n_heads  = model.config.num_heads            # 8 for T5-small

    for j, ex in enumerate(examples):
        if j % 10 == 0:
            print(f"  [{group_name}] {j+1}/{len(examples)} ...")

        inp = tokenizer(
            ex["commands"],
            truncation=True, max_length=MAX_INPUT_LEN,
            return_tensors="pt"
        ).to(device)

        around_pos = None
        if compute_around_pos:
            around_pos = around_token_pos(ex["commands"], tokenizer)

        with torch.no_grad():
            out = model.generate(
                input_ids=inp["input_ids"],
                attention_mask=inp["attention_mask"],
                max_length=MAX_TARGET_LEN,
                output_attentions=True,
                return_dict_in_generate=True,
            )

        # out.cross_attentions: tuple over decoder steps
        # Each element: tuple over decoder layers
        # Each layer tensor: [1, n_heads, 1, T_enc]
        cross_attns = out.cross_attentions   # len = n_decoder_steps
        n_steps = len(cross_attns)
        T_enc = inp["input_ids"].shape[1]

        # Per-step mean cross-attention to around_pos
        attn_to_around = np.zeros(n_steps) if around_pos is not None else None
        # Per-step mean entropy
        attn_entropy   = np.zeros(n_steps)
        # Per-layer, per-step mean cross-attn to around_pos
        attn_by_layer  = np.zeros((n_layers, n_steps)) if around_pos is not None else None

        for t, step_attns in enumerate(cross_attns):
            # step_attns: tuple of n_layers tensors [1, n_heads, 1, T_enc]
            all_heads = []  # collect [n_layers*n_heads, T_enc]
            for l, layer_attn in enumerate(step_attns):
                # layer_attn: [1, n_heads, 1, T_enc]
                a = layer_attn[0, :, 0, :].cpu().numpy()  # [n_heads, T_enc]
                all_heads.append(a)
                if around_pos is not None and around_pos < T_enc:
                    attn_by_layer[l, t] = a[:, around_pos].mean()

            all_heads = np.concatenate(all_heads, axis=0)  # [n_layers*n_heads, T_enc]
            mean_over_heads = all_heads.mean(0)  # [T_enc]

            if around_pos is not None and around_pos < T_enc:
                attn_to_around[t] = mean_over_heads[around_pos]

            # Entropy of mean-over-heads distribution
            p = mean_over_heads + 1e-12
            p = p / p.sum()
            attn_entropy[t] = -np.sum(p * np.log(p))

        results.append({
            "cmd":                ex["commands"],
            "correct":            ex["correct"],
            "around_pos":         around_pos,
            "n_decoder_steps":    n_steps,
            "cross_attn_to_around": attn_to_around.tolist() if attn_to_around is not None else None,
            "cross_attn_entropy": attn_entropy.tolist(),
            "cross_attn_by_layer": attn_by_layer.tolist() if attn_by_layer is not None else None,
        })

    return results

results_a = capture_cross_attention(group_a, "A (around+fail)", compute_around_pos=True)
results_b = capture_cross_attention(group_b, "B (around+succ)", compute_around_pos=True)
results_c = capture_cross_attention(group_c, "C (no_around+succ)", compute_around_pos=False)

# ── Phase 5: H1 — mean attention to around ────────────────────────────────────────────────────────────────────────
print("\n" + "="*70)
print("  PHASE 5 — H1: MEAN CROSS-ATTENTION TO 'AROUND' TOKEN")
print("="*70)

def mean_attn_to_around(results):
    vals = []
    for r in results:
        if r["cross_attn_to_around"] is not None:
            vals.append(np.mean(r["cross_attn_to_around"]))
    return np.array(vals)

a_vals = mean_attn_to_around(results_a)
b_vals = mean_attn_to_around(results_b)

print(f"\n  Group A (has_around=1 + fail):    "
      f"mean={a_vals.mean():.5f}  std={a_vals.std():.5f}  N={len(a_vals)}")
print(f"  Group B (has_around=1 + success): "
      f"mean={b_vals.mean():.5f}  std={b_vals.std():.5f}  N={len(b_vals)}")

h1_stat, h1_p = scipy_stats.mannwhitneyu(b_vals, a_vals, alternative="greater")
h1_pass = (b_vals.mean() > a_vals.mean()) and (h1_p < 0.05)
print(f"\n  H1 Mann-Whitney (B > A): U={h1_stat:.1f}, p={h1_p:.4f}")
print(f"  H1: {'PASS' if h1_pass else 'FAIL'} "
      f"({'B > A, routing failure confirmed' if h1_pass else 'B ≈ A, routing is not the bottleneck'})")

# ── Phase 6: H2 — periodicity in Group B ────────────────────────────────────────────────────────────────────────────
print("\n" + "="*70)
print("  PHASE 6 — H2: PERIODICITY OF ATTENTION TO 'AROUND' (period=2)")
print("="*70)
print(f"\n  Correct output for 'jump around X' = (I_TURN_X I_JUMP)×4")
print(f"  Expected period: every 2 decoder steps, attention should peak at 'around'")

def periodicity_score(series, period=2):
    """Pearson r between series and a cosine with given period."""
    n = len(series)
    if n < period * 2:
        return 0.0, 1.0
    t = np.arange(n)
    cosine = np.cos(2 * np.pi * t / period)
    r, p = scipy_stats.pearsonr(series, cosine)
    return r, p

period_a = [periodicity_score(r["cross_attn_to_around"]) for r in results_a
            if r["cross_attn_to_around"] is not None]
period_b = [periodicity_score(r["cross_attn_to_around"]) for r in results_b
            if r["cross_attn_to_around"] is not None]

r_a = np.array([x[0] for x in period_a])
r_b = np.array([x[0] for x in period_b])

print(f"\n  Periodicity r (period=2) — Group A (fail):    "
      f"mean={r_a.mean():.4f}  std={r_a.std():.4f}")
print(f"  Periodicity r (period=2) — Group B (success): "
      f"mean={r_b.mean():.4f}  std={r_b.std():.4f}")

h2_stat, h2_p = scipy_stats.mannwhitneyu(r_b, r_a, alternative="greater") if len(r_b) > 1 else (0, 1.0)
h2_pass = (r_b.mean() > r_a.mean()) and (h2_p < 0.05)
print(f"\n  H2 Mann-Whitney (B > A periodicity): U={h2_stat:.1f}, p={h2_p:.4f}")
print(f"  H2: {'PASS' if h2_pass else 'FAIL'} "
      f"({'Period-2 rhythm stronger in success cases' if h2_pass else 'No clear periodic difference'})")

# ── Phase 7: H3 — entropy ──────────────────────────────────────────────────────────────────────────────────
print("\n" + "="*70)
print("  PHASE 7 — H3: CROSS-ATTENTION ENTROPY (A vs C)")
print("="*70)

def mean_entropy(results):
    return np.array([np.mean(r["cross_attn_entropy"]) for r in results])

ent_a = mean_entropy(results_a)
ent_c = mean_entropy(results_c)

print(f"\n  Group A (has_around=1 + fail):    "
      f"mean_entropy={ent_a.mean():.4f}  std={ent_a.std():.4f}  N={len(ent_a)}")
print(f"  Group C (has_around=0 + success): "
      f"mean_entropy={ent_c.mean():.4f}  std={ent_c.std():.4f}  N={len(ent_c)}")

h3_stat, h3_p = scipy_stats.mannwhitneyu(ent_a, ent_c, alternative="greater")
h3_pass = (ent_a.mean() > ent_c.mean()) and (h3_p < 0.05)
print(f"\n  H3 Mann-Whitney (A > C entropy): U={h3_stat:.1f}, p={h3_p:.4f}")
print(f"  H3: {'PASS' if h3_pass else 'FAIL'} "
      f"({'Failing cases more diffuse' if h3_pass else 'Entropy does not differentiate'})")

# ── Phase 8: H4 — layer localization ─────────────────────────────────────────────────────────────────────────────
print("\n" + "="*70)
print("  PHASE 8 — H4: LAYER LOCALIZATION OF ATTENTION DIVERGENCE")
print("="*70)

n_layers = model.config.num_decoder_layers  # 6

print(f"\n  Per-decoder-layer mean cross-attn to 'around' position:")
print(f"  {'Layer':<8}  {'A (fail)':>10}  {'B (succ)':>10}  {'B-A':>8}")

layer_a_means = []
layer_b_means = []
for l in range(n_layers):
    a_layer = [np.mean(r["cross_attn_by_layer"][l]) for r in results_a
               if r["cross_attn_by_layer"] is not None]
    b_layer = [np.mean(r["cross_attn_by_layer"][l]) for r in results_b
               if r["cross_attn_by_layer"] is not None]
    a_m = np.mean(a_layer) if a_layer else float("nan")
    b_m = np.mean(b_layer) if b_layer else float("nan")
    layer_a_means.append(a_m)
    layer_b_means.append(b_m)
    print(f"  Layer {l+1:<5}  {a_m:>10.5f}  {b_m:>10.5f}  {b_m-a_m:>+8.5f}")

# H4: divergence max in layers 3-6 (indices 2-5)
if all(not np.isnan(x) for x in layer_b_means):
    diffs = [layer_b_means[l] - layer_a_means[l] for l in range(n_layers)]
    max_diff_layer = np.argmax([abs(d) for d in diffs]) + 1   # 1-indexed
    h4_pass = (max_diff_layer >= 3)  # layers 3-6 = indices 2-5
    print(f"\n  Max divergence at layer {max_diff_layer}")
    print(f"  H4: {'PASS' if h4_pass else 'FAIL'} "
          f"({'Divergence in mid-to-late decoder layers' if h4_pass else 'Divergence in early layers'})")
else:
    h4_pass = False
    max_diff_layer = -1
    print(f"\n  H4: FAIL (insufficient Group B data)")

# ── Phase 9: K1 check ────────────────────────────────────────────────────────────────────────────────────
print("\n" + "="*70)
print("  PHASE 9 — KILL CONDITION K1")
print("="*70)

# K1: cross-attention to around is identical in A and B
k1_fires = not h1_pass   # K1 fires if H1 fails (B ≤ A)
print(f"\n  K1 (routing is identical in fail vs success): {'FIRES' if k1_fires else 'does not fire'}")
if k1_fires:
    print(f"  Interpretation: failure is NOT in attention routing.")
    print(f"  Failure is in value projections or FFN (OOD value for 'jump' at 'around' position).")
    print(f"  Next step: encoder value analysis at 'around' position for has_around=1 examples.")

# ── Phase 10: Summary ────────────────────────────────────────────────────────────────────────────────────
print("\n" + "="*70)
print("  PHASE 10 — SUMMARY")
print("="*70)

print(f"\n  {'─'*60}")
print(f"  HYPOTHESIS VERDICTS")
print(f"  {'─'*60}")
print(f"  H1: B > A mean attn-to-around (routing failure)    — {'PASS' if h1_pass else 'FAIL'}")
print(f"      A={a_vals.mean():.5f}, B={b_vals.mean():.5f}, p={h1_p:.4f}")
print(f"  H2: B > A periodicity r=2 (counting rhythm)        — {'PASS' if h2_pass else 'FAIL'}")
print(f"      A={r_a.mean():.4f}, B={r_b.mean():.4f}, p={h2_p:.4f}")
print(f"  H3: A > C entropy (diffuse routing in fail)        — {'PASS' if h3_pass else 'FAIL'}")
print(f"      A={ent_a.mean():.4f}, C={ent_c.mean():.4f}, p={h3_p:.4f}")
print(f"  H4: Max divergence in layers 3-6                   — {'PASS' if h4_pass else 'FAIL'}")
print(f"      Max divergence at layer {max_diff_layer}")
print(f"  K1 (routing identical = value failure):            {'FIRES' if k1_fires else 'does not fire'}")

if h1_pass:
    verdict = ("ROUTING FAILURE: decoder cross-attention to 'around' is "
               "weaker in failing cases. The decoder fails to route through "
               "the 'around' encoder position when generating the 4× cycle.")
else:
    verdict = ("VALUE FAILURE (K1 fires): decoder cross-attention to 'around' "
               "is similar in failing and succeeding cases. The failure is in "
               "the value projection — the encoder's 'around+jump' representation "
               "is OOD relative to 'around+walk/run/look'.")

print(f"\n  VERDICT: {verdict}")

# ── Save ───────────────────────────────────────────────────────────────────────────────────────────────────────────
output = {
    "script_id": SCRIPT_ID,
    "group_sizes": {"A": len(group_a), "B": len(group_b), "C": len(group_c)},
    "failure_mode_analysis": {
        "n_analyzed": len(sample_fail),
        "categories": dict(failure_categories),
        "half_count_2of4_pct": float(half_count / total_fail),
        "h0_180deg_support": bool(h0_180deg),
        "note": ("half_count_2of4 = predicted 2 I_JUMP instead of 4 — "
                 "consistent with T5 using 180 degree semantics for 'around'"),
    },
    "h1": {
        "pass": bool(h1_pass), "A_mean": float(a_vals.mean()),
        "B_mean": float(b_vals.mean()), "p": float(h1_p),
    },
    "h2": {
        "pass": bool(h2_pass), "A_periodicity": float(r_a.mean()),
        "B_periodicity": float(r_b.mean()), "p": float(h2_p),
    },
    "h3": {
        "pass": bool(h3_pass), "A_entropy": float(ent_a.mean()),
        "C_entropy": float(ent_c.mean()), "p": float(h3_p),
    },
    "h4": {
        "pass": bool(h4_pass), "max_divergence_layer": int(max_diff_layer),
        "per_layer_A": [float(x) for x in layer_a_means],
        "per_layer_B": [float(x) for x in layer_b_means],
    },
    "k1_fires": bool(k1_fires),
    "verdict": verdict,
}

with open("049_results.json", "w") as f:
    import json
    json.dump(output, f, indent=2)
save_to_drive("049_results.json")

print(f"\n  Output: 049_results.json")
print(f"  {'='*70}")
print(f"  S-049 complete.")
print(f"  {'='*70}")
