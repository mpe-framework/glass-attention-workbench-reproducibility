#!/usr/bin/env python3
"""
S-062_H4_SUCCESS_CHECK.py — Specificity check for S-061 W_V repair
Applied Categorical Physics Workbench
Troy Teno | May 2026 | Open Access

Pre-registered hypotheses:
  workbench/proposals/S-062_H4_SUCCESS_CHECK_PROPOSAL.md
DO NOT modify hypotheses after seeing results.

─────────────────────────────────────────────────────────────────────
BACKGROUND:
  S-061 confirmed H1/H2/H3 (value substitution causally confirmed).
  H4 (specificity) was not evaluable: the success classifier erroneously
  required n_walk_tgt >= 1, excluding all "jump around" success examples.
  Result: n_success=0. This script fixes the classifier and evaluates H4.

  Pre-registered condition (H4): mean_Δmargin_success (joint, α=1.0) > −1.0
  Kill condition (K2):           mean_Δmargin_success < −2.0

SPEED OPTIMISATION:
  S-061 Phase 3 ran generate() on all 7706 test examples (~60 min).
  S-062 pre-filters to examples containing both "around" AND "jump"
  before running inference. This cuts runtime to ~5-15 minutes.

CORRECTION GEOMETRY:
  Identical to S-061. u = attention-weighted mean encoder direction
  over 30 fail examples. v_target = pinv(W_O_h) @ W_U_eff, normalized
  to ‖v_bad‖. Joint rank-1 ΔW_V at α=1.0 applied in-place.

Version history:
  V0.1.0 — initial build
─────────────────────────────────────────────────────────────────────
"""

import numpy as np
import json
import os
import random
import urllib.request

SCRIPT_ID = "S-062_H4_SUCCESS_CHECK_V0.1.0"
SEED = 42
np.random.seed(SEED)
random.seed(SEED)

CHECKPOINT_DIR = "043_t5_scan_checkpoint"
MAX_INPUT_LEN  = 50
MAX_TARGET_LEN = 100
N_FAIL         = 30
N_SUCCESS      = 25

# Pre-registered threshold (H4) — amended before run (see proposal)
# Original S-061 threshold (> -1.0) was too loose for a specificity claim.
# Correct prediction for a rank-1 OOD-direction correction: near-zero effect.
H4_NEAR_ZERO_LO = -0.3   # H4 passes if mean Δm_success >= this ...
H4_NEAR_ZERO_HI = +0.3   # ... and <= this (two-sided near-zero test)
K2_SUCCESS_FLOOR = -0.5  # K2 fires if mean Δm_success < this (meaningful degradation)
OOD_THRESH       = 0.01

# Head coordinates (0-based, same as S-061)
HEADS = {
    "L5H5": {"block": 4, "head": 5},
    "L4H6": {"block": 3, "head": 6},
}

# Token IDs
I_JUMP_ID = 683
I_WALK_ID = 12054
TURN_END_IDS = {6245, 27262}
DIV_IDS = {683, 12054, 448, 5017}

# ── Google Drive restoration (Colab) ─────────────────────────────────────────────
DRIVE_DIR = None
try:
    from google.colab import drive as _drive
    _drive.mount("/content/drive", force_remount=False)
    DRIVE_DIR = "/content/drive/MyDrive/workbench_artifacts"
    import shutil as _shutil
    _ckpt_src = os.path.join(DRIVE_DIR, CHECKPOINT_DIR)
    if not os.path.exists(CHECKPOINT_DIR) and os.path.exists(_ckpt_src):
        _shutil.copytree(_ckpt_src, CHECKPOINT_DIR)
        print(f"  Restored checkpoint from Drive: {CHECKPOINT_DIR}/")
    print(f"  Drive mounted. Artifacts dir: {DRIVE_DIR}")
except Exception:
    pass

def _save_to_drive(*filenames):
    if DRIVE_DIR is None:
        return
    import shutil
    wb_results = os.path.join(DRIVE_DIR, "workbench", "results")
    os.makedirs(wb_results, exist_ok=True)
    for fname in filenames:
        if os.path.exists(fname):
            dst = os.path.join(wb_results, os.path.basename(fname))
            shutil.copy(fname, dst)
            print(f"  Saved to Drive: {dst}")

print("=" * 70)
print(f"  {SCRIPT_ID}")
print("  H4 specificity check: joint α=1.0 repair vs success group")
print("=" * 70)

# ─────────────────────────────────────────────────────────────────────
# PHASE 1 — LOAD MODEL
# ─────────────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("  PHASE 1 — LOAD MODEL")
print("=" * 70)

import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer

print("Importing libraries...")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"  Device: {device}")

tokenizer = T5Tokenizer.from_pretrained(CHECKPOINT_DIR, local_files_only=True)
model = T5ForConditionalGeneration.from_pretrained(
    CHECKPOINT_DIR, local_files_only=True, output_attentions=False
)
model.to(device)
model.eval()
print(f"  Loaded '{CHECKPOINT_DIR}'.")

d_model = model.config.d_model   # 512
d_kv    = model.config.d_kv      # 64
n_heads = model.config.num_heads  # 8
BOS_ID  = model.config.decoder_start_token_id

# ─────────────────────────────────────────────────────────────────────
# PHASE 2 — LOAD SCAN (pre-filtered)
# ─────────────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("  PHASE 2 — LOAD SCAN (pre-filtered to 'around'+'jump' examples)")
print("=" * 70)

def _load_scan_txt(url):
    print(f"  Downloading: {url}")
    with urllib.request.urlopen(url) as f:
        lines = f.read().decode("utf-8").strip().split("\n")
    out = []
    for line in lines:
        line = line.strip()
        if not line:
            continue
        cmd_part, act_part = line.split(" OUT: ", 1)
        out.append({
            "commands": cmd_part.replace("IN: ", "").strip(),
            "actions":  act_part.strip(),
        })
    return out

_SCAN_BASE = ("https://raw.githubusercontent.com/"
              "brendenlake/SCAN/master/add_prim_split/")
all_test = _load_scan_txt(_SCAN_BASE + "tasks_test_addprim_jump.txt")
print(f"  Full test set: {len(all_test)} examples")

# Pre-filter: only run inference on "around" + "jump" examples.
# Both fail and success groups require "around" in the command.
# This avoids running generate() on the ~7000 non-around examples.
around_jump = [ex for ex in all_test
               if "around" in ex["commands"].lower()
               and "jump" in ex["commands"].lower()]
print(f"  Pre-filtered (around+jump): {len(around_jump)} examples")
print(f"  Skipping {len(all_test) - len(around_jump)} non-around-jump examples")

# ─────────────────────────────────────────────────────────────────────
# PHASE 3 — CLASSIFY (fail + success groups)
# ─────────────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("  PHASE 3 — CLASSIFY (pre-filtered set only)")
print("=" * 70)


def tokenize_example(ex):
    cmd = "translate English to actions: " + ex["commands"]
    enc = tokenizer(cmd, return_tensors="pt",
                    max_length=MAX_INPUT_LEN, truncation=True, padding=False)
    return enc


def run_generate(enc_input):
    with torch.no_grad():
        out = model.generate(
            input_ids=enc_input["input_ids"].to(device),
            attention_mask=enc_input["attention_mask"].to(device),
            max_new_tokens=MAX_TARGET_LEN,
        )
    return out[0].cpu().tolist()


fail_examples    = []
success_examples = []

print(f"  Running inference on {len(around_jump)} pre-filtered examples...")
for ex_idx, ex in enumerate(around_jump):
    if ex_idx % 50 == 0:
        print(f"  {ex_idx+1}/{len(around_jump)} ...")

    enc_input = tokenize_example(ex)
    gen_ids   = run_generate(enc_input)
    tgt_ids   = tokenizer.encode(ex["actions"], add_special_tokens=False)

    has_around  = "around" in ex["commands"].lower()
    n_jump_tgt  = tgt_ids.count(I_JUMP_ID)
    n_walk_tgt  = tgt_ids.count(I_WALK_ID)
    n_jump_gen  = gen_ids.count(I_JUMP_ID)
    n_walk_gen  = gen_ids.count(I_WALK_ID)
    is_correct  = (gen_ids[1:] == tgt_ids + [tokenizer.eos_token_id] or
                   gen_ids[1:-1] == tgt_ids)

    # Fail: substituted-walk (same as S-061)
    if (has_around and not is_correct and
            n_jump_tgt >= 1 and n_walk_tgt >= 1 and
            n_jump_gen == 0 and n_walk_gen >= 1):
        fail_examples.append((ex, enc_input))

    # Success: CORRECTED classifier — no n_walk_tgt requirement
    # "jump around" targets have only I_JUMP + I_TURN_LEFT, no I_WALK.
    elif (has_around and is_correct and n_jump_tgt >= 1):
        success_examples.append((ex, enc_input))

print(f"\n  substituted_walk fail examples: {len(fail_examples)}")
print(f"  has_around+correct examples:    {len(success_examples)}")

# Shuffle and cap — same SEED as S-061 for reproducibility
rng_fail = random.Random(SEED)
rng_fail.shuffle(fail_examples)
rng_succ = random.Random(SEED)
rng_succ.shuffle(success_examples)

fail_examples    = fail_examples[:N_FAIL]
success_examples = success_examples[:N_SUCCESS]
print(f"  Using {len(fail_examples)} fail, {len(success_examples)} success")

if len(fail_examples) != 30:
    print(f"  WARNING: expected 30 fail examples, got {len(fail_examples)}")
    print(f"  Cross-check with S-061 may be invalid.")
if len(success_examples) == 0:
    raise RuntimeError(
        "Still 0 success examples after classifier fix! "
        "Check that the pre-filtered set contains has_around+correct examples."
    )

# ─────────────────────────────────────────────────────────────────────
# PHASE 4 — W_U_EFF
# ─────────────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("  PHASE 4 — W_U_EFF")
print("=" * 70)

W_U      = model.shared.weight.detach().cpu().float()
W_U_diff = W_U[I_JUMP_ID] - W_U[I_WALK_ID]
LN_WEIGHT = model.decoder.final_layer_norm.weight.detach().cpu().float()
W_U_eff   = W_U_diff * LN_WEIGHT
print(f"  ‖W_U_diff‖ = {W_U_diff.norm():.4f}")
print(f"  ‖W_U_eff‖  = {W_U_eff.norm():.4f}")

# ─────────────────────────────────────────────────────────────────────
# HELPER FUNCTIONS
# ─────────────────────────────────────────────────────────────────────

def find_div_step(gen_ids):
    for i, tok in enumerate(gen_ids):
        if i == 0:
            continue
        if tok in DIV_IDS:
            return i, tok
    return None, None


def get_margin(logits):
    return (logits[I_JUMP_ID] - logits[I_WALK_ID]).item()


def get_p_sum(logits):
    probs = torch.softmax(logits, dim=-1)
    return (probs[I_JUMP_ID] + probs[I_WALK_ID]).item()


def run_forward(enc_input, dec_prefix_ids, output_attentions=False):
    dec_tensor = torch.tensor([dec_prefix_ids], dtype=torch.long, device=device)
    with torch.no_grad():
        out = model(
            input_ids=enc_input["input_ids"].to(device),
            attention_mask=enc_input["attention_mask"].to(device),
            decoder_input_ids=dec_tensor,
            output_attentions=output_attentions,
        )
    return out

# ─────────────────────────────────────────────────────────────────────
# PHASE 5 — FAIL GROUP BASELINES + ENCODER DIRECTIONS
# ─────────────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("  PHASE 5 — FAIL GROUP BASELINES + ENCODER DIRECTIONS")
print("=" * 70)

enc_weighted_accum = {hname: [] for hname in HEADS}
fail_records = []
n_skip = 0

print(f"  Processing {len(fail_examples)} fail examples...")
for ex_idx, (ex, enc_input) in enumerate(fail_examples):
    gen_ids = run_generate(enc_input)
    div_pos, div_tok = find_div_step(gen_ids)
    if div_pos is None:
        print(f"    [skip ex {ex_idx}] no divergence token found")
        n_skip += 1
        continue
    if div_tok != I_WALK_ID:
        print(f"    [skip ex {ex_idx}] div_tok={div_tok} != I_WALK_ID")
        n_skip += 1
        continue

    fail_prefix = gen_ids[:div_pos]
    out = run_forward(enc_input, fail_prefix, output_attentions=True)
    logits  = out.logits[0, -1, :]
    baseline_margin = get_margin(logits)

    enc_hidden = out.encoder_last_hidden_state[0].cpu().float()
    for hname, hinfo in HEADS.items():
        block_idx   = hinfo["block"]
        head_idx    = hinfo["head"]
        ca          = out.cross_attentions[block_idx]
        attn_weights = ca[0, head_idx, -1, :].cpu().float()
        enc_weighted_accum[hname].append(attn_weights @ enc_hidden)

    fail_records.append({
        "ex_idx":          ex_idx,
        "fail_prefix":     fail_prefix,
        "baseline_margin": baseline_margin,
        "enc_input":       enc_input,
    })

baseline_fail_margins = [r["baseline_margin"] for r in fail_records]
mean_bl_fail = np.mean(baseline_fail_margins)
print(f"  Valid fail examples: {len(fail_records)}  (skipped: {n_skip})")
print(f"  Baseline fail group: mean margin = {mean_bl_fail:.4f}")
print(f"  Cross-check vs S-061 baseline (expected ≈ −6.51): "
      f"{'OK' if abs(mean_bl_fail - (-6.5108)) < 0.5 else 'MISMATCH — check classifier'}")

# ─────────────────────────────────────────────────────────────────────
# PHASE 5b — SUCCESS GROUP BASELINES
# ─────────────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("  PHASE 5b — SUCCESS GROUP BASELINES")
print("=" * 70)

success_records = []
n_skip_s = 0

print(f"  Processing {len(success_examples)} success examples...")
for ex_idx, (ex, enc_input) in enumerate(success_examples):
    gen_ids = run_generate(enc_input)
    div_pos, div_tok = find_div_step(gen_ids)
    if div_pos is None:
        print(f"    [skip ex {ex_idx}] no divergence token found")
        n_skip_s += 1
        continue
    if div_tok != I_JUMP_ID:
        print(f"    [skip ex {ex_idx}] div_tok={div_tok} != I_JUMP_ID")
        n_skip_s += 1
        continue

    success_prefix = gen_ids[:div_pos]
    out = run_forward(enc_input, success_prefix, output_attentions=False)
    logits = out.logits[0, -1, :]

    success_records.append({
        "ex_idx":          ex_idx,
        "success_prefix":  success_prefix,
        "baseline_margin": get_margin(logits),
        "p_sum_baseline":  get_p_sum(logits),
        "enc_input":       enc_input,
    })

baseline_succ_margins = [r["baseline_margin"] for r in success_records]
mean_bl_succ = float(np.mean(baseline_succ_margins)) if success_records else float("nan")
n_succ_pos   = sum(1 for m in baseline_succ_margins if m > 0)
print(f"  Valid success examples: {len(success_records)}  (skipped: {n_skip_s})")
print(f"  Baseline success group: mean margin = {mean_bl_succ:.4f}")
print(f"  I_JUMP wins at baseline: {n_succ_pos}/{len(success_records)}")

# ─────────────────────────────────────────────────────────────────────
# PHASE 6 — COMPUTE JOINT α=1.0 CORRECTION (identical to S-061)
# ─────────────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("  PHASE 6 — W_V CORRECTION GEOMETRY (joint, α=1.0)")
print("=" * 70)

corrections = {}

for hname, hinfo in HEADS.items():
    block_idx = hinfo["block"]
    head_idx  = hinfo["head"]
    print(f"\n  Head: {hname} (block {block_idx}, head {head_idx})")

    enc_vecs = enc_weighted_accum[hname]
    enc_mean = torch.stack(enc_vecs, dim=0).mean(dim=0)
    u = enc_mean / enc_mean.norm()

    enc_dec_attn = model.decoder.block[block_idx].layer[1].EncDecAttention
    W_V_full = enc_dec_attn.v.weight.detach().cpu().float()
    W_O_full = enc_dec_attn.o.weight.detach().cpu().float()

    h_start = head_idx * d_kv
    h_end   = h_start + d_kv
    W_V_h   = W_V_full[h_start:h_end, :]
    W_O_h   = W_O_full[:, h_start:h_end]

    v_bad         = W_V_h @ u
    W_O_h_pinv    = torch.linalg.pinv(W_O_h)
    v_target_raw  = W_O_h_pinv @ W_U_eff.cpu()
    v_target      = v_target_raw * (v_bad.norm() / v_target_raw.norm())
    v_correction  = v_target - v_bad

    # Verify geometry matches S-061
    proj_target   = W_O_h @ v_target
    proj_bad      = W_O_h @ v_bad
    W_U_eff_norm  = W_U_eff.cpu() / W_U_eff.cpu().norm()
    cos_target    = torch.dot(proj_target / proj_target.norm(), W_U_eff_norm).item()
    cos_bad       = torch.dot(proj_bad    / proj_bad.norm(),    W_U_eff_norm).item()
    print(f"    cos(W_O@v_target, W_U_eff) = {cos_target:+.4f}  (S-061 L5H5: +0.3421, L4H6: +0.3768)")
    print(f"    cos(W_O@v_bad,    W_U_eff) = {cos_bad:+.4f}  (S-061 L5H5: -0.0485, L4H6: -0.0038)")

    dWV_a1 = 1.0 * torch.outer(v_correction, u)   # α=1.0

    corrections[hname] = {
        "block_idx":   block_idx,
        "head_idx":    head_idx,
        "h_start":     h_start,
        "h_end":       h_end,
        "dWV_a1":      dWV_a1,
        "cos_target":  cos_target,
        "cos_bad":     cos_bad,
    }

print("\n  Correction geometry complete.")

# ─────────────────────────────────────────────────────────────────────
# PHASE 7 — APPLY JOINT CORRECTION + RUN SUCCESS GROUP
# ─────────────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("  PHASE 7 — JOINT α=1.0 CORRECTION ON SUCCESS GROUP")
print("=" * 70)

# Apply corrections in-place
originals = {}
for hname, corr in corrections.items():
    block_idx = corr["block_idx"]
    h_start   = corr["h_start"]
    h_end     = corr["h_end"]
    v_param   = model.decoder.block[block_idx].layer[1].EncDecAttention.v.weight
    originals[hname] = v_param.data[h_start:h_end, :].clone()
    v_param.data[h_start:h_end, :] += corr["dWV_a1"].to(device)
    print(f"  Applied correction: {hname}")

# Run success group
print(f"\n  Running {len(success_records)} success examples (corrected model)...")
success_patched = []
for rec in success_records:
    out    = run_forward(rec["enc_input"], rec["success_prefix"], output_attentions=False)
    logits = out.logits[0, -1, :]
    patched_margin   = get_margin(logits)
    delta_margin     = patched_margin - rec["baseline_margin"]
    flip_regression  = 1 if patched_margin < 0 else 0
    ood              = 1 if get_p_sum(logits) < OOD_THRESH else 0

    success_patched.append({
        "ex_idx":          rec["ex_idx"],
        "baseline_margin": rec["baseline_margin"],
        "patched_margin":  patched_margin,
        "delta_margin":    delta_margin,
        "flip_regression": flip_regression,
        "ood":             ood,
    })

# Also sanity-check the fail group under the correction (should match S-061)
print(f"  Running {len(fail_records)} fail examples (corrected model — cross-check)...")
fail_patched = []
for rec in fail_records:
    out    = run_forward(rec["enc_input"], rec["fail_prefix"], output_attentions=False)
    logits = out.logits[0, -1, :]
    patched_margin = get_margin(logits)
    delta_margin   = patched_margin - rec["baseline_margin"]
    flip           = 1 if patched_margin > 0 else 0

    fail_patched.append({
        "ex_idx":          rec["ex_idx"],
        "baseline_margin": rec["baseline_margin"],
        "patched_margin":  patched_margin,
        "delta_margin":    delta_margin,
        "flip":            flip,
    })

# Restore W_V
for hname, corr in corrections.items():
    block_idx = corr["block_idx"]
    h_start   = corr["h_start"]
    h_end     = corr["h_end"]
    v_param   = model.decoder.block[block_idx].layer[1].EncDecAttention.v.weight
    v_param.data[h_start:h_end, :] = originals[hname].to(device)
print("  W_V restored.")

# ─────────────────────────────────────────────────────────────────────
# PHASE 8 — AGGREGATE STATISTICS + VERDICT
# ─────────────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("  PHASE 8 — AGGREGATE STATISTICS + VERDICT")
print("=" * 70)

# Success group
succ_dms      = [r["delta_margin"]    for r in success_patched]
n_flip_reg    = sum(r["flip_regression"] for r in success_patched)
n_ood_succ    = sum(r["ood"]          for r in success_patched)
mean_dm_succ  = float(np.mean(succ_dms))

print(f"\n  SUCCESS GROUP (n={len(success_records)}):")
print(f"    Baseline mean margin:  {mean_bl_succ:+.4f}")
print(f"    Corrected mean margin: {float(np.mean([r['patched_margin'] for r in success_patched])):+.4f}")
print(f"    mean Δm_success:       {mean_dm_succ:+.4f}")
print(f"    Flip regression:       {n_flip_reg}/{len(success_records)}  "
      f"(success→I_WALK after repair — should be ≈0)")
print(f"    OOD:                   {n_ood_succ}/{len(success_records)}")

# Fail group cross-check
fail_dms      = [r["delta_margin"] for r in fail_patched]
n_flip_fail   = sum(r["flip"] for r in fail_patched)
mean_dm_fail  = float(np.mean(fail_dms))
print(f"\n  FAIL GROUP cross-check (n={len(fail_records)}):")
print(f"    mean Δm_fail:  {mean_dm_fail:+.4f}  (S-061 reported: +2.9024)")
print(f"    flips:         {n_flip_fail}/{len(fail_records)}  (S-061 reported: 16/30)")

# Verdict — H4: near-zero two-sided test [-0.3, +0.3]
h4_pass  = H4_NEAR_ZERO_LO <= mean_dm_succ <= H4_NEAR_ZERO_HI
k2_fires = mean_dm_succ < K2_SUCCESS_FLOOR

print("\n" + "─" * 70)
print("  VERDICT")
print("─" * 70)
print(f"\n  Three required numbers:")
print(f"    (1) mean Δm_success  = {mean_dm_succ:+.4f}")
print(f"    (2) flip_regression  = {n_flip_reg}/{len(success_records)}"
      f"  (success examples that flip to I_WALK-preferred)")
print(f"    (3) OOD count        = {n_ood_succ}/{len(success_records)}"
      f"  (P(I_JUMP)+P(I_WALK) < {OOD_THRESH})")
print(f"\n  H4 — near-zero effect on success group [{H4_NEAR_ZERO_LO}, {H4_NEAR_ZERO_HI}]:")
print(f"    mean Δm_success = {mean_dm_succ:+.4f}  {'PASS' if h4_pass else 'FAIL'}")
print(f"\n  K2 — meaningful degradation < {K2_SUCCESS_FLOOR}:")
print(f"    K2: {'FIRES' if k2_fires else 'CLEAR'}")

if h4_pass:
    print(f"\n  *** H4 PASS: repair is specific. Effect on success group is near zero")
    print(f"      (Δm_success = {mean_dm_succ:+.4f} ∈ [{H4_NEAR_ZERO_LO}, {H4_NEAR_ZERO_HI}]).")
elif mean_dm_succ < H4_NEAR_ZERO_LO:
    print(f"\n  *** H4 FAIL (low): mean Δm_success = {mean_dm_succ:+.4f} < {H4_NEAR_ZERO_LO}.")
    print(f"      Correction is leaking into in-distribution representations.")
    print(f"      OOD and in-distribution jump directions are not sufficiently distinct.")
else:
    print(f"\n  *** H4 FAIL (high): mean Δm_success = {mean_dm_succ:+.4f} > {H4_NEAR_ZERO_HI}.")
    print(f"      Correction is making a broader change than expected. Requires interpretation.")

# ─────────────────────────────────────────────────────────────────────
# PHASE 9 — SAVE RESULTS
# ─────────────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("  PHASE 9 — SAVE RESULTS")
print("=" * 70)

results = {
    "script_id":        SCRIPT_ID,
    "seed":             SEED,
    "n_fail_valid":     len(fail_records),
    "n_success_valid":  len(success_records),
    "m1_baseline": {
        "fail_mean_margin":    float(mean_bl_fail),
        "success_mean_margin": float(mean_bl_succ) if success_records else None,
        "n_success_pos":       n_succ_pos,
    },
    "geometry": {
        hname: {
            "cos_v_target_W_U": float(corr["cos_target"]),
            "cos_v_bad_W_U":    float(corr["cos_bad"]),
        }
        for hname, corr in corrections.items()
    },
    "joint_a1_fail_crosscheck": {
        "mean_dm_fail":  float(mean_dm_fail),
        "n_flip_fail":   n_flip_fail,
        "s061_expected_dm_fail":  2.9024,
        "s061_expected_flips":    16,
    },
    "joint_a1_success": {
        "mean_dm_success":    float(mean_dm_succ),
        "n_flip_regression":  n_flip_reg,
        "n_ood":              n_ood_succ,
        "per_example":        success_patched,
    },
    "verdicts": {
        "H4": {"pass": bool(h4_pass), "val": float(mean_dm_succ),
               "threshold_lo": H4_NEAR_ZERO_LO, "threshold_hi": H4_NEAR_ZERO_HI},
        "K2": {"fires": bool(k2_fires), "val": float(mean_dm_succ),
               "threshold": K2_SUCCESS_FLOOR},
    },
}

out_path = "workbench/results/062_results.json"
os.makedirs(os.path.dirname(out_path), exist_ok=True)
with open(out_path, "w") as f:
    json.dump(results, f, indent=2)
print(f"  Results saved: {out_path}")
_save_to_drive(out_path)

# ─────────────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print(f"  {SCRIPT_ID} — COMPLETE")
print("=" * 70)
print(f"\n  Fail cross-check:  Δm_fail={mean_dm_fail:+.4f} "
      f"(S-061: +2.9024)  flips={n_flip_fail}/30 (S-061: 16)")
print(f"  Success result:    Δm_success={mean_dm_succ:+.4f} "
      f"  flip_regression={n_flip_reg}/{len(success_records)}")
print(f"\n  H4: {'PASS' if h4_pass else 'FAIL'}  "
      f"K2: {'FIRES' if k2_fires else 'CLEAR'}")
print(f"\n  Results: {out_path}")
print("\n" + "─" * 70)
print("  After running: update findings/METHODS_REPORT_S061.md with H4 result.")
print("  Update COORDINATION.md; push to S-Track; tell Troy.")
print("─" * 70)
