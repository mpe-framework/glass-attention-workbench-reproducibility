#!/usr/bin/env python3
"""
S-061_WV_GEOMETRY_INTERVENTION.py — W_V Geometry Repair at L4H6/L5H5
Applied Categorical Physics Workbench
Troy Teno | May 2026 | Open Access

Pre-registered hypotheses:
  workbench/proposals/S-061_WV_GEOMETRY_INTERVENTION_PROPOSAL.md
DO NOT modify hypotheses after seeing results.

─────────────────────────────────────────────────────────────────────
SETUP (run once in Colab or terminal):
  pip install torch transformers datasets numpy sentencepiece -q

BACKGROUND:
  The mechanistic account (S-051 → S-059b → S-060 → G-057):
    - L4H6 and L5H5 exhibit value substitution: at the divergence step,
      they attend to the OOD jump encoder representation and their W_V
      maps it to a pro-walk value direction.
    - L6H2/L6H6 are compensatory (S-060 negative specificity; G-057
      toy-scale replication). Do NOT touch them.
    - S-060 showed within-example hook subtraction disrupts LN rescaling
      for large contributors. The repair must be a weight-level change.

  This script applies a rank-1 W_V correction at L4H6 and/or L5H5:

      ΔW_V_h = α × outer(v_correction, u)

  where:
    u   = attention-weighted mean encoder hidden state at div step
          (over 29 fail examples) — normalized. This is exactly what
          W_V operates on for each example.
    v_bad    = W_V_h @ u  (current value direction for jump)
    v_target = pinv(W_O_h) @ W_U_eff  (target: pre-image of pro-jump
               direction under W_O_h, via pseudo-inverse — NOT transpose.
               W_O_h is [512,64]; pinv gives true orthogonal projection
               onto col(W_O_h), not singular-value-biased transpose.)
    v_correction = v_target_normed - v_bad  (in d_kv space)

  The correction is baked into W_V before the forward pass — no hooks,
  no LN disruption (the corrected residual stream passes through LN
  naturally). Weights are restored after each arm/α measurement.

  Arms:
    L5H5-only: modify W_V at block 4, head 5
    L4H6-only: modify W_V at block 3, head 6
    joint:     modify both simultaneously
  α sweep: {0.25, 0.50, 0.75, 1.00}

  Pre-registered hypotheses:
    H1: L5H5-only mean Δmargin_fail > 0 at α=1.0
    H2: L4H6-only mean Δmargin_fail > 0 at α=1.0
    H3: joint ≥ max(L5H5-only, L4H6-only) at α=1.0
    H4: best arm mean Δmargin_success > −1.0 at best fail α
  Kill conditions:
    K1: all arms mean Δmargin_fail ≤ 0 at all α
    K2: best arm mean Δmargin_success < −2.0 at best fail α
    K3: > 5/29 fail or > 5/25 success OOD in any arm/α

Version history:
  V0.1.0 — initial build (pseudo-inverse for v_target;
            attention-weighted encoder direction for u)
─────────────────────────────────────────────────────────────────────
"""

import numpy as np
import json
import os
import random
from collections import defaultdict

SCRIPT_ID = "S-061_WV_GEOMETRY_INTERVENTION_V0.1.0"
SEED = 42
np.random.seed(SEED)
random.seed(SEED)

CHECKPOINT_DIR = "043_t5_scan_checkpoint"

# ── Google Drive restoration (Colab) ─────────────────────────────────────────────
# Mirrors the pattern in 043_SCAN_BC.py: mount Drive and restore checkpoint
# if it isn't already present locally.
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
    pass  # Not in Colab, or Drive already mounted — proceed with local path

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
MAX_INPUT_LEN  = 50
MAX_TARGET_LEN = 100
N_FAIL         = 30
N_SUCCESS      = 25

# Pre-registered thresholds
H1_DELTA_MARGIN_MIN  = 0.0   # L5H5-only fail Δmargin > 0 at α=1.0
H2_DELTA_MARGIN_MIN  = 0.0   # L4H6-only fail Δmargin > 0 at α=1.0
H4_SUCCESS_FLOOR     = -1.0  # success degradation > −1.0
K1_ALL_NONPOS        = 0.0   # fires if all arms ≤ this at all α
K2_SUCCESS_FLOOR     = -2.0  # fires if success < −2.0
K3_OOD_FAIL_MAX      = 5     # OOD threshold for fail group
K3_OOD_SUCCESS_MAX   = 5     # OOD threshold for success group
OOD_THRESH           = 0.01  # P(I_JUMP) + P(I_WALK) < this → OOD

# Head coordinates (block index 0-based, head index 0-based)
# L4H6 = decoder layer 4 (1-indexed) = block 3 (0-indexed), head 6
# L5H5 = decoder layer 5 (1-indexed) = block 4 (0-indexed), head 5
HEADS = {
    "L5H5": {"block": 4, "head": 5},
    "L4H6": {"block": 3, "head": 6},
}

ALPHA_SWEEP = [0.25, 0.50, 0.75, 1.00]

ARMS = {
    "L5H5_only": ["L5H5"],
    "L4H6_only": ["L4H6"],
    "joint":     ["L5H5", "L4H6"],
}

# Token IDs
I_JUMP_ID = 683
I_WALK_ID = 12054
TURN_END_IDS = {6245, 27262}
DIV_IDS = {683, 12054, 448, 5017}

print("=" * 70)
print(f"  {SCRIPT_ID}")
print("  W_V geometry repair: L4H6 and L5H5")
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

# Model dimensions
d_model  = model.config.d_model         # 512
d_kv     = model.config.d_kv            # 64
n_heads  = model.config.num_heads       # 8
BOS_ID   = model.config.decoder_start_token_id  # 0

print(f"  T5 decoder: d_model={d_model}, d_kv={d_kv}, n_heads={n_heads}")

# ─────────────────────────────────────────────────────────────────────
# PHASE 2 — ACTION TOKEN IDs
# ─────────────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("  PHASE 2 — ACTION TOKEN IDs")
print("=" * 70)

for name, ids_str in [
    ("I_JUMP",      "I_JUMP"),
    ("I_WALK",      "I_WALK"),
    ("I_RUN",       "I_RUN"),
    ("I_LOOK",      "I_LOOK"),
    ("I_TURN_LEFT", "I_TURN_LEFT"),
]:
    toks = tokenizer.encode(ids_str, add_special_tokens=False)
    print(f"  {ids_str:15s} → {toks}")

print(f"  I_JUMP_ID={I_JUMP_ID}, I_WALK_ID={I_WALK_ID}")
print(f"  TURN_END_IDS: {TURN_END_IDS}")
print(f"  BOS_ID: {BOS_ID}")

# ─────────────────────────────────────────────────────────────────────
# PHASE 3 — LOAD SCAN + CLASSIFY
# ─────────────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("  PHASE 3 — LOAD SCAN + CLASSIFY")
print("=" * 70)

# datasets library no longer supports SCAN loading scripts.
# Download test split directly from the canonical GitHub repo.
import urllib.request

def _load_scan_txt(url):
    print(f"  Downloading: {url}")
    with urllib.request.urlopen(url) as f:
        lines = f.read().decode("utf-8").strip().split("\n")
    out = []
    for line in lines:
        line = line.strip()
        if not line:
            continue
        # Format: "IN: jump twice OUT: JUMP JUMP"
        cmd_part, act_part = line.split(" OUT: ", 1)
        out.append({
            "commands": cmd_part.replace("IN: ", "").strip(),
            "actions":  act_part.strip(),
        })
    return out

_SCAN_BASE = ("https://raw.githubusercontent.com/"
              "brendenlake/SCAN/master/add_prim_split/")
test_data = _load_scan_txt(_SCAN_BASE + "tasks_test_addprim_jump.txt")
print(f"  Test set: {len(test_data)} examples (from GitHub)")


def tokenize_example(ex):
    cmd = "translate English to actions: " + ex["commands"]
    enc = tokenizer(cmd, return_tensors="pt",
                    max_length=MAX_INPUT_LEN, truncation=True, padding=False)
    return enc


def run_generate(enc_input, max_new_tokens=MAX_TARGET_LEN):
    with torch.no_grad():
        out = model.generate(
            input_ids=enc_input["input_ids"].to(device),
            attention_mask=enc_input["attention_mask"].to(device),
            max_new_tokens=max_new_tokens,
        )
    return out[0].cpu().tolist()


def is_substituted_walk(gen_ids, target_ids):
    """True if gen is (I_TURN_LEFT I_WALK)×N and target has n_jump≥1, n_walk≥1."""
    has_jump_in_gen  = I_JUMP_ID in gen_ids
    has_walk_in_gen  = I_WALK_ID in gen_ids
    has_jump_in_tgt  = I_JUMP_ID in target_ids
    has_walk_in_tgt  = I_WALK_ID in target_ids
    is_around = "around" in " ".join([str(x) for x in target_ids])
    # Simpler: model outputs walk when target has jump
    return (not has_jump_in_gen) and has_walk_in_gen and has_jump_in_tgt and has_walk_in_tgt


def is_compound_jump(ex):
    cmd = ex["commands"].lower()
    acts = ex["actions"].lower()
    return ("jump" in cmd and
            "I_JUMP" in ex["actions"] and
            any(w in cmd for w in ["around", "opposite", "and", "after", "twice", "thrice"]))


print("  Running inference to classify examples...")
rng = random.Random(SEED)

fail_examples = []
success_examples = []

jump_test = [ex for ex in test_data if "jump" in ex["commands"].lower()]
print(f"  Jump test examples: {len(jump_test)}")

for ex in jump_test:
    enc_input = tokenize_example(ex)
    gen_ids = run_generate(enc_input)
    tgt_ids = tokenizer.encode(ex["actions"], add_special_tokens=False)

    has_around = "around" in ex["commands"].lower()
    n_jump_tgt = tgt_ids.count(I_JUMP_ID)
    n_walk_tgt = tgt_ids.count(I_WALK_ID)
    n_jump_gen = gen_ids.count(I_JUMP_ID)
    n_walk_gen = gen_ids.count(I_WALK_ID)

    is_correct = (gen_ids[1:] == tgt_ids + [tokenizer.eos_token_id] or
                  gen_ids[1:-1] == tgt_ids)  # allow EOS variation

    if (has_around and not is_correct and
            n_jump_tgt >= 1 and n_walk_tgt >= 1 and
            n_jump_gen == 0 and n_walk_gen >= 1):
        fail_examples.append((ex, enc_input))
    elif (has_around and is_correct and
          n_jump_tgt >= 1 and n_walk_tgt >= 1):
        success_examples.append((ex, enc_input))

print(f"  substituted_walk fail examples: {len(fail_examples)}")
print(f"  has_around+correct examples:    {len(success_examples)}")

# Shuffle and cap
rng_state = random.Random(SEED)
rng_state.shuffle(fail_examples)
rng_state2 = random.Random(SEED)
rng_state2.shuffle(success_examples)

fail_examples    = fail_examples[:N_FAIL]
success_examples = success_examples[:N_SUCCESS]
print(f"  Using {len(fail_examples)} fail, {len(success_examples)} success (SEED={SEED})")

# ─────────────────────────────────────────────────────────────────────
# PHASE 4 — COMPUTE W_U_EFF FOR v_target
# ─────────────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("  PHASE 4 — W_U_EFF (W_U_diff * LN_weight)")
print("=" * 70)

# W_U[I_JUMP] - W_U[I_WALK]
W_U = model.shared.weight.detach().cpu().float()  # [vocab, d_model]
W_U_diff = W_U[I_JUMP_ID] - W_U[I_WALK_ID]       # [d_model]

# Apply LN weight (direction correction for final RMSNorm)
LN_WEIGHT = model.decoder.final_layer_norm.weight.detach().cpu().float()  # [d_model]
W_U_eff   = W_U_diff * LN_WEIGHT                  # [d_model], fixed part of LN correction

print(f"  ‖W_U_diff‖ = {W_U_diff.norm():.4f}")
print(f"  ‖W_U_eff‖  = {W_U_eff.norm():.4f}")
print(f"  Using W_U_eff (LN-weighted) for v_target computation")

# ─────────────────────────────────────────────────────────────────────
# HELPER FUNCTIONS
# ─────────────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("  PHASE 4b — HELPER FUNCTIONS")
print("=" * 70)


def find_div_step(gen_ids):
    """Find first decoder position where a divergence token is generated.
    Returns (div_step_idx_in_prefix, div_tok_id) or None."""
    # gen_ids includes BOS at position 0
    # The action sequence starts after BOS
    # We look for the first occurrence of a DIV_ID token
    for i, tok in enumerate(gen_ids):
        if i == 0:
            continue  # skip BOS
        if tok in DIV_IDS:
            # The prefix (decoder input to predict this token) is gen_ids[:i]
            return i, tok
    return None, None


def get_margin(logits, jump_id=I_JUMP_ID, walk_id=I_WALK_ID):
    """logit(I_JUMP) - logit(I_WALK) from last position logits."""
    return (logits[jump_id] - logits[walk_id]).item()


def get_p_sum(logits, jump_id=I_JUMP_ID, walk_id=I_WALK_ID):
    probs = torch.softmax(logits, dim=-1)
    return (probs[jump_id] + probs[walk_id]).item()


def run_forward(enc_input, dec_prefix_ids, output_attentions=False):
    """Run a single forward pass. dec_prefix_ids should include BOS."""
    dec_tensor = torch.tensor([dec_prefix_ids], dtype=torch.long, device=device)
    enc_ids    = enc_input["input_ids"].to(device)
    enc_mask   = enc_input["attention_mask"].to(device)
    with torch.no_grad():
        out = model(
            input_ids=enc_ids,
            attention_mask=enc_mask,
            decoder_input_ids=dec_tensor,
            output_attentions=output_attentions,
        )
    return out


print("  Helpers defined.")

# ─────────────────────────────────────────────────────────────────────
# PHASE 5 — COLLECT BASELINES + ENCODER JUMP DIRECTIONS
# ─────────────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("  PHASE 5 — BASELINES + ENCODER JUMP DIRECTIONS (fail group)")
print("=" * 70)

# Accumulate attention-weighted encoder hidden states per head
enc_weighted_accum = {hname: [] for hname in HEADS}

fail_records = []  # {ex_idx, div_prefix, gen_ids, baseline_margin, ...}
n_skip = 0

print(f"  Processing {len(fail_examples)} fail examples...")
for ex_idx, (ex, enc_input) in enumerate(fail_examples):
    if ex_idx % 5 == 0:
        print(f"  Example {ex_idx+1}/{len(fail_examples)} ...")

    # Generate to find divergence step
    gen_ids = run_generate(enc_input)
    div_pos, div_tok = find_div_step(gen_ids)
    if div_pos is None:
        print(f"    [skip ex {ex_idx}] no divergence token found")
        n_skip += 1
        continue

    # Fail prefix = decoder input up to (not including) div token
    fail_prefix = gen_ids[:div_pos]  # includes BOS at position 0
    # Verify fail token is I_WALK
    if div_tok != I_WALK_ID:
        print(f"    [skip ex {ex_idx}] div_tok={div_tok} != I_WALK_ID")
        n_skip += 1
        continue

    # Run forward with attention output to get baseline + enc directions
    out = run_forward(enc_input, fail_prefix, output_attentions=True)
    logits = out.logits[0, -1, :]  # [vocab]
    baseline_margin = get_margin(logits)
    p_sum_baseline  = get_p_sum(logits)

    # Extract encoder last hidden state: [enc_len, d_model]
    enc_hidden = out.encoder_last_hidden_state[0].cpu().float()  # [enc_len, d_model]

    # For each target head: compute attention-weighted encoder direction
    # cross_attentions[block_idx] shape: [1, n_heads, dec_len, enc_len]
    for hname, hinfo in HEADS.items():
        block_idx = hinfo["block"]
        head_idx  = hinfo["head"]
        # cross_attentions is indexed by decoder layer (same as block index for T5)
        ca = out.cross_attentions[block_idx]  # [1, n_heads, dec_len, enc_len]
        attn_weights = ca[0, head_idx, -1, :].cpu().float()  # [enc_len]
        enc_weighted = attn_weights @ enc_hidden  # [d_model]
        enc_weighted_accum[hname].append(enc_weighted)

    fail_records.append({
        "ex_idx":          ex_idx,
        "fail_prefix":     fail_prefix,
        "baseline_margin": baseline_margin,
        "p_sum_baseline":  p_sum_baseline,
        "enc_input":       enc_input,
    })

print(f"\n  Valid fail examples: {len(fail_records)}  (skipped: {n_skip})")
print(f"  Baseline fail group:")
baseline_fail_margins = [r["baseline_margin"] for r in fail_records]
print(f"    mean margin = {np.mean(baseline_fail_margins):.4f}")
print(f"    mean P_sum  = {np.mean([r['p_sum_baseline'] for r in fail_records]):.4f}")

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
    if ex_idx % 5 == 0:
        print(f"  Example {ex_idx+1}/{len(success_examples)} ...")

    gen_ids = run_generate(enc_input)
    div_pos, div_tok = find_div_step(gen_ids)
    if div_pos is None:
        print(f"    [skip ex {ex_idx}] no divergence token found")
        n_skip_s += 1
        continue
    if div_tok != I_JUMP_ID:
        print(f"    [skip ex {ex_idx}] div_tok={div_tok} != I_JUMP_ID (success should predict JUMP)")
        n_skip_s += 1
        continue

    success_prefix = gen_ids[:div_pos]  # includes BOS
    out = run_forward(enc_input, success_prefix, output_attentions=False)
    logits = out.logits[0, -1, :]
    baseline_margin = get_margin(logits)
    p_sum_baseline  = get_p_sum(logits)

    success_records.append({
        "ex_idx":          ex_idx,
        "success_prefix":  success_prefix,
        "baseline_margin": baseline_margin,
        "p_sum_baseline":  p_sum_baseline,
        "enc_input":       enc_input,
    })

print(f"\n  Valid success examples: {len(success_records)}  (skipped: {n_skip_s})")
baseline_succ_margins = [r["baseline_margin"] for r in success_records]
print(f"  Baseline success group:")
print(f"    mean margin = {np.mean(baseline_succ_margins):.4f}")
n_succ_pos = sum(1 for m in baseline_succ_margins if m > 0)
print(f"    fraction with margin > 0 (flip=I_JUMP wins): {n_succ_pos}/{len(success_records)}")

# ─────────────────────────────────────────────────────────────────────
# PHASE 6 — COMPUTE W_V CORRECTION GEOMETRY
# ─────────────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("  PHASE 6 — W_V CORRECTION GEOMETRY")
print("=" * 70)

corrections = {}  # hname -> {u, v_bad, v_target, delta_WV_per_alpha, norm_ratio_per_alpha}

for hname, hinfo in HEADS.items():
    block_idx = hinfo["block"]
    head_idx  = hinfo["head"]

    print(f"\n  Head: {hname} (block {block_idx}, head {head_idx})")

    # --- Compute u: mean attention-weighted encoder direction ---
    enc_vecs = enc_weighted_accum[hname]  # list of [d_model] tensors
    if len(enc_vecs) == 0:
        print(f"    [ERROR] No encoder vectors accumulated for {hname}")
        continue
    enc_mean = torch.stack(enc_vecs, dim=0).mean(dim=0)  # [d_model]
    u = enc_mean / enc_mean.norm()  # normalized
    print(f"    u: mean enc direction, ‖enc_mean‖ = {enc_mean.norm():.4f}")

    # --- Extract W_V_h and W_O_h ---
    enc_dec_attn = model.decoder.block[block_idx].layer[1].EncDecAttention
    W_V_full = enc_dec_attn.v.weight.detach().cpu().float()  # [n_heads*d_kv, d_model]
    W_O_full = enc_dec_attn.o.weight.detach().cpu().float()  # [d_model, n_heads*d_kv]

    h_start = head_idx * d_kv
    h_end   = h_start + d_kv

    W_V_h = W_V_full[h_start:h_end, :]  # [d_kv, d_model]
    W_O_h = W_O_full[:, h_start:h_end]  # [d_model, d_kv]

    print(f"    W_V_h shape: {tuple(W_V_h.shape)} (expected [{d_kv}, {d_model}])")
    print(f"    W_O_h shape: {tuple(W_O_h.shape)} (expected [{d_model}, {d_kv}])")

    # --- Compute v_bad (current value direction for jump) ---
    v_bad = W_V_h @ u  # [d_kv]
    print(f"    ‖v_bad‖ = {v_bad.norm():.4f}")

    # --- Compute v_target via pseudo-inverse of W_O_h ---
    # W_O_h: [d_model, d_kv] = [512, 64]
    # pinv(W_O_h): [d_kv, d_model] = [64, 512]
    # v_target = pinv(W_O_h) @ W_U_eff: minimum-norm least-squares solution
    # such that W_O_h @ v_target ≈ proj_{col(W_O_h)}(W_U_eff)
    print(f"    Computing pinv(W_O_h)...")
    W_O_h_pinv = torch.linalg.pinv(W_O_h)  # [d_kv, d_model]
    W_U_eff_cpu = W_U_eff.cpu()
    v_target_raw = W_O_h_pinv @ W_U_eff_cpu  # [d_kv]

    # Normalize v_target to ‖v_bad‖ (preserve scale)
    v_target = v_target_raw * (v_bad.norm() / v_target_raw.norm())
    print(f"    ‖v_target_raw‖ = {v_target_raw.norm():.4f}  → normalized to ‖v_bad‖ = {v_bad.norm():.4f}")

    # --- Verify alignment ---
    # W_O_h @ v_target should be aligned with W_U_eff
    proj_target = W_O_h @ v_target   # [d_model]
    proj_bad    = W_O_h @ v_bad      # [d_model]
    W_U_eff_norm = W_U_eff_cpu / W_U_eff_cpu.norm()
    cos_target = torch.dot(proj_target / proj_target.norm(), W_U_eff_norm).item()
    cos_bad    = torch.dot(proj_bad    / proj_bad.norm(),    W_U_eff_norm).item()
    print(f"    cos(W_O_h @ v_target, W_U_eff) = {cos_target:+.4f}  (should be > 0)")
    print(f"    cos(W_O_h @ v_bad,    W_U_eff) = {cos_bad:+.4f}  (should be < 0 for value-sub heads)")

    # Predicted δlogit contribution from correction (approximate, ignoring LN scale):
    delta_v_full = proj_target - proj_bad  # [d_model]
    pred_delta_logit = torch.dot(W_U_eff_norm * W_U_eff_cpu.norm(), delta_v_full).item()
    print(f"    Predicted Δδlogit at α=1 (linear approx, pre-LN): {pred_delta_logit:+.4f}")

    # --- Compute rank-1 ΔW_V for each α ---
    v_correction = v_target - v_bad  # [d_kv]
    print(f"    ‖v_correction‖ = {v_correction.norm():.4f}")

    delta_WV_per_alpha = {}
    norm_ratio_per_alpha = {}
    W_V_h_frob = W_V_h.norm(p="fro").item()

    for alpha in ALPHA_SWEEP:
        dWV = alpha * torch.outer(v_correction, u)  # [d_kv, d_model]
        delta_WV_per_alpha[alpha] = dWV
        ratio = dWV.norm(p="fro").item() / W_V_h_frob
        norm_ratio_per_alpha[alpha] = ratio

    print(f"    ‖ΔW_V‖_F / ‖W_V_h‖_F:")
    for alpha in ALPHA_SWEEP:
        print(f"      α={alpha:.2f}: {norm_ratio_per_alpha[alpha]:.4f} (M5)")

    corrections[hname] = {
        "u":                   u,
        "v_bad":               v_bad,
        "v_target":            v_target,
        "v_correction":        v_correction,
        "cos_target":          cos_target,
        "cos_bad":             cos_bad,
        "pred_delta_logit_a1": pred_delta_logit,
        "delta_WV_per_alpha":  delta_WV_per_alpha,
        "norm_ratio_per_alpha": norm_ratio_per_alpha,
        "block_idx":           block_idx,
        "head_idx":            head_idx,
        "h_start":             h_start,
        "h_end":               h_end,
    }

print("\n  Correction geometry complete.")

# ─────────────────────────────────────────────────────────────────────
# PHASE 7 — RUN CORRECTION ARMS
# ─────────────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("  PHASE 7 — CORRECTION ARMS")
print(f"  {len(ARMS)} arms × {len(ALPHA_SWEEP)} α values")
print(f"  × ({len(fail_records)} fail + {len(success_records)} success) = "
      f"{len(ARMS) * len(ALPHA_SWEEP) * (len(fail_records) + len(success_records))} forward passes")
print("=" * 70)

# Results: arm_name → alpha → {"fail": [...], "success": [...]}
arm_results = {}

for arm_name, arm_head_names in ARMS.items():
    arm_results[arm_name] = {}

    for alpha in ALPHA_SWEEP:
        arm_results[arm_name][alpha] = {"fail": [], "success": []}

        print(f"\n  Arm: {arm_name} | α={alpha:.2f}")

        # --- Apply W_V corrections in-place ---
        originals = {}
        for hname in arm_head_names:
            corr = corrections[hname]
            block_idx = corr["block_idx"]
            h_start   = corr["h_start"]
            h_end     = corr["h_end"]
            dWV       = corr["delta_WV_per_alpha"][alpha].to(device)

            v_param = model.decoder.block[block_idx].layer[1].EncDecAttention.v.weight
            originals[hname] = v_param.data[h_start:h_end, :].clone()
            v_param.data[h_start:h_end, :] += dWV

        # --- Run fail group ---
        print(f"    Running {len(fail_records)} fail examples...")
        for rec in fail_records:
            out = run_forward(rec["enc_input"], rec["fail_prefix"], output_attentions=False)
            logits = out.logits[0, -1, :]
            patched_margin = get_margin(logits)
            p_jump = torch.softmax(logits, dim=-1)[I_JUMP_ID].item()
            p_walk = torch.softmax(logits, dim=-1)[I_WALK_ID].item()
            p_sum  = p_jump + p_walk
            delta_margin = patched_margin - rec["baseline_margin"]
            flip = 1 if patched_margin > 0 else 0
            ood  = 1 if p_sum < OOD_THRESH else 0

            arm_results[arm_name][alpha]["fail"].append({
                "ex_idx":          rec["ex_idx"],
                "baseline_margin": rec["baseline_margin"],
                "patched_margin":  patched_margin,
                "delta_margin":    delta_margin,
                "flip":            flip,
                "p_jump":          p_jump,
                "p_walk":          p_walk,
                "p_sum":           p_sum,
                "ood":             ood,
            })

        # --- Run success group ---
        print(f"    Running {len(success_records)} success examples...")
        for rec in success_records:
            out = run_forward(rec["enc_input"], rec["success_prefix"], output_attentions=False)
            logits = out.logits[0, -1, :]
            patched_margin = get_margin(logits)
            p_jump = torch.softmax(logits, dim=-1)[I_JUMP_ID].item()
            p_walk = torch.softmax(logits, dim=-1)[I_WALK_ID].item()
            p_sum  = p_jump + p_walk
            delta_margin = patched_margin - rec["baseline_margin"]
            # For success: a "flip" means the corrected model now prefers I_WALK (regression)
            flip_regression = 1 if patched_margin < 0 else 0
            ood = 1 if p_sum < OOD_THRESH else 0

            arm_results[arm_name][alpha]["success"].append({
                "ex_idx":          rec["ex_idx"],
                "baseline_margin": rec["baseline_margin"],
                "patched_margin":  patched_margin,
                "delta_margin":    delta_margin,
                "flip_regression": flip_regression,
                "p_jump":          p_jump,
                "p_walk":          p_walk,
                "p_sum":           p_sum,
                "ood":             ood,
            })

        # --- Restore W_V ---
        for hname in arm_head_names:
            corr = corrections[hname]
            block_idx = corr["block_idx"]
            h_start   = corr["h_start"]
            h_end     = corr["h_end"]
            v_param = model.decoder.block[block_idx].layer[1].EncDecAttention.v.weight
            v_param.data[h_start:h_end, :] = originals[hname].to(device)

        # Quick summary
        fail_dms = [r["delta_margin"] for r in arm_results[arm_name][alpha]["fail"]]
        succ_dms = [r["delta_margin"] for r in arm_results[arm_name][alpha]["success"]]
        n_flip_fail = sum(r["flip"] for r in arm_results[arm_name][alpha]["fail"])
        n_ood_fail  = sum(r["ood"]  for r in arm_results[arm_name][alpha]["fail"])
        n_ood_succ  = sum(r["ood"]  for r in arm_results[arm_name][alpha]["success"])
        print(f"    Δm_fail = {np.mean(fail_dms):+.4f} | Δm_succ = {np.mean(succ_dms):+.4f} | "
              f"flips_fail={n_flip_fail}/{len(fail_records)} | "
              f"ood_fail={n_ood_fail} ood_succ={n_ood_succ}")

print("\n  All arms complete.")

# ─────────────────────────────────────────────────────────────────────
# PHASE 8 — AGGREGATE STATISTICS + VERDICTS
# ─────────────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("  PHASE 8 — AGGREGATE STATISTICS")
print("=" * 70)

# M1 — Baselines
print("\n--- M1: Baseline ---")
n_fail_valid    = len(fail_records)
n_success_valid = len(success_records)
mean_bl_fail = np.mean(baseline_fail_margins)
mean_bl_succ = np.mean(baseline_succ_margins)
n_fail_pos   = sum(1 for m in baseline_fail_margins if m > 0)
n_succ_pos   = sum(1 for m in baseline_succ_margins if m > 0)
print(f"  Fail group:    n={n_fail_valid}, mean margin={mean_bl_fail:.4f}, "
      f"flip frac={n_fail_pos}/{n_fail_valid} (should be 0)")
print(f"  Success group: n={n_success_valid}, mean margin={mean_bl_succ:.4f}, "
      f"I_JUMP wins: {n_succ_pos}/{n_success_valid}")

# M2/M3/M4 — Per-arm, per-α
print("\n--- M2/M3/M4: Per-arm, per-α results ---")
print(f"\n{'Arm':<12} {'α':>5}  {'Δm_fail':>9}  {'Δm_succ':>9}  "
      f"{'spec_ratio':>10}  {'flip_gain':>10}  {'ood_fail':>8}  {'ood_succ':>8}")
print("-" * 82)

agg = {}
for arm_name in ARMS:
    agg[arm_name] = {}
    for alpha in ALPHA_SWEEP:
        fail_recs = arm_results[arm_name][alpha]["fail"]
        succ_recs = arm_results[arm_name][alpha]["success"]
        fail_dms  = [r["delta_margin"] for r in fail_recs]
        succ_dms  = [r["delta_margin"] for r in succ_recs]
        n_flip    = sum(r["flip"] for r in fail_recs)
        n_flip_r  = sum(r["flip_regression"] for r in succ_recs)
        n_ood_f   = sum(r["ood"] for r in fail_recs)
        n_ood_s   = sum(r["ood"] for r in succ_recs)
        mean_dm_f = np.mean(fail_dms)
        mean_dm_s = np.mean(succ_dms)
        # Specificity ratio: fail improvement / |success degradation|
        denom = abs(mean_dm_s) if abs(mean_dm_s) > 1e-6 else 1e-6
        spec_ratio = mean_dm_f / denom if mean_dm_s <= 0 else float("inf")

        flip_gain_frac = n_flip / len(fail_recs)

        agg[arm_name][alpha] = {
            "mean_dm_fail":    mean_dm_f,
            "mean_dm_success": mean_dm_s,
            "spec_ratio":      spec_ratio,
            "n_flip_fail":     n_flip,
            "n_flip_regression": n_flip_r,
            "flip_gain_frac":  flip_gain_frac,
            "n_ood_fail":      n_ood_f,
            "n_ood_success":   n_ood_s,
        }

        print(f"  {arm_name:<12} {alpha:>5.2f}  {mean_dm_f:>+9.4f}  {mean_dm_s:>+9.4f}  "
              f"{spec_ratio:>10.4f}  {flip_gain_frac:>10.4f}  "
              f"{n_ood_f:>4}/{len(fail_recs):<3}  {n_ood_s:>4}/{len(succ_recs):<3}")
    print()

# M5 — Norm ratios (already printed in Phase 6, just collect for JSON)
print("\n--- M5: W_V perturbation norms ---")
for hname, corr in corrections.items():
    print(f"  {hname}:  cos(v_target, W_U_eff direction) = {corr['cos_target']:+.4f}")
    print(f"          cos(v_bad,    W_U_eff direction) = {corr['cos_bad']:+.4f}")
    print(f"          predicted Δδlogit at α=1 ≈ {corr['pred_delta_logit_a1']:+.4f}")
    for alpha in ALPHA_SWEEP:
        print(f"          α={alpha:.2f}: ‖ΔW_V‖_F/‖W_V_h‖_F = "
              f"{corr['norm_ratio_per_alpha'][alpha]:.5f}")

# ─────────────────────────────────────────────────────────────────────
# HYPOTHESIS VERDICTS
# ─────────────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("  PHASE 8b — HYPOTHESIS VERDICTS")
print("=" * 70)

# H1: L5H5-only fail Δmargin > 0 at α=1.0
h1_val  = agg["L5H5_only"][1.00]["mean_dm_fail"]
h1_pass = h1_val > H1_DELTA_MARGIN_MIN
print(f"\n  H1 — L5H5-only mean Δm_fail > 0 at α=1.0:")
print(f"    mean Δm_fail = {h1_val:+.4f}  {'PASS' if h1_pass else 'FAIL'}")

# H2: L4H6-only fail Δmargin > 0 at α=1.0
h2_val  = agg["L4H6_only"][1.00]["mean_dm_fail"]
h2_pass = h2_val > H2_DELTA_MARGIN_MIN
print(f"\n  H2 — L4H6-only mean Δm_fail > 0 at α=1.0:")
print(f"    mean Δm_fail = {h2_val:+.4f}  {'PASS' if h2_pass else 'FAIL'}")

# H3: joint ≥ max(L5H5-only, L4H6-only) at α=1.0
h3_joint = agg["joint"][1.00]["mean_dm_fail"]
h3_max   = max(agg["L5H5_only"][1.00]["mean_dm_fail"],
               agg["L4H6_only"][1.00]["mean_dm_fail"])
h3_pass  = h3_joint >= h3_max
print(f"\n  H3 — joint ≥ max(individual) at α=1.0:")
print(f"    joint={h3_joint:+.4f}, max(individual)={h3_max:+.4f}  {'PASS' if h3_pass else 'FAIL'}")

# H4: best arm success degradation > −1.0 at best fail α
# Find best fail arm/α combination
best_fail_dm = -float("inf")
best_arm_for_H4 = None
best_alpha_for_H4 = None
for arm_name in ARMS:
    for alpha in ALPHA_SWEEP:
        dm = agg[arm_name][alpha]["mean_dm_fail"]
        if dm > best_fail_dm:
            best_fail_dm = dm
            best_arm_for_H4 = arm_name
            best_alpha_for_H4 = alpha

h4_succ_dm = agg[best_arm_for_H4][best_alpha_for_H4]["mean_dm_success"]
h4_pass    = h4_succ_dm > H4_SUCCESS_FLOOR
print(f"\n  H4 — success not degraded > {H4_SUCCESS_FLOOR} at best fail arm/α:")
print(f"    Best fail: arm={best_arm_for_H4}, α={best_alpha_for_H4:.2f}, Δm_fail={best_fail_dm:+.4f}")
print(f"    Δm_success at that arm/α = {h4_succ_dm:+.4f}  {'PASS' if h4_pass else 'FAIL'}")

# K1: all arms Δm_fail ≤ 0 at all α
all_fail_dms = [agg[arm][alpha]["mean_dm_fail"]
                for arm in ARMS for alpha in ALPHA_SWEEP]
k1_fires = all(dm <= K1_ALL_NONPOS for dm in all_fail_dms)
max_fail_dm = max(all_fail_dms)
print(f"\n  K1 — all arms Δm_fail ≤ 0 at all α:")
print(f"    max Δm_fail across all arms/α = {max_fail_dm:+.4f}")
print(f"  K1: {'FIRES' if k1_fires else 'CLEAR'}")
if k1_fires:
    print("  *** K1 FIRES: rank-1 W_V correction does not improve fail margin.")
    print("  *** Value substitution mechanism not confirmed via this repair.")

# K2: best arm success < −2.0 at best fail α
k2_fires = h4_succ_dm < K2_SUCCESS_FLOOR
print(f"\n  K2 — success degradation < {K2_SUCCESS_FLOOR} at best fail arm/α:")
print(f"    Δm_success = {h4_succ_dm:+.4f}  K2: {'FIRES' if k2_fires else 'CLEAR'}")
if k2_fires:
    print("  *** K2 FIRES: repair substantially degrades success group.")
    print("  *** Jump direction overlaps too much with in-distribution representations.")

# K3: >5 OOD in any arm/α for fail or success
k3_fires = False
k3_details = []
for arm_name in ARMS:
    for alpha in ALPHA_SWEEP:
        n_ood_f = agg[arm_name][alpha]["n_ood_fail"]
        n_ood_s = agg[arm_name][alpha]["n_ood_success"]
        if n_ood_f > K3_OOD_FAIL_MAX:
            k3_fires = True
            k3_details.append(f"{arm_name}/α={alpha:.2f}: {n_ood_f} fail OOD")
        if n_ood_s > K3_OOD_SUCCESS_MAX:
            k3_fires = True
            k3_details.append(f"{arm_name}/α={alpha:.2f}: {n_ood_s} success OOD")
print(f"\n  K3 — > {K3_OOD_FAIL_MAX} OOD in any arm/α:")
if k3_fires:
    for d in k3_details:
        print(f"    {d}")
    print(f"  K3: FIRES")
else:
    print(f"  K3: CLEAR")

# H5 — dose-response curve (exploratory, no pass/fail threshold)
print(f"\n  H5 (exploratory) — dose-response Δm_fail vs α:")
for arm_name in ARMS:
    print(f"    {arm_name}:")
    for alpha in ALPHA_SWEEP:
        dm = agg[arm_name][alpha]["mean_dm_fail"]
        print(f"      α={alpha:.2f}: Δm_fail={dm:+.4f}")

# Summary
print("\n" + "─" * 70)
print("  VERDICT SUMMARY")
print("─" * 70)
print(f"  H1 (L5H5-only Δm_fail > 0 at α=1.0):         {'PASS' if h1_pass else 'FAIL'}")
print(f"  H2 (L4H6-only Δm_fail > 0 at α=1.0):         {'PASS' if h2_pass else 'FAIL'}")
print(f"  H3 (joint ≥ max(individual) at α=1.0):        {'PASS' if h3_pass else 'FAIL'}")
print(f"  H4 (success not degraded > {H4_SUCCESS_FLOOR}):           {'PASS' if h4_pass else 'FAIL'}")
print(f"  K1 (all arms non-positive):                   {'FIRES' if k1_fires else 'CLEAR'}")
print(f"  K2 (success < {K2_SUCCESS_FLOOR}):                      {'FIRES' if k2_fires else 'CLEAR'}")
print(f"  K3 (> {K3_OOD_FAIL_MAX} OOD in any arm/α):                {'FIRES' if k3_fires else 'CLEAR'}")

# ─────────────────────────────────────────────────────────────────────
# PHASE 9 — SAVE RESULTS
# ─────────────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("  PHASE 9 — SAVE RESULTS")
print("=" * 70)

results = {
    "script_id": SCRIPT_ID,
    "seed": SEED,
    "n_fail_valid":    n_fail_valid,
    "n_fail_skip":     n_skip,
    "n_success_valid": n_success_valid,
    "n_success_skip":  n_skip_s,
    "m1_baseline": {
        "fail":    {"n": n_fail_valid,    "mean_margin": float(mean_bl_fail),
                    "n_pos_flip": n_fail_pos},
        "success": {"n": n_success_valid, "mean_margin": float(mean_bl_succ),
                    "n_jump_wins": n_succ_pos},
    },
    "m5_geometry": {
        hname: {
            "cos_v_target_W_U": float(corr["cos_target"]),
            "cos_v_bad_W_U":    float(corr["cos_bad"]),
            "pred_delta_logit_a1": float(corr["pred_delta_logit_a1"]),
            "norm_ratios": {
                str(alpha): float(corr["norm_ratio_per_alpha"][alpha])
                for alpha in ALPHA_SWEEP
            },
        }
        for hname, corr in corrections.items()
    },
    "m2_m3_agg": {
        arm_name: {
            str(alpha): {
                "mean_dm_fail":      float(agg[arm_name][alpha]["mean_dm_fail"]),
                "mean_dm_success":   float(agg[arm_name][alpha]["mean_dm_success"]),
                "spec_ratio":        float(agg[arm_name][alpha]["spec_ratio"]),
                "n_flip_fail":       agg[arm_name][alpha]["n_flip_fail"],
                "flip_gain_frac":    float(agg[arm_name][alpha]["flip_gain_frac"]),
                "n_flip_regression": agg[arm_name][alpha]["n_flip_regression"],
                "n_ood_fail":        agg[arm_name][alpha]["n_ood_fail"],
                "n_ood_success":     agg[arm_name][alpha]["n_ood_success"],
            }
            for alpha in ALPHA_SWEEP
        }
        for arm_name in ARMS
    },
    "verdicts": {
        "H1": {"pass": bool(h1_pass), "val": float(h1_val)},
        "H2": {"pass": bool(h2_pass), "val": float(h2_val)},
        "H3": {"pass": bool(h3_pass), "joint_val": float(h3_joint), "max_individual": float(h3_max)},
        "H4": {"pass": bool(h4_pass), "best_arm": best_arm_for_H4,
               "best_alpha": float(best_alpha_for_H4), "dm_success": float(h4_succ_dm)},
        "K1": {"fires": bool(k1_fires), "max_dm_fail": float(max_fail_dm)},
        "K2": {"fires": bool(k2_fires), "dm_success_at_best_fail": float(h4_succ_dm)},
        "K3": {"fires": bool(k3_fires), "details": k3_details},
    },
    "per_arm_per_alpha_per_example": {
        arm_name: {
            str(alpha): {
                "fail":    arm_results[arm_name][alpha]["fail"],
                "success": arm_results[arm_name][alpha]["success"],
            }
            for alpha in ALPHA_SWEEP
        }
        for arm_name in ARMS
    },
}

out_path = "workbench/results/061_results.json"
os.makedirs(os.path.dirname(out_path), exist_ok=True)
with open(out_path, "w") as f:
    json.dump(results, f, indent=2)
print(f"  Results saved: {out_path}")

# Also save to Drive if available
_save_to_drive(out_path)

# ─────────────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print(f"  {SCRIPT_ID} — COMPLETE")
print("=" * 70)
print(f"\n  n_fail={n_fail_valid}  n_success={n_success_valid}")
print(f"\n  Baseline: fail margin={mean_bl_fail:.4f} | success margin={mean_bl_succ:.4f}")
print(f"\n  Geometry:")
for hname, corr in corrections.items():
    print(f"    {hname}: cos(v_target, W_U_eff)={corr['cos_target']:+.4f}  "
          f"cos(v_bad, W_U_eff)={corr['cos_bad']:+.4f}")
print(f"\n  Results summary:")
print(f"  {'Arm':<12}  {'α':>5}  {'Δm_fail':>9}  {'Δm_succ':>9}  {'flips':>6}")
for arm_name in ARMS:
    for alpha in ALPHA_SWEEP:
        a = agg[arm_name][alpha]
        print(f"  {arm_name:<12}  {alpha:>5.2f}  "
              f"{a['mean_dm_fail']:>+9.4f}  {a['mean_dm_success']:>+9.4f}  "
              f"{a['n_flip_fail']:>4}/{n_fail_valid}")
    print()
print(f"\n  Verdicts: H1={'PASS' if h1_pass else 'FAIL'}  H2={'PASS' if h2_pass else 'FAIL'}  "
      f"H3={'PASS' if h3_pass else 'FAIL'}  H4={'PASS' if h4_pass else 'FAIL'}")
print(f"  Kill:     K1={'FIRES' if k1_fires else 'CLEAR'}  "
      f"K2={'FIRES' if k2_fires else 'CLEAR'}  "
      f"K3={'FIRES' if k3_fires else 'CLEAR'}")
print(f"\n  Results: {out_path}")
print("\n" + "─" * 70)
print("  After running: write findings/METHODS_REPORT_S061.md")
print("  Update COORDINATION.md; push to S-Track; tell Troy.")
print("─" * 70)
