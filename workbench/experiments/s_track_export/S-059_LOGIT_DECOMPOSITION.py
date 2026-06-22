#!/usr/bin/env python3
"""
S-059_LOGIT_DECOMPOSITION.py — Logit Decomposition: Which Heads Drive the Wrong Decision?
Applied Categorical Physics Workbench
Troy Teno | May 2026 | Open Access

Pre-registered hypotheses:
  workbench/proposals/S-059_LOGIT_DECOMPOSITION_PROPOSAL.md
DO NOT modify hypotheses after seeing results.

─────────────────────────────────────────────────────────────────────
SETUP (run once in Colab or terminal):
  pip install torch transformers datasets numpy scipy sentencepiece -q

BACKGROUND:
  S-058 found that cross-example activation patching is non-specific:
  every arm including the L3H0 control produced 100% flip rate, with
  P(I_JUMP) ≈ P(I_WALK) ≈ 0 post-patch (OOD state). Root cause: 72%
  of L4H6's attention integrates non-jump encoder positions; the
  transplant vector carries global context differences between commands.

  S-059 uses a direct measurement instead: logit decomposition. For
  each of the 48 decoder cross-attention heads, at the action-slot
  divergence step, compute:

      δlogit_h = (W_U[I_JUMP] − W_U[I_WALK]) · contrib_h

  where contrib_h is the head's V/O projection contribution to the
  decoder residual stream (same compute_head_contrib method as S-057/S-058).
  This gives, in model-native logit units, how much each head pushes the
  decision toward I_JUMP (positive) or I_WALK (negative).

  No patching. No donor examples. No matching. Direct measurement on the
  original fail trajectory.

  Pre-registered hypotheses:
    H1: L4H6, L5H2, L5H5 are in top-5 by |δlogit_h| in fail group,
        and all three have mean δlogit_h < 0 (pro-walk)
    H2: Δ(δlogit_h) = mean_success − mean_fail ≥ +0.20 for each
        of L4H6, L5H2, L5H5
    H3: target heads (L4H6+L5H2+L5H5) account for ≥ 30% of total
        reconstructed negative margin in fail group
    H4: |δlogit_L3H0| < |δlogit_L4H6| AND |δlogit_L3H4| < |δlogit_L4H6|
        (sanity check: diagnostic controls smaller than target heads)

  Kill conditions:
    K1: target heads have positive mean δlogit in fail group (pro-jump,
        not pro-walk) — mechanism account is wrong
    K2: Σδlogit_h / observed_margin < 0.20 in absolute terms —
        decomposition does not capture dominant signal (FFN/embedding)

Version history:
  V0.1.0 — initial build
─────────────────────────────────────────────────────────────────────
"""

import numpy as np
import json
import os
import random
from collections import defaultdict

SCRIPT_ID = "S-059_LOGIT_DECOMPOSITION_V0.1.0"
# V0.1.0 is sealed — do not modify. V0.2.0 adds LayerNorm correction.
# See findings/METHODS_REPORT_S059.md for V0.1.0 results and methodological finding.
SEED = 42
np.random.seed(SEED)
random.seed(SEED)

CHECKPOINT_DIR = "043_t5_scan_checkpoint"
MAX_INPUT_LEN  = 50
MAX_TARGET_LEN = 100
N_FAIL         = 30
N_SUCCESS      = 25

# Pre-registered thresholds
H2_DELTA_MIN   = 0.20   # Δ(δlogit_h) = success − fail threshold per target head
H3_SHARE_MIN   = 0.30   # target heads' share of total reconstructed negative margin
K2_RECON_MIN   = 0.20   # reconstruction fraction of observed margin (absolute)

# Head definitions: (decoder_block_0indexed, head_0indexed, label)
# Target heads (value-substitution candidates from S-051/S-057)
L4H6 = (3, 6, "L4H6")
L5H2 = (4, 2, "L5H2")
L5H5 = (4, 5, "L5H5")
# Control heads
L3H0 = (2, 0, "L3H0")   # global-context head — diagnostic, not causal
L3H4 = (2, 4, "L3H4")   # suppressive head — diagnostic per G-056

TARGET_HEADS   = [L4H6, L5H2, L5H5]
CONTROL_HEADS  = [L3H0, L3H4]

# ── Google Drive ────────────────────────────────────────────────────────────────────────────
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
        print(f"  Restored checkpoint from Drive.")
    print(f"  Drive mounted.")
except Exception:
    print("  Drive not available.")

def save_to_drive(*filenames):
    if DRIVE_DIR is None:
        return
    import shutil
    for fname in filenames:
        if os.path.exists(fname):
            dst_dir = os.path.join(DRIVE_DIR, os.path.dirname(fname))
            os.makedirs(dst_dir, exist_ok=True)
            shutil.copy(fname, os.path.join(DRIVE_DIR, fname))
            print(f"  Saved to Drive: {fname}")

print(f"\n{'='*70}")
print(f"  {SCRIPT_ID}")
print(f"  Logit decomposition: which heads drive the wrong decision?")
print(f"{'='*70}")

# ── Imports ─────────────────────────────────────────────────────────────────────────────
print("\nImporting libraries...")
try:
    import torch
    import torch.nn.functional as F
    from transformers import T5ForConditionalGeneration, T5Tokenizer
except ImportError as e:
    raise SystemExit(f"\nMissing library: {e}\n"
                     "Run: pip install torch transformers datasets scipy sentencepiece -q\n")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"  Device: {device}")
torch.manual_seed(SEED)

# ── Phase 1: Load checkpoint ───────────────────────────────────────────────────────────────────
print("\n" + "="*70)
print("  PHASE 1 — LOAD S-043 CHECKPOINT")
print("="*70)

if not os.path.exists(CHECKPOINT_DIR):
    raise SystemExit(f"\n  Checkpoint not found: {CHECKPOINT_DIR}\n"
                     "  Run S-043 first or restore from Drive.\n")

tokenizer = T5Tokenizer.from_pretrained(CHECKPOINT_DIR)
model     = T5ForConditionalGeneration.from_pretrained(CHECKPOINT_DIR).to(device)
model.eval()
print(f"  Loaded {CHECKPOINT_DIR!r}.")

# ── Phase 2: Token IDs and unembedding direction ───────────────────────────────────────────────
print("\n" + "="*70)
print("  PHASE 2 — ACTION TOKEN IDs AND UNEMBEDDING DIRECTION VECTOR")
print("="*70)

core_actions    = ["I_JUMP", "I_WALK", "I_RUN", "I_LOOK"]
ALL_ACTION_NAMES = core_actions + ["I_TURN_LEFT", "I_TURN_RIGHT"]

raw_ids = {name: tokenizer.encode(name, add_special_tokens=False)
           for name in ALL_ACTION_NAMES}
print(f"\n  Full tokenizations:")
for name, ids in raw_ids.items():
    print(f"  {name:<15} → {ids}")

DIV_IDS       = {name: raw_ids[name][2] for name in core_actions}
TURN_LAST_IDS = {name: raw_ids[name][-1] for name in ["I_TURN_LEFT", "I_TURN_RIGHT"]}
TURN_END_IDS  = set(TURN_LAST_IDS.values())
DIV_TOKEN_SET = set(DIV_IDS.values())
id_to_action  = {v: k for k, v in DIV_IDS.items()}

I_JUMP_ID = DIV_IDS["I_JUMP"]
I_WALK_ID = DIV_IDS["I_WALK"]

print(f"\n  Divergence-point IDs: {DIV_IDS}")
print(f"  I_JUMP_ID={I_JUMP_ID}, I_WALK_ID={I_WALK_ID}")
print(f"  TURN_END_IDS:         {TURN_END_IDS}")

n_dec_layers = model.config.num_decoder_layers   # 6
n_heads      = model.config.num_heads            # 8
d_kv         = model.config.d_kv                 # 64
d_model      = model.config.d_model              # 512
n_total_heads = n_dec_layers * n_heads           # 48
print(f"\n  T5 decoder: {n_dec_layers} layers × {n_heads} heads = {n_total_heads} total heads")
print(f"  d_kv={d_kv}, d_model={d_model}")

# Compute unembedding direction: W_U[I_JUMP] − W_U[I_WALK]
# T5 ties embeddings: lm_head.weight == shared.weight
# W_U[t] is the row of lm_head.weight corresponding to token t.
# δlogit_h = W_U_diff · contrib_h separates I_JUMP from I_WALK in logit space.
with torch.no_grad():
    W_U_jump = model.lm_head.weight[I_JUMP_ID].cpu().float()   # [d_model]
    W_U_walk = model.lm_head.weight[I_WALK_ID].cpu().float()   # [d_model]
    W_U_diff = W_U_jump - W_U_walk                              # [d_model]
    EMBED_WALK = model.shared.weight[I_WALK_ID].cpu().float()
    EMBED_JUMP = model.shared.weight[I_JUMP_ID].cpu().float()

print(f"\n  ‖W_U[I_JUMP]‖ = {W_U_jump.norm():.2f}")
print(f"  ‖W_U[I_WALK]‖ = {W_U_walk.norm():.2f}")
print(f"  ‖W_U_diff‖    = {W_U_diff.norm():.2f}")
print(f"\n  ‖embed(I_WALK div)‖ = {EMBED_WALK.norm():.2f}")
print(f"  ‖embed(I_JUMP div)‖ = {EMBED_JUMP.norm():.2f}")
# Verify embedding ties (lm_head.weight == shared.weight for T5)
tie_diff = (model.lm_head.weight[I_JUMP_ID] - model.shared.weight[I_JUMP_ID]).norm().item()
print(f"\n  Embedding tie check (lm_head vs shared, I_JUMP): diff={tie_diff:.6f}")

# ── Phase 3: Load SCAN + classify ───────────────────────────────────────────────────────────────
print("\n" + "="*70)
print("  PHASE 3 — LOAD SCAN + CLASSIFY EXAMPLES")
print("="*70)

train_raw = test_raw = None
try:
    from datasets import load_dataset
    for _cfg in ("addprim_jump", "add_prim_jump"):
        try:
            _ds = load_dataset("scan", _cfg)
            train_raw = list(_ds["train"])
            test_raw  = list(_ds["test"])
            print(f"  Loaded via HuggingFace datasets ({_cfg}).")
            break
        except Exception:
            pass
except ImportError:
    pass

if train_raw is None:
    import urllib.request
    _BASE = ("https://raw.githubusercontent.com/"
             "brendenlake/SCAN/master/add_prim_split/")
    def _load(url):
        with urllib.request.urlopen(url) as f:
            lines = f.read().decode("utf-8").strip().split("\n")
        out = []
        for line in lines:
            if not line.strip():
                continue
            cmd_part, act_part = line.split(" OUT: ", 1)
            out.append({"commands": cmd_part.replace("IN: ", "").strip(),
                        "actions": act_part.strip()})
        return out
    train_raw = _load(_BASE + "tasks_train_addprim_jump.txt")
    test_raw  = _load(_BASE + "tasks_test_addprim_jump.txt")
    print(f"  Downloaded from GitHub. Train={len(train_raw)}, Test={len(test_raw)}")

test_jc = [ex for ex in test_raw
           if "jump" in ex["commands"].split() and len(ex["commands"].split()) > 1]
print(f"  Jump-compound test examples: {len(test_jc)}")

print(f"\n  Running inference to classify fail / success groups...")
BATCH = 32
sub_walk_examples = []
success_examples  = []
ACTION_TOKS = {"I_JUMP", "I_WALK", "I_RUN", "I_LOOK"}

with torch.no_grad():
    for i in range(0, len(test_jc), BATCH):
        batch = test_jc[i:i+BATCH]
        inp = tokenizer(
            [ex["commands"] for ex in batch],
            padding=True, truncation=True,
            max_length=MAX_INPUT_LEN, return_tensors="pt"
        ).to(device)
        gen_ids = model.generate(
            input_ids=inp["input_ids"],
            attention_mask=inp["attention_mask"],
            max_length=MAX_TARGET_LEN,
        )
        preds = tokenizer.batch_decode(gen_ids, skip_special_tokens=True)
        for ex, pred in zip(batch, preds):
            gold    = ex["actions"].strip()
            correct = (pred.strip() == gold)
            ex["predicted"] = pred.strip()
            ex["correct"]   = correct
            has_around = "around" in ex["commands"].lower().split()
            n_pred_j = pred.split().count("I_JUMP")
            n_pred_w = pred.split().count("I_WALK")
            if has_around and not correct and n_pred_j == 0 and n_pred_w > 0:
                sub_walk_examples.append(ex)
            elif has_around and correct:
                success_examples.append(ex)
        if (i // BATCH) % 20 == 0:
            print(f"    {i+len(batch)}/{len(test_jc)}")

random.seed(SEED)
random.shuffle(sub_walk_examples)
sub_walk_examples = sub_walk_examples[:N_FAIL]
random.shuffle(success_examples)
success_examples  = success_examples[:N_SUCCESS]

print(f"\n  substituted_walk fail examples:   {len(sub_walk_examples)}")
print(f"  has_around + success examples:    {len(success_examples)}")

# ── Phase 4: Unembedding direction and helper functions ────────────────────────────────────────────
print("\n" + "="*70)
print("  PHASE 4 — HELPER FUNCTIONS")
print("="*70)

def find_div_steps(token_ids):
    """
    Action-slot divergence-point step positions in a generated sequence.
    Returns list of (step_idx, action_label).
    The divergence point is the 3rd subword of an action token (S-050 fix).
    """
    points = []
    for i in range(3, len(token_ids)):
        if (token_ids[i]   in DIV_TOKEN_SET and
            token_ids[i-1] == 834 and
            token_ids[i-2] == 27 and
            token_ids[i-3] in TURN_END_IDS):
            points.append((i, id_to_action.get(token_ids[i], "UNKNOWN")))
    return points

def compute_head_contrib(l_block, h_idx, step_cross_attns, enc_hidden_final):
    """
    Head contribution to decoder residual stream via V/O projection.
    contrib_h = W_O_h @ (attn_h @ W_V_h @ enc_hidden)
    Returns [d_model] float32 CPU tensor.
    Same method as S-057 M4 and S-058.
    """
    enc_attn = model.decoder.block[l_block].layer[1].EncDecAttention
    with torch.no_grad():
        V_all    = enc_hidden_final @ enc_attn.v.weight.T    # [1, T_enc, n_heads*d_kv]
        attn_w   = step_cross_attns[l_block]                  # [1, n_heads, 1, T_enc]
        head_attn = attn_w[0, h_idx, 0, :]                    # [T_enc]
        head_V    = V_all[0, :, h_idx*d_kv:(h_idx+1)*d_kv]   # [T_enc, d_kv]
        head_val  = (head_attn.unsqueeze(0) @ head_V).squeeze(0)  # [d_kv]
        head_o_sl = enc_attn.o.weight[:, h_idx*d_kv:(h_idx+1)*d_kv]  # [d_model, d_kv]
    return (head_o_sl @ head_val).cpu().float()

def compute_dlogit(contrib_h):
    """
    δlogit_h = (W_U[I_JUMP] - W_U[I_WALK]) · contrib_h
    Dot product in d_model space.
    Positive → head pushes toward I_JUMP.
    Negative → head pushes toward I_WALK.
    """
    return float(torch.dot(W_U_diff, contrib_h).item())

def run_generate_with_states(cmd):
    """
    Run model.generate() on a command, collecting all states.
    Returns dict: enc_hidden, cross_attns_per_step, sequences, scores.
    """
    inp = tokenizer(cmd, truncation=True, max_length=MAX_INPUT_LEN,
                    return_tensors="pt").to(device)
    with torch.no_grad():
        out = model.generate(
            input_ids=inp["input_ids"],
            attention_mask=inp["attention_mask"],
            max_length=MAX_TARGET_LEN,
            output_attentions=True,
            output_hidden_states=True,
            return_dict_in_generate=True,
            output_scores=True,
        )
    return {
        "enc_hidden":  out.encoder_hidden_states[-1],   # [1, T_enc, d_model]
        "cross_attns": out.cross_attentions,             # list[step] of tuple[layer]
        "sequences":   out.sequences[0].tolist()[1:],    # strip BOS
        "scores":      out.scores,                       # list[step] of [1, vocab_size]
    }

print("  Helper functions defined.")
print(f"  Will measure all {n_total_heads} decoder cross-attention heads")
print(f"  (layers 0-{n_dec_layers-1}, heads 0-{n_heads-1})")

# ── Phase 5: Logit decomposition — fail group ──────────────────────────────────────────────────
print("\n" + "="*70)
print("  PHASE 5 — LOGIT DECOMPOSITION: FAIL GROUP (M1)")
print(f"  {len(sub_walk_examples)} substituted_walk examples")
print("="*70)

# Storage: per-example results
# dlogits_fail[l][h] = list of δlogit values across examples
dlogits_fail      = [[[] for _ in range(n_heads)] for _ in range(n_dec_layers)]
observed_margins_fail = []   # actual logit(I_JUMP) − logit(I_WALK) from model scores
recon_margins_fail    = []   # sum of all head δlogit contributions
n_valid_fail = 0
n_skip_fail  = 0
skip_reasons_fail = defaultdict(int)

print(f"\n  Running forward passes for {len(sub_walk_examples)} fail examples...")

for ex_idx, ex in enumerate(sub_walk_examples):
    if ex_idx % 5 == 0:
        print(f"  Example {ex_idx+1}/{len(sub_walk_examples)} ...")

    data = run_generate_with_states(ex["commands"])

    # Find first action-slot divergence step
    div_steps = find_div_steps(data["sequences"])
    if not div_steps:
        n_skip_fail += 1
        skip_reasons_fail["no_div_steps"] += 1
        continue
    div_step_idx, div_action = div_steps[0]

    if div_step_idx >= len(data["cross_attns"]):
        n_skip_fail += 1
        skip_reasons_fail["div_step_out_of_range"] += 1
        continue

    # Observed margin from model's own logits
    logits_at_div = data["scores"][div_step_idx][0]  # [vocab_size]
    obs_margin = float((logits_at_div[I_JUMP_ID] - logits_at_div[I_WALK_ID]).item())
    observed_margins_fail.append(obs_margin)

    # Compute δlogit for all 48 heads
    enc_hidden = data["enc_hidden"]
    cross_attns = data["cross_attns"][div_step_idx]

    recon_margin = 0.0
    for l in range(n_dec_layers):
        for h in range(n_heads):
            contrib = compute_head_contrib(l, h, cross_attns, enc_hidden)
            dl = compute_dlogit(contrib)
            dlogits_fail[l][h].append(dl)
            recon_margin += dl

    recon_margins_fail.append(recon_margin)
    n_valid_fail += 1

print(f"\n  Valid fail examples: {n_valid_fail}")
print(f"  Skipped: {n_skip_fail}")
if n_skip_fail:
    for reason, count in skip_reasons_fail.items():
        print(f"    {reason}: {count}")

# ── Phase 6: Logit decomposition — success group ─────────────────────────────────────────────────
print("\n" + "="*70)
print("  PHASE 6 — LOGIT DECOMPOSITION: SUCCESS GROUP (M2)")
print(f"  {len(success_examples)} has_around+correct examples")
print("="*70)

dlogits_success      = [[[] for _ in range(n_heads)] for _ in range(n_dec_layers)]
observed_margins_success = []
recon_margins_success    = []
n_valid_success = 0
n_skip_success  = 0
skip_reasons_success = defaultdict(int)

print(f"\n  Running forward passes for {len(success_examples)} success examples...")

for ex_idx, ex in enumerate(success_examples):
    if ex_idx % 5 == 0:
        print(f"  Example {ex_idx+1}/{len(success_examples)} ...")

    data = run_generate_with_states(ex["commands"])

    div_steps = find_div_steps(data["sequences"])
    if not div_steps:
        n_skip_success += 1
        skip_reasons_success["no_div_steps"] += 1
        continue
    div_step_idx, div_action = div_steps[0]

    if div_step_idx >= len(data["cross_attns"]):
        n_skip_success += 1
        skip_reasons_success["div_step_out_of_range"] += 1
        continue

    logits_at_div = data["scores"][div_step_idx][0]
    obs_margin = float((logits_at_div[I_JUMP_ID] - logits_at_div[I_WALK_ID]).item())
    observed_margins_success.append(obs_margin)

    enc_hidden = data["enc_hidden"]
    cross_attns = data["cross_attns"][div_step_idx]

    recon_margin = 0.0
    for l in range(n_dec_layers):
        for h in range(n_heads):
            contrib = compute_head_contrib(l, h, cross_attns, enc_hidden)
            dl = compute_dlogit(contrib)
            dlogits_success[l][h].append(dl)
            recon_margin += dl

    recon_margins_success.append(recon_margin)
    n_valid_success += 1

print(f"\n  Valid success examples: {n_valid_success}")
print(f"  Skipped: {n_skip_success}")
if n_skip_success:
    for reason, count in skip_reasons_success.items():
        print(f"    {reason}: {count}")

# ── Phase 7: Aggregate statistics and measurements ────────────────────────────────────────────────
print("\n" + "="*70)
print("  PHASE 7 — AGGREGATE STATISTICS AND MEASUREMENTS")
print("="*70)

# Compute mean/std per head for both groups
def head_stats(dlogits_by_layer):
    """Compute mean, std, n for each (l, h)."""
    stats = {}
    for l in range(n_dec_layers):
        for h in range(n_heads):
            vals = dlogits_by_layer[l][h]
            label = f"L{l+1}H{h}"
            if vals:
                stats[label] = {
                    "layer": l+1, "head": h,
                    "mean": float(np.mean(vals)),
                    "std":  float(np.std(vals)),
                    "n":    len(vals),
                    "vals": vals,
                }
            else:
                stats[label] = {"layer": l+1, "head": h, "mean": 0.0, "std": 0.0, "n": 0, "vals": []}
    return stats

stats_fail    = head_stats(dlogits_fail)
stats_success = head_stats(dlogits_success)

# ── M1: Per-head δlogit fail group — ranked table ───────────────────────────────────────────────────
print("\n--- M1: Per-head δlogit at divergence step (fail group) ---")
print(f"  (N={n_valid_fail} valid examples)")
print()

# Rank by |mean δlogit|
all_heads_fail = sorted(stats_fail.items(),
                         key=lambda kv: abs(kv[1]["mean"]),
                         reverse=True)

print(f"  {'Rank':<5} {'Head':<8} {'mean δlogit':<14} {'std':<10} {'sign':<8}")
print(f"  {'----':<5} {'----':<8} {'-----------':<14} {'---':<10} {'----':<8}")
for rank, (label, s) in enumerate(all_heads_fail, 1):
    sign = "pro-jump" if s["mean"] > 0 else "pro-walk"
    print(f"  {rank:<5} {label:<8} {s['mean']:>+12.4f}   {s['std']:<10.4f} {sign}")

# ── M2: Per-head δlogit success group — ranked table + delta ─────────────────────────────────────────
print("\n--- M2: Per-head δlogit at divergence step (success group) ---")
print(f"  (N={n_valid_success} valid examples)")
print()

all_heads_success = sorted(stats_success.items(),
                            key=lambda kv: abs(kv[1]["mean"]),
                            reverse=True)

# Compute fail-vs-success delta for all heads
dlogit_delta = {}
for label in stats_fail:
    mf = stats_fail[label]["mean"]
    ms = stats_success[label]["mean"] if label in stats_success else 0.0
    dlogit_delta[label] = ms - mf   # positive = more pro-jump in success

print(f"  {'Rank':<5} {'Head':<8} {'mean δlogit':<14} {'std':<10} {'Δ(succ-fail)':<15} {'sign':<8}")
print(f"  {'----':<5} {'----':<8} {'-----------':<14} {'---':<10} {'------------':<15} {'----':<8}")
for rank, (label, s) in enumerate(all_heads_success, 1):
    sign = "pro-jump" if s["mean"] > 0 else "pro-walk"
    delta = dlogit_delta.get(label, 0.0)
    print(f"  {rank:<5} {label:<8} {s['mean']:>+12.4f}   {s['std']:<10.4f} {delta:>+13.4f}   {sign}")

# ── M3: Logit margin reconstruction ──────────────────────────────────────────────────────────────────────
print("\n--- M3: Logit margin reconstruction (fail group) ---")

mean_obs_margin   = float(np.mean(observed_margins_fail))
mean_recon_margin = float(np.mean(recon_margins_fail))
mean_residual     = mean_obs_margin - mean_recon_margin
# Fraction of observed margin explained by head contributions
# (use sign-aware ratio: if both negative, ratio should be positive)
if abs(mean_obs_margin) > 1e-9:
    recon_fraction = mean_recon_margin / mean_obs_margin
else:
    recon_fraction = 0.0

print(f"  Mean observed margin (jump - walk):     {mean_obs_margin:>+10.4f}")
print(f"  Mean reconstructed margin (Σδlogit_h): {mean_recon_margin:>+10.4f}")
print(f"  Mean residual (obs - recon):            {mean_residual:>+10.4f}")
print(f"  Reconstruction fraction:                {recon_fraction:>10.4f}")
print()
print(f"  K2 check: |recon_fraction| >= {K2_RECON_MIN}: ", end="")
k2_fires = abs(recon_fraction) < K2_RECON_MIN
print("FIRES" if k2_fires else "CLEAR")

if k2_fires:
    print(f"\n  *** K2 FIRES: reconstruction fraction = {recon_fraction:.4f} < {K2_RECON_MIN}")
    print(f"  *** The V/O projection method does not capture the dominant signal.")
    print(f"  *** FFN or embedding terms dominate the logit margin.")
    print(f"  *** Per pre-registration protocol: report residual distribution and relay to G-track.")
    print(f"  *** Continuing to report all measurements for reference...")

# Success group reconstruction
mean_obs_margin_succ   = float(np.mean(observed_margins_success)) if observed_margins_success else 0.0
mean_recon_margin_succ = float(np.mean(recon_margins_success)) if recon_margins_success else 0.0
mean_residual_succ     = mean_obs_margin_succ - mean_recon_margin_succ
recon_fraction_succ = mean_recon_margin_succ / mean_obs_margin_succ if abs(mean_obs_margin_succ) > 1e-9 else 0.0
print(f"\n  Success group reconstruction:")
print(f"  Mean observed margin:                   {mean_obs_margin_succ:>+10.4f}")
print(f"  Mean reconstructed margin:              {mean_recon_margin_succ:>+10.4f}")
print(f"  Reconstruction fraction:                {recon_fraction_succ:>10.4f}")

# ── M4: Target head breakdown ──────────────────────────────────────────────────────────────────────────────
print("\n--- M4: Target head and control head breakdown ---")

# Map head labels to (l_block_0indexed, h_idx)
# L4H6 → layer 4 (1-indexed) = l_block=3 (0-indexed), H6 = head 6 → label "L4H6"
# L5H2 → layer 5 (1-indexed) = l_block=4, H2 → label "L5H2"
# L5H5 → layer 5, H5 → label "L5H5"
# L3H0 → layer 3 (1-indexed) = l_block=2, H0 → label "L3H0"
# L3H4 → layer 3, H4 → label "L3H4"
KEY_HEADS = ["L4H6", "L5H2", "L5H5", "L3H0", "L3H4"]

# Get rank of each key head in the fail group ranking
rank_map_fail = {label: rank for rank, (label, _) in enumerate(all_heads_fail, 1)}

# Compute total reconstructed margin and baseline for share calculation
target_head_sum_fail = sum(stats_fail[h]["mean"] for h in ["L4H6", "L5H2", "L5H5"])
# For H3, we want the share of the total negative reconstructed margin
# Use per-example target-head contributions vs total recon margin

print(f"\n  {'Head':<8} {'δlogit fail':<14} {'δlogit succ':<14} {'Δ(succ-fail)':<15} {'rank/48':<9} {'share of recon'}")
print(f"  {'----':<8} {'-----------':<14} {'-----------':<14} {'------------':<15} {'-------':<9} {'-'*14}")
for h_label in KEY_HEADS:
    sf = stats_fail[h_label]
    ss = stats_success.get(h_label, {"mean": 0.0})
    delta = dlogit_delta.get(h_label, 0.0)
    rank  = rank_map_fail.get(h_label, -1)
    # Share of reconstructed margin (using means)
    if abs(mean_recon_margin) > 1e-9:
        share = sf["mean"] / mean_recon_margin
    else:
        share = 0.0
    print(f"  {h_label:<8} {sf['mean']:>+12.4f}   {ss['mean']:>+12.4f}   {delta:>+13.4f}   {rank:<9} {share:>+.4f}")

# Find the lowest-|δlogit| head (random baseline)
lowest_head_label = all_heads_fail[-1][0]
lowest_sf = stats_fail[lowest_head_label]
lowest_ss = stats_success.get(lowest_head_label, {"mean": 0.0})
lowest_delta = dlogit_delta.get(lowest_head_label, 0.0)
lowest_share = lowest_sf["mean"] / mean_recon_margin if abs(mean_recon_margin) > 1e-9 else 0.0
print(f"  {lowest_head_label:<8} {lowest_sf['mean']:>+12.4f}   {lowest_ss['mean']:>+12.4f}   {lowest_delta:>+13.4f}   {48:<9} {lowest_share:>+.4f}  ← lowest |δlogit| baseline")

# ── M5: Cumulative margin by head rank ────────────────────────────────────────────────────────────────────
print("\n--- M5: Cumulative margin by head rank (fail group, most negative first) ---")

# Sort by mean δlogit, most negative first
heads_by_dlogit = sorted(stats_fail.items(), key=lambda kv: kv[1]["mean"])
cumulative = 0.0
frac_50_head = frac_75_head = frac_90_head = None
neg_total = sum(s["mean"] for _, s in heads_by_dlogit if s["mean"] < 0)
print(f"\n  Total reconstructed margin from negative-δlogit heads: {neg_total:.4f}")
print(f"\n  {'Rank':<5} {'Head':<8} {'δlogit':<12} {'cumulative':<12} {'cum/neg_total'}")
print(f"  {'----':<5} {'----':<8} {'------':<12} {'----------':<12} {'-'*14}")
neg_running = 0.0
neg_count   = 0
for rank, (label, s) in enumerate(heads_by_dlogit, 1):
    if s["mean"] < 0:
        neg_running += s["mean"]
        neg_count   += 1
        frac = neg_running / neg_total if abs(neg_total) > 1e-9 else 0.0
        if frac_50_head is None and frac >= 0.50: frac_50_head = (neg_count, label, frac)
        if frac_75_head is None and frac >= 0.75: frac_75_head = (neg_count, label, frac)
        if frac_90_head is None and frac >= 0.90: frac_90_head = (neg_count, label, frac)
        print(f"  {neg_count:<5} {label:<8} {s['mean']:>+10.4f}   {neg_running:>+10.4f}   {frac:.4f}")

print(f"\n  50% of negative margin explained by: {frac_50_head}")
print(f"  75% of negative margin explained by: {frac_75_head}")
print(f"  90% of negative margin explained by: {frac_90_head}")

# ── Phase 8: Hypothesis verdicts ──────────────────────────────────────────────────────────────────────
print("\n" + "="*70)
print("  PHASE 8 — HYPOTHESIS VERDICTS")
print("="*70)

# H1: L4H6, L5H2, L5H5 are in top-5 by |δlogit| AND all three have mean < 0
h1_target_ranks = {h: rank_map_fail[h] for h in ["L4H6", "L5H2", "L5H5"]}
h1_in_top5 = all(r <= 5 for r in h1_target_ranks.values())
h1_all_negative = all(stats_fail[h]["mean"] < 0 for h in ["L4H6", "L5H2", "L5H5"])
h1_pass = h1_in_top5 and h1_all_negative

print(f"\n  H1 — L4H6/L5H2/L5H5 in top-5 by |δlogit| AND all negative in fail:")
for h in ["L4H6", "L5H2", "L5H5"]:
    s = stats_fail[h]
    print(f"    {h}: mean={s['mean']:>+.4f}  rank={h1_target_ranks[h]}/48")
print(f"  All in top-5: {h1_in_top5}  |  All negative: {h1_all_negative}")
print(f"  H1: {'PASS' if h1_pass else 'FAIL'}  [{h1_in_top5 and h1_all_negative}]")

# K1: Target heads have positive mean δlogit in fail group (wrong direction)
k1_fires = any(stats_fail[h]["mean"] > 0 for h in ["L4H6", "L5H2", "L5H5"])
print(f"\n  K1 — Target heads positive in fail (pro-jump, mechanism wrong):")
for h in ["L4H6", "L5H2", "L5H5"]:
    print(f"    {h}: mean={stats_fail[h]['mean']:>+.4f}")
print(f"  K1: {'FIRES' if k1_fires else 'CLEAR'}")

if k1_fires:
    print(f"\n  *** K1 FIRES: at least one target head is pro-jump in fail cases.")
    print(f"  *** The value-substitution mechanism as characterized is wrong.")
    print(f"  *** Stop; relay to G-track for reanalysis.")

# H2: Δ(δlogit_h) = success − fail ≥ +0.20 for each of L4H6, L5H2, L5H5
h2_deltas = {h: dlogit_delta[h] for h in ["L4H6", "L5H2", "L5H5"]}
h2_per_head = {h: d >= H2_DELTA_MIN for h, d in h2_deltas.items()}
h2_pass = all(h2_per_head.values())

print(f"\n  H2 — Δ(δlogit_h) = success−fail ≥ +{H2_DELTA_MIN} for each target head:")
for h in ["L4H6", "L5H2", "L5H5"]:
    d = h2_deltas[h]
    print(f"    {h}: Δ={d:>+.4f}  (threshold={H2_DELTA_MIN:>+.2f})  {'≥' if d >= H2_DELTA_MIN else '<'} threshold")
print(f"  H2: {'PASS' if h2_pass else 'FAIL'}  [all {[v for v in h2_per_head.values()]}]")

# H3: target heads account for ≥ 30% of total reconstructed negative margin
# Use mean δlogit values and mean reconstructed margin
# Condition: (δlogit_L4H6 + δlogit_L5H2 + δlogit_L5H5) / margin_reconstructed ≥ 0.30
# (margin_reconstructed < 0, so this computes share of the negative total)
h3_target_sum = sum(stats_fail[h]["mean"] for h in ["L4H6", "L5H2", "L5H5"])
if abs(mean_recon_margin) > 1e-9:
    h3_share = h3_target_sum / mean_recon_margin
else:
    h3_share = 0.0
h3_pass = h3_share >= H3_SHARE_MIN

print(f"\n  H3 — Target heads account for ≥{H3_SHARE_MIN:.0%} of reconstructed negative margin:")
print(f"    δlogit_L4H6 = {stats_fail['L4H6']['mean']:>+.4f}")
print(f"    δlogit_L5H2 = {stats_fail['L5H2']['mean']:>+.4f}")
print(f"    δlogit_L5H5 = {stats_fail['L5H5']['mean']:>+.4f}")
print(f"    Target sum  = {h3_target_sum:>+.4f}")
print(f"    Recon margin = {mean_recon_margin:>+.4f}")
print(f"    Share = {h3_share:>+.4f}  (threshold = {H3_SHARE_MIN:.2f})")
print(f"  H3: {'PASS' if h3_pass else 'FAIL'}  [{h3_share:.4f} {'≥' if h3_pass else '<'} {H3_SHARE_MIN}]")

# H4: |δlogit_L3H0| < |δlogit_L4H6| AND |δlogit_L3H4| < |δlogit_L4H6|
h4_l4h6_abs = abs(stats_fail["L4H6"]["mean"])
h4_l3h0_abs = abs(stats_fail["L3H0"]["mean"])
h4_l3h4_abs = abs(stats_fail["L3H4"]["mean"])
h4_l3h0_pass = h4_l3h0_abs < h4_l4h6_abs
h4_l3h4_pass = h4_l3h4_abs < h4_l4h6_abs
h4_pass = h4_l3h0_pass and h4_l3h4_pass

print(f"\n  H4 — Control heads |δlogit| < L4H6 |δlogit| (sanity check):")
print(f"    |δlogit_L4H6| = {h4_l4h6_abs:.4f}")
print(f"    |δlogit_L3H0| = {h4_l3h0_abs:.4f}  {'<' if h4_l3h0_pass else '≥'} L4H6  ({'' if h4_l3h0_pass else 'NOT '}satisfied)")
print(f"    |δlogit_L3H4| = {h4_l3h4_abs:.4f}  {'<' if h4_l3h4_pass else '≥'} L4H6  ({'' if h4_l3h4_pass else 'NOT '}satisfied)")
print(f"  H4: {'PASS' if h4_pass else 'FAIL'}  [L3H0 {'<' if h4_l3h0_pass else '≥'} L4H6; L3H4 {'<' if h4_l3h4_pass else '≥'} L4H6]")

# Summary
print(f"\n  ─── VERDICT SUMMARY ───")
print(f"  H1 (L4H6/L5H2/L5H5 top-5, negative):          {'PASS' if h1_pass else 'FAIL'}")
print(f"  H2 (target heads Δ ≥ +0.20 success-fail):       {'PASS' if h2_pass else 'FAIL'}")
print(f"  H3 (target heads ≥ 30% of neg margin):           {'PASS' if h3_pass else 'FAIL'}")
print(f"  H4 (control heads smaller than L4H6):            {'PASS' if h4_pass else 'FAIL'}")
print(f"  K1 (target heads pro-jump — mechanism wrong):    {'FIRES' if k1_fires else 'CLEAR'}")
print(f"  K2 (reconstruction < 20% — FFN dominates):       {'FIRES' if k2_fires else 'CLEAR'}")

# ── Phase 9: Save results ──────────────────────────────────────────────────────────────────────────────
print("\n" + "="*70)
print("  PHASE 9 — SAVE RESULTS")
print("="*70)

RESULTS_PATH = "workbench/results/059_results.json"
os.makedirs(os.path.dirname(RESULTS_PATH), exist_ok=True)

# Build serializable dlogit tables
def build_head_table(stats_dict):
    table = {}
    for label, s in stats_dict.items():
        table[label] = {
            "layer": s["layer"], "head": s["head"],
            "mean": s["mean"], "std": s["std"], "n": s["n"],
        }
    return table

# Per-example arrays for fail and success (all 48 heads)
# Store as {head_label: [δlogit per example]}
def build_per_example_arrays(dlogits_by_layer):
    arrays = {}
    for l in range(n_dec_layers):
        for h in range(n_heads):
            label = f"L{l+1}H{h}"
            arrays[label] = dlogits_by_layer[l][h]
    return arrays

results = {
    "script_id": SCRIPT_ID,
    "seed": SEED,
    "n_valid_fail": n_valid_fail,
    "n_valid_success": n_valid_success,

    # M1: fail group head stats
    "fail_head_table": build_head_table(stats_fail),
    "fail_head_ranking": [label for label, _ in all_heads_fail],

    # M2: success group head stats + delta
    "success_head_table": build_head_table(stats_success),
    "dlogit_delta": {label: dlogit_delta[label] for label in stats_fail},

    # M3: margin reconstruction
    "m3_fail": {
        "mean_observed_margin": mean_obs_margin,
        "mean_recon_margin": mean_recon_margin,
        "mean_residual": mean_residual,
        "recon_fraction": recon_fraction,
        "observed_margins": observed_margins_fail,
        "recon_margins": recon_margins_fail,
    },
    "m3_success": {
        "mean_observed_margin": mean_obs_margin_succ,
        "mean_recon_margin": mean_recon_margin_succ,
        "mean_residual": mean_residual_succ,
        "recon_fraction": recon_fraction_succ,
    },

    # M4: target head breakdown
    "m4_key_heads": {h: {
        "mean_fail": stats_fail[h]["mean"],
        "mean_success": stats_success[h]["mean"] if h in stats_success else 0.0,
        "delta": dlogit_delta.get(h, 0.0),
        "rank_fail": rank_map_fail.get(h, -1),
        "share_of_recon": stats_fail[h]["mean"] / mean_recon_margin if abs(mean_recon_margin) > 1e-9 else 0.0,
    } for h in KEY_HEADS + [lowest_head_label]},

    # M5: cumulative margin
    "m5_cumulative": {
        "frac_50_head": frac_50_head,
        "frac_75_head": frac_75_head,
        "frac_90_head": frac_90_head,
        "neg_total": neg_total,
    },

    # H/K verdicts
    "verdicts": {
        "H1": {"pass": h1_pass, "in_top5": h1_in_top5, "all_negative": h1_all_negative,
               "target_ranks": h1_target_ranks},
        "H2": {"pass": h2_pass, "deltas": h2_deltas, "threshold": H2_DELTA_MIN,
               "per_head": h2_per_head},
        "H3": {"pass": h3_pass, "share": h3_share, "target_sum": h3_target_sum,
               "threshold": H3_SHARE_MIN},
        "H4": {"pass": h4_pass, "l4h6_abs": h4_l4h6_abs, "l3h0_abs": h4_l3h0_abs,
               "l3h4_abs": h4_l3h4_abs},
        "K1": {"fires": k1_fires},
        "K2": {"fires": k2_fires, "recon_fraction": recon_fraction},
    },

    # Per-example δlogit arrays
    "per_example_fail": build_per_example_arrays(dlogits_fail),
    "per_example_success": build_per_example_arrays(dlogits_success),
}

with open(RESULTS_PATH, "w") as f:
    json.dump(results, f, indent=2)
print(f"  Results saved: {RESULTS_PATH}")

save_to_drive(RESULTS_PATH)

# ── Final summary ─────────────────────────────────────────────────────────────────────────────────
print("\n" + "="*70)
print(f"  {SCRIPT_ID} — COMPLETE")
print("="*70)
print(f"\n  Groups:  {n_valid_fail} fail  |  {n_valid_success} success")
print(f"\n  Observed margin (fail):       {mean_obs_margin:>+10.4f}")
print(f"  Reconstructed margin (fail):  {mean_recon_margin:>+10.4f}")
print(f"  Reconstruction fraction:      {recon_fraction:>10.4f}")
print(f"\n  Top 5 heads by |δlogit| in fail group:")
for rank, (label, s) in enumerate(all_heads_fail[:5], 1):
    sign = "pro-jump" if s["mean"] > 0 else "pro-walk"
    print(f"    {rank}. {label:<8}  {s['mean']:>+.4f}  ({sign})")
print(f"\n  Target head contributions (fail):")
for h in ["L4H6", "L5H2", "L5H5"]:
    s = stats_fail[h]
    d = dlogit_delta[h]
    print(f"    {h}: fail={s['mean']:>+.4f}  succ={stats_success[h]['mean']:>+.4f}  Δ={d:>+.4f}  rank={rank_map_fail[h]}/48")
print(f"\n  Verdicts: H1={'PASS' if h1_pass else 'FAIL'}  H2={'PASS' if h2_pass else 'FAIL'}  H3={'PASS' if h3_pass else 'FAIL'}  H4={'PASS' if h4_pass else 'FAIL'}")
print(f"  Kill cond: K1={'FIRES' if k1_fires else 'CLEAR'}  K2={'FIRES' if k2_fires else 'CLEAR'}")
print()
print(f"  Results: {RESULTS_PATH}")
print()

# ── Tell Troy what's next ───────────────────────────────────────────────────────────────────────────────
print("─" * 70)
print("  NEXT STEPS")
print("─" * 70)
if not k1_fires and not k2_fires:
    print("""
  S-059 complete. Write methods report:
    findings/METHODS_REPORT_S059.md

  If H1 + H2 + H3 all pass:
    → Value-substitution mechanism confirmed in logit units.
    → Next: targeted repair — modify W_V at L4H6/L5H2/L5H5 per G-056 geometry.
    → Relay to G-track: S-059 confirms target heads; G-057 can proceed
      with toy-scale logit decomposition.

  If H1 passes but H3 fails:
    → Value-substitution partially correct but distributed.
    → Report full ranking and relay to G-track for geometric analysis.

  Update COORDINATION.md, push to S-Track, tell Troy.
""")
elif k1_fires:
    print("""
  K1 FIRED: target heads are pro-jump in fail cases.
  The value-substitution mechanism as characterized is wrong.
  Do NOT proceed to repair experiments.
  Write methods report, update COORDINATION.md, relay to G-track.
""")
elif k2_fires:
    print("""
  K2 FIRED: reconstruction fraction < 20%.
  FFN or embedding terms dominate the logit margin.
  Report residual distribution in methods report.
  Relay to G-track before designing any head-targeted repair.
""")

print("─" * 70)
