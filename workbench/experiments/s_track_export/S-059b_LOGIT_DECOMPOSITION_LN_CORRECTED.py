#!/usr/bin/env python3
"""
S-059b_LOGIT_DECOMPOSITION_LN_CORRECTED.py — Logit Decomposition V0.2.0
Applied Categorical Physics Workbench
Troy Teno | May 2026 | Open Access

Pre-registered hypotheses:
  workbench/proposals/S-059_LOGIT_DECOMPOSITION_PROPOSAL.md
  (same hypotheses as V0.1.0 — amendment: LayerNorm correction only)
DO NOT modify hypotheses after seeing results.

─────────────────────────────────────────────────────────────────────
SETUP (run once in Colab or terminal):
  pip install torch transformers datasets numpy scipy sentencepiece -q

AMENDMENT FROM V0.1.0:
  V0.1.0 (S-059) computed δlogit_h = W_U_diff · contrib_h, projecting
  residual stream contributions directly through the unembedding matrix.
  This is incorrect for T5: the final decoder RMSNorm (model.decoder.
  final_layer_norm) is applied between the residual stream and lm_head.
  The V0.1.0 reconstruction fraction was ~32,193 (off by that factor).

  V0.2.0 correction: use the per-example effective unembedding direction:

      W_U_eff = (W_U_diff * LN_weight) / RMS(h_pre_ln)   [per example]
      δlogit_h = W_U_eff · contrib_h

  where h_pre_ln is the pre-LayerNorm decoder hidden state at the
  divergence step, obtained from decoder_hidden_states[-2][0,0,:].

  A reconstruction verification step confirms the formula by computing
  the logit margin from h_pre_ln and comparing to model.scores.

  All hypotheses, groups (29 fail / 25 success), and measurements
  (M1–M5, H1–H4, K1–K2) are unchanged from V0.1.0.

Version history:
  V0.1.0 (S-059) — initial build; missing LayerNorm correction;
                    reconstruction fraction ≈ 32,193; sealed as
                    methodological partial finding
  V0.2.0 (S-059b) — LayerNorm correction applied; correct logit units
─────────────────────────────────────────────────────────────────────
"""

import numpy as np
import json
import os
import random
from collections import defaultdict

SCRIPT_ID = "S-059_LOGIT_DECOMPOSITION_V0.2.0"
SEED = 42
np.random.seed(SEED)
random.seed(SEED)

CHECKPOINT_DIR = "043_t5_scan_checkpoint"
MAX_INPUT_LEN  = 50
MAX_TARGET_LEN = 100
N_FAIL         = 30
N_SUCCESS      = 25

# Pre-registered thresholds (unchanged from V0.1.0)
H2_DELTA_MIN   = 0.20   # Δ(δlogit_h) = success − fail per target head (logit units)
H3_SHARE_MIN   = 0.30   # target heads' share of total reconstructed negative margin
K2_RECON_MIN   = 0.20   # cross-attn head reconstruction fraction of observed margin

# Head definitions: (decoder_block_0indexed, head_0indexed, label)
L4H6 = (3, 6, "L4H6")
L5H2 = (4, 2, "L5H2")
L5H5 = (4, 5, "L5H5")
L3H0 = (2, 0, "L3H0")
L3H4 = (2, 4, "L3H4")

TARGET_HEADS  = [L4H6, L5H2, L5H5]
CONTROL_HEADS = [L3H0, L3H4]

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
print(f"  Logit decomposition with LayerNorm correction")
print(f"{'='*70}")

# ── Imports ─────────────────────────────────────────────────────────────────────────────
print("\nImporting libraries...")
try:
    import torch
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

# ── Phase 2: Token IDs, unembedding direction, LayerNorm parameters ──────────────────────────────
print("\n" + "="*70)
print("  PHASE 2 — TOKEN IDs, UNEMBEDDING DIRECTION, LAYERNORM PARAMETERS")
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
print(f"  TURN_END_IDS: {TURN_END_IDS}")

n_dec_layers  = model.config.num_decoder_layers   # 6
n_heads       = model.config.num_heads            # 8
d_kv          = model.config.d_kv                 # 64
d_model       = model.config.d_model              # 512
n_total_heads = n_dec_layers * n_heads            # 48
print(f"\n  T5 decoder: {n_dec_layers} layers × {n_heads} heads = {n_total_heads} total heads")
print(f"  d_kv={d_kv}, d_model={d_model}")

# Unembedding direction: W_U[I_JUMP] − W_U[I_WALK]
with torch.no_grad():
    W_U_jump = model.lm_head.weight[I_JUMP_ID].cpu().float()
    W_U_walk = model.lm_head.weight[I_WALK_ID].cpu().float()
    W_U_diff = W_U_jump - W_U_walk                             # [d_model]

print(f"\n  ‖W_U[I_JUMP]‖ = {W_U_jump.norm():.2f}")
print(f"  ‖W_U[I_WALK]‖ = {W_U_walk.norm():.2f}")
print(f"  ‖W_U_diff‖    = {W_U_diff.norm():.2f}")

# LayerNorm parameters: T5 uses RMSNorm (no bias, no mean subtraction)
# model.decoder.final_layer_norm: T5LayerNorm
# forward: h_out = h_in * weight / sqrt(mean(h_in²) + variance_epsilon)
LN_WEIGHT = model.decoder.final_layer_norm.weight.detach().cpu().float()  # [d_model]
LN_EPS    = getattr(model.decoder.final_layer_norm, "variance_epsilon", 1e-6)
print(f"\n  LayerNorm (RMSNorm) parameters:")
print(f"  weight: mean={LN_WEIGHT.mean():.4f}, std={LN_WEIGHT.std():.4f}, ‖LN_weight‖={LN_WEIGHT.norm():.2f}")
print(f"  variance_epsilon: {LN_EPS}")

# Effective unembedding: W_U_eff = (W_U_diff * LN_weight) / RMS(h_pre_ln)
# This is per-example (RMS depends on the hidden state). LN_WEIGHT part is fixed:
W_U_diff_times_LN = W_U_diff * LN_WEIGHT   # [d_model] — reused in inner loop
print(f"  ‖W_U_diff * LN_weight‖ = {W_U_diff_times_LN.norm():.2f}")

print(f"\n  NOTE: Per-example W_U_eff = W_U_diff_times_LN / RMS(h_pre_ln)")
print(f"  This accounts for the final decoder RMSNorm before lm_head.")

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

# ── Phase 4: Helper functions ──────────────────────────────────────────────────────────────────────────────
print("\n" + "="*70)
print("  PHASE 4 — HELPER FUNCTIONS")
print("="*70)

def find_div_steps(token_ids):
    """Action-slot divergence-point step positions (3rd subword of action token)."""
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
    Identical to S-057 M4 and S-058 / S-059 V0.1.0.
    """
    enc_attn = model.decoder.block[l_block].layer[1].EncDecAttention
    with torch.no_grad():
        V_all     = enc_hidden_final @ enc_attn.v.weight.T    # [1, T_enc, n_heads*d_kv]
        attn_w    = step_cross_attns[l_block]                  # [1, n_heads, 1, T_enc]
        head_attn = attn_w[0, h_idx, 0, :]                    # [T_enc]
        head_V    = V_all[0, :, h_idx*d_kv:(h_idx+1)*d_kv]   # [T_enc, d_kv]
        head_val  = (head_attn.unsqueeze(0) @ head_V).squeeze(0)  # [d_kv]
        head_o_sl = enc_attn.o.weight[:, h_idx*d_kv:(h_idx+1)*d_kv]  # [d_model, d_kv]
    return (head_o_sl @ head_val).cpu().float()

def compute_rms(h):
    """T5 RMSNorm RMS: sqrt(mean(h²) + eps)."""
    return float((h.pow(2).mean() + LN_EPS).sqrt().item())

def compute_dlogit_corrected(contrib_h, W_U_eff):
    """
    LayerNorm-corrected δlogit:
      δlogit_h = W_U_eff · contrib_h
    where W_U_eff = (W_U_diff * LN_weight) / RMS(h_pre_ln)  [per example].
    This gives the head's contribution in actual logit units.
    """
    return float(torch.dot(W_U_eff, contrib_h).item())

def run_generate_with_states(cmd):
    """
    Run model.generate() with full state collection.
    Returns enc_hidden, cross_attns, sequences, scores, dec_hidden_states.

    dec_hidden_states structure (from transformers T5):
      Tuple of length n_decode_steps.
      Each element: tuple of (n_dec_layers+1) tensors [1, 1, d_model].
        Index 0: embedding output (input to block 0)
        Index k (1..n_dec_layers): output of block k-1 (pre-LN for k=n_dec_layers)
        Index -1 (= n_dec_layers+1): POST-final-LayerNorm output
      Note: if len == n_dec_layers+1 (no post-LN appended), index -1 = pre-LN.
      The reconstruction verification step confirms which index to use.
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
        "enc_hidden":        out.encoder_hidden_states[-1],   # [1, T_enc, d_model]
        "cross_attns":       out.cross_attentions,             # list[step] tuple[layer]
        "sequences":         out.sequences[0].tolist()[1:],    # strip BOS
        "scores":            out.scores,                       # list[step][1, vocab_size]
        "dec_hidden_states": out.decoder_hidden_states,        # list[step] tuple[layer+1]
    }

print("  Helper functions defined.")
print(f"  Will measure all {n_total_heads} decoder cross-attention heads")

# ── Phase 4b: Determine correct dec_hidden_states index via verification ──────────────────────────────
print("\n" + "="*70)
print("  PHASE 4b — DETERMINE CORRECT DEC_HIDDEN_STATES INDEX")
print("  (Verify which index gives pre-LayerNorm hidden state)")
print("="*70)

# Run one example to find the correct hidden state index
_ex = sub_walk_examples[0]
_data = run_generate_with_states(_ex["commands"])
_div_steps = find_div_steps(_data["sequences"])

# Get observed margin for verification
if _div_steps:
    _div_idx = _div_steps[0][0]
    _obs_margin = float((_data["scores"][_div_idx][0][I_JUMP_ID]
                         - _data["scores"][_div_idx][0][I_WALK_ID]).item())
    _step_states = _data["dec_hidden_states"][_div_idx]  # tuple of states for this step

    print(f"\n  Verification example: '{_ex['commands'][:50]}'")
    print(f"  Observed margin (jump-walk): {_obs_margin:.6f}")
    print(f"  n_states_per_step: {len(_step_states)}")
    print(f"\n  Testing each hidden state index for logit reconstruction:")
    print(f"  {'Index':<7} {'‖h‖':<12} {'recon_margin':<16} {'diff from obs'}")
    print(f"  {'-----':<7} {'---':<12} {'------------':<16} {'-------------'}")

    best_idx = None
    best_err = float('inf')
    for idx in range(len(_step_states)):
        h = _step_states[idx][0, 0, :].cpu().float()   # [d_model]
        h_norm = float(h.norm().item())
        # Compute logit margin from this hidden state WITHOUT LN (it might already be post-LN)
        logit_j = float(torch.dot(W_U_jump, h).item())
        logit_w = float(torch.dot(W_U_walk, h).item())
        margin_direct = logit_j - logit_w
        err_direct = abs(margin_direct - _obs_margin)
        # Compute logit margin applying LN manually (treating this as pre-LN)
        rms = compute_rms(h)
        h_normed = h / rms * LN_WEIGHT
        logit_j_ln = float(torch.dot(W_U_jump, h_normed).item())
        logit_w_ln = float(torch.dot(W_U_walk, h_normed).item())
        margin_with_ln = logit_j_ln - logit_w_ln
        err_with_ln = abs(margin_with_ln - _obs_margin)
        print(f"  [{idx:>2}]  ‖h‖={h_norm:<10.2f} direct={margin_direct:>+10.4f} (err={err_direct:.4f})  "
              f"+LN={margin_with_ln:>+10.4f} (err={err_with_ln:.4f})")
        if err_with_ln < best_err:
            best_err = err_with_ln
            best_idx = idx

    print(f"\n  Best index (with LN applied): [{best_idx}]  err={best_err:.6f}")
    DEC_PRE_LN_IDX = best_idx
else:
    print(f"  WARNING: verification example has no div steps; defaulting to index -2")
    DEC_PRE_LN_IDX = -2   # typical T5 structure: [-2] = last block output (pre-LN)

print(f"\n  Using DEC_PRE_LN_IDX = {DEC_PRE_LN_IDX}")
print(f"  (pre-LayerNorm hidden state at each decode step)")

# ── Phase 5: Logit decomposition — fail group (LayerNorm corrected) ──────────────────────────────────────────
print("\n" + "="*70)
print("  PHASE 5 — LOGIT DECOMPOSITION: FAIL GROUP (M1, LN-corrected)")
print(f"  {len(sub_walk_examples)} substituted_walk examples")
print("="*70)

dlogits_fail      = [[[] for _ in range(n_heads)] for _ in range(n_dec_layers)]
observed_margins_fail = []
recon_margins_fail    = []
rms_values_fail       = []   # track RMS values for diagnostic
recon_errors_fail     = []   # |recon_margin - obs_margin| for reconstruction check
n_valid_fail = 0
n_skip_fail  = 0
skip_reasons_fail = defaultdict(int)

print(f"\n  Running forward passes for {len(sub_walk_examples)} fail examples...")

for ex_idx, ex in enumerate(sub_walk_examples):
    if ex_idx % 5 == 0:
        print(f"  Example {ex_idx+1}/{len(sub_walk_examples)} ...")

    data = run_generate_with_states(ex["commands"])

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

    # Observed margin
    logits_at_div = data["scores"][div_step_idx][0]
    obs_margin = float((logits_at_div[I_JUMP_ID] - logits_at_div[I_WALK_ID]).item())
    observed_margins_fail.append(obs_margin)

    # Pre-LayerNorm hidden state → per-example W_U_eff
    h_pre_ln  = data["dec_hidden_states"][div_step_idx][DEC_PRE_LN_IDX][0, 0, :].cpu().float()
    rms       = compute_rms(h_pre_ln)
    W_U_eff   = W_U_diff_times_LN / rms   # [d_model], per-example effective direction
    rms_values_fail.append(rms)

    # Verify reconstruction (diagnostic only — does not affect measurements)
    h_normed = h_pre_ln / rms * LN_WEIGHT
    margin_check = float(torch.dot(W_U_diff, h_normed).item())
    recon_errors_fail.append(abs(margin_check - obs_margin))

    # Compute δlogit for all 48 heads using W_U_eff
    enc_hidden  = data["enc_hidden"]
    cross_attns = data["cross_attns"][div_step_idx]

    recon_margin = 0.0
    for l in range(n_dec_layers):
        for h in range(n_heads):
            contrib = compute_head_contrib(l, h, cross_attns, enc_hidden)
            dl = compute_dlogit_corrected(contrib, W_U_eff)
            dlogits_fail[l][h].append(dl)
            recon_margin += dl

    recon_margins_fail.append(recon_margin)
    n_valid_fail += 1

print(f"\n  Valid fail examples: {n_valid_fail}")
print(f"  Skipped: {n_skip_fail}")
if n_skip_fail:
    for reason, count in skip_reasons_fail.items():
        print(f"    {reason}: {count}")

mean_rms_fail = float(np.mean(rms_values_fail))
mean_recon_err_fail = float(np.mean(recon_errors_fail)) if recon_errors_fail else 0.0
print(f"\n  Diagnostic: mean RMS(h_pre_ln) = {mean_rms_fail:.4f}")
print(f"  Diagnostic: mean |margin_check - obs_margin| = {mean_recon_err_fail:.6f}")
print(f"  (near-zero = LN formula is verified correct)")

# ── Phase 6: Logit decomposition — success group ───────────────────────────────────────────────────────────────
print("\n" + "="*70)
print("  PHASE 6 — LOGIT DECOMPOSITION: SUCCESS GROUP (M2, LN-corrected)")
print(f"  {len(success_examples)} has_around+correct examples")
print("="*70)

dlogits_success      = [[[] for _ in range(n_heads)] for _ in range(n_dec_layers)]
observed_margins_success = []
recon_margins_success    = []
rms_values_success       = []
recon_errors_success     = []
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

    h_pre_ln  = data["dec_hidden_states"][div_step_idx][DEC_PRE_LN_IDX][0, 0, :].cpu().float()
    rms       = compute_rms(h_pre_ln)
    W_U_eff   = W_U_diff_times_LN / rms
    rms_values_success.append(rms)

    h_normed = h_pre_ln / rms * LN_WEIGHT
    margin_check = float(torch.dot(W_U_diff, h_normed).item())
    recon_errors_success.append(abs(margin_check - obs_margin))

    enc_hidden  = data["enc_hidden"]
    cross_attns = data["cross_attns"][div_step_idx]

    recon_margin = 0.0
    for l in range(n_dec_layers):
        for h in range(n_heads):
            contrib = compute_head_contrib(l, h, cross_attns, enc_hidden)
            dl = compute_dlogit_corrected(contrib, W_U_eff)
            dlogits_success[l][h].append(dl)
            recon_margin += dl

    recon_margins_success.append(recon_margin)
    n_valid_success += 1

print(f"\n  Valid success examples: {n_valid_success}")
print(f"  Skipped: {n_skip_success}")
if n_skip_success:
    for reason, count in skip_reasons_success.items():
        print(f"    {reason}: {count}")

mean_rms_success = float(np.mean(rms_values_success))
mean_recon_err_success = float(np.mean(recon_errors_success)) if recon_errors_success else 0.0
print(f"\n  Diagnostic: mean RMS(h_pre_ln) = {mean_rms_success:.4f}")
print(f"  Diagnostic: mean |margin_check - obs_margin| = {mean_recon_err_success:.6f}")

# ── Phase 7: Aggregate statistics ─────────────────────────────────────────────────────────────────────────────
print("\n" + "="*70)
print("  PHASE 7 — AGGREGATE STATISTICS AND MEASUREMENTS (LN-corrected)")
print("="*70)

def head_stats(dlogits_by_layer):
    stats = {}
    for l in range(n_dec_layers):
        for h in range(n_heads):
            vals  = dlogits_by_layer[l][h]
            label = f"L{l+1}H{h}"
            if vals:
                stats[label] = {"layer": l+1, "head": h,
                                 "mean": float(np.mean(vals)),
                                 "std":  float(np.std(vals)),
                                 "n": len(vals), "vals": vals}
            else:
                stats[label] = {"layer": l+1, "head": h,
                                 "mean": 0.0, "std": 0.0, "n": 0, "vals": []}
    return stats

stats_fail    = head_stats(dlogits_fail)
stats_success = head_stats(dlogits_success)

all_heads_fail    = sorted(stats_fail.items(),    key=lambda kv: abs(kv[1]["mean"]), reverse=True)
all_heads_success = sorted(stats_success.items(), key=lambda kv: abs(kv[1]["mean"]), reverse=True)

dlogit_delta = {}
for label in stats_fail:
    mf = stats_fail[label]["mean"]
    ms = stats_success[label]["mean"] if label in stats_success else 0.0
    dlogit_delta[label] = ms - mf

mean_obs_margin   = float(np.mean(observed_margins_fail))
mean_recon_margin = float(np.mean(recon_margins_fail))
mean_residual     = mean_obs_margin - mean_recon_margin
recon_fraction    = mean_recon_margin / mean_obs_margin if abs(mean_obs_margin) > 1e-9 else 0.0

mean_obs_margin_succ   = float(np.mean(observed_margins_success)) if observed_margins_success else 0.0
mean_recon_margin_succ = float(np.mean(recon_margins_success)) if recon_margins_success else 0.0
recon_fraction_succ    = (mean_recon_margin_succ / mean_obs_margin_succ
                          if abs(mean_obs_margin_succ) > 1e-9 else 0.0)

rank_map_fail = {label: rank for rank, (label, _) in enumerate(all_heads_fail, 1)}
KEY_HEADS     = ["L4H6", "L5H2", "L5H5", "L3H0", "L3H4"]

# ── M1 ───────────────────────────────────────────────────────────────────────────────────────────────────
print(f"\n--- M1: Per-head δlogit at divergence step (fail group, LN-corrected) ---")
print(f"  (N={n_valid_fail} valid examples, values in actual logit units)")
print()
print(f"  {'Rank':<5} {'Head':<8} {'mean δlogit':<14} {'std':<10} {'sign'}")
print(f"  {'----':<5} {'----':<8} {'-----------':<14} {'---':<10} {'----'}")
for rank, (label, s) in enumerate(all_heads_fail, 1):
    sign = "pro-jump" if s["mean"] > 0 else "pro-walk"
    print(f"  {rank:<5} {label:<8} {s['mean']:>+12.4f}   {s['std']:<10.4f} {sign}")

# ── M2 ────────────────────────────────────────────────────────────────────────────────────────────────────
print(f"\n--- M2: Per-head δlogit (success group, LN-corrected) + fail-success delta ---")
print(f"  (N={n_valid_success} valid examples)")
print()
print(f"  {'Rank':<5} {'Head':<8} {'mean δlogit':<14} {'std':<10} {'Δ(succ-fail)':<15} {'sign'}")
print(f"  {'----':<5} {'----':<8} {'-----------':<14} {'---':<10} {'------------':<15} {'----'}")
for rank, (label, s) in enumerate(all_heads_success, 1):
    sign  = "pro-jump" if s["mean"] > 0 else "pro-walk"
    delta = dlogit_delta.get(label, 0.0)
    print(f"  {rank:<5} {label:<8} {s['mean']:>+12.4f}   {s['std']:<10.4f} {delta:>+13.4f}   {sign}")

# ── M3 ────────────────────────────────────────────────────────────────────────────────────────────────────
print(f"\n--- M3: Logit margin reconstruction (LN-corrected, cross-attn heads only) ---")
print(f"  (Reconstruction = sum of 48 cross-attn head δlogits; residual = FFN+embed+self-attn)")
print()
print(f"  Fail group:")
print(f"    Mean observed margin (jump - walk):          {mean_obs_margin:>+10.4f}")
print(f"    Mean reconstructed margin (Σδlogit_h):       {mean_recon_margin:>+10.4f}")
print(f"    Mean residual (obs - recon):                 {mean_residual:>+10.4f}")
print(f"    Reconstruction fraction (recon / obs):       {recon_fraction:>10.4f}")
print()
k2_fires = abs(recon_fraction) < K2_RECON_MIN
print(f"  K2 check: |recon_fraction| >= {K2_RECON_MIN}: ", end="")
print("FIRES" if k2_fires else "CLEAR")
if k2_fires:
    print(f"\n  *** K2 FIRES: cross-attn heads account for < {K2_RECON_MIN:.0%} of observed margin.")
    print(f"  *** FFN/embedding/self-attn dominate the logit decision.")
    print(f"  *** Note: even K2 CLEAR does not guarantee that cross-attn dominates — the residual")
    print(f"  *** includes self-attention and FFN contributions which may partially cancel.")

print(f"\n  Success group:")
print(f"    Mean observed margin:                        {mean_obs_margin_succ:>+10.4f}")
print(f"    Mean reconstructed margin:                   {mean_recon_margin_succ:>+10.4f}")
print(f"    Reconstruction fraction:                     {recon_fraction_succ:>10.4f}")

# ── M4 ────────────────────────────────────────────────────────────────────────────────────────────────────
print(f"\n--- M4: Target head and control head breakdown (LN-corrected) ---")

target_head_sum_fail = sum(stats_fail[h]["mean"] for h in ["L4H6", "L5H2", "L5H5"])
lowest_head_label    = all_heads_fail[-1][0]

print(f"\n  {'Head':<8} {'δlogit fail':<14} {'δlogit succ':<14} {'Δ(succ-fail)':<15} {'rank/48':<9} {'share'}")
print(f"  {'----':<8} {'-----------':<14} {'-----------':<14} {'------------':<15} {'-------':<9} {'-----'}")
for h_label in KEY_HEADS:
    sf = stats_fail[h_label]
    ss = stats_success.get(h_label, {"mean": 0.0})
    delta = dlogit_delta.get(h_label, 0.0)
    rank  = rank_map_fail.get(h_label, -1)
    share = sf["mean"] / mean_recon_margin if abs(mean_recon_margin) > 1e-9 else 0.0
    print(f"  {h_label:<8} {sf['mean']:>+12.4f}   {ss['mean']:>+12.4f}   {delta:>+13.4f}   {rank:<9} {share:>+.4f}")

lowest_sf    = stats_fail[lowest_head_label]
lowest_ss    = stats_success.get(lowest_head_label, {"mean": 0.0})
lowest_delta = dlogit_delta.get(lowest_head_label, 0.0)
lowest_share = lowest_sf["mean"] / mean_recon_margin if abs(mean_recon_margin) > 1e-9 else 0.0
print(f"  {lowest_head_label:<8} {lowest_sf['mean']:>+12.4f}   {lowest_ss['mean']:>+12.4f}   "
      f"{lowest_delta:>+13.4f}   {48:<9} {lowest_share:>+.4f}  ← lowest |δlogit| baseline")

# ── M5 ────────────────────────────────────────────────────────────────────────────────────────────────────
print(f"\n--- M5: Cumulative margin by head rank (fail group, most negative first) ---")

heads_by_dlogit = sorted(stats_fail.items(), key=lambda kv: kv[1]["mean"])
neg_total = sum(s["mean"] for _, s in heads_by_dlogit if s["mean"] < 0)
print(f"\n  Total reconstructed margin from negative-δlogit heads: {neg_total:.4f}")
print(f"\n  {'Rank':<5} {'Head':<8} {'δlogit':<12} {'cumulative':<12} {'cum/neg_total'}")
print(f"  {'----':<5} {'----':<8} {'------':<12} {'----------':<12} {'-'*14}")
frac_50_head = frac_75_head = frac_90_head = None
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
print("  PHASE 8 — HYPOTHESIS VERDICTS (LN-corrected, logit units)")
print("="*70)

# H1
h1_target_ranks = {h: rank_map_fail[h] for h in ["L4H6", "L5H2", "L5H5"]}
h1_in_top5      = all(r <= 5 for r in h1_target_ranks.values())
h1_all_negative = all(stats_fail[h]["mean"] < 0 for h in ["L4H6", "L5H2", "L5H5"])
h1_pass         = h1_in_top5 and h1_all_negative

print(f"\n  H1 — L4H6/L5H2/L5H5 in top-5 by |δlogit| AND all negative in fail:")
for h in ["L4H6", "L5H2", "L5H5"]:
    s = stats_fail[h]
    print(f"    {h}: mean={s['mean']:>+.4f}  rank={h1_target_ranks[h]}/48")
print(f"  All in top-5: {h1_in_top5}  |  All negative: {h1_all_negative}")
print(f"  H1: {'PASS' if h1_pass else 'FAIL'}")

# K1
k1_fires = any(stats_fail[h]["mean"] > 0 for h in ["L4H6", "L5H2", "L5H5"])
print(f"\n  K1 — Target heads positive in fail:")
for h in ["L4H6", "L5H2", "L5H5"]:
    print(f"    {h}: mean={stats_fail[h]['mean']:>+.4f}")
print(f"  K1: {'FIRES' if k1_fires else 'CLEAR'}")
if k1_fires:
    print(f"  *** K1 FIRES: at least one target head is pro-jump in fail cases.")

# H2
h2_deltas   = {h: dlogit_delta[h] for h in ["L4H6", "L5H2", "L5H5"]}
h2_per_head = {h: d >= H2_DELTA_MIN for h, d in h2_deltas.items()}
h2_pass     = all(h2_per_head.values())

print(f"\n  H2 — Δ(δlogit_h) = success−fail ≥ +{H2_DELTA_MIN} for each target head:")
for h in ["L4H6", "L5H2", "L5H5"]:
    d = h2_deltas[h]
    print(f"    {h}: Δ={d:>+.4f}  {'≥' if d >= H2_DELTA_MIN else '<'} {H2_DELTA_MIN}")
print(f"  H2: {'PASS' if h2_pass else 'FAIL'}")

# H3
h3_target_sum = sum(stats_fail[h]["mean"] for h in ["L4H6", "L5H2", "L5H5"])
h3_share      = h3_target_sum / mean_recon_margin if abs(mean_recon_margin) > 1e-9 else 0.0
h3_pass       = h3_share >= H3_SHARE_MIN

print(f"\n  H3 — Target heads ≥{H3_SHARE_MIN:.0%} of reconstructed negative margin:")
print(f"    Target sum  = {h3_target_sum:>+.4f}")
print(f"    Recon margin = {mean_recon_margin:>+.4f}")
print(f"    Share = {h3_share:>+.4f}  (threshold = {H3_SHARE_MIN:.2f})")
print(f"  H3: {'PASS' if h3_pass else 'FAIL'}")

# H4
h4_l4h6_abs = abs(stats_fail["L4H6"]["mean"])
h4_l3h0_abs = abs(stats_fail["L3H0"]["mean"])
h4_l3h4_abs = abs(stats_fail["L3H4"]["mean"])
h4_pass     = (h4_l3h0_abs < h4_l4h6_abs) and (h4_l3h4_abs < h4_l4h6_abs)

print(f"\n  H4 — Control heads |δlogit| < L4H6 |δlogit|:")
print(f"    |δlogit_L4H6| = {h4_l4h6_abs:.4f}")
print(f"    |δlogit_L3H0| = {h4_l3h0_abs:.4f}  {'<' if h4_l3h0_abs < h4_l4h6_abs else '≥'} L4H6")
print(f"    |δlogit_L3H4| = {h4_l3h4_abs:.4f}  {'<' if h4_l3h4_abs < h4_l4h6_abs else '≥'} L4H6")
print(f"  H4: {'PASS' if h4_pass else 'FAIL'}")

print(f"\n  ─── VERDICT SUMMARY ───")
print(f"  H1 (L4H6/L5H2/L5H5 top-5, negative):           {'PASS' if h1_pass else 'FAIL'}")
print(f"  H2 (target heads Δ ≥ +0.20 success-fail):        {'PASS' if h2_pass else 'FAIL'}")
print(f"  H3 (target heads ≥ 30% of neg margin):            {'PASS' if h3_pass else 'FAIL'}")
print(f"  H4 (control heads smaller than L4H6):             {'PASS' if h4_pass else 'FAIL'}")
print(f"  K1 (target heads pro-jump — mechanism wrong):     {'FIRES' if k1_fires else 'CLEAR'}")
print(f"  K2 (reconstruction < 20% — FFN dominates):        {'FIRES' if k2_fires else 'CLEAR'}")

# ── Phase 9: Save results ──────────────────────────────────────────────────────────────────────────────
print("\n" + "="*70)
print("  PHASE 9 — SAVE RESULTS")
print("="*70)

RESULTS_PATH = "workbench/results/059_v2_results.json"
os.makedirs(os.path.dirname(RESULTS_PATH), exist_ok=True)

def build_head_table(stats_dict):
    return {label: {"layer": s["layer"], "head": s["head"],
                    "mean": s["mean"], "std": s["std"], "n": s["n"]}
            for label, s in stats_dict.items()}

def build_per_example_arrays(dlogits_by_layer):
    arrays = {}
    for l in range(n_dec_layers):
        for h in range(n_heads):
            label = f"L{l+1}H{h}"
            arrays[label] = dlogits_by_layer[l][h]
    return arrays

results = {
    "script_id":       SCRIPT_ID,
    "seed":            SEED,
    "version":         "V0.2.0",
    "ln_corrected":    True,
    "dec_pre_ln_idx":  DEC_PRE_LN_IDX,
    "mean_rms_fail":   mean_rms_fail,
    "mean_rms_success": float(np.mean(rms_values_success)) if rms_values_success else 0.0,
    "mean_recon_err_fail": mean_recon_err_fail,
    "n_valid_fail":    n_valid_fail,
    "n_valid_success": n_valid_success,

    "fail_head_table":    build_head_table(stats_fail),
    "fail_head_ranking":  [label for label, _ in all_heads_fail],
    "success_head_table": build_head_table(stats_success),
    "dlogit_delta":       {label: dlogit_delta[label] for label in stats_fail},

    "m3_fail": {
        "mean_observed_margin": mean_obs_margin,
        "mean_recon_margin":    mean_recon_margin,
        "mean_residual":        mean_residual,
        "recon_fraction":       recon_fraction,
        "observed_margins":     observed_margins_fail,
        "recon_margins":        recon_margins_fail,
    },
    "m3_success": {
        "mean_observed_margin": mean_obs_margin_succ,
        "mean_recon_margin":    mean_recon_margin_succ,
        "recon_fraction":       recon_fraction_succ,
    },

    "m4_key_heads": {h: {
        "mean_fail":       stats_fail[h]["mean"],
        "mean_success":    stats_success[h]["mean"] if h in stats_success else 0.0,
        "delta":           dlogit_delta.get(h, 0.0),
        "rank_fail":       rank_map_fail.get(h, -1),
        "share_of_recon":  stats_fail[h]["mean"] / mean_recon_margin if abs(mean_recon_margin) > 1e-9 else 0.0,
    } for h in KEY_HEADS + [lowest_head_label]},

    "m5_cumulative": {
        "frac_50_head": frac_50_head,
        "frac_75_head": frac_75_head,
        "frac_90_head": frac_90_head,
        "neg_total": neg_total,
    },

    "verdicts": {
        "H1": {"pass": h1_pass, "in_top5": h1_in_top5, "all_negative": h1_all_negative,
               "target_ranks": h1_target_ranks},
        "H2": {"pass": h2_pass, "deltas": h2_deltas, "threshold": H2_DELTA_MIN},
        "H3": {"pass": h3_pass, "share": h3_share, "target_sum": h3_target_sum,
               "threshold": H3_SHARE_MIN},
        "H4": {"pass": h4_pass, "l4h6_abs": h4_l4h6_abs, "l3h0_abs": h4_l3h0_abs,
               "l3h4_abs": h4_l3h4_abs},
        "K1": {"fires": k1_fires},
        "K2": {"fires": k2_fires, "recon_fraction": recon_fraction},
    },

    "per_example_fail":    build_per_example_arrays(dlogits_fail),
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
print(f"\n  LayerNorm correction: RMS(h_pre_ln) ≈ {mean_rms_fail:.4f} (fail)")
print(f"  Reconstruction error (LN formula check): {mean_recon_err_fail:.6f}")
print(f"\n  Observed margin (fail):      {mean_obs_margin:>+10.4f}")
print(f"  Reconstructed margin (fail): {mean_recon_margin:>+10.4f}")
print(f"  Reconstruction fraction:     {recon_fraction:>10.4f}")
print(f"  (= cross-attn share of total logit margin; residual = FFN+embed+self-attn)")
print(f"\n  Top 5 heads by |δlogit| in fail group (logit units):")
for rank, (label, s) in enumerate(all_heads_fail[:5], 1):
    sign = "pro-jump" if s["mean"] > 0 else "pro-walk"
    print(f"    {rank}. {label:<8}  {s['mean']:>+.4f}  ({sign})")
print(f"\n  Target head contributions (logit units):")
for h in ["L4H6", "L5H2", "L5H5"]:
    s = stats_fail[h]
    d = dlogit_delta[h]
    print(f"    {h}: fail={s['mean']:>+.4f}  succ={stats_success[h]['mean']:>+.4f}  "
          f"Δ={d:>+.4f}  rank={rank_map_fail[h]}/48")
print(f"\n  Verdicts: H1={'PASS' if h1_pass else 'FAIL'}  H2={'PASS' if h2_pass else 'FAIL'}  "
      f"H3={'PASS' if h3_pass else 'FAIL'}  H4={'PASS' if h4_pass else 'FAIL'}")
print(f"  Kill cond: K1={'FIRES' if k1_fires else 'CLEAR'}  K2={'FIRES' if k2_fires else 'CLEAR'}")
print()
print(f"  Results: {RESULTS_PATH}")
print()
print("─" * 70)
print("  After running: compare outputs to the corresponding sealed methods report in findings/; private coordination files are not included in this public export.")
print("─" * 70)
