#!/usr/bin/env python3
"""
S-060_CAUSAL_ABLATION.py — Causal Ablation: Are L6H2/L6H6 Causing the Wrong Decision?
Applied Categorical Physics Workbench
Troy Teno | May 2026 | Open Access

Pre-registered hypotheses:
  workbench/proposals/S-060_CAUSAL_ABLATION_PROPOSAL.md
DO NOT modify hypotheses after seeing results.

─────────────────────────────────────────────────────────────────────
SETUP (run once in Colab or terminal):
  pip install torch transformers datasets numpy sentencepiece -q

BACKGROUND:
  S-059b (logit decomposition, LayerNorm-corrected) identified the L6
  cluster as the dominant calibrated pro-walk signal:
    L6H2 = −51.2, L5H6 = −40.6, L6H6 = −37.7, L6H5 = −29.6
  L4H6 (−10.4) and L5H5 (−19.7) are directionally confirmed original
  targets. L5H2 (−0.10) is removed from the primary target set.

  S-058 showed cross-example transplants are non-specific (OOD). S-060
  uses WITHIN-EXAMPLE NEUTRALIZATION: zero out each head's V/O projection
  contribution within the fail example itself, using a forward hook on
  EncDecAttention. No donor examples. No matching. No cross-example
  context contamination.

  Each target arm is compared against a LAYER-MATCHED CONTROL arm
  (a head at the same decoder block with near-zero δlogit). The
  SPECIFICITY SIGNAL is:

      Δspecificity = mean_Δmargin_target − mean_Δmargin_control

  A large Δspecificity means the target head is causally upstream
  of the wrong logit decision, not just any head at that layer.

  Arms:
    A: L6H2 (target) vs L6H3 (control, same block 5)
    B: L6H6 (target) vs L6H1 (control, same block 5)
    C: L6H2+L6H6 jointly vs L6H3+L6H1 jointly
    D: L4H6 (target) vs L4H3 (control, same block 3)
    E: L5H5 (target) vs L5H4 (control, same block 4)
    F: L6H2+L6H6+L4H6+L5H5 jointly vs all four controls jointly

  Pre-registered hypotheses:
    H1: Arm A Δspecificity ≥ 2.0 AND Arm B Δspecificity ≥ 2.0
    H2: Arm D Δspecificity ≥ 1.0 AND Arm E Δspecificity ≥ 1.0
        AND mean_Δmargin_A > mean_Δmargin_D
        AND mean_Δmargin_B > mean_Δmargin_E
    H3: Arm F target flip fraction ≥ 0.25
    H4: Arm F target OOD-safe (< 5/29 with P(I_JUMP)+P(I_WALK) < 0.01)

  Kill conditions:
    K1: No arm produces Δspecificity ≥ 0.50
    K2: Arm F flip < 0.10 AND H4 passes (OOD-safe)
    K3: > 10/29 examples OOD in any arm

Version history:
  V0.1.0 — initial build
─────────────────────────────────────────────────────────────────────
"""

import numpy as np
import json
import os
import random
from collections import defaultdict

SCRIPT_ID = "S-060_CAUSAL_ABLATION_V0.1.0"
SEED = 42
np.random.seed(SEED)
random.seed(SEED)

CHECKPOINT_DIR = "043_t5_scan_checkpoint"
MAX_INPUT_LEN  = 50
MAX_TARGET_LEN = 100
N_FAIL         = 30   # take top N_FAIL after shuffle with SEED=42

# Pre-registered thresholds
H1_SPECIFICITY_MIN  = 2.0   # Δspecificity for L6H2/L6H6 arms (logit units)
H2_SPECIFICITY_MIN  = 1.0   # Δspecificity for L4H6/L5H5 arms
H3_FLIP_MIN         = 0.25  # Arm F target flip fraction
H4_OOD_MAX_COUNT    = 5     # max OOD examples in Arm F for H4 to pass
K1_SPEC_MIN         = 0.50  # K1 fires if NO arm reaches this Δspecificity
K2_FLIP_MAX         = 0.10  # K2 flip threshold for Arm F
K3_OOD_MAX_COUNT    = 10    # K3 fires if > this many OOD in any arm
OOD_THRESH          = 0.01  # P(I_JUMP) + P(I_WALK) < this → OOD flag

# Head definitions: (decoder_block_0indexed, head_0indexed, label)
# L-labels are 1-indexed (L6 = block 5 in 0-indexed T5-small)
L6H2 = (5, 2, "L6H2")   # primary target 1 (δlogit −51.2 in S-059b)
L6H6 = (5, 6, "L6H6")   # primary target 2 (δlogit −37.7)
L6H3 = (5, 3, "L6H3")   # control for L6H2 (δlogit −1.43)
L6H1 = (5, 1, "L6H1")   # control for L6H6 (δlogit −4.26)
L4H6 = (3, 6, "L4H6")   # reference target 1 (δlogit −10.4)
L5H5 = (4, 5, "L5H5")   # reference target 2 (δlogit −19.7)
L4H3 = (3, 3, "L4H3")   # control for L4H6 (δlogit +0.53)
L5H4 = (4, 4, "L5H4")   # control for L5H5 (δlogit −0.62)

# All heads that need contrib computed per example
ALL_MEASURE_HEADS = [L6H2, L6H6, L6H3, L6H1, L4H6, L5H5, L4H3, L5H4]

# Arm definitions (all arms use within-example neutralization)
ARMS = {
    "A_tgt": [L6H2],
    "A_ctl": [L6H3],
    "B_tgt": [L6H6],
    "B_ctl": [L6H1],
    "C_tgt": [L6H2, L6H6],
    "C_ctl": [L6H3, L6H1],
    "D_tgt": [L4H6],
    "D_ctl": [L4H3],
    "E_tgt": [L5H5],
    "E_ctl": [L5H4],
    "F_tgt": [L6H2, L6H6, L4H6, L5H5],
    "F_ctl": [L6H3, L6H1, L4H3, L5H4],
}

# Comparison pairs: (target_arm, control_arm, pair_label)
COMPARISONS = [
    ("A_tgt", "A_ctl", "A"),
    ("B_tgt", "B_ctl", "B"),
    ("C_tgt", "C_ctl", "C"),
    ("D_tgt", "D_ctl", "D"),
    ("E_tgt", "E_ctl", "E"),
    ("F_tgt", "F_ctl", "F"),
]

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
print(f"  Causal ablation: L6H2/L6H6 vs layer-matched controls")
print(f"{'='*70}")

# ── Imports ─────────────────────────────────────────────────────────────────────────────
print("\nImporting libraries...")
try:
    import torch
    from transformers import T5ForConditionalGeneration, T5Tokenizer
except ImportError as e:
    raise SystemExit(f"\nMissing library: {e}\n"
                     "Run: pip install torch transformers datasets sentencepiece -q\n")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"  Device: {device}")
torch.manual_seed(SEED)

# ── Phase 1: Load checkpoint ───────────────────────────────────────────────────────────────────
print("\n" + "="*70)
print("  PHASE 1 — LOAD S-043 CHECKPOINT")
print("="*70)

if not os.path.exists(CHECKPOINT_DIR):
    raise SystemExit(f"\n  Checkpoint not found: {CHECKPOINT_DIR}\n"
                     "  Restore from Drive or run S-043 first.\n")

tokenizer = T5Tokenizer.from_pretrained(CHECKPOINT_DIR)
model     = T5ForConditionalGeneration.from_pretrained(CHECKPOINT_DIR).to(device)
model.eval()
print(f"  Loaded {CHECKPOINT_DIR!r}.")

# ── Phase 2: Token IDs ──────────────────────────────────────────────────────────────────────────────
print("\n" + "="*70)
print("  PHASE 2 — ACTION TOKEN IDs")
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
BOS_ID    = model.config.decoder_start_token_id   # 0 for T5

print(f"\n  Divergence-point IDs: {DIV_IDS}")
print(f"  I_JUMP_ID={I_JUMP_ID}, I_WALK_ID={I_WALK_ID}")
print(f"  TURN_END_IDS: {TURN_END_IDS}")
print(f"  BOS_ID (decoder_start_token_id): {BOS_ID}")

n_dec_layers = model.config.num_decoder_layers   # 6
n_heads      = model.config.num_heads            # 8
d_kv         = model.config.d_kv                 # 64
d_model      = model.config.d_model              # 512
print(f"\n  T5 decoder: {n_dec_layers} layers × {n_heads} heads, d_kv={d_kv}, d_model={d_model}")

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

print(f"\n  Running inference to classify fail group...")
BATCH = 32
sub_walk_examples = []

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
        if (i // BATCH) % 20 == 0:
            print(f"    {i+len(batch)}/{len(test_jc)}")

random.seed(SEED)
random.shuffle(sub_walk_examples)
sub_walk_examples = sub_walk_examples[:N_FAIL]
print(f"\n  substituted_walk fail examples: {len(sub_walk_examples)}")

# ── Phase 4: Helper functions ──────────────────────────────────────────────────────────────────────────────
print("\n" + "="*70)
print("  PHASE 4 — HELPER FUNCTIONS")
print("="*70)

def find_div_steps(token_ids):
    """First action-slot divergence step in a generated sequence (BOS stripped)."""
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
    Head V/O projection contribution to decoder residual stream.
    contrib_h = W_O_h @ (attn_h @ W_V_h @ enc_hidden)
    Returns [d_model] float32 CPU tensor.
    Identical method to S-057/S-058/S-059.
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

def run_generate_with_states(cmd):
    """Run model.generate() collecting cross-attentions and scores."""
    inp = tokenizer(cmd, truncation=True, max_length=MAX_INPUT_LEN,
                    return_tensors="pt").to(device)
    with torch.no_grad():
        out = model.generate(
            input_ids=inp["input_ids"],
            attention_mask=inp["attention_mask"],
            max_length=MAX_TARGET_LEN,
            output_attentions=True,
            output_hidden_states=False,
            return_dict_in_generate=True,
            output_scores=True,
        )
    return {
        "enc_input":   inp,                                     # tokenizer output
        "enc_hidden":  None,                                    # computed on demand
        "cross_attns": out.cross_attentions,                    # list[step] tuple[layer]
        "sequences":   out.sequences[0].tolist()[1:],           # strip BOS
        "scores":      out.scores,                              # list[step][1, vocab_size]
        "inp_input_ids":       inp["input_ids"],
        "inp_attention_mask":  inp["attention_mask"],
    }

def get_enc_hidden(cmd_data):
    """Compute encoder hidden states (cached on demand)."""
    if cmd_data["enc_hidden"] is not None:
        return cmd_data["enc_hidden"]
    with torch.no_grad():
        enc_out = model.encoder(
            input_ids=cmd_data["inp_input_ids"].to(device),
            attention_mask=cmd_data["inp_attention_mask"].to(device),
            return_dict=True,
        )
    cmd_data["enc_hidden"] = enc_out.last_hidden_state   # [1, T_enc, d_model]
    return cmd_data["enc_hidden"]

def get_logit_outcome(logits_1d):
    """From [vocab_size] logit tensor, extract P(I_JUMP), P(I_WALK), margin, flip."""
    probs  = torch.softmax(logits_1d.float(), dim=-1)
    p_jump = float(probs[I_JUMP_ID].item())
    p_walk = float(probs[I_WALK_ID].item())
    margin = float(logits_1d[I_JUMP_ID].item()) - float(logits_1d[I_WALK_ID].item())
    flip   = 1 if p_jump > p_walk else 0
    p_sum  = p_jump + p_walk
    ood    = 1 if p_sum < OOD_THRESH else 0
    return {"p_jump": p_jump, "p_walk": p_walk, "margin": margin,
            "flip": flip, "p_sum": p_sum, "ood": ood}

def run_neutralized_arm(arm_heads, contribs_fail, fail_prefix, enc_input):
    """
    Run a within-example neutralization arm using forward hook.

    For each head in arm_heads, registers a hook on EncDecAttention at that
    block to subtract contrib_fail[head_label] from the last decoder position.

    Runs model.forward() with [BOS] + fail_prefix as decoder_input_ids.
    Returns get_logit_outcome dict for the last decoder position.
    """
    # Build per-layer neutralization vectors
    layer_neut = defaultdict(lambda: torch.zeros(d_model, dtype=torch.float32))
    for (l_block, h_idx, label) in arm_heads:
        layer_neut[l_block] += contribs_fail[label]   # will subtract via hook

    # Register hooks
    hooks = []
    for l_block, neut_vec in layer_neut.items():
        enc_dec_attn = model.decoder.block[l_block].layer[1].EncDecAttention
        nv = neut_vec.to(device)
        def make_hook(v):
            def hook_fn(module, input, output):
                # output[0]: [batch, dec_seq_len, d_model]
                patched = output[0].clone()
                patched[0, -1, :] -= v   # subtract head contribution at last pos
                return (patched,) + output[1:]
            return hook_fn
        h = enc_dec_attn.register_forward_hook(make_hook(nv))
        hooks.append(h)

    # Build decoder_input_ids: [BOS] + fail_prefix
    dec_ids = [BOS_ID] + fail_prefix
    dec_tensor = torch.tensor([dec_ids], dtype=torch.long, device=device)

    try:
        with torch.no_grad():
            out = model(
                input_ids=enc_input["input_ids"].to(device),
                attention_mask=enc_input["attention_mask"].to(device),
                decoder_input_ids=dec_tensor,
            )
        patched_logits = out.logits[0, -1, :].cpu()   # last decoder position
        result = get_logit_outcome(patched_logits)
    except Exception as e:
        result = {"p_jump": 0.0, "p_walk": 0.0, "margin": 0.0,
                  "flip": 0, "p_sum": 0.0, "ood": 1, "error": str(e)}
    finally:
        for h in hooks:
            h.remove()

    return result

print("  Helper functions defined.")

# ── Phase 5: Collect baselines and head contributions ───────────────────────────────────────────────
print("\n" + "="*70)
print("  PHASE 5 — COLLECT BASELINES AND HEAD CONTRIBUTIONS")
print(f"  {len(sub_walk_examples)} fail examples")
print("="*70)

per_example = []
n_valid = 0
n_skip  = 0
skip_reasons = defaultdict(int)

# Verification: compare model.generate() baseline with model.forward() no-hook baseline
# Report discrepancy to flag any forward vs generate inconsistency
fwd_verify_errors = []

print(f"\n  Running forward passes for {len(sub_walk_examples)} examples...")
print(f"  (collecting generate baseline + head contributions)")

for ex_idx, ex in enumerate(sub_walk_examples):
    if ex_idx % 5 == 0:
        print(f"  Example {ex_idx+1}/{len(sub_walk_examples)} ...")

    result = {
        "cmd":       ex["commands"],
        "valid":     True,
        "skip_reason": None,
        "contribs":  {},
        "baseline":  None,
        "fwd_baseline": None,
    }

    # ── generate() pass ──
    data = run_generate_with_states(ex["commands"])
    div_steps = find_div_steps(data["sequences"])

    if not div_steps:
        result["valid"] = False
        result["skip_reason"] = "no_div_steps"
        per_example.append(result)
        n_skip += 1
        skip_reasons["no_div_steps"] += 1
        continue

    div_step_idx, div_action = div_steps[0]

    if div_step_idx >= len(data["cross_attns"]):
        result["valid"] = False
        result["skip_reason"] = "div_step_out_of_range"
        per_example.append(result)
        n_skip += 1
        skip_reasons["div_step_out_of_range"] += 1
        continue

    result["div_step_idx"] = div_step_idx
    result["div_action"]   = div_action

    # Baseline from generate()
    gen_logits = data["scores"][div_step_idx][0]   # [vocab_size]
    result["baseline"] = get_logit_outcome(gen_logits)

    # ── Compute head contributions ──
    enc_hidden  = get_enc_hidden(data)
    cross_attns = data["cross_attns"][div_step_idx]

    for (l_block, h_idx, label) in ALL_MEASURE_HEADS:
        result["contribs"][label] = compute_head_contrib(
            l_block, h_idx, cross_attns, enc_hidden
        )

    # ── Forward pass verification (no hooks) ──
    fail_prefix = data["sequences"][:div_step_idx]
    if fail_prefix:
        fwd_result = run_neutralized_arm([], {}, fail_prefix, data["enc_input"])
        result["fwd_baseline"] = fwd_result
        fwd_err = abs(fwd_result["margin"] - result["baseline"]["margin"])
        fwd_verify_errors.append(fwd_err)
        if ex_idx < 3:  # show first 3 for diagnostic
            print(f"    [verify ex {ex_idx}] gen_margin={result['baseline']['margin']:>+.4f}  "
                  f"fwd_margin={fwd_result['margin']:>+.4f}  err={fwd_err:.4f}")

    result["fail_prefix"] = fail_prefix
    result["enc_input"]   = data["enc_input"]
    per_example.append(result)
    n_valid += 1

print(f"\n  Valid examples: {n_valid}")
print(f"  Skipped: {n_skip}")
if n_skip:
    for r, c in skip_reasons.items():
        print(f"    {r}: {c}")

if fwd_verify_errors:
    mean_fwd_err = float(np.mean(fwd_verify_errors))
    max_fwd_err  = float(np.max(fwd_verify_errors))
    print(f"\n  Forward-vs-generate verification:")
    print(f"  Mean error: {mean_fwd_err:.4f}")
    print(f"  Max error:  {max_fwd_err:.4f}")
    if mean_fwd_err > 1.0:
        print(f"  WARNING: Large forward/generate discrepancy. Patched margins "
              f"are relative to forward() baseline, not generate() baseline.")
    else:
        print(f"  OK: forward and generate baselines are consistent.")

# ── Phase 6: Run all ablation arms ───────────────────────────────────────────────────────────────────────
print("\n" + "="*70)
print("  PHASE 6 — ABLATION ARMS (within-example neutralization)")
print(f"  {n_valid} valid examples × {len(ARMS)} arms")
print("="*70)

# arm_results[arm_name] = list of per-example results
arm_results = {arm: [] for arm in ARMS}

valid_examples = [r for r in per_example if r["valid"]]

print(f"\n  Running {len(valid_examples) * len(ARMS)} patched forward passes...")

for ex_idx, result in enumerate(valid_examples):
    if ex_idx % 5 == 0:
        print(f"  Example {ex_idx+1}/{len(valid_examples)} ...")

    fail_prefix = result["fail_prefix"]
    enc_input   = result["enc_input"]
    contribs    = result["contribs"]
    baseline    = result["baseline"]

    for arm_name, arm_heads in ARMS.items():
        if not fail_prefix:
            arm_results[arm_name].append({
                "skip": True, "reason": "empty_prefix",
                "baseline_margin": baseline["margin"],
            })
            continue

        patched = run_neutralized_arm(arm_heads, contribs, fail_prefix, enc_input)
        delta_margin = patched["margin"] - (
            result["fwd_baseline"]["margin"] if result["fwd_baseline"] else baseline["margin"]
        )
        arm_results[arm_name].append({
            "skip": False,
            "baseline_margin":  baseline["margin"],
            "patched_margin":   patched["margin"],
            "delta_margin":     delta_margin,
            "flip":             patched["flip"],
            "p_jump":           patched["p_jump"],
            "p_walk":           patched["p_walk"],
            "p_sum":            patched["p_sum"],
            "ood":              patched["ood"],
        })

print(f"\n  Done.")

# ── Phase 7: Aggregate statistics ────────────────────────────────────────────────────────────────────────────
print("\n" + "="*70)
print("  PHASE 7 — AGGREGATE STATISTICS")
print("="*70)

def arm_stats(arm_name):
    """Compute aggregate stats for one arm across valid examples."""
    records = [r for r in arm_results[arm_name] if not r.get("skip")]
    if not records:
        return {"n": 0, "flip_frac": 0.0, "mean_delta_margin": 0.0,
                "std_delta_margin": 0.0, "mean_p_jump": 0.0, "mean_p_walk": 0.0,
                "mean_p_sum": 0.0, "n_ood": 0, "ood_frac": 0.0}
    deltas  = [r["delta_margin"] for r in records]
    flips   = [r["flip"] for r in records]
    p_sums  = [r["p_sum"] for r in records]
    n_ood   = sum(r["ood"] for r in records)
    return {
        "n": len(records),
        "flip_frac":         float(np.mean(flips)),
        "mean_delta_margin": float(np.mean(deltas)),
        "std_delta_margin":  float(np.std(deltas)),
        "mean_p_jump":       float(np.mean([r["p_jump"] for r in records])),
        "mean_p_walk":       float(np.mean([r["p_walk"] for r in records])),
        "mean_p_sum":        float(np.mean(p_sums)),
        "n_ood": n_ood,
        "ood_frac": n_ood / len(records),
    }

stats = {arm: arm_stats(arm) for arm in ARMS}

# ── M1 — Baseline ──────────────────────────────────────────────────────────────────────────────────────
print(f"\n--- M1: Baseline (unpatched fail group, from model.generate()) ---")
baseline_records = [r for r in per_example if r["valid"] and r["baseline"]]
bl_margins = [r["baseline"]["margin"] for r in baseline_records]
bl_p_jump  = [r["baseline"]["p_jump"]  for r in baseline_records]
bl_p_walk  = [r["baseline"]["p_walk"]  for r in baseline_records]
bl_flips   = [r["baseline"]["flip"]    for r in baseline_records]

print(f"\n  n = {len(baseline_records)} valid examples")
print(f"  mean P(I_WALK) = {np.mean(bl_p_walk):.4f}")
print(f"  mean P(I_JUMP) = {np.mean(bl_p_jump):.4f}")
print(f"  mean logit margin (jump - walk) = {np.mean(bl_margins):.4f}")
print(f"  baseline flip fraction = {np.mean(bl_flips):.3f}  (should be 0.000)")

# ── M2 — Per-arm results ──────────────────────────────────────────────────────────────────────────────────────
print(f"\n--- M2: Per-arm flip fraction and mean Δmargin ---")
print(f"\n  {'Arm':<10} {'n':<5} {'flip_frac':<12} {'Δmargin':<12} {'std':<10} {'n_ood':<7} {'p_sum'}")
print(f"  {'---':<10} {'--':<5} {'---------':<12} {'-------':<12} {'---':<10} {'-----':<7} {'-----'}")
for arm_name in ARMS:
    s = stats[arm_name]
    print(f"  {arm_name:<10} {s['n']:<5} {s['flip_frac']:<12.4f} "
          f"{s['mean_delta_margin']:>+10.4f}   {s['std_delta_margin']:<10.4f} "
          f"{s['n_ood']:<7} {s['mean_p_sum']:.4f}")

# ── M3 — Specificity comparison ─────────────────────────────────────────────────────────────────────────────────
print(f"\n--- M3: Specificity comparison (target − control) ---")
print(f"\n  {'Pair':<5} {'Δmargin_tgt':<14} {'Δmargin_ctl':<14} {'Δspecificity':<15} "
      f"{'flip_tgt':<10} {'flip_ctl'}")
print(f"  {'----':<5} {'-----------':<14} {'-----------':<14} {'------------':<15} "
      f"{'--------':<10} {'--------'}")
specificity = {}
for tgt_arm, ctl_arm, pair_label in COMPARISONS:
    s_tgt = stats[tgt_arm]
    s_ctl = stats[ctl_arm]
    delta_spec  = s_tgt["mean_delta_margin"] - s_ctl["mean_delta_margin"]
    delta_flip  = s_tgt["flip_frac"] - s_ctl["flip_frac"]
    specificity[pair_label] = {
        "delta_margin_tgt": s_tgt["mean_delta_margin"],
        "delta_margin_ctl": s_ctl["mean_delta_margin"],
        "delta_specificity": delta_spec,
        "flip_tgt": s_tgt["flip_frac"],
        "flip_ctl": s_ctl["flip_frac"],
        "delta_flip": delta_flip,
    }
    print(f"  {pair_label:<5} {s_tgt['mean_delta_margin']:>+12.4f}   "
          f"{s_ctl['mean_delta_margin']:>+12.4f}   {delta_spec:>+13.4f}   "
          f"{s_tgt['flip_frac']:.4f}     {s_ctl['flip_frac']:.4f}")

# ── M4 — OOD check ───────────────────────────────────────────────────────────────────────────────────────────────
print(f"\n--- M4: OOD check (P(I_JUMP) + P(I_WALK) < {OOD_THRESH}) ---")
print(f"\n  {'Arm':<10} {'n_ood':<8} {'ood_frac':<10} {'mean_p_sum':<12} {'status'}")
print(f"  {'---':<10} {'-----':<8} {'--------':<10} {'----------':<12} {'------'}")
for arm_name in ARMS:
    s = stats[arm_name]
    k3_flag = "K3-FIRES" if s["n_ood"] > K3_OOD_MAX_COUNT else "ok"
    print(f"  {arm_name:<10} {s['n_ood']:<8} {s['ood_frac']:<10.4f} "
          f"{s['mean_p_sum']:<12.4f} {k3_flag}")

# ── M5 — Norm check ───────────────────────────────────────────────────────────────────────────────────────────────
print(f"\n--- M5: Head contribution norms (diagnostic) ---")
# Report mean ‖contrib_h‖ for each measured head
print(f"\n  {'Head':<8} {'mean ‖contrib‖':<16} {'std ‖contrib‖':<16} {'S-059b δlogit'}")
print(f"  {'----':<8} {'-------------':<16} {'-------------':<16} {'-------------'}")
s059b_dlogit = {"L6H2": -51.21, "L6H6": -37.67, "L6H3": -1.43, "L6H1": -4.26,
                "L4H6": -10.41, "L5H5": -19.65, "L4H3": 0.53,  "L5H4": -0.62}
for (l_block, h_idx, label) in ALL_MEASURE_HEADS:
    norms = [float(r["contribs"][label].norm().item())
             for r in per_example if r["valid"] and label in r["contribs"]]
    if norms:
        print(f"  {label:<8} {np.mean(norms):>14.2f}   {np.std(norms):>14.2f}   "
              f"{s059b_dlogit.get(label, '—'):>+.2f}")

# ── Phase 8: Hypothesis verdicts ──────────────────────────────────────────────────────────────────────
print("\n" + "="*70)
print("  PHASE 8 — HYPOTHESIS VERDICTS")
print("="*70)

# H1: L6H2 and L6H6 Δspecificity ≥ H1_SPECIFICITY_MIN
spec_A = specificity["A"]["delta_specificity"]
spec_B = specificity["B"]["delta_specificity"]
h1_A   = spec_A >= H1_SPECIFICITY_MIN
h1_B   = spec_B >= H1_SPECIFICITY_MIN
h1_pass = h1_A and h1_B

print(f"\n  H1 — L6H2/L6H6 Δspecificity ≥ {H1_SPECIFICITY_MIN} logit units each:")
print(f"    Arm A (L6H2): Δspecificity = {spec_A:>+.4f}  {'≥' if h1_A else '<'} {H1_SPECIFICITY_MIN}")
print(f"    Arm B (L6H6): Δspecificity = {spec_B:>+.4f}  {'≥' if h1_B else '<'} {H1_SPECIFICITY_MIN}")
print(f"  H1: {'PASS' if h1_pass else 'FAIL'}")

# H2: L4H6/L5H5 Δspecificity ≥ H2_SPECIFICITY_MIN; AND L6 > L4/L5
spec_D    = specificity["D"]["delta_specificity"]
spec_E    = specificity["E"]["delta_specificity"]
h2_D      = spec_D >= H2_SPECIFICITY_MIN
h2_E      = spec_E >= H2_SPECIFICITY_MIN
dm_A      = stats["A_tgt"]["mean_delta_margin"]
dm_B      = stats["B_tgt"]["mean_delta_margin"]
dm_D      = stats["D_tgt"]["mean_delta_margin"]
dm_E      = stats["E_tgt"]["mean_delta_margin"]
h2_l6_gt  = (dm_A > dm_D) and (dm_B > dm_E)
h2_pass   = h2_D and h2_E and h2_l6_gt

print(f"\n  H2 — L4H6/L5H5 Δspecificity ≥ {H2_SPECIFICITY_MIN} AND L6 > L4/L5:")
print(f"    Arm D (L4H6): Δspecificity = {spec_D:>+.4f}  {'≥' if h2_D else '<'} {H2_SPECIFICITY_MIN}")
print(f"    Arm E (L5H5): Δspecificity = {spec_E:>+.4f}  {'≥' if h2_E else '<'} {H2_SPECIFICITY_MIN}")
print(f"    L6>L4: A ({dm_A:>+.2f}) > D ({dm_D:>+.2f}): {dm_A > dm_D}")
print(f"    L6>L5: B ({dm_B:>+.2f}) > E ({dm_E:>+.2f}): {dm_B > dm_E}")
print(f"  H2: {'PASS' if h2_pass else 'FAIL'}")

# H3: Arm F target flip fraction ≥ H3_FLIP_MIN
flip_F = stats["F_tgt"]["flip_frac"]
h3_pass = flip_F >= H3_FLIP_MIN
print(f"\n  H3 — Arm F (all four targets) flip fraction ≥ {H3_FLIP_MIN}:")
print(f"    Arm F_tgt flip_frac = {flip_F:.4f}  {'≥' if h3_pass else '<'} {H3_FLIP_MIN}")
print(f"  H3: {'PASS' if h3_pass else 'FAIL'}")

# H4: Arm F OOD-safe (< H4_OOD_MAX_COUNT)
n_ood_F = stats["F_tgt"]["n_ood"]
h4_pass = n_ood_F < H4_OOD_MAX_COUNT
print(f"\n  H4 — Arm F OOD-safe (< {H4_OOD_MAX_COUNT} examples with P_sum < {OOD_THRESH}):")
print(f"    Arm F_tgt n_ood = {n_ood_F}  {'<' if h4_pass else '≥'} {H4_OOD_MAX_COUNT}")
print(f"  H4: {'PASS' if h4_pass else 'FAIL'}")

# K1: no arm produces Δspecificity ≥ K1_SPEC_MIN
all_specs = [specificity[p]["delta_specificity"] for _, _, p in COMPARISONS]
k1_fires = all(s < K1_SPEC_MIN for s in all_specs)
print(f"\n  K1 — No arm produces Δspecificity ≥ {K1_SPEC_MIN}:")
for _, _, p in COMPARISONS:
    s = specificity[p]["delta_specificity"]
    print(f"    Arm {p}: Δspecificity = {s:>+.4f}")
print(f"  K1: {'FIRES' if k1_fires else 'CLEAR'}")
if k1_fires:
    print(f"  *** K1 FIRES: within-example neutralization does not isolate causal heads.")
    print(f"  *** The heads may not be independently causally upstream. Relay to G-track.")

# K2: Arm F flip < K2_FLIP_MAX AND H4 passes (OOD-safe)
k2_fires = (flip_F < K2_FLIP_MAX) and h4_pass
print(f"\n  K2 — Arm F flip < {K2_FLIP_MAX} AND OOD-safe:")
print(f"    flip_F = {flip_F:.4f}  ({'<' if flip_F < K2_FLIP_MAX else '≥'} {K2_FLIP_MAX})")
print(f"    OOD-safe (H4): {'yes' if h4_pass else 'no'}")
print(f"  K2: {'FIRES' if k2_fires else 'CLEAR'}")
if k2_fires:
    print(f"  *** K2 FIRES: joint ablation ineffective despite OOD-safe status.")
    print(f"  *** Cross-attention heads are not the locus of the failure.")
    print(f"  *** Major revision to mechanism account required. Relay to Troy.")

# K3: > K3_OOD_MAX_COUNT OOD in any arm
k3_fires_arms = [arm for arm in ARMS if stats[arm]["n_ood"] > K3_OOD_MAX_COUNT]
k3_fires = len(k3_fires_arms) > 0
print(f"\n  K3 — > {K3_OOD_MAX_COUNT} OOD in any arm:")
if k3_fires:
    print(f"  *** K3 FIRES for: {k3_fires_arms}")
    print(f"  *** Results for these arms are unreliable for those examples.")
else:
    print(f"    All arms within OOD limit.")
print(f"  K3: {'FIRES' if k3_fires else 'CLEAR'}")

print(f"\n  ─── VERDICT SUMMARY ───")
print(f"  H1 (L6H2/L6H6 Δspecificity ≥ 2.0):             {'PASS' if h1_pass else 'FAIL'}")
print(f"  H2 (L4H6/L5H5 ≥ 1.0 AND L6 > L4/L5):           {'PASS' if h2_pass else 'FAIL'}")
print(f"  H3 (Arm F flip fraction ≥ 0.25):                 {'PASS' if h3_pass else 'FAIL'}")
print(f"  H4 (Arm F OOD-safe, < 5 examples):              {'PASS' if h4_pass else 'FAIL'}")
print(f"  K1 (no specificity ≥ 0.50 — non-specific):      {'FIRES' if k1_fires else 'CLEAR'}")
print(f"  K2 (Arm F flip < 0.10 AND OOD-safe):            {'FIRES' if k2_fires else 'CLEAR'}")
print(f"  K3 (> 10 OOD in any arm):                       {'FIRES' if k3_fires else 'CLEAR'}")

# ── Phase 9: Save results ──────────────────────────────────────────────────────────────────────────────
print("\n" + "="*70)
print("  PHASE 9 — SAVE RESULTS")
print("="*70)

RESULTS_PATH = "workbench/results/060_results.json"
os.makedirs(os.path.dirname(RESULTS_PATH), exist_ok=True)

results = {
    "script_id": SCRIPT_ID,
    "seed": SEED,
    "n_valid": n_valid,
    "n_skip": n_skip,
    "fwd_verify": {
        "mean_err": float(np.mean(fwd_verify_errors)) if fwd_verify_errors else None,
        "max_err":  float(np.max(fwd_verify_errors)) if fwd_verify_errors else None,
    },
    "m1_baseline": {
        "n": len(baseline_records),
        "mean_margin":  float(np.mean(bl_margins)),
        "mean_p_walk":  float(np.mean(bl_p_walk)),
        "mean_p_jump":  float(np.mean(bl_p_jump)),
        "flip_frac":    float(np.mean(bl_flips)),
    },
    "m2_arm_stats":    {arm: stats[arm] for arm in ARMS},
    "m3_specificity":  specificity,
    "m4_ood": {
        arm: {"n_ood": stats[arm]["n_ood"], "ood_frac": stats[arm]["ood_frac"]}
        for arm in ARMS
    },
    "verdicts": {
        "H1": {"pass": h1_pass, "spec_A": spec_A, "spec_B": spec_B},
        "H2": {"pass": h2_pass, "spec_D": spec_D, "spec_E": spec_E,
               "l6_gt_l4": h2_l6_gt},
        "H3": {"pass": h3_pass, "flip_F": flip_F},
        "H4": {"pass": h4_pass, "n_ood_F": n_ood_F},
        "K1": {"fires": k1_fires, "all_specs": {p: specificity[p]["delta_specificity"]
                                                  for _, _, p in COMPARISONS}},
        "K2": {"fires": k2_fires, "flip_F": flip_F, "ood_safe": h4_pass},
        "K3": {"fires": k3_fires, "firing_arms": k3_fires_arms},
    },
    "per_arm_per_example": {
        arm: [
            {k: v for k, v in r.items() if k != "skip" and not isinstance(v, bool)}
            for r in arm_results[arm] if not r.get("skip")
        ]
        for arm in ARMS
    },
}

with open(RESULTS_PATH, "w") as f:
    json.dump(results, f, indent=2)
print(f"  Results saved: {RESULTS_PATH}")
save_to_drive(RESULTS_PATH)

# ── Final summary ─────────────────────────────────────────────────────────────────────────────────
print("\n" + "="*70)
print(f"  {SCRIPT_ID} — COMPLETE")
print("="*70)
print(f"\n  n_valid = {n_valid}  |  n_skip = {n_skip}")
print(f"\n  Baseline: mean margin = {np.mean(bl_margins):>+.4f}  "
      f"P(I_WALK)={np.mean(bl_p_walk):.4f}  flip_frac={np.mean(bl_flips):.3f}")
print(f"\n  Specificity table:")
print(f"  {'Pair':<5}  {'Δspec':<12}  {'Δm_tgt':<12}  {'Δm_ctl':<12}  flip_tgt  flip_ctl")
for _, _, p in COMPARISONS:
    sp = specificity[p]
    print(f"  {p:<5}  {sp['delta_specificity']:>+10.4f}  "
          f"{sp['delta_margin_tgt']:>+10.4f}  {sp['delta_margin_ctl']:>+10.4f}  "
          f"{sp['flip_tgt']:.4f}    {sp['flip_ctl']:.4f}")
print(f"\n  Verdicts: H1={'PASS' if h1_pass else 'FAIL'}  "
      f"H2={'PASS' if h2_pass else 'FAIL'}  "
      f"H3={'PASS' if h3_pass else 'FAIL'}  "
      f"H4={'PASS' if h4_pass else 'FAIL'}")
print(f"  Kill:     K1={'FIRES' if k1_fires else 'CLEAR'}  "
      f"K2={'FIRES' if k2_fires else 'CLEAR'}  "
      f"K3={'FIRES' if k3_fires else 'CLEAR'}")
print()
print(f"  Results: {RESULTS_PATH}")
print()
print("─" * 70)
print("  After running: compare outputs to the corresponding sealed methods report in findings/; private coordination files are not included in this public export.")
print("─" * 70)
