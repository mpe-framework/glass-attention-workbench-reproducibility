#!/usr/bin/env python3
"""
S-051_LEVER_ARM_DISSECTION.py — Three Interventions on the I_WALK Mechanism
Applied Categorical Physics Workbench
Troy Teno | May 2026 | Open Access

Pre-registered hypotheses:
  workbench/proposals/S-051_LEVER_ARM_DISSECTION_PROPOSAL.md
DO NOT modify hypotheses after seeing results.

─────────────────────────────────────────────────────────────────────
SETUP (run once in Colab or terminal):
  pip install torch transformers datasets numpy scipy sentencepiece -q

BACKGROUND:
  S-050b established: I_WALK is selected at fail action slots via two factors:
    1. Largest divergence-point embedding norm (520 vs 494/424/407)
    2. Only positive cosine alignment with h_div (+0.024 vs negative for others)
  The embedding norm amplifies the 0.033 cosine spread into a 192-unit logit gap → P(I_WALK) = 0.91.

THREE INTERVENTIONS (one generation pass captures all):

  A — NORMALIZATION: Remove the lever arm. Recompute action-slot probabilities
      using unit-norm action embeddings. Does P(I_WALK) collapse?
      Pre-registered: P(I_WALK) drops below 0.50 (from 0.91) — norm is load-bearing.

  B — LAYER ATTRIBUTION: At each decoder layer, compute cos(h^l, embed(action div)).
      Find where h_div acquires its I_WALK directional bias.
      Pre-registered: bias emerges at mid-decoder (layers 3-5), not at input.

  C — HEAD ATTRIBUTION: Per-head cross-attention contribution to residual stream.
      Find which head(s) set the I_WALK cosine direction.
      Pre-registered: 1-2 heads dominate; they attend to "around" encoder position;
      they flip to I_JUMP direction in success cases.
─────────────────────────────────────────────────────────────────────
"""

import numpy as np
import json
import os
import random
from collections import defaultdict

SCRIPT_ID = "S-051_LEVER_ARM_DISSECTION_V0.1.0"
SEED = 42
np.random.seed(SEED)
random.seed(SEED)

CHECKPOINT_DIR = "043_t5_scan_checkpoint"
MAX_INPUT_LEN  = 50
MAX_TARGET_LEN = 100
N_FAIL         = 30
N_SUCCESS      = 25

# ── Google Drive ───────────────────────────────────────────────────────────────
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
    print(f"  Drive mounted.")
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
print(f"  Three interventions: norm, layer, head attribution")
print(f"{'='*70}")

# ── Imports ───────────────────────────────────────────────────────────────────
print("\nImporting libraries...")
try:
    import torch
    import torch.nn.functional as F
    from transformers import T5ForConditionalGeneration, T5Tokenizer
    from scipy import stats as scipy_stats
except ImportError as e:
    raise SystemExit(f"\nMissing library: {e}\n")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"  Device: {device}")
torch.manual_seed(SEED)

# ── Phase 1: Load checkpoint ─────────────────────────────────────────────────────
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

# ── Phase 2: Action token divergence IDs and embeddings ───────────────────────
print("\n" + "="*70)
print("  PHASE 2 — ACTION TOKEN EMBEDDINGS")
print("="*70)

core_actions = ["I_JUMP", "I_WALK", "I_RUN", "I_LOOK"]
ALL_NAMES    = core_actions + ["I_TURN_LEFT", "I_TURN_RIGHT"]

raw_ids = {name: tokenizer.encode(name, add_special_tokens=False)
           for name in ALL_NAMES}
print(f"\n  Full tokenizations:")
for name, ids in raw_ids.items():
    print(f"  {name:<15} → {ids}")

DIV_IDS = {name: raw_ids[name][2] for name in core_actions}
TURN_LAST_IDS = {name: raw_ids[name][-1]
                 for name in ["I_TURN_LEFT", "I_TURN_RIGHT"]}
TURN_END_IDS  = set(TURN_LAST_IDS.values())
DIV_TOKEN_SET = set(DIV_IDS.values())

print(f"\n  Divergence-point IDs: {DIV_IDS}")
print(f"  TURN_END_IDS:         {TURN_END_IDS}")

embed_matrix = model.shared.weight.detach()   # [vocab_size, d_model]
div_embeds   = {name: embed_matrix[div_id] for name, div_id in DIV_IDS.items()}

# Unit-norm versions for Option A
div_embeds_normed = {name: e / e.norm() for name, e in div_embeds.items()}

print(f"\n  Embedding norms (confirming S-050b):")
for name in core_actions:
    print(f"  {name:<15} ‖embed(div)‖ = {div_embeds[name].norm():.2f}")

# Decoder architecture parameters
n_dec_layers = model.config.num_decoder_layers   # 6 for T5-small
n_heads      = model.config.num_heads            # 8
d_kv         = model.config.d_kv                 # 64
d_model      = model.config.d_model              # 512
print(f"\n  T5 decoder: {n_dec_layers} layers, {n_heads} heads, d_kv={d_kv}, d_model={d_model}")

# ── Phase 3: Load SCAN + classify ─────────────────────────────────────────────
print("\n" + "="*70)
print("  PHASE 3 — LOAD SCAN + CLASSIFY EXAMPLES")
print("="*70)

train_raw = test_raw = None
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

print(f"\n  Running inference to find fail/success groups...")
BATCH = 32
sub_walk_examples = []
success_examples  = []

with torch.no_grad():
    for i in range(0, len(test_jc), BATCH):
        batch = test_jc[i : i + BATCH]
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
success_examples  = success_examples[:N_SUCCESS]
print(f"\n  substituted_walk examples: {len(sub_walk_examples)}")
print(f"  has_around + success:      {len(success_examples)}")

# ── Phase 4: Divergence-point detection ────────────────────────────────────────
id_to_action = {v: k for k, v in DIV_IDS.items()}

def find_div_points(gen_token_ids):
    turn_end_to_dir = {TURN_LAST_IDS["I_TURN_LEFT"]:  "left",
                       TURN_LAST_IDS["I_TURN_RIGHT"]: "right"}
    points = []
    for i in range(3, len(gen_token_ids)):
        if (gen_token_ids[i]   in DIV_TOKEN_SET and
            gen_token_ids[i-1] == 834 and
            gen_token_ids[i-2] == 27 and
            gen_token_ids[i-3] in TURN_END_IDS):
            action_name = id_to_action.get(gen_token_ids[i], "UNKNOWN")
            turn_dir    = turn_end_to_dir.get(gen_token_ids[i-3], "unknown")
            points.append((i, action_name, turn_dir))
    return points

# ── Phase 5: Main measurement pass ────────────────────────────────────────────
print("\n" + "="*70)
print("  PHASE 5 — SINGLE GENERATION PASS (A + B + C in one)")
print("="*70)
print(f"\n  {len(sub_walk_examples)} fail + {len(success_examples)} success examples")
print(f"  (Slow — full attention + hidden states per example)\n")

records_fail    = []
records_success = []

def measure_example(ex, group_label):
    cmd = ex["commands"]
    inp = tokenizer(cmd, truncation=True, max_length=MAX_INPUT_LEN,
                    return_tensors="pt").to(device)
    T_enc = inp["input_ids"].shape[1]

    enc_tokens = tokenizer.convert_ids_to_tokens(inp["input_ids"][0])
    jump_pos   = next((i for i, t in enumerate(enc_tokens) if "jump"  in t.lower()), None)
    around_pos = next((i for i, t in enumerate(enc_tokens) if "around" in t.lower()), None)

    with torch.no_grad():
        out = model.generate(
            input_ids=inp["input_ids"],
            attention_mask=inp["attention_mask"],
            max_length=MAX_TARGET_LEN,
            output_attentions=True,
            output_hidden_states=True,
            return_dict_in_generate=True,
        )

    gen_ids   = out.sequences[0].tolist()
    n_steps   = len(out.cross_attentions)
    seq_no_bos = gen_ids[1:]

    # Encoder final hidden states (used by all decoder cross-attn layers)
    # out.encoder_hidden_states is a tuple of n_enc_layers+1 tensors [1, T_enc, d_model]
    enc_hidden_final = out.encoder_hidden_states[-1]   # [1, T_enc, d_model]

    div_points = find_div_points(seq_no_bos)
    slot_records = []

    for (div_pos, action_name, turn_dir) in div_points:
        step = div_pos
        if step >= n_steps:
            continue

        # ── Decoder hidden states at this step (all layers) ────────────────────
        # out.decoder_hidden_states[step] = tuple of n_layers+1 tensors [1, 1, d_model]
        # Index 0 = embedding input; indices 1..6 = after each transformer block
        step_hidden = out.decoder_hidden_states[step]
        h_final     = step_hidden[-1][0, 0, :]   # [d_model] — last decoder layer

        # ── OPTION A: Normalized action logits ─────────────────────────────────
        # Original logits (from embed_matrix @ h_final)
        all_logits = (embed_matrix @ h_final).float()
        probs_orig = torch.softmax(all_logits, dim=0)
        orig_probs_action = {name: float(probs_orig[div_id])
                             for name, div_id in DIV_IDS.items()}

        # Normalized logits: h · (e/‖e‖) for each action div token
        norm_logits = {name: float((h_final @ div_embeds_normed[name]).item())
                       for name in core_actions}
        # Softmax over just the 4 normalized action logits
        nl_tensor = torch.tensor([norm_logits[n] for n in core_actions])
        norm_probs_list = torch.softmax(nl_tensor, dim=0).tolist()
        norm_probs = {name: norm_probs_list[i]
                      for i, name in enumerate(core_actions)}

        # ── OPTION B: Layer-wise cosine tracking ──────────────────────────────
        layer_cosines = {}   # {layer_idx: {action_name: cos_val}}
        for layer_idx, h_l_tensor in enumerate(step_hidden):
            h_l = h_l_tensor[0, 0, :]   # [d_model]
            layer_cosines[layer_idx] = {
                name: float(F.cosine_similarity(
                    h_l.unsqueeze(0), div_embeds[name].unsqueeze(0)).item())
                for name in core_actions
            }

        # ── OPTION C: Per-head cross-attention attribution ───────────────────────
        # For each decoder layer, compute per-head contribution to residual stream
        # contribution[l][h] = W_O[:,h*d_kv:(h+1)*d_kv] @ (attn_w[h] @ V_h)
        # where V_h = enc_hidden_final @ v_proj.weight.T split by head

        head_cosines = {}   # {(layer_idx, head_idx): {action_name: cos_val}}
        head_attn_to_around = {}   # {(layer_idx, head_idx): float}
        head_attn_to_jump   = {}

        step_cross_attns = out.cross_attentions[step]
        # step_cross_attns: tuple of n_dec_layers tensors [1, n_heads, 1, T_enc]

        for l_idx in range(n_dec_layers):
            enc_attn = model.decoder.block[l_idx].layer[1].EncDecAttention

            # V projection: enc_hidden_final → [1, T_enc, n_heads * d_kv]
            with torch.no_grad():
                V_all = enc_hidden_final @ enc_attn.v.weight.T  # [1, T_enc, n_heads*d_kv]

            attn_w = step_cross_attns[l_idx]  # [1, n_heads, 1, T_enc]

            for h_idx in range(n_heads):
                head_attn_weights = attn_w[0, h_idx, 0, :]   # [T_enc]
                head_V = V_all[0, :, h_idx*d_kv:(h_idx+1)*d_kv]  # [T_enc, d_kv]

                # Weighted sum of values: [d_kv]
                head_val = (head_attn_weights.unsqueeze(0) @ head_V).squeeze(0)

                # Output projection slice for this head: [d_model, d_kv]
                head_o_slice = enc_attn.o.weight[:, h_idx*d_kv:(h_idx+1)*d_kv]

                # Contribution to residual stream: [d_model]
                head_contrib = head_o_slice @ head_val

                key = (l_idx, h_idx)
                head_cosines[key] = {
                    name: float(F.cosine_similarity(
                        head_contrib.unsqueeze(0),
                        div_embeds[name].unsqueeze(0)).item())
                    for name in core_actions
                }

                # Also record attention to key encoder positions
                head_attn_to_around[key] = (float(head_attn_weights[around_pos].item())
                                            if around_pos is not None else float("nan"))
                head_attn_to_jump[key]   = (float(head_attn_weights[jump_pos].item())
                                            if jump_pos   is not None else float("nan"))

        # ── Cross-attention summary (K1 probe) ────────────────────────────────
        all_cross_heads = np.concatenate(
            [step_cross_attns[l][0, :, 0, :].cpu().numpy()
             for l in range(n_dec_layers)], axis=0  # [n_dec_layers*n_heads, T_enc]
        )
        mean_attn = all_cross_heads.mean(0)
        attn_jump  = float(mean_attn[jump_pos])   if jump_pos   is not None else float("nan")
        attn_around= float(mean_attn[around_pos]) if around_pos is not None else float("nan")

        slot_records.append({
            "div_pos":      div_pos,
            "action_name":  action_name,
            "turn_dir":     turn_dir,
            "h_norm":       float(h_final.norm().item()),
            "attn_jump":    attn_jump,
            "attn_around":  attn_around,
            # Option A
            "orig_probs":   orig_probs_action,
            "norm_probs":   norm_probs,
            "norm_logits":  norm_logits,
            # Option B
            "layer_cosines": layer_cosines,
            # Option C (serialized to nested list to avoid JSON key issues)
            "head_cosines_walk": {
                f"{l}_{h}": head_cosines[(l, h)]["I_WALK"]
                for l in range(n_dec_layers) for h in range(n_heads)
            },
            "head_cosines_jump": {
                f"{l}_{h}": head_cosines[(l, h)]["I_JUMP"]
                for l in range(n_dec_layers) for h in range(n_heads)
            },
            "head_attn_around": {
                f"{l}_{h}": head_attn_to_around[(l, h)]
                for l in range(n_dec_layers) for h in range(n_heads)
            },
            "head_attn_jump": {
                f"{l}_{h}": head_attn_to_jump[(l, h)]
                for l in range(n_dec_layers) for h in range(n_heads)
            },
        })

    return {"cmd": cmd, "group": group_label, "slots": slot_records}

for j, ex in enumerate(sub_walk_examples):
    if j % 5 == 0:
        print(f"  [fail] {j+1}/{len(sub_walk_examples)} ...")
    records_fail.append(measure_example(ex, "fail"))

for j, ex in enumerate(success_examples):
    if j % 5 == 0:
        print(f"  [success] {j+1}/{len(success_examples)} ...")
    records_success.append(measure_example(ex, "success"))

# ── Phase 6: Aggregate ──────────────────────────────────────────────────────
print("\n" + "="*70)
print("  PHASE 6 — AGGREGATE: OPTION A (NORMALIZATION)")
print("="*70)

def all_slots(records):
    return [slot for rec in records for slot in rec["slots"]]

fail_slots    = all_slots(records_fail)
success_slots = all_slots(records_success)
print(f"\n  Total divergence points: fail={len(fail_slots)}, success={len(success_slots)}")

# ── Option A report ────────────────────────────────────────────────────────────
print(f"\n  ── Original vs Normalized probabilities at action slots ─────────────")
print(f"  {'Action':<12}  {'Orig(fail)':>11}  {'Norm(fail)':>11}  {'Orig(succ)':>11}  {'Norm(succ)':>11}")

ha1_pass = ha2_pass = ha3_pass = False
orig_means_fail = {}
norm_means_fail = {}
orig_means_succ = {}
norm_means_succ = {}

for name in core_actions:
    of = np.mean([s["orig_probs"][name] for s in fail_slots])
    nf = np.mean([s["norm_probs"][name] for s in fail_slots])
    os_ = np.mean([s["orig_probs"][name] for s in success_slots])
    ns = np.mean([s["norm_probs"][name] for s in success_slots])
    orig_means_fail[name] = float(of)
    norm_means_fail[name] = float(nf)
    orig_means_succ[name] = float(os_)
    norm_means_succ[name] = float(ns)
    print(f"  {name:<12}  {of:>11.5f}  {nf:>11.5f}  {os_:>11.5f}  {ns:>11.5f}")

norm_rank_fail = sorted(core_actions, key=lambda n: -norm_means_fail[n])
orig_rank_fail = sorted(core_actions, key=lambda n: -orig_means_fail[n])
print(f"\n  Original rank (fail): {' > '.join(orig_rank_fail)}")
print(f"  Normalized rank (fail): {' > '.join(norm_rank_fail)}")

ha1_pass = norm_means_fail["I_WALK"] < 0.50
ha2_pass = (norm_means_fail["I_WALK"] / max(norm_means_fail["I_RUN"], 1e-9)) < 2.0
ha3_pass = norm_rank_fail[0] == "I_WALK"

print(f"\n  H_A1 (norm P(I_WALK) < 0.50):                {'PASS' if ha1_pass else 'FAIL'}")
print(f"       norm P(I_WALK) = {norm_means_fail['I_WALK']:.5f}")
print(f"  H_A2 (P(I_WALK)/P(I_RUN) < 2.0 after norm): {'PASS' if ha2_pass else 'FAIL'}")
print(f"       ratio = {norm_means_fail['I_WALK'] / max(norm_means_fail['I_RUN'], 1e-9):.3f}")
print(f"  H_A3 (I_WALK still highest after norm):      {'PASS' if ha3_pass else 'FAIL'}")
print(f"  K_A fires (P(I_WALK)>0.70 after norm):       {'YES — cosine alone drives selection' if norm_means_fail['I_WALK'] > 0.70 else 'NO'}")

# ── Option B report ────────────────────────────────────────────────────────────
print("\n" + "="*70)
print("  OPTION B — LAYER-WISE COSINE ATTRIBUTION")
print("="*70)

n_layers_total = len(records_fail[0]["slots"][0]["layer_cosines"]) if records_fail[0]["slots"] else 7

print(f"\n  cos(h^layer, embed(div_token)) — mean over slots")
print(f"  {'Layer':<8}", end="")
for name in core_actions:
    print(f"  {name:>12} F / S", end="")
print()

layer_walk_cos_fail    = []
layer_walk_cos_success = []
layer_jump_cos_fail    = []
layer_jump_cos_success = []

walk_goes_positive_layer = None

for l_idx in range(n_layers_total):
    print(f"  L{l_idx:<6}", end="")
    for name in core_actions:
        f_vals = [s["layer_cosines"][l_idx][name] for s in fail_slots
                  if l_idx in s["layer_cosines"]]
        s_vals = [s["layer_cosines"][l_idx][name] for s in success_slots
                  if l_idx in s["layer_cosines"]]
        fm = float(np.mean(f_vals)) if f_vals else float("nan")
        sm = float(np.mean(s_vals)) if s_vals else float("nan")
        print(f"  {fm:>+8.4f}/{sm:>+8.4f}", end="")
        if name == "I_WALK":
            layer_walk_cos_fail.append(fm)
            layer_walk_cos_success.append(sm)
        if name == "I_JUMP":
            layer_jump_cos_fail.append(fm)
            layer_jump_cos_success.append(sm)
    print()

# Find where I_WALK cosine first goes positive in fail
for l_idx, cos_val in enumerate(layer_walk_cos_fail):
    if not np.isnan(cos_val) and cos_val > 0 and walk_goes_positive_layer is None:
        walk_goes_positive_layer = l_idx

# Find where I_JUMP cosine first goes positive in success
jump_goes_positive_layer = None
for l_idx, cos_val in enumerate(layer_jump_cos_success):
    if not np.isnan(cos_val) and cos_val > 0 and jump_goes_positive_layer is None:
        jump_goes_positive_layer = l_idx

hb1_pass = (layer_walk_cos_fail[0] if layer_walk_cos_fail else float("nan"))
hb1_pass = (abs(hb1_pass) < 0.02) if not np.isnan(hb1_pass) else False

print(f"\n  I_WALK cosine by layer (fail):    {[f'{x:.4f}' for x in layer_walk_cos_fail]}")
print(f"  I_WALK cosine by layer (success): {[f'{x:.4f}' for x in layer_walk_cos_success]}")
print(f"  I_JUMP cosine by layer (fail):    {[f'{x:.4f}' for x in layer_jump_cos_fail]}")
print(f"  I_JUMP cosine by layer (success): {[f'{x:.4f}' for x in layer_jump_cos_success]}")

print(f"\n  I_WALK cosine first positive (fail) at layer: {walk_goes_positive_layer}")
print(f"  I_JUMP cosine first positive (success) at layer: {jump_goes_positive_layer}")

hb2_pass = (walk_goes_positive_layer is not None and walk_goes_positive_layer >= 2)
hb3_pass = (walk_goes_positive_layer is not None and jump_goes_positive_layer is not None
            and walk_goes_positive_layer == jump_goes_positive_layer)

print(f"\n  H_B1 (layer 0 cosine near zero):              {'PASS' if hb1_pass else 'FAIL'}")
print(f"       Layer-0 cos(h^0, I_WALK div) = {layer_walk_cos_fail[0] if layer_walk_cos_fail else 'N/A':.4f}")
print(f"  H_B2 (bias emerges at mid-decoder, ≥layer 2): {'PASS' if hb2_pass else 'FAIL'}")
print(f"  H_B3 (fail/success fork at same layer):       {'PASS' if hb3_pass else 'FAIL'}")
print(f"  K_B fires (bias at layer 0):                  {'YES — pretraining origin' if not hb1_pass else 'NO'}")

# ── Option C report ────────────────────────────────────────────────────────────
print("\n" + "="*70)
print("  OPTION C — PER-HEAD CROSS-ATTENTION ATTRIBUTION")
print("="*70)

print(f"\n  Per-head cosine contribution toward embed(I_WALK div) — fail group")
print(f"  (Positive = head pulls h_div toward I_WALK; negative = away)")
print(f"\n  {'Head':<10}  {'Walk(F)':>9}  {'Jump(F)':>9}  {'Walk(S)':>9}  {'Jump(S)':>9}  {'Attn(arnd)':>11}  {'Attn(jump)':>10}")

head_walk_fail    = {}
head_jump_fail    = {}
head_walk_success = {}
head_jump_success = {}
head_attn_around_fail = {}

for l in range(n_dec_layers):
    for h in range(n_heads):
        key = f"{l}_{h}"
        wf = np.mean([s["head_cosines_walk"].get(key, float("nan")) for s in fail_slots])
        jf = np.mean([s["head_cosines_jump"].get(key, float("nan")) for s in fail_slots])
        ws = np.mean([s["head_cosines_walk"].get(key, float("nan")) for s in success_slots])
        js = np.mean([s["head_cosines_jump"].get(key, float("nan")) for s in success_slots])
        aa = np.mean([s["head_attn_around"].get(key, float("nan")) for s in fail_slots])
        aj = np.mean([s["head_attn_jump"].get(key, float("nan")) for s in fail_slots])
        head_walk_fail[key]    = float(np.nanmean([wf]))
        head_jump_fail[key]    = float(np.nanmean([jf]))
        head_walk_success[key] = float(np.nanmean([ws]))
        head_jump_success[key] = float(np.nanmean([js]))
        head_attn_around_fail[key] = float(np.nanmean([aa]))
        print(f"  L{l}H{h:<6}  {wf:>+9.4f}  {jf:>+9.4f}  {ws:>+9.4f}  {js:>+9.4f}  {aa:>11.4f}  {aj:>10.4f}")

# Top heads by I_WALK cosine in fail cases
sorted_heads_walk = sorted(head_walk_fail.keys(),
                           key=lambda k: -head_walk_fail[k])
print(f"\n  Top-5 heads by I_WALK cosine (fail):")
for k in sorted_heads_walk[:5]:
    l, h = int(k.split("_")[0]), int(k.split("_")[1])
    print(f"    L{l}H{h}:  I_WALK cos={head_walk_fail[k]:+.4f}  "
          f"I_JUMP cos={head_jump_fail[k]:+.4f}  "
          f"(success: I_WALK={head_walk_success[k]:+.4f}, I_JUMP={head_jump_success[k]:+.4f})  "
          f"attn_around={head_attn_around_fail[k]:.4f}")

# Gini coefficient of I_WALK cosines across heads (H_C1)
def gini(vals):
    a = np.abs(vals)
    a = np.sort(a)
    n = len(a)
    if n == 0 or a.sum() == 0:
        return float("nan")
    idx = np.arange(1, n+1)
    return float((2 * (idx * a).sum() / (n * a.sum())) - (n+1)/n)

walk_cos_values = np.array(list(head_walk_fail.values()))
gini_coeff = gini(walk_cos_values)
hc1_pass = gini_coeff > 0.4

# H_C2: does top head attend to "around"?
top_head_key = sorted_heads_walk[0]
top_attn_around = head_attn_around_fail.get(top_head_key, float("nan"))
hc2_pass = (top_attn_around > 0.20)  # dominant head attends to around > 20%

# H_C3: does top head flip toward I_JUMP in success?
top_head_walk_success = head_walk_success.get(top_head_key, float("nan"))
top_head_jump_success = head_jump_success.get(top_head_key, float("nan"))
top_head_walk_fail    = head_walk_fail.get(top_head_key, float("nan"))
hc3_pass = (top_head_jump_success > top_head_walk_success and
            top_head_jump_success > top_head_walk_fail)

print(f"\n  Gini coefficient of I_WALK cosines across {n_dec_layers*n_heads} heads: {gini_coeff:.3f}")
print(f"\n  H_C1 (Gini > 0.4 — attribution non-uniform):       {'PASS' if hc1_pass else 'FAIL'}")
print(f"  H_C2 (top head attends to 'around' > 20%):          {'PASS' if hc2_pass else 'FAIL'}")
print(f"       top head={top_head_key}, attn_around={top_attn_around:.4f}")
print(f"  H_C3 (top head flips to I_JUMP direction in succ):  {'PASS' if hc3_pass else 'FAIL'}")
print(f"       top head I_WALK: fail={top_head_walk_fail:+.4f}, success={top_head_walk_success:+.4f}")
print(f"       top head I_JUMP: fail={head_jump_fail.get(top_head_key, 0):+.4f}, success={top_head_jump_success:+.4f}")
print(f"  K_C fires (Gini < 0.2 — uniform):                   {'YES — no specialist head' if gini_coeff < 0.2 else 'NO'}")

# ── Phase 7: Summary ───────────────────────────────────────────────────────────────
print("\n" + "="*70)
print("  PHASE 7 — SUMMARY AND VERDICT")
print("="*70)

print(f"\n  ── HYPOTHESIS VERDICTS ────────────────────────────────────────────")
print(f"  H_A1: P(I_WALK) < 0.50 after normalization — {'PASS' if ha1_pass else 'FAIL'} ({norm_means_fail['I_WALK']:.4f})")
print(f"  H_A2: P(I_WALK)/P(I_RUN) < 2.0 after norm  — {'PASS' if ha2_pass else 'FAIL'}")
print(f"  H_A3: I_WALK still highest after norm        — {'PASS' if ha3_pass else 'FAIL'}")
print(f"  H_B1: Layer-0 cosine near zero               — {'PASS' if hb1_pass else 'FAIL'}")
print(f"  H_B2: I_WALK bias emerges at ≥layer 2        — {'PASS' if hb2_pass else 'FAIL'} (layer {walk_goes_positive_layer})")
print(f"  H_B3: Fail/success fork at same layer        — {'PASS' if hb3_pass else 'FAIL'}")
print(f"  H_C1: Attribution non-uniform (Gini>0.4)     — {'PASS' if hc1_pass else 'FAIL'} ({gini_coeff:.3f})")
print(f"  H_C2: Top head attends to 'around' >20%     — {'PASS' if hc2_pass else 'FAIL'}")
print(f"  H_C3: Top head flips to I_JUMP in success    — {'PASS' if hc3_pass else 'FAIL'}")

print(f"\n  ── LEVER ARM VERDICT ───────────────────────────────────────────────────")
print(f"  Original P(I_WALK): {orig_means_fail['I_WALK']:.5f}")
print(f"  Normalized P(I_WALK): {norm_means_fail['I_WALK']:.5f}")
p_drop = orig_means_fail['I_WALK'] - norm_means_fail['I_WALK']
print(f"  Drop after removing norm lever: {p_drop:.5f} "
      f"({'norm is load-bearing' if ha1_pass else 'cosine alone drives selection'})")
print(f"  I_WALK bias emergence layer (fail): {walk_goes_positive_layer}")
print(f"  I_JUMP bias emergence layer (success): {jump_goes_positive_layer}")
print(f"  Attribution: top head = {top_head_key}, Gini = {gini_coeff:.3f}")

# ── Save ────────────────────────────────────────────────────────────────────────
output = {
    "script_id": SCRIPT_ID,
    "n_fail_slots":    int(len(fail_slots)),
    "n_success_slots": int(len(success_slots)),
    "option_a": {
        "orig_probs_fail": orig_means_fail,
        "norm_probs_fail": norm_means_fail,
        "orig_probs_succ": orig_means_succ,
        "norm_probs_succ": norm_means_succ,
        "norm_rank_fail":  norm_rank_fail,
        "ha1_pass": bool(ha1_pass),
        "ha2_pass": bool(ha2_pass),
        "ha3_pass": bool(ha3_pass),
    },
    "option_b": {
        "walk_cos_by_layer_fail":    [float(x) for x in layer_walk_cos_fail],
        "walk_cos_by_layer_success": [float(x) for x in layer_walk_cos_success],
        "jump_cos_by_layer_fail":    [float(x) for x in layer_jump_cos_fail],
        "jump_cos_by_layer_success": [float(x) for x in layer_jump_cos_success],
        "walk_positive_layer_fail":    walk_goes_positive_layer,
        "jump_positive_layer_success": jump_goes_positive_layer,
        "hb1_pass": bool(hb1_pass),
        "hb2_pass": bool(hb2_pass),
        "hb3_pass": bool(hb3_pass),
    },
    "option_c": {
        "gini_coefficient": float(gini_coeff),
        "top_head": top_head_key,
        "top_head_walk_fail":    float(top_head_walk_fail),
        "top_head_walk_success": float(top_head_walk_success),
        "top_head_jump_fail":    float(head_jump_fail.get(top_head_key, float("nan"))),
        "top_head_jump_success": float(top_head_jump_success),
        "top_head_attn_around":  float(top_attn_around),
        "top5_heads": [
            {"head": k,
             "walk_fail":    float(head_walk_fail[k]),
             "jump_fail":    float(head_jump_fail[k]),
             "walk_success": float(head_walk_success[k]),
             "jump_success": float(head_jump_success[k]),
             "attn_around":  float(head_attn_around_fail[k])}
            for k in sorted_heads_walk[:5]
        ],
        "hc1_pass": bool(hc1_pass),
        "hc2_pass": bool(hc2_pass),
        "hc3_pass": bool(hc3_pass),
    },
}

with open("051_results.json", "w") as f:
    json.dump(output, f, indent=2)
save_to_drive("051_results.json")

print(f"\n  Output: 051_results.json")
print(f"  {'='*70}")
print(f"  S-051 complete.")
print(f"  {'='*70}")
