#!/usr/bin/env python3
"""
S-058_CAUSAL_PATCH_VALUE_GEOMETRY.py — Activation Patch at L4H6/L5H2/L5H5
Applied Categorical Physics Workbench
Troy Teno | May 2026 | Open Access

Pre-registered hypotheses:
  workbench/proposals/S-058_CAUSAL_PATCH_VALUE_GEOMETRY_PROPOSAL.md
DO NOT modify hypotheses after seeing results.

─────────────────────────────────────────────────────────────────────
SETUP (run once in Colab or terminal):
  pip install torch transformers datasets numpy scipy sentencepiece -q

BACKGROUND:
  S-057 closed the L3 cluster. The failure chain is causally localized
  to L4H6 (decoder block 3, head 6), L5H2 (block 4, head 2), and
  L5H5 (block 4, head 5): the three value-substitution heads that attend
  to the jump encoder position but output contributions pointing toward
  I_WALK rather than I_JUMP.

  S-058 tests whether transplanting the value-head output contributions
  from matched success trajectories into fail trajectories flips the
  action-slot logit decision from I_WALK toward I_JUMP.

  Intervention arms:
    Arm A  — patch L4H6 + L5H2 + L5H5 simultaneously (primary repair test)
    Arm B1 — patch L4H6 alone
    Arm B2 — patch L5H2 alone
    Arm B3 — patch L5H5 alone
    Arm C  — patch L3H0 (global-context control; should have near-zero effect)
    Arm D  — neutralize L3H4 (suppressive-head control per sandbox_018)

  Pre-registered hypotheses:
    H1: Arm A flip fraction >= 0.50
    H2: Arm A flip fraction - Arm C flip fraction >= 0.20
    H3: max(single-head flip fractions B1/B2/B3) >= 0.25
    H4: mean(logit_jump - logit_walk | Arm A) > 0
    H5: Arm D mean delta_margin < 0.5 AND material-effect fraction < 0.10

  Kill conditions:
    K1: Arm A flip < 0.20 AND Arm C flip >= 0.15  (intervention ineffective)
    K2: Arm A flip >= 0.50 AND Arm C flip >= 0.40  (norm/scale confound suspected)
    K3: fewer than 15 matched pairs available

Version history:
  V0.1.0 — initial build
  V0.2.0 — K3 fired (1/30 matched): expanded success donor pool from
            N=25 to all available has_around+correct examples (458).
            Matching criteria unchanged.
  V0.3.0 — K3 fired again (5/30 matched): full-template matching too
            strict. Revised to jump-component matching per proposal
            amendment. Required: same movement+multiplier+output-position.
            Preferred: same direction, similar n_actions_gold.
─────────────────────────────────────────────────────────────────────
"""

import numpy as np
import json
import os
import random
from collections import defaultdict

SCRIPT_ID = "S-058_CAUSAL_PATCH_VALUE_GEOMETRY_V0.3.0"
SEED = 42
np.random.seed(SEED)
random.seed(SEED)

CHECKPOINT_DIR = "043_t5_scan_checkpoint"
MAX_INPUT_LEN  = 50
MAX_TARGET_LEN = 100
N_FAIL         = 30
# K3 fired on V0.1.0 with N_SUCCESS=25: only 1/30 fail examples matched.
# Root cause: 25-example success pool too sparse to cover the full range
# of command templates in the 30-example fail pool. Fix: expand success
# pool to draw from all available has_around+correct test examples.
# Matching criteria are UNCHANGED (template + n_actions_gold required;
# seq_len preferred). Only the donor pool size changes.
N_SUCCESS      = None   # None = use all available has_around+correct examples

# Head definitions: (decoder_block_0indexed, head_0indexed, label)
L4H6 = (3, 6, "L4H6")   # value-substitution specialist (S-051)
L5H2 = (4, 2, "L5H2")   # value-substitution specialist (S-051)
L5H5 = (4, 5, "L5H5")   # value-substitution specialist (S-051)
L3H0 = (2, 0, "L3H0")   # global-context control (S-056 K1)
L3H4 = (2, 4, "L3H4")   # suppressive-head control (sandbox_018 / S-057)

# Arm definitions: list of (block, head_idx, label) to patch per arm
# Arm D uses neutralize=True (zero out fail contrib instead of transplant)
ARMS = {
    "A":  {"heads": [L4H6, L5H2, L5H5], "neutralize": False},
    "B1": {"heads": [L4H6],              "neutralize": False},
    "B2": {"heads": [L5H2],              "neutralize": False},
    "B3": {"heads": [L5H5],              "neutralize": False},
    "C":  {"heads": [L3H0],              "neutralize": False},
    "D":  {"heads": [L3H4],              "neutralize": True},
}

# Pre-registered thresholds
H1_THRESHOLD       = 0.50   # Arm A flip fraction
H2_MARGIN          = 0.20   # Arm A - Arm C flip fraction difference
H3_SINGLE_MIN      = 0.25   # max single-head flip fraction
H5_DELTA_MAX       = 0.50   # mean delta_margin for Arm D (logit units)
H5_EFFECT_FRAC_MAX = 0.10   # material-effect fraction for Arm D
NORM_RATIO_MAX     = 2.0    # M4 scale check threshold
K1_ARM_A_MAX       = 0.20   # K1: Arm A flip below this
K1_ARM_C_MIN       = 0.15   # K1: Arm C flip above this
K2_ARM_A_MIN       = 0.50   # K2: Arm A flip above this
K2_ARM_C_MIN       = 0.40   # K2: Arm C flip above this
K3_MIN_PAIRS       = 15     # K3: minimum matched pairs

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
            os.makedirs(dst_dir, exist_ok=True)   # create nested dirs if needed
            shutil.copy(fname, os.path.join(DRIVE_DIR, fname))
            print(f"  Saved to Drive: {fname}")

print(f"\n{'='*70}")
print(f"  {SCRIPT_ID}")
print(f"  Activation patch: L4H6/L5H2/L5H5 value geometry")
print(f"{'='*70}")

# ── Imports ───────────────────────────────────────────────────────────────────
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

# ── Phase 1: Load checkpoint ───────────────────────────────────────────────────
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

# ── Phase 2: Token IDs and divergence embeddings ───────────────────────────
print("\n" + "="*70)
print("  PHASE 2 — ACTION TOKEN IDs AND DIVERGENCE EMBEDDINGS")
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
print(f"\n  T5 decoder: {n_dec_layers} layers × {n_heads} heads, d_kv={d_kv}, d_model={d_model}")

with torch.no_grad():
    EMBED_WALK = model.shared.weight[I_WALK_ID].cpu().float()
    EMBED_JUMP = model.shared.weight[I_JUMP_ID].cpu().float()
print(f"\n  ‖embed(I_WALK div)‖ = {EMBED_WALK.norm():.2f}")
print(f"  ‖embed(I_JUMP div)‖ = {EMBED_JUMP.norm():.2f}")

# ── Phase 3: Load SCAN + classify ─────────────────────────────────────────────
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
# N_SUCCESS=None uses all available has_around+correct examples as the donor pool.
# K3 fired in V0.1.0 with N_SUCCESS=25 (only 1/30 matched). Pool expanded per
# K3 resolution path in the proposal. Matching criteria are unchanged.
if N_SUCCESS is not None:
    success_examples = success_examples[:N_SUCCESS]
print(f"\n  substituted_walk fail examples: {len(sub_walk_examples)}")
print(f"  has_around + success (donor pool): {len(success_examples)}")

# ── Phase 4: Pre-registered Matching Procedure (V0.3.0 revised) ─────────────
print("\n" + "="*70)
print("  PHASE 4 — PRE-REGISTERED MATCHING PROCEDURE (V0.3.0 revised)")
print("  (Matching locked before any patching forward passes)")
print("  K3 fired twice on full-template matching; criteria revised per")
print("  proposal amendment. Jump-component structure is now the required")
print("  match criterion. Full-template match preferred but not required.")
print("="*70)

MULT_MAP = {"twice": 2, "thrice": 3}
PRIMITIVES = {"jump", "walk", "run", "look"}
ACTION_TOKS = {"I_JUMP", "I_WALK", "I_RUN", "I_LOOK"}

def parse_jump_component(cmd, gold_actions):
    """
    Extract the jump component's structural features.

    In SCAN:
      "A and B"   → output A then B  (A is first in output)
      "A after B" → output B then A  (B is first in output)

    Returns dict with keys:
      movement      — 'around' / 'opposite' / 'simple'
      direction     — 'left' / 'right' / None
      multiplier    — 1 / 2 / 3
      n_jump_slots  — number of I_JUMP action tokens in gold output
      jump_is_first — True if jump component appears first in output sequence
      n_actions_gold — total primitive action count in gold
    """
    words = cmd.lower().split()

    # Identify compound operator and split parts
    if "after" in words:
        sep = words.index("after")
        part_A, part_B = words[:sep], words[sep+1:]
        compound = "after"
    elif "and" in words:
        sep = words.index("and")
        part_A, part_B = words[:sep], words[sep+1:]
        compound = "and"
    else:
        part_A, part_B = words, []
        compound = "none"

    jump_in_A = "jump" in part_A
    jump_part  = part_A if jump_in_A else part_B

    # Output order: "A after B" → B first; "A and B" → A first
    if compound == "after":
        jump_is_first = not jump_in_A   # jump in A → second; jump in B → first
    elif compound == "and":
        jump_is_first = jump_in_A       # jump in A → first; jump in B → second
    else:
        jump_is_first = True

    # Jump component: movement type
    if "around" in jump_part:
        movement, base_reps = "around", 4
    elif "opposite" in jump_part:
        movement, base_reps = "opposite", 2
    else:
        movement, base_reps = "simple", 1

    # Multiplier
    mult = 1
    for mword, mval in MULT_MAP.items():
        if mword in jump_part:
            mult = mval
            break

    # Direction
    dirs = [w for w in jump_part if w in ("left", "right")]
    direction = dirs[0] if dirs else None

    # n_jump_slots (how many I_JUMP tokens in gold output)
    n_jump_slots = base_reps * mult

    # Total action count from gold
    gold_toks = gold_actions.split()
    n_actions_gold = sum(1 for t in gold_toks if t in ACTION_TOKS)

    return {
        "movement":      movement,
        "direction":     direction,
        "multiplier":    mult,
        "n_jump_slots":  n_jump_slots,
        "jump_is_first": jump_is_first,
        "n_actions_gold": n_actions_gold,
    }

# Pre-compute features for success pool
success_jfeats = [parse_jump_component(ex["commands"], ex["actions"])
                  for ex in success_examples]

matched_pairs = []   # list of (fail_ex, donor_ex, match_quality)
unmatched     = []

print(f"\n  Matching {len(sub_walk_examples)} fail examples against "
      f"{len(success_examples)} success donors (V0.3.0 jump-component matching)...")
print(f"  Required: movement + multiplier + jump_is_first (jump-component structure)")
print(f"  Preferred: direction (+2), n_actions_gold within ±8 (+1)")

for fail_ex in sub_walk_examples:
    ff = parse_jump_component(fail_ex["commands"], fail_ex["actions"])
    best_donor  = None
    best_score  = -1

    for s_ex, sf in zip(success_examples, success_jfeats):
        # Required: same jump component structure
        if sf["movement"]      != ff["movement"]:      continue
        if sf["multiplier"]    != ff["multiplier"]:    continue
        if sf["jump_is_first"] != ff["jump_is_first"]: continue
        # Score preferred criteria
        score = 0
        if sf["direction"] == ff["direction"]:         score += 2
        if abs(sf["n_actions_gold"] - ff["n_actions_gold"]) <= 8: score += 1
        if score > best_score:
            best_donor = s_ex
            best_score = score

    if best_donor is not None:
        if best_score >= 3:   quality = "jump+dir+count"
        elif best_score >= 2: quality = "jump+dir"
        elif best_score >= 1: quality = "jump+count"
        else:                 quality = "jump_only"
        matched_pairs.append((fail_ex, best_donor, quality))
        print(f"  MATCH [{quality}]: '{fail_ex['commands'][:45]}'"
              f"  ↔  '{best_donor['commands'][:45]}'")
    else:
        unmatched.append(fail_ex)
        print(f"  NO MATCH: '{fail_ex['commands'][:50]}'")

n_matched = len(matched_pairs)
print(f"\n  Matched: {n_matched} / {len(sub_walk_examples)}")
print(f"  Unmatched: {len(unmatched)}")
if matched_pairs:
    quality_counts = defaultdict(int)
    for _, _, q in matched_pairs:
        quality_counts[q] += 1
    print(f"  Match quality: {dict(quality_counts)}")

# K3 kill check
if n_matched < K3_MIN_PAIRS:
    print(f"\n  *** K3 FIRES: only {n_matched} matched pairs (minimum {K3_MIN_PAIRS})")
    print(f"  *** Donor pool size was {len(success_examples)} examples.")
    print(f"  *** STOP. Criteria 4 and 5 cannot be relaxed further.")
    raise SystemExit("K3: insufficient matched pairs.")

print(f"  K3: CLEAR ({n_matched} >= {K3_MIN_PAIRS})")

# ── Phase 5: Helper functions ──────────────────────────────────────────────────

def find_div_steps(token_ids):
    """
    Action-slot divergence-point step positions in a generated sequence.
    Returns list of (step_idx, action_label).
    """
    points = []
    for i in range(3, len(token_ids)):
        if (token_ids[i]   in DIV_TOKEN_SET and
            token_ids[i-1] == 834 and
            token_ids[i-2] == 27 and
            token_ids[i-3] in TURN_END_IDS):
            points.append((i, id_to_action.get(token_ids[i], "UNKNOWN")))
    return points

def get_jump_enc_pos(cmd):
    """
    Find the encoder position of the jump token in a tokenized command.
    Returns the first position where 'jump' appears distinctively.
    """
    j_ids = tokenizer.encode(cmd, add_special_tokens=False)
    # Compare against a walk version to find the differing position
    walk_cmd = " ".join("walk" if w == "jump" else w for w in cmd.split())
    w_ids = tokenizer.encode(walk_cmd, add_special_tokens=False)
    for i in range(min(len(j_ids), len(w_ids))):
        if j_ids[i] != w_ids[i]:
            return i
    return None

def compute_head_contrib(l_block, h_idx, step_cross_attns, enc_hidden_final):
    """
    Head contribution to decoder residual stream via V/O projection.
    V/O method: W_O[h] @ (attn[h] @ W_V[h] @ enc_hidden).
    Returns [d_model] float32 CPU tensor.
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

def run_generate_with_states(cmd):
    """
    Run model.generate() on a command, returning full states.
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
        "enc_hidden":   out.encoder_hidden_states[-1],   # [1, T_enc, d_model]
        "cross_attns":  out.cross_attentions,             # list[step] of tuple[layer][1,H,1,T_enc]
        "sequences":    out.sequences[0].tolist()[1:],    # strip BOS
        "scores":       out.scores,                       # list[step] of [1, vocab_size]
    }

def get_logit_outcome(logits_1d):
    """
    From a 1D logit tensor (vocab_size), extract P(I_JUMP), P(I_WALK), margin.
    Returns dict.
    """
    probs = torch.softmax(logits_1d, dim=-1)
    p_jump   = float(probs[I_JUMP_ID].item())
    p_walk   = float(probs[I_WALK_ID].item())
    margin   = float(logits_1d[I_JUMP_ID].item() - logits_1d[I_WALK_ID].item())
    flip     = 1 if p_jump > p_walk else 0
    return {"p_jump": p_jump, "p_walk": p_walk, "margin": margin, "flip": flip}

# ── Phase 6: Collect base contributions (fail + donor) ────────────────────────
print("\n" + "="*70)
print("  PHASE 6 — COLLECT BASE CONTRIBUTIONS (fail + donor forward passes)")
print("="*70)

ALL_PATCH_HEADS = [L4H6, L5H2, L5H5, L3H0, L3H4]

# Per-example storage
per_example = []   # list of dicts, one per matched pair

print(f"\n  Running base forward passes for {n_matched} matched pairs...")

for pair_idx, (fail_ex, donor_ex, quality) in enumerate(matched_pairs):
    if pair_idx % 5 == 0:
        print(f"  Pair {pair_idx+1}/{n_matched} ...")

    result = {
        "fail_cmd":    fail_ex["commands"],
        "donor_cmd":   donor_ex["commands"],
        "match_quality": quality,
        "valid":       True,
        "skip_reason": None,
    }

    # ── Run fail and donor generate passes ──
    fail_data  = run_generate_with_states(fail_ex["commands"])
    donor_data = run_generate_with_states(donor_ex["commands"])

    # ── Find first action-slot divergence step in fail sequence ──
    fail_div_steps = find_div_steps(fail_data["sequences"])
    if not fail_div_steps:
        result["valid"]       = False
        result["skip_reason"] = "no_div_steps_fail"
        per_example.append(result)
        continue

    div_step_idx, div_action = fail_div_steps[0]  # first action slot
    result["div_step_idx"] = div_step_idx
    result["div_action"]   = div_action

    # Verify div_step_idx is within cross_attentions range
    if div_step_idx >= len(fail_data["cross_attns"]):
        result["valid"]       = False
        result["skip_reason"] = "div_step_out_of_range"
        per_example.append(result)
        continue

    # ── Baseline logits (fail, unpatched) ──
    baseline_logits = fail_data["scores"][div_step_idx][0]  # [vocab_size]
    result["baseline"] = get_logit_outcome(baseline_logits)

    # ── Find donor divergence step ──
    donor_div_steps = find_div_steps(donor_data["sequences"])
    if not donor_div_steps:
        result["valid"]       = False
        result["skip_reason"] = "no_div_steps_donor"
        per_example.append(result)
        continue

    donor_div_step_idx = donor_div_steps[0][0]
    if donor_div_step_idx >= len(donor_data["cross_attns"]):
        result["valid"]       = False
        result["skip_reason"] = "donor_div_step_out_of_range"
        per_example.append(result)
        continue

    # ── Compute contributions for all patch heads ──
    fail_ca   = fail_data["cross_attns"][div_step_idx]
    donor_ca  = donor_data["cross_attns"][donor_div_step_idx]
    fail_enc  = fail_data["enc_hidden"]
    donor_enc = donor_data["enc_hidden"]

    contribs_fail  = {}
    contribs_donor = {}
    norm_ratios    = {}

    for (l_block, h_idx, label) in ALL_PATCH_HEADS:
        cf = compute_head_contrib(l_block, h_idx, fail_ca,  fail_enc)
        cd = compute_head_contrib(l_block, h_idx, donor_ca, donor_enc)
        contribs_fail[label]  = cf
        contribs_donor[label] = cd
        norm_f = float(cf.norm().item())
        norm_d = float(cd.norm().item())
        norm_ratios[label] = norm_d / (norm_f + 1e-9)

    result["contribs_fail"]  = contribs_fail
    result["contribs_donor"] = contribs_donor
    result["norm_ratios"]    = norm_ratios

    # ── Store fail encoder input IDs for the patched run ──
    result["fail_enc_input"]   = tokenizer(
        fail_ex["commands"], truncation=True, max_length=MAX_INPUT_LEN,
        return_tensors="pt"
    )
    result["fail_dec_prefix"]  = fail_data["sequences"][:div_step_idx]
    result["fail_enc_hidden"]  = fail_enc

    per_example.append(result)

n_valid = sum(1 for r in per_example if r["valid"])
n_skipped = sum(1 for r in per_example if not r["valid"])
print(f"\n  Valid:   {n_valid}")
print(f"  Skipped: {n_skipped}")
if n_skipped:
    reasons = defaultdict(int)
    for r in per_example:
        if not r["valid"]:
            reasons[r["skip_reason"]] += 1
    for reason, count in reasons.items():
        print(f"    {reason}: {count}")

# ── Phase 7: Run patched forward passes ───────────────────────────────────────────
print("\n" + "="*70)
print("  PHASE 7 — PATCHED FORWARD PASSES (hook-based activation patching)")
print("="*70)

def run_patched_arm(result_dict, arm_name, arm_def):
    """
    Run patched forward pass for one arm on one example.

    Strategy: register forward hooks on EncDecAttention at patched layers.
    Each hook adds (contrib_donor[h] - contrib_fail[h]) — or (-contrib_fail[h]
    for neutralization arms — to the last decoder position's cross-attn output.
    Then run model.forward() with decoder_input_ids = prefix up to div step.

    Returns get_logit_outcome dict, or None if invalid.
    """
    if not result_dict["valid"]:
        return None

    heads     = arm_def["heads"]
    neutralize = arm_def["neutralize"]

    # Build per-layer diff vectors: sum contributions from all heads in that layer
    layer_diffs = defaultdict(lambda: torch.zeros(d_model, dtype=torch.float32))
    for (l_block, h_idx, label) in heads:
        cf = result_dict["contribs_fail"][label]
        cd = result_dict["contribs_donor"][label]
        if neutralize:
            diff = -cf   # zero out the fail contribution
        else:
            diff = cd - cf  # transplant: replace fail with donor
        layer_diffs[l_block] += diff

    # Register hooks
    hooks = []
    for l_block, diff_vec in layer_diffs.items():
        enc_dec_attn = model.decoder.block[l_block].layer[1].EncDecAttention
        diff_dev = diff_vec.to(device)

        def make_hook(d):
            def hook_fn(module, input, output):
                # output[0] is attn_output: [batch, seq_len, d_model]
                patched = output[0].clone()
                patched[0, -1, :] += d
                return (patched,) + output[1:]
            return hook_fn

        h = enc_dec_attn.register_forward_hook(make_hook(diff_dev))
        hooks.append(h)

    # Build decoder prefix tensor: tokens up to (not including) div_step
    dec_prefix = result_dict["fail_dec_prefix"]
    if not dec_prefix:
        for h in hooks:
            h.remove()
        return None

    dec_input_ids = torch.tensor([dec_prefix], dtype=torch.long).to(device)
    enc_input = {k: v.to(device) for k, v in result_dict["fail_enc_input"].items()}

    try:
        with torch.no_grad():
            out = model(
                input_ids=enc_input["input_ids"],
                attention_mask=enc_input.get("attention_mask"),
                decoder_input_ids=dec_input_ids,
            )
        # logits: [1, seq_len, vocab_size] — last position predicts the div token
        logits_last = out.logits[0, -1, :]   # [vocab_size]
        outcome = get_logit_outcome(logits_last)
    except Exception as e:
        outcome = None
        print(f"  WARNING: patched forward failed: {e}")
    finally:
        for h in hooks:
            h.remove()

    return outcome

print(f"\n  Running {len(ARMS)} arms × {n_valid} valid examples...\n")

# Initialize arm results storage
arm_results = {arm: [] for arm in ARMS}

for i, res in enumerate(per_example):
    if not res["valid"]:
        for arm in ARMS:
            arm_results[arm].append(None)
        continue

    if i % 5 == 0:
        print(f"  Example {i+1}/{len(per_example)} ...")

    for arm_name, arm_def in ARMS.items():
        outcome = run_patched_arm(res, arm_name, arm_def)
        arm_results[arm_name].append(outcome)

# ── Phase 8: M1-M6 Compute and Report ───────────────────────────────────────────
print("\n" + "="*70)
print("  PHASE 8 — MEASUREMENTS M1–M6")
print("="*70)

# ── M5: Pre-patch baseline ───────────────────────────────────────────────────────
print(f"\n  M5 — Pre-patch baseline (unpatched fail examples):")
baseline_outcomes = [r["baseline"] for r in per_example if r["valid"]]
if baseline_outcomes:
    bl_p_jump  = [o["p_jump"]  for o in baseline_outcomes]
    bl_p_walk  = [o["p_walk"]  for o in baseline_outcomes]
    bl_margins = [o["margin"]  for o in baseline_outcomes]
    bl_flips   = [o["flip"]    for o in baseline_outcomes]
    print(f"  n={len(baseline_outcomes)}")
    print(f"  mean P(I_WALK) = {np.mean(bl_p_walk):.4f}  "
          f"mean P(I_JUMP) = {np.mean(bl_p_jump):.4f}")
    print(f"  mean logit margin (jump - walk) = {np.mean(bl_margins):.4f}")
    print(f"  baseline flip fraction = {np.mean(bl_flips):.3f}  "
          f"(should be near 0)")
else:
    print("  No valid baseline outcomes.")
    baseline_outcomes = []

# ── M1: Per-example Arm A logit outcomes ────────────────────────────────────────
print(f"\n  M1 — Arm A (L4H6+L5H2+L5H5) logit outcomes:")
arm_A_outcomes = [o for o in arm_results["A"] if o is not None]
if arm_A_outcomes:
    A_flips   = [o["flip"]   for o in arm_A_outcomes]
    A_margins = [o["margin"] for o in arm_A_outcomes]
    A_p_jump  = [o["p_jump"] for o in arm_A_outcomes]
    A_p_walk  = [o["p_walk"] for o in arm_A_outcomes]
    print(f"  n={len(arm_A_outcomes)}")
    print(f"  Flip fraction:           {np.mean(A_flips):.3f}  "
          f"({int(np.sum(A_flips))}/{len(A_flips)})")
    print(f"  Mean P(I_JUMP) post:     {np.mean(A_p_jump):.4f}  "
          f"(baseline: {np.mean(bl_p_jump):.4f})")
    print(f"  Mean P(I_WALK) post:     {np.mean(A_p_walk):.4f}  "
          f"(baseline: {np.mean(bl_p_walk):.4f})")
    print(f"  Mean logit margin post:  {np.mean(A_margins):.4f}  "
          f"(baseline: {np.mean(bl_margins):.4f})")
    # Also track examples where I_JUMP rank improved
    arm_A_rank_improved = sum(
        1 for o, b in zip(arm_A_outcomes,
                          [r["baseline"] for r in per_example if r["valid"]])
        if o["p_jump"] > b["p_jump"]
    )
    print(f"  P(I_JUMP) rank improved: {arm_A_rank_improved}/{len(arm_A_outcomes)}")

# ── M2: Control arm comparison (Arm C) ────────────────────────────────────────────
print(f"\n  M2 — Arm C (L3H0 global-context control) logit outcomes:")
arm_C_outcomes = [o for o in arm_results["C"] if o is not None]
if arm_C_outcomes:
    C_flips   = [o["flip"]   for o in arm_C_outcomes]
    C_margins = [o["margin"] for o in arm_C_outcomes]
    print(f"  n={len(arm_C_outcomes)}")
    print(f"  Flip fraction:           {np.mean(C_flips):.3f}  "
          f"({int(np.sum(C_flips))}/{len(C_flips)})")
    print(f"  Mean logit margin post:  {np.mean(C_margins):.4f}")

# ── M3: Single-head contribution breakdown (Arms B1-B3) ──────────────────────────
print(f"\n  M3 — Single-head arms (B1/B2/B3) contribution breakdown:")
single_head_results = {}
for arm_name, label in [("B1", "L4H6"), ("B2", "L5H2"), ("B3", "L5H5")]:
    outcomes = [o for o in arm_results[arm_name] if o is not None]
    if outcomes:
        flips   = [o["flip"]   for o in outcomes]
        margins = [o["margin"] for o in outcomes]
        single_head_results[arm_name] = {
            "label":        label,
            "n":            len(outcomes),
            "flip_fraction": float(np.mean(flips)),
            "mean_margin":  float(np.mean(margins)),
        }
        print(f"  {arm_name} ({label:<6}): flip={np.mean(flips):.3f}  "
              f"({int(np.sum(flips))}/{len(flips)})  "
              f"mean_margin={np.mean(margins):.4f}")

# ── M4: Scale / norm check ──────────────────────────────────────────────────────────
print(f"\n  M4 — Norm ratio check (donor norm / fail norm, all patch heads):")
print(f"  {'Head':<6} {'Mean ratio':>12} {'Max ratio':>10} {'Flagged (>2×)':>14}")
norm_check_results = {}
for (_, _, label) in ALL_PATCH_HEADS:
    ratios = [r["norm_ratios"][label] for r in per_example
              if r["valid"] and "norm_ratios" in r]
    if ratios:
        n_flagged = sum(1 for nr in ratios if nr > NORM_RATIO_MAX)
        norm_check_results[label] = {
            "mean_ratio": float(np.mean(ratios)),
            "max_ratio":  float(np.max(ratios)),
            "n_flagged":  int(n_flagged),
            "n_total":    len(ratios),
        }
        print(f"  {label:<6} {np.mean(ratios):>12.3f} {np.max(ratios):>10.3f} "
              f"{n_flagged:>14} / {len(ratios)}")

# ── M6: Arm D (L3H4 suppressive-head control) ──────────────────────────────────
print(f"\n  M6 — Arm D (L3H4 neutralization) suppressive-head control:")
arm_D_outcomes = [o for o in arm_results["D"] if o is not None]
if arm_D_outcomes:
    D_flips   = [o["flip"]   for o in arm_D_outcomes]
    D_margins = [o["margin"] for o in arm_D_outcomes]

    baseline_margins_valid = [r["baseline"]["margin"] for r in per_example
                               if r["valid"]]
    delta_margins = [dm - bm for dm, bm in zip(D_margins, baseline_margins_valid)]
    mean_delta    = float(np.mean(delta_margins))

    # Material effect: delta_margin > 0.5 logit units (pro-jump direction)
    n_material  = sum(1 for dm in delta_margins if dm > H5_DELTA_MAX)
    effect_frac = float(n_material / len(delta_margins)) if delta_margins else float("nan")

    print(f"  n={len(arm_D_outcomes)}")
    print(f"  Flip fraction:           {np.mean(D_flips):.3f}  "
          f"({int(np.sum(D_flips))}/{len(D_flips)})")
    print(f"  Mean logit margin post:  {np.mean(D_margins):.4f}  "
          f"(baseline: {np.mean(baseline_margins_valid):.4f})")
    print(f"  Mean delta_margin:       {mean_delta:.4f}")
    print(f"  Material effect (>0.5):  {n_material}/{len(delta_margins)}  "
          f"fraction={effect_frac:.3f}")

# ── Phase 9: Kill condition checks ────────────────────────────────────────────
print("\n" + "="*70)
print("  PHASE 9 — KILL CONDITION CHECKS")
print("="*70)

arm_A_flip_frac = float(np.mean(A_flips))   if arm_A_outcomes else float("nan")
arm_C_flip_frac = float(np.mean(C_flips))   if arm_C_outcomes else float("nan")
arm_D_mean_delta = mean_delta               if arm_D_outcomes else float("nan")
arm_D_effect_frac = effect_frac            if arm_D_outcomes else float("nan")

k1_fires = arm_A_flip_frac < K1_ARM_A_MAX and arm_C_flip_frac >= K1_ARM_C_MIN
k2_fires = arm_A_flip_frac >= K2_ARM_A_MIN and arm_C_flip_frac >= K2_ARM_C_MIN

print(f"\n  K1 (Arm A flip < {K1_ARM_A_MAX} AND Arm C flip >= {K1_ARM_C_MIN}):")
print(f"    Arm A flip = {arm_A_flip_frac:.3f}  Arm C flip = {arm_C_flip_frac:.3f}")
print(f"    K1: {'FIRES' if k1_fires else 'CLEAR'}")

print(f"\n  K2 (Arm A flip >= {K2_ARM_A_MIN} AND Arm C flip >= {K2_ARM_C_MIN}):")
print(f"    K2: {'FIRES — stop, diagnose norm confound before claiming result' if k2_fires else 'CLEAR'}")

if k1_fires:
    print(f"\n  *** K1 FIRES: activation patch at L4H6/L5H2/L5H5 does not work.")
    print(f"  *** Do NOT proceed to weight editing on these heads.")
    print(f"  *** Relay to G-track: activation patch ineffective;")
    print(f"  *** G-056 correction geometry work becomes the primary diagnostic.")

if k2_fires:
    print(f"\n  *** K2 FIRES: both Arm A and Arm C produce high flip fractions.")
    print(f"  *** Check M4 norm ratios — likely a norm/scale confound.")
    print(f"  *** Do NOT claim a causal result until confound is ruled out.")

# ── Phase 10: Hypothesis verdicts ──────────────────────────────────────────────
print("\n" + "="*70)
print("  PHASE 10 — HYPOTHESIS VERDICTS")
print("="*70)

# H1: Arm A flip fraction >= 0.50
h1_pass = arm_A_flip_frac >= H1_THRESHOLD

# H2: Arm A - Arm C >= 0.20
h2_diff = arm_A_flip_frac - arm_C_flip_frac
h2_pass = h2_diff >= H2_MARGIN

# H3: max single-head flip >= 0.25
single_flip_fracs = {
    arm: d["flip_fraction"] for arm, d in single_head_results.items()
}
max_single_flip = max(single_flip_fracs.values()) if single_flip_fracs else float("nan")
h3_pass = max_single_flip >= H3_SINGLE_MIN

# H4: mean logit margin > 0 after Arm A
h4_pass = float(np.mean(A_margins)) > 0.0 if arm_A_outcomes else False

# H5: Arm D mean delta < 0.5 AND material-effect fraction < 0.10
h5_delta_ok  = arm_D_mean_delta  < H5_DELTA_MAX
h5_effect_ok = arm_D_effect_frac < H5_EFFECT_FRAC_MAX
h5_pass = h5_delta_ok and h5_effect_ok

print(f"\n  H1  (Arm A flip >= {H1_THRESHOLD:.2f}):                  "
      f"{'PASS' if h1_pass else 'FAIL'}  [{arm_A_flip_frac:.3f}]")
print(f"  H2  (Arm A - Arm C >= {H2_MARGIN:.2f}):              "
      f"{'PASS' if h2_pass else 'FAIL'}  [diff={h2_diff:.3f}]")
print(f"  H3  (max single-head flip >= {H3_SINGLE_MIN:.2f}):         "
      f"{'PASS' if h3_pass else 'FAIL'}  [max={max_single_flip:.3f}]")
print(f"  H4  (mean Arm A logit margin > 0):          "
      f"{'PASS' if h4_pass else 'FAIL'}  [{float(np.mean(A_margins)):.4f}]")
print(f"  H5  (Arm D near-zero: delta<{H5_DELTA_MAX}, effect<{H5_EFFECT_FRAC_MAX}): "
      f"{'PASS' if h5_pass else 'FAIL'}  "
      f"[delta={arm_D_mean_delta:.4f}, effect_frac={arm_D_effect_frac:.3f}]")
print(f"\n  K1: {'FIRES' if k1_fires else 'CLEAR'}  "
      f"  K2: {'FIRES' if k2_fires else 'CLEAR'}  "
      f"  K3: CLEAR ({n_matched} pairs)")

if not h5_pass:
    print(f"\n  *** H5 FAIL: L3H4 neutralization has a material effect.")
    print(f"  *** Stop and reassess before interpreting Arm A primary result.")
    print(f"  *** Relay to G-track: L3H4 may need to be added to repair target set.")

if h1_pass and h2_pass and not k2_fires and h5_pass:
    print(f"\n  *** PRIMARY RESULT CONFIRMED:")
    print(f"  *** Activation patch at L4H6/L5H2/L5H5 flips logit decision in")
    print(f"  *** {int(np.sum(A_flips))}/{len(A_flips)} patched fail examples.")
    print(f"  *** Value-substitution mechanism confirmed as causally sufficient.")
    print(f"  *** Next step: weight editing using G-056 correction geometry.")

# ── Save results ──────────────────────────────────────────────────────────────────────
def arm_summary(outcomes, baseline_list=None):
    if not outcomes:
        return {}
    flips   = [o["flip"]   for o in outcomes]
    margins = [o["margin"] for o in outcomes]
    p_jumps = [o["p_jump"] for o in outcomes]
    p_walks = [o["p_walk"] for o in outcomes]
    out = {
        "n":            len(outcomes),
        "flip_fraction": float(np.mean(flips)),
        "n_flipped":    int(np.sum(flips)),
        "mean_margin":  float(np.mean(margins)),
        "mean_p_jump":  float(np.mean(p_jumps)),
        "mean_p_walk":  float(np.mean(p_walks)),
        "per_example_flips": [int(f) for f in flips],
        "per_example_margins": [float(m) for m in margins],
    }
    if baseline_list is not None:
        deltas = [m - b["margin"] for m, b in zip(margins, baseline_list)]
        out["mean_delta_margin"] = float(np.mean(deltas))
        out["n_material_effect"] = int(sum(1 for d in deltas if d > H5_DELTA_MAX))
        out["effect_fraction"]   = float(out["n_material_effect"] / len(deltas))
    return out

valid_baselines = [r["baseline"] for r in per_example if r["valid"]]

output = {
    "script_id": SCRIPT_ID,
    "n_matched_pairs": int(n_matched),
    "n_valid": int(n_valid),
    "n_skipped": int(n_skipped),

    "matching": {
        "pairs": [
            {
                "fail_cmd":      r["fail_cmd"],
                "donor_cmd":     r["donor_cmd"],
                "quality":       r["match_quality"],
                "valid":         r["valid"],
                "skip_reason":   r.get("skip_reason"),
            }
            for r in per_example
        ]
    },

    "M5_baseline": {
        "n":            len(baseline_outcomes),
        "mean_p_walk":  float(np.mean(bl_p_walk))   if baseline_outcomes else None,
        "mean_p_jump":  float(np.mean(bl_p_jump))   if baseline_outcomes else None,
        "mean_margin":  float(np.mean(bl_margins))  if baseline_outcomes else None,
        "flip_fraction": float(np.mean(bl_flips))   if baseline_outcomes else None,
    },

    "M1_arm_A": arm_summary(arm_A_outcomes),
    "M2_arm_C": arm_summary(arm_C_outcomes),
    "M3_single_heads": single_head_results,
    "M3_arm_B1": arm_summary([o for o in arm_results["B1"] if o is not None]),
    "M3_arm_B2": arm_summary([o for o in arm_results["B2"] if o is not None]),
    "M3_arm_B3": arm_summary([o for o in arm_results["B3"] if o is not None]),

    "M4_norm_ratios": norm_check_results,

    "M6_arm_D": arm_summary(
        arm_D_outcomes,
        baseline_list=valid_baselines
    ),

    "hypotheses": {
        "H1": "PASS" if h1_pass else "FAIL",
        "H2": "PASS" if h2_pass else "FAIL",
        "H3": "PASS" if h3_pass else "FAIL",
        "H4": "PASS" if h4_pass else "FAIL",
        "H5": "PASS" if h5_pass else "FAIL",
        "K1": "FIRES" if k1_fires else "CLEAR",
        "K2": "FIRES" if k2_fires else "CLEAR",
        "K3": "CLEAR",
    },

    "key_values": {
        "arm_A_flip_fraction": float(arm_A_flip_frac),
        "arm_C_flip_fraction": float(arm_C_flip_frac),
        "h2_arm_diff":         float(h2_diff),
        "max_single_head_flip": float(max_single_flip),
        "arm_A_mean_margin":   float(np.mean(A_margins)) if arm_A_outcomes else None,
        "arm_D_mean_delta_margin": float(arm_D_mean_delta),
        "arm_D_effect_fraction":   float(arm_D_effect_frac),
        "single_head_B1_flip": single_head_results.get("B1", {}).get("flip_fraction"),
        "single_head_B2_flip": single_head_results.get("B2", {}).get("flip_fraction"),
        "single_head_B3_flip": single_head_results.get("B3", {}).get("flip_fraction"),
    },
}

results_path = os.path.join("workbench", "results", "058_results.json")
os.makedirs(os.path.dirname(results_path), exist_ok=True)
with open(results_path, "w") as f:
    json.dump(output, f, indent=2)
save_to_drive(results_path)

print(f"\n  Output: {results_path}")
print(f"  {'='*70}")
print(f"  S-058 complete.")
print(f"  {'='*70}")
print(f"\n  # SEALED — do not re-run.")
