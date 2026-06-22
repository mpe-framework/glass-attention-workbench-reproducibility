#!/usr/bin/env python3
"""
S-056_L3_HEAD_CLUSTER.py — L3 Head Cluster Characterization
Applied Categorical Physics Workbench
Troy Teno | May 2026 | Open Access

Pre-registered hypotheses:
  workbench/proposals/S-056_L3_HEAD_CLUSTER_PROPOSAL.md
DO NOT modify hypotheses after seeing results.

─────────────────────────────────────────────────────────────────────
SETUP (run once in Colab or terminal):
  pip install torch transformers datasets numpy scipy sentencepiece -q

BACKGROUND:
  S-055 measured per-head Born filter defect across all 48 decoder
  cross-attention heads. The expected peak (layers 4–5, from S-051)
  did not appear. Instead, L3H0 (3.867×), L3H3 (1.950×), L3H4 (2.230×)
  dominated the top-5 — none were pre-registered.

  S-056 characterizes L3H0/H3/H4 with four measurements:
    M1: attention weight to jump encoder position
    M2: attention weight to around encoder position
    M3: cosine attribution to I_WALK / I_JUMP divergence embeddings
    M4: per-example Born filter defect distribution for L3H0

  H1: L3H0 attn_to_jump > 0.3 in fail group (jump-attending head)
  H2: L3H0 shows value substitution signature in cosine space
  H3: L3H0 attn_to_jump > attn_to_around in fail group
  H4: ≥75% of success examples have L3H0 defect > median fail defect

  K1: attn_to_jump for L3H0 < 0.1 (not a jump-attending head)
  K2: |cos_walk| < 0.05 AND |cos_jump| < 0.05 (orthogonal to vocabulary)
─────────────────────────────────────────────────────────────────────
"""

import numpy as np
import json
import os
import random
from collections import defaultdict

SCRIPT_ID = "S-056_L3_HEAD_CLUSTER_V0.1.0"
SEED = 42
np.random.seed(SEED)
random.seed(SEED)

CHECKPOINT_DIR = "043_t5_scan_checkpoint"
MAX_INPUT_LEN  = 50
MAX_TARGET_LEN = 100
N_FAIL         = 30
N_SUCCESS      = 25

# L3 cluster: newly identified in S-055 (0-based block, 0-based head)
# L3H0 = block[2] head 0; L3H3 = block[2] head 3; L3H4 = block[2] head 4
L3_HEADS = [
    (2, 0, "L3H0"),
    (2, 3, "L3H3"),
    (2, 4, "L3H4"),
]

# Jump-attending heads from S-051 — reference / comparison
JUMP_HEADS = [
    (3, 6, "L4H6"),
    (4, 2, "L5H2"),
    (4, 5, "L5H5"),
]

ALL_TARGET_HEADS = L3_HEADS + JUMP_HEADS

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
print(f"  L3 head cluster characterization: attention routing + cosine attribution")
print(f"{'='*70}")

# ── Imports ───────────────────────────────────────────────────────────────────
print("\nImporting libraries...")
try:
    import torch
    import torch.nn.functional as F
    from transformers import T5ForConditionalGeneration, T5Tokenizer
except ImportError as e:
    raise SystemExit(f"\nMissing library: {e}\n")

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

# ── Phase 2: Token IDs, architecture, divergence embeddings ─────────────────
print("\n" + "="*70)
print("  PHASE 2 — ACTION TOKEN IDs AND DIVERGENCE EMBEDDINGS")
print("="*70)

core_actions = ["I_JUMP", "I_WALK", "I_RUN", "I_LOOK"]
ALL_NAMES    = core_actions + ["I_TURN_LEFT", "I_TURN_RIGHT"]

raw_ids = {name: tokenizer.encode(name, add_special_tokens=False)
           for name in ALL_NAMES}
print(f"\n  Full tokenizations:")
for name, ids in raw_ids.items():
    print(f"  {name:<15} → {ids}")

DIV_IDS       = {name: raw_ids[name][2] for name in core_actions}
TURN_LAST_IDS = {name: raw_ids[name][-1]
                 for name in ["I_TURN_LEFT", "I_TURN_RIGHT"]}
TURN_END_IDS  = set(TURN_LAST_IDS.values())
DIV_TOKEN_SET = set(DIV_IDS.values())
id_to_action  = {v: k for k, v in DIV_IDS.items()}

print(f"\n  Divergence-point IDs: {DIV_IDS}")
print(f"  TURN_END_IDS:         {TURN_END_IDS}")

n_dec_layers = model.config.num_decoder_layers   # 6
n_heads      = model.config.num_heads            # 8
d_kv         = model.config.d_kv                 # 64
d_model      = model.config.d_model              # 512
print(f"\n  T5 decoder: {n_dec_layers} layers, {n_heads} heads, d_kv={d_kv}, d_model={d_model}")

# Divergence-point embeddings for cosine attribution (M3)
with torch.no_grad():
    EMBED_WALK = model.shared.weight[DIV_IDS["I_WALK"]].cpu().float()   # [d_model]
    EMBED_JUMP = model.shared.weight[DIV_IDS["I_JUMP"]].cpu().float()   # [d_model]
EMBED_WALK_NORM = EMBED_WALK / (EMBED_WALK.norm() + 1e-9)
EMBED_JUMP_NORM = EMBED_JUMP / (EMBED_JUMP.norm() + 1e-9)
print(f"\n  Divergence embeddings loaded.")
print(f"  ‖embed(I_WALK div)‖ = {EMBED_WALK.norm():.2f}")
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

def make_walk_cmd(cmd):
    return " ".join("walk" if w == "jump" else w for w in cmd.split())

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

# ── Phase 4: Helper functions ──────────────────────────────────────────────────

def find_div_points(gen_token_ids):
    """Find action-slot divergence-point positions in generated sequence."""
    points = []
    for i in range(3, len(gen_token_ids)):
        if (gen_token_ids[i]   in DIV_TOKEN_SET and
            gen_token_ids[i-1] == 834 and
            gen_token_ids[i-2] == 27 and
            gen_token_ids[i-3] in TURN_END_IDS):
            action_name = id_to_action.get(gen_token_ids[i], "UNKNOWN")
            points.append((i, action_name))
    return points

def find_key_enc_positions(jump_cmd, walk_cmd):
    """
    Returns (jump_enc_pos, around_enc_pos).
    jump_enc_pos: first encoder token position where jump_cmd and walk_cmd differ.
    around_enc_pos: first re-sync token after the divergence (= 'around' token start).
    Both are indices into the encoder sequence (without EOS; valid within T_enc-1).
    Returns (None, None) if not found.
    """
    j_ids = tokenizer.encode(jump_cmd, add_special_tokens=False)
    w_ids = tokenizer.encode(walk_cmd, add_special_tokens=False)

    # Jump position: first divergence
    jump_pos = None
    for i in range(min(len(j_ids), len(w_ids))):
        if j_ids[i] != w_ids[i]:
            jump_pos = i
            break
    if jump_pos is None:
        return None, None

    # Around position: first re-sync after the divergence.
    # In SCAN has_around commands, "jump around X" is always a bigram —
    # "around" is the token immediately after jump's token(s). We find
    # re-sync at the first position > jump_pos where j_ids[i] == w_ids[i].
    around_pos = None
    for delta in range(1, 6):
        cand_j = jump_pos + delta
        cand_w = jump_pos + delta  # walk and jump diverge for same # tokens (1, usually)
        # Check if sequences re-sync (both have same token = "around" start)
        if (cand_j < len(j_ids) and cand_w < len(w_ids) and
                j_ids[cand_j] == w_ids[cand_w]):
            around_pos = cand_j
            break

    return jump_pos, around_pos

def compute_head_contrib(l_block, h_idx, step_cross_attns, enc_hidden_final):
    """
    Compute head contribution to decoder residual stream at this step.
    Same as S-055 / S-051 Option C.
    Returns: contrib [d_model] float32 on CPU.
    """
    enc_attn = model.decoder.block[l_block].layer[1].EncDecAttention
    with torch.no_grad():
        V_all = enc_hidden_final @ enc_attn.v.weight.T   # [1, T_enc, n_heads*d_kv]
    attn_w       = step_cross_attns[l_block]               # [1, n_heads, 1, T_enc]
    head_attn    = attn_w[0, h_idx, 0, :]                  # [T_enc]
    head_V       = V_all[0, :, h_idx*d_kv:(h_idx+1)*d_kv] # [T_enc, d_kv]
    head_val     = (head_attn.unsqueeze(0) @ head_V).squeeze(0)  # [d_kv]
    head_o_slice = enc_attn.o.weight[:, h_idx*d_kv:(h_idx+1)*d_kv]  # [d_model, d_kv]
    return (head_o_slice @ head_val).cpu().float()

def cos_sim(a, b):
    """Cosine similarity between two vectors."""
    return float((a @ b) / (a.norm() * b.norm() + 1e-9))

# ── Phase 5: Main measurement loop ────────────────────────────────────────────
print("\n" + "="*70)
print("  PHASE 5 — PAIRED FORWARD PASSES (jump vs walk)")
print("="*70)
print(f"\n  Running {len(sub_walk_examples)} fail + {len(success_examples)} success pairs")
print(f"  Measuring: attn routing (M1/M2) + cosine attribution (M3) + defect (M4)\n")

# Storage: one dict per target head; keys are "fail" / "success"
# Values: lists of per-example means over action slots
head_storage = {}
for (l_block, h_idx, label) in ALL_TARGET_HEADS:
    head_storage[label] = {
        "fail":    {"attn_jump": [], "attn_around": [], "cos_walk": [], "cos_jump": [], "defect": []},
        "success": {"attn_jump": [], "attn_around": [], "cos_walk": [], "cos_jump": [], "defect": []},
    }

n_skip_no_positions = 0
n_skip_no_slots     = 0

def measure_pair(jump_cmd, walk_cmd, group_label, idx, total):
    if idx % 5 == 0:
        print(f"  [{group_label}] {idx+1}/{total} ...")

    # Find encoder key positions
    jump_enc_pos, around_enc_pos = find_key_enc_positions(jump_cmd, walk_cmd)
    if jump_enc_pos is None:
        return None

    def encode(cmd):
        return tokenizer(cmd, truncation=True, max_length=MAX_INPUT_LEN,
                         return_tensors="pt").to(device)

    inp_j = encode(jump_cmd)
    inp_w = encode(walk_cmd)

    with torch.no_grad():
        out_j = model.generate(
            input_ids=inp_j["input_ids"],
            attention_mask=inp_j["attention_mask"],
            max_length=MAX_TARGET_LEN,
            output_attentions=True,
            output_hidden_states=True,
            return_dict_in_generate=True,
        )
        out_w = model.generate(
            input_ids=inp_w["input_ids"],
            attention_mask=inp_w["attention_mask"],
            max_length=MAX_TARGET_LEN,
            output_attentions=True,
            output_hidden_states=True,
            return_dict_in_generate=True,
        )

    enc_h_j = out_j.encoder_hidden_states[-1]   # [1, T_enc_j, d_model]
    enc_h_w = out_w.encoder_hidden_states[-1]   # [1, T_enc_w, d_model]

    seq_j = out_j.sequences[0].tolist()[1:]
    seq_w = out_w.sequences[0].tolist()[1:]

    pts_j = find_div_points(seq_j)
    pts_w = find_div_points(seq_w)

    n_slots    = min(len(pts_j), len(pts_w))
    if n_slots == 0:
        return None

    n_steps_j = len(out_j.decoder_hidden_states)
    n_steps_w = len(out_w.decoder_hidden_states)

    # Per-slot measurements for each target head
    slot_data = {label: {"attn_jump": [], "attn_around": [], "cos_walk": [], "cos_jump": [], "defect": []}
                 for (_, _, label) in ALL_TARGET_HEADS}

    for k in range(n_slots):
        s_j = pts_j[k][0]
        s_w = pts_w[k][0]
        if s_j >= n_steps_j or s_w >= n_steps_w:
            continue

        ca_j = out_j.cross_attentions[s_j]   # tuple[n_dec_layers] of [1,n_heads,1,T_enc]
        ca_w = out_w.cross_attentions[s_w]

        for (l_block, h_idx, label) in ALL_TARGET_HEADS:
            attn_weights_j = ca_j[l_block][0, h_idx, 0, :]   # [T_enc_j]
            T_enc_j = attn_weights_j.shape[0]

            # M1: attention to jump encoder position (in jump_cmd run)
            a_jump = float(attn_weights_j[jump_enc_pos].item()) if jump_enc_pos < T_enc_j else 0.0

            # M2: attention to around encoder position (in jump_cmd run)
            a_around = 0.0
            if around_enc_pos is not None and around_enc_pos < T_enc_j:
                a_around = float(attn_weights_j[around_enc_pos].item())

            # M3: cosine attribution (in jump_cmd run only)
            contrib_j = compute_head_contrib(l_block, h_idx, ca_j, enc_h_j)
            c_walk = cos_sim(contrib_j, EMBED_WALK_NORM)
            c_jump = cos_sim(contrib_j, EMBED_JUMP_NORM)

            # M4 / Born filter defect: L2 between jump and walk contributions
            contrib_w = compute_head_contrib(l_block, h_idx, ca_w, enc_h_w)
            defect_val = float((contrib_j - contrib_w).norm().item())

            slot_data[label]["attn_jump"].append(a_jump)
            slot_data[label]["attn_around"].append(a_around)
            slot_data[label]["cos_walk"].append(c_walk)
            slot_data[label]["cos_jump"].append(c_jump)
            slot_data[label]["defect"].append(defect_val)

    # Accumulate per-example means
    result = {}
    for (_, _, label) in ALL_TARGET_HEADS:
        sd = slot_data[label]
        if not sd["attn_jump"]:
            result[label] = None
            continue
        result[label] = {k: float(np.mean(v)) for k, v in sd.items()}
    return result

# ── Run groups ────────────────────────────────────────────────────────────────────
def run_group(examples, group_label):
    global n_skip_no_positions, n_skip_no_slots
    n_valid = 0
    for idx, ex in enumerate(examples):
        walk_cmd = make_walk_cmd(ex["commands"])
        result = measure_pair(ex["commands"], walk_cmd, group_label, idx, len(examples))
        if result is None:
            n_skip_no_positions += 1
            continue
        all_none = all(v is None for v in result.values())
        if all_none:
            n_skip_no_slots += 1
            continue
        for (_, _, label) in ALL_TARGET_HEADS:
            if result.get(label) is not None:
                for key, val in result[label].items():
                    head_storage[label][group_label][key].append(val)
        n_valid += 1
    return n_valid

print(f"\n  --- Fail group ---")
n_valid_fail = run_group(sub_walk_examples, "fail")
print(f"\n  --- Success group ---")
n_valid_success = run_group(success_examples, "success")

print(f"\n  Valid fail examples:    {n_valid_fail}")
print(f"  Valid success examples: {n_valid_success}")
if n_skip_no_positions:
    print(f"  Skipped (no enc positions found): {n_skip_no_positions}")
if n_skip_no_slots:
    print(f"  Skipped (no aligned slots): {n_skip_no_slots}")

# ── Phase 6: Aggregate and evaluate ───────────────────────────────────────────
print("\n" + "="*70)
print("  PHASE 6 — AGGREGATE + HYPOTHESIS EVALUATION")
print("="*70)

EPSILON = 1e-9

def group_mean(label, group, key):
    vals = head_storage[label][group][key]
    return float(np.mean(vals)) if vals else float("nan")

# Print M1/M2 attention routing table
print(f"\n  M1/M2 — Attention Routing (mean attn weight to encoder positions)")
print(f"  {'Head':<8} {'Fail→Jump':>12} {'Fail→Around':>12} "
      f"{'Succ→Jump':>12} {'Succ→Around':>12}")
for (_, _, label) in ALL_TARGET_HEADS:
    fj = group_mean(label, "fail",    "attn_jump")
    fa = group_mean(label, "fail",    "attn_around")
    sj = group_mean(label, "success", "attn_jump")
    sa = group_mean(label, "success", "attn_around")
    print(f"  {label:<8} {fj:>12.4f} {fa:>12.4f} {sj:>12.4f} {sa:>12.4f}")

# Print M3 cosine attribution table
print(f"\n  M3 — Cosine Attribution (cos toward I_WALK / I_JUMP divergence embeddings)")
print(f"  {'Head':<8} {'Fw(fail)':>10} {'Fj(fail)':>10} "
      f"{'Fw(succ)':>10} {'Fj(succ)':>10} "
      f"{'diff(fail)':>12} {'diff(succ)':>12}")
for (_, _, label) in ALL_TARGET_HEADS:
    fw_f = group_mean(label, "fail",    "cos_walk")
    fj_f = group_mean(label, "fail",    "cos_jump")
    fw_s = group_mean(label, "success", "cos_walk")
    fj_s = group_mean(label, "success", "cos_jump")
    diff_f = fw_f - fj_f
    diff_s = fw_s - fj_s
    print(f"  {label:<8} {fw_f:>10.4f} {fj_f:>10.4f} "
          f"{fw_s:>10.4f} {fj_s:>10.4f} "
          f"{diff_f:>12.4f} {diff_s:>12.4f}")

# Print M4 Born filter defect (should match S-055 per-head values)
print(f"\n  M4 — Born Filter Defect (L2 between head contributions, jump vs walk cmd)")
print(f"  {'Head':<8} {'Fail':>10} {'Success':>10} {'Ratio':>10}")
for (_, _, label) in ALL_TARGET_HEADS:
    fd = group_mean(label, "fail",    "defect")
    sd = group_mean(label, "success", "defect")
    ratio = sd / (fd + EPSILON) if not (np.isnan(fd) or np.isnan(sd)) else float("nan")
    print(f"  {label:<8} {fd:>10.4f} {sd:>10.4f} {ratio:>10.3f}×")

# ── Hypothesis evaluation ──────────────────────────────────────────────────────
print(f"\n  ── HYPOTHESIS VERDICTS ─────────────────────────────────────────")

l3h0_fail_attn_jump   = group_mean("L3H0", "fail",    "attn_jump")
l3h0_fail_attn_around = group_mean("L3H0", "fail",    "attn_around")
l3h0_fail_cos_walk    = group_mean("L3H0", "fail",    "cos_walk")
l3h0_fail_cos_jump    = group_mean("L3H0", "fail",    "cos_jump")
l3h0_succ_cos_walk    = group_mean("L3H0", "success", "cos_walk")
l3h0_succ_cos_jump    = group_mean("L3H0", "success", "cos_jump")

# H1: L3H0 is a jump-attending head (attn_to_jump > 0.3 in fail group)
h1_pass = (l3h0_fail_attn_jump > 0.3)
print(f"\n  H1 (L3H0 attn_to_jump > 0.3 in fail group):")
print(f"       attn_to_jump_fail = {l3h0_fail_attn_jump:.4f}: {'PASS' if h1_pass else 'FAIL'}")

# H2: value substitution signature in cosine space
# (cos_walk_fail - cos_jump_fail) > (cos_walk_success - cos_jump_success)
diff_fail    = l3h0_fail_cos_walk - l3h0_fail_cos_jump
diff_success = l3h0_succ_cos_walk - l3h0_succ_cos_jump
h2_pass = diff_fail > diff_success
print(f"\n  H2 (L3H0 value substitution signature):")
print(f"       diff_fail    = cos_walk({l3h0_fail_cos_walk:.4f}) - cos_jump({l3h0_fail_cos_jump:.4f}) = {diff_fail:.4f}")
print(f"       diff_success = cos_walk({l3h0_succ_cos_walk:.4f}) - cos_jump({l3h0_succ_cos_jump:.4f}) = {diff_success:.4f}")
print(f"       diff_fail > diff_success: {'PASS' if h2_pass else 'FAIL'}")

# H3: L3H0 attends more to jump than around in fail group
h3_pass = l3h0_fail_attn_jump > l3h0_fail_attn_around
print(f"\n  H3 (L3H0 attn_to_jump > attn_to_around in fail group):")
print(f"       attn_to_jump={l3h0_fail_attn_jump:.4f}, attn_to_around={l3h0_fail_attn_around:.4f}: "
      f"{'PASS' if h3_pass else 'FAIL'}")

# H4: per-example consistency — ≥75% of success examples above median fail defect
fail_defects    = head_storage["L3H0"]["fail"]["defect"]
success_defects = head_storage["L3H0"]["success"]["defect"]
if fail_defects and success_defects:
    median_fail = float(np.median(fail_defects))
    pct_above   = float(np.mean([d > median_fail for d in success_defects]))
    h4_pass     = pct_above >= 0.75
    print(f"\n  H4 (≥75% of success examples above median fail defect):")
    print(f"       median_fail_defect = {median_fail:.4f}")
    print(f"       fraction_above     = {pct_above:.3f} ({int(pct_above*len(success_defects))}/{len(success_defects)}): "
          f"{'PASS' if h4_pass else 'FAIL'}")
else:
    h4_pass = None
    pct_above = float("nan")
    median_fail = float("nan")
    print(f"\n  H4: insufficient data")

# Kill conditions
# K1: attn_to_jump < 0.1 for L3H0
k1_fires = l3h0_fail_attn_jump < 0.1
# K2: |cos_walk| < 0.05 AND |cos_jump| < 0.05 in fail group
k2_fires = (abs(l3h0_fail_cos_walk) < 0.05 and abs(l3h0_fail_cos_jump) < 0.05)

print(f"\n  K1 fires (L3H0 attn_to_jump < 0.1 — not jump-attending): {'YES' if k1_fires else 'NO'}")
print(f"  K2 fires (L3H0 cosine to vocab < 0.05 — orthogonal to vocabulary): {'YES' if k2_fires else 'NO'}")

# Per-example L3H0 distribution
print(f"\n  L3H0 per-example defect distribution:")
if fail_defects:
    print(f"    Fail:    n={len(fail_defects)}, mean={np.mean(fail_defects):.4f}, "
          f"median={np.median(fail_defects):.4f}, std={np.std(fail_defects):.4f}, "
          f"[p25={np.percentile(fail_defects,25):.4f}, p75={np.percentile(fail_defects,75):.4f}]")
if success_defects:
    print(f"    Success: n={len(success_defects)}, mean={np.mean(success_defects):.4f}, "
          f"median={np.median(success_defects):.4f}, std={np.std(success_defects):.4f}, "
          f"[p25={np.percentile(success_defects,25):.4f}, p75={np.percentile(success_defects,75):.4f}]")

# ── Save results ──────────────────────────────────────────────────────────────────────
head_summary = {}
for (_, _, label) in ALL_TARGET_HEADS:
    head_summary[label] = {
        "fail_attn_jump":   group_mean(label, "fail",    "attn_jump"),
        "fail_attn_around": group_mean(label, "fail",    "attn_around"),
        "success_attn_jump":   group_mean(label, "success", "attn_jump"),
        "success_attn_around": group_mean(label, "success", "attn_around"),
        "fail_cos_walk":    group_mean(label, "fail",    "cos_walk"),
        "fail_cos_jump":    group_mean(label, "fail",    "cos_jump"),
        "success_cos_walk": group_mean(label, "success", "cos_walk"),
        "success_cos_jump": group_mean(label, "success", "cos_jump"),
        "fail_defect":      group_mean(label, "fail",    "defect"),
        "success_defect":   group_mean(label, "success", "defect"),
    }
    fd = head_summary[label]["fail_defect"]
    sd = head_summary[label]["success_defect"]
    head_summary[label]["defect_ratio"] = (
        sd / (fd + EPSILON) if not (np.isnan(fd) or np.isnan(sd)) else float("nan")
    )

output = {
    "script_id":      SCRIPT_ID,
    "n_fail_valid":   int(n_valid_fail),
    "n_success_valid": int(n_valid_success),
    "head_results":   head_summary,
    "l3h0_per_example": {
        "fail_defects":    [float(x) for x in fail_defects],
        "success_defects": [float(x) for x in success_defects],
        "fail_median":     float(median_fail) if not np.isnan(median_fail) else None,
        "pct_success_above_fail_median": float(pct_above) if not np.isnan(pct_above) else None,
    },
    "hypotheses": {
        "H1": "PASS" if h1_pass else "FAIL",
        "H2": "PASS" if h2_pass else "FAIL",
        "H3": "PASS" if h3_pass else "FAIL",
        "H4": ("PASS" if h4_pass else "FAIL") if h4_pass is not None else "INSUFFICIENT_DATA",
        "K1": "FIRES" if k1_fires else "CLEAR",
        "K2": "FIRES" if k2_fires else "CLEAR",
    },
    "key_values": {
        "l3h0_fail_attn_jump":   float(l3h0_fail_attn_jump),
        "l3h0_fail_attn_around": float(l3h0_fail_attn_around),
        "l3h0_diff_fail":        float(diff_fail),
        "l3h0_diff_success":     float(diff_success),
        "l3h0_pct_above_median": float(pct_above) if not np.isnan(pct_above) else None,
    },
}

with open("056_results.json", "w") as f:
    json.dump(output, f, indent=2)
save_to_drive("056_results.json")

print(f"\n  Output: 056_results.json")
print(f"  {'='*70}")
print(f"  S-056 complete.")
print(f"  {'='*70}")
print(f"\n  # SEALED — do not re-run.")
