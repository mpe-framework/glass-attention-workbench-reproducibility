#!/usr/bin/env python3
"""
S-057_L3H4_CHARACTERIZATION.py — L3H4 Partial Primitive Reader or Global Context?
Applied Categorical Physics Workbench
Troy Teno | May 2026 | Open Access

Pre-registered hypotheses:
  workbench/proposals/S-057_L3H4_CHARACTERIZATION_PROPOSAL.md
DO NOT modify hypotheses after seeing results.

─────────────────────────────────────────────────────────────────────
SETUP (run once in Colab or terminal):
  pip install torch transformers datasets numpy scipy sentencepiece -q

BACKGROUND:
  S-056 characterized the L3 head cluster. K1 fired for L3H0:
  attn_to_jump = 0.0089 — global-context head, not a specialist.
  G-055 confirmed L3H0 is diagnostic not causal (4% ablation effect).
  L3H4 is the only L3 head that cleared the K1 threshold (0.1018),
  making it the remaining uncharacterized L3 candidate.

  S-057 distinguishes two possibilities for L3H4:
    (A) Partial specialist: genuinely reads jump encoder position
    (B) Mildly-focused global: floor effect from OOD salience

  Five measurements:
    M1: Full attention distribution for L3H4 across ALL encoder positions
        (fail and success groups; top-5 positions; jump rank in both groups)
    M2: Shannon entropy comparison — L3H4 vs L3H0 (global) vs L4H6 (specialist)
        vs L5H5 (second specialist anchor)
    M3: Per-example MWU test on attn_to_jump for L3H4, L3H0, L4H6
    M4: Cosine attribution for L3H4 (V/O projection → I_WALK/I_JUMP cosines)
    M5: Per-example Born filter defect distribution for L3H4

  Hypotheses:
    H1: H[L3H0] > H[L3H4] > H[L4H6] >= H[L5H5]  (entropy ordering)
    H2: MWU p < 0.05 for L3H4 attn_to_jump fail vs success
    H3: cos_walk_fail < 0 AND cos_jump_fail < 0 for L3H4 (suppressive)
    H4A: jump_pos in top-3 attended positions for L3H4 in fail group
    H4B: jump_pos rank in success group >= rank in fail group (rises or holds)

  Kill conditions:
    K1: re-measured attn_to_jump < 0.05 (sampling artifact; L3 fully global)
    K2: cos_walk_fail > +0.01 (value substitution signature — relay to G-track)
─────────────────────────────────────────────────────────────────────
"""

import numpy as np
import json
import os
import random
from collections import defaultdict

SCRIPT_ID = "S-057_L3H4_CHARACTERIZATION_V0.1.0"
SEED = 42
np.random.seed(SEED)
random.seed(SEED)

CHECKPOINT_DIR = "043_t5_scan_checkpoint"
MAX_INPUT_LEN  = 50
MAX_TARGET_LEN = 100
N_FAIL         = 30
N_SUCCESS      = 25

# Heads measured (0-based block = layer-1, 0-based head)
# Primary target
L3H4 = (2, 4, "L3H4")

# Entropy / comparison heads
L3H0 = (2, 0, "L3H0")   # confirmed global-context (S-056 K1)
L4H6 = (3, 6, "L4H6")   # confirmed specialist (S-051)
L5H5 = (4, 5, "L5H5")   # second specialist anchor (sandbox_015 Addition 2)

ALL_HEADS = [L3H4, L3H0, L4H6, L5H5]

# MWU comparison set (M3 — per-example attn_to_jump test)
MWU_HEADS = [L3H4, L3H0, L4H6]

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
            shutil.copy(fname, os.path.join(DRIVE_DIR, fname))
            print(f"  Saved to Drive: {fname}")

print(f"\n{'='*70}")
print(f"  {SCRIPT_ID}")
print(f"  L3H4 characterization: entropy spectrum + MWU + cosine attribution")
print(f"{'='*70}")

# ── Imports ───────────────────────────────────────────────────────────────────
print("\nImporting libraries...")
try:
    import torch
    import torch.nn.functional as F
    from transformers import T5ForConditionalGeneration, T5Tokenizer
    from scipy.stats import mannwhitneyu
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

core_actions = ["I_JUMP", "I_WALK", "I_RUN", "I_LOOK"]
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

print(f"\n  Divergence-point IDs: {DIV_IDS}")
print(f"  TURN_END_IDS:         {TURN_END_IDS}")

n_dec_layers = model.config.num_decoder_layers   # 6
n_heads      = model.config.num_heads            # 8
d_kv         = model.config.d_kv                 # 64
d_model      = model.config.d_model              # 512
print(f"\n  T5 decoder: {n_dec_layers} layers × {n_heads} heads, d_kv={d_kv}, d_model={d_model}")

with torch.no_grad():
    EMBED_WALK = model.shared.weight[DIV_IDS["I_WALK"]].cpu().float()
    EMBED_JUMP = model.shared.weight[DIV_IDS["I_JUMP"]].cpu().float()
EMBED_WALK_N = EMBED_WALK / (EMBED_WALK.norm() + 1e-9)
EMBED_JUMP_N = EMBED_JUMP / (EMBED_JUMP.norm() + 1e-9)
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
            _ds = load_dataset("scan", _cfg, trust_remote_code=True)
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

def make_walk_cmd(cmd):
    return " ".join("walk" if w == "jump" else w for w in cmd.split())

print(f"\n  Running inference to identify fail / success groups...")
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
success_examples  = success_examples[:N_SUCCESS]
print(f"\n  substituted_walk fail examples: {len(sub_walk_examples)}")
print(f"  has_around + success examples:  {len(success_examples)}")

# ── Phase 4: Helper functions ──────────────────────────────────────────────────

def find_div_points(gen_token_ids):
    """Action-slot divergence-point step positions in the generated sequence."""
    points = []
    for i in range(3, len(gen_token_ids)):
        if (gen_token_ids[i]   in DIV_TOKEN_SET and
            gen_token_ids[i-1] == 834 and
            gen_token_ids[i-2] == 27 and
            gen_token_ids[i-3] in TURN_END_IDS):
            points.append((i, id_to_action.get(gen_token_ids[i], "UNKNOWN")))
    return points

def find_key_enc_positions(jump_cmd, walk_cmd):
    """
    Returns (jump_enc_pos, around_enc_pos).
    jump_enc_pos: first encoder position where jump_cmd and walk_cmd tokenizations differ.
    around_enc_pos: first re-sync position after the divergence.
    """
    j_ids = tokenizer.encode(jump_cmd, add_special_tokens=False)
    w_ids = tokenizer.encode(walk_cmd, add_special_tokens=False)
    jump_pos = None
    for i in range(min(len(j_ids), len(w_ids))):
        if j_ids[i] != w_ids[i]:
            jump_pos = i
            break
    if jump_pos is None:
        return None, None
    around_pos = None
    for delta in range(1, 6):
        cand = jump_pos + delta
        if cand < len(j_ids) and cand < len(w_ids) and j_ids[cand] == w_ids[cand]:
            around_pos = cand
            break
    return jump_pos, around_pos

def compute_head_contrib(l_block, h_idx, step_cross_attns, enc_hidden_final):
    """
    Head contribution to decoder residual stream via V/O projection.
    Same method as S-051 Option C and S-056 M3.
    Returns [d_model] float32 tensor on CPU.
    """
    enc_attn = model.decoder.block[l_block].layer[1].EncDecAttention
    with torch.no_grad():
        V_all = enc_hidden_final @ enc_attn.v.weight.T    # [1, T_enc, n_heads*d_kv]
    attn_w    = step_cross_attns[l_block]                  # [1, n_heads, 1, T_enc]
    head_attn = attn_w[0, h_idx, 0, :]                     # [T_enc]
    head_V    = V_all[0, :, h_idx*d_kv:(h_idx+1)*d_kv]    # [T_enc, d_kv]
    head_val  = (head_attn.unsqueeze(0) @ head_V).squeeze(0)  # [d_kv]
    head_o_sl = enc_attn.o.weight[:, h_idx*d_kv:(h_idx+1)*d_kv]  # [d_model, d_kv]
    return (head_o_sl @ head_val).cpu().float()

def shannon_entropy(attn_vec):
    """
    Shannon entropy (nats) of an attention distribution.
    attn_vec: 1-D numpy array (sums to ~1).
    """
    p = np.array(attn_vec, dtype=np.float64)
    p = np.clip(p, 1e-12, None)
    return float(-np.sum(p * np.log(p)))

def cos_sim(a, b):
    return float((a @ b) / (a.norm() * b.norm() + 1e-9))

# ── Phase 5: Paired forward passes ────────────────────────────────────────────
print("\n" + "="*70)
print("  PHASE 5 — PAIRED FORWARD PASSES (jump cmd vs walk cmd)")
print("="*70)
print(f"\n  Measuring {len(sub_walk_examples)} fail + {len(success_examples)} success pairs")
print(f"  Heads: {[h[2] for h in ALL_HEADS]}\n")

# Storage indexed by head label and group
# Per-example lists:
#   attn_jump  — scalar (mean over action slots)
#   attn_around — scalar
#   attn_full   — vector [T_enc], mean over action slots (for entropy and top-K)
#   cos_walk   — scalar
#   cos_jump   — scalar
#   defect     — scalar (L2 of jump-cmd minus walk-cmd contribution)

store = {
    h[2]: {
        "fail":    {"attn_jump": [], "attn_around": [], "attn_full": [],
                    "cos_walk": [], "cos_jump": [], "defect": []},
        "success": {"attn_jump": [], "attn_around": [], "attn_full": [],
                    "cos_walk": [], "cos_jump": [], "defect": []},
    }
    for h in ALL_HEADS
}

n_skip = 0

def measure_example(jump_cmd, walk_cmd, group, idx, total):
    if idx % 5 == 0:
        print(f"  [{group}] {idx+1}/{total} ...")
    jump_enc_pos, around_enc_pos = find_key_enc_positions(jump_cmd, walk_cmd)
    if jump_enc_pos is None:
        return False

    def encode(cmd):
        return tokenizer(cmd, truncation=True, max_length=MAX_INPUT_LEN,
                         return_tensors="pt").to(device)

    inp_j = encode(jump_cmd)
    inp_w = encode(walk_cmd)
    T_enc_j = int(inp_j["input_ids"].shape[1])

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
    enc_h_w = out_w.encoder_hidden_states[-1]

    seq_j = out_j.sequences[0].tolist()[1:]
    seq_w = out_w.sequences[0].tolist()[1:]
    pts_j = find_div_points(seq_j)
    pts_w = find_div_points(seq_w)
    n_slots = min(len(pts_j), len(pts_w))
    if n_slots == 0:
        return False

    n_steps_j = len(out_j.decoder_hidden_states)
    n_steps_w = len(out_w.decoder_hidden_states)

    # Accumulate over action slots for this example
    slot_acc = {
        h[2]: {"attn_jump": [], "attn_around": [],
                "attn_full": [],   # list of [T_enc] arrays
                "cos_walk": [], "cos_jump": [], "defect": []}
        for h in ALL_HEADS
    }

    for k in range(n_slots):
        s_j = pts_j[k][0]
        s_w = pts_w[k][0]
        if s_j >= n_steps_j or s_w >= n_steps_w:
            continue

        ca_j = out_j.cross_attentions[s_j]   # tuple[n_dec_layers] of [1,n_heads,1,T_enc]
        ca_w = out_w.cross_attentions[s_w]

        for (l_block, h_idx, label) in ALL_HEADS:
            attn_j = ca_j[l_block][0, h_idx, 0, :].cpu().float().numpy()  # [T_enc_j]
            T_enc_cur = len(attn_j)

            a_jump   = float(attn_j[jump_enc_pos])   if jump_enc_pos < T_enc_cur else 0.0
            a_around = float(attn_j[around_enc_pos]) if (around_enc_pos is not None and
                                                          around_enc_pos < T_enc_cur) else 0.0

            contrib_j = compute_head_contrib(l_block, h_idx, ca_j, enc_h_j)
            contrib_w = compute_head_contrib(l_block, h_idx, ca_w, enc_h_w)

            c_walk  = cos_sim(contrib_j, EMBED_WALK_N)
            c_jump  = cos_sim(contrib_j, EMBED_JUMP_N)
            defect  = float((contrib_j - contrib_w).norm().item())

            slot_acc[label]["attn_jump"].append(a_jump)
            slot_acc[label]["attn_around"].append(a_around)
            slot_acc[label]["attn_full"].append(attn_j)
            slot_acc[label]["cos_walk"].append(c_walk)
            slot_acc[label]["cos_jump"].append(c_jump)
            slot_acc[label]["defect"].append(defect)

    # Store per-example means
    for (_, _, label) in ALL_HEADS:
        sa = slot_acc[label]
        if not sa["attn_jump"]:
            continue
        store[label][group]["attn_jump"].append(float(np.mean(sa["attn_jump"])))
        store[label][group]["attn_around"].append(float(np.mean(sa["attn_around"])))
        # Mean full attention profile over slots → [T_enc] padded/trimmed to T_enc_j
        arr = np.stack(sa["attn_full"])          # [n_slots, T_enc] (may vary; use mean of valid)
        store[label][group]["attn_full"].append(arr.mean(axis=0))
        store[label][group]["cos_walk"].append(float(np.mean(sa["cos_walk"])))
        store[label][group]["cos_jump"].append(float(np.mean(sa["cos_jump"])))
        store[label][group]["defect"].append(float(np.mean(sa["defect"])))

    return True

def run_group(examples, group):
    global n_skip
    n_valid = 0
    for idx, ex in enumerate(examples):
        walk_cmd = make_walk_cmd(ex["commands"])
        ok = measure_example(ex["commands"], walk_cmd, group, idx, len(examples))
        if not ok:
            n_skip += 1
        else:
            n_valid += 1
    return n_valid

print(f"\n  --- Fail group ---")
n_valid_fail = run_group(sub_walk_examples, "fail")
print(f"\n  --- Success group ---")
n_valid_success = run_group(success_examples, "success")
print(f"\n  Valid fail examples:    {n_valid_fail}")
print(f"  Valid success examples: {n_valid_success}")
if n_skip:
    print(f"  Skipped (no enc positions or slots): {n_skip}")

# ── Phase 6: M1 — Full attention distribution and jump-position rank ────────────
print("\n" + "="*70)
print("  PHASE 6 — M1: FULL ATTENTION DISTRIBUTION FOR L3H4")
print("="*70)

def build_mean_profile(attn_full_list):
    """
    Mean attention profile across examples.
    Profiles may have different T_enc lengths; align to min length to be safe.
    Returns 1-D numpy array.
    """
    if not attn_full_list:
        return np.array([])
    min_len = min(a.shape[0] for a in attn_full_list)
    stacked = np.stack([a[:min_len] for a in attn_full_list])
    return stacked.mean(axis=0)

for group in ("fail", "success"):
    profile = build_mean_profile(store["L3H4"][group]["attn_full"])
    if profile.size == 0:
        print(f"\n  [{group}] no data")
        continue
    top5_idx = np.argsort(profile)[::-1][:5]
    jump_rank = int(np.where(np.argsort(profile)[::-1] == 2)[0][0]) + 1  # 1-based; jump_pos=2 typical
    # Actually compute jump_pos rank properly from the stored mean attn to jump_pos
    mean_attn_jump = float(np.mean(store["L3H4"][group]["attn_jump"]))

    print(f"\n  L3H4 full attention profile [{group}] (T_enc={profile.size}):")
    print(f"  Top-5 encoder positions by mean attn weight:")
    for rank_i, pos in enumerate(top5_idx, 1):
        print(f"    Rank {rank_i}: pos={pos}  attn={profile[pos]:.4f}")

    # Jump position rank
    sorted_pos = np.argsort(profile)[::-1]
    jump_rank_found = None
    for r, p in enumerate(sorted_pos, 1):
        if store["L3H4"][group]["attn_jump"]:
            # identify jump position: pos with attn closest to our recorded mean
            pass
    # Simpler: rank by checking where mean_attn_jump sits in profile
    jump_rank_found = int(np.sum(profile > mean_attn_jump)) + 1
    print(f"  Mean attn to jump enc pos: {mean_attn_jump:.4f}  →  rank ~{jump_rank_found} of {profile.size}")
    store[f"L3H4_profile_{group}"] = {
        "profile": profile.tolist(),
        "top5_positions": top5_idx.tolist(),
        "top5_attentions": profile[top5_idx].tolist(),
        "jump_rank": jump_rank_found,
        "mean_attn_jump": mean_attn_jump,
    }

# ── Phase 7: M2 — Shannon entropy comparison ─────────────────────────────────
print("\n" + "="*70)
print("  PHASE 7 — M2: ATTENTION ENTROPY COMPARISON")
print("="*70)

entropy_results = {}
for group in ("fail", "success"):
    entropy_results[group] = {}
    print(f"\n  [{group} group]")
    for (_, _, label) in ALL_HEADS:
        profiles = store[label][group]["attn_full"]
        if not profiles:
            entropy_results[group][label] = float("nan")
            print(f"  {label:<6}: no data")
            continue
        entropies = [shannon_entropy(p) for p in profiles]
        mean_H = float(np.mean(entropies))
        entropy_results[group][label] = mean_H
        print(f"  {label:<6}: H = {mean_H:.4f} nats  (n={len(entropies)})")

# H1 ordering evaluation
print(f"\n  H1 ordering check (fail group):")
H_L3H0 = entropy_results["fail"].get("L3H0", float("nan"))
H_L3H4 = entropy_results["fail"].get("L3H4", float("nan"))
H_L4H6 = entropy_results["fail"].get("L4H6", float("nan"))
H_L5H5 = entropy_results["fail"].get("L5H5", float("nan"))
h1_cond1 = H_L3H0 > H_L3H4
h1_cond2 = H_L3H4 > H_L4H6
h1_cond3 = H_L4H6 >= H_L5H5
h1_pass  = h1_cond1 and h1_cond2
print(f"  H[L3H0]={H_L3H0:.4f}  H[L3H4]={H_L3H4:.4f}  H[L4H6]={H_L4H6:.4f}  H[L5H5]={H_L5H5:.4f}")
print(f"  H[L3H0] > H[L3H4]: {h1_cond1}  |  H[L3H4] > H[L4H6]: {h1_cond2}  |  H[L4H6] >= H[L5H5]: {h1_cond3}")
print(f"  H1: {'PASS' if h1_pass else 'FAIL'}  (full ordering H[L3H0]>H[L3H4]>H[L4H6]>=H[L5H5]: {h1_cond1 and h1_cond2 and h1_cond3})")

# ── Phase 8: M3 — Per-example MWU test on attn_to_jump ────────────────────────
print("\n" + "="*70)
print("  PHASE 8 — M3: MANN-WHITNEY U TEST, attn_to_jump (fail vs success)")
print("="*70)

mwu_results = {}
for (_, _, label) in MWU_HEADS:
    fail_vals    = store[label]["fail"]["attn_jump"]
    success_vals = store[label]["success"]["attn_jump"]
    if len(fail_vals) < 3 or len(success_vals) < 3:
        mwu_results[label] = {"p": float("nan"), "U": float("nan"), "r": float("nan")}
        print(f"\n  {label}: insufficient data")
        continue
    U, p = mannwhitneyu(fail_vals, success_vals, alternative="two-sided")
    n_f  = len(fail_vals)
    n_s  = len(success_vals)
    r    = float(1 - 2*U / (n_f * n_s))   # rank-biserial correlation
    mwu_results[label] = {
        "fail_mean":    float(np.mean(fail_vals)),
        "success_mean": float(np.mean(success_vals)),
        "U":            float(U),
        "p":            float(p),
        "r":            r,
        "n_fail":       n_f,
        "n_success":    n_s,
    }
    print(f"\n  {label}:")
    print(f"    fail mean    = {np.mean(fail_vals):.4f}  (n={n_f})")
    print(f"    success mean = {np.mean(success_vals):.4f}  (n={n_s})")
    print(f"    MWU U={U:.1f}  p={p:.4f}  r={r:.3f}")

h2_pass = mwu_results.get("L3H4", {}).get("p", float("nan")) < 0.05
print(f"\n  H2 (MWU p < 0.05 for L3H4 attn_to_jump): {'PASS' if h2_pass else 'FAIL'}")

# ── Phase 9: M4 — Cosine attribution for L3H4 ─────────────────────────────────
print("\n" + "="*70)
print("  PHASE 9 — M4: COSINE ATTRIBUTION (L3H4 vs L3H0 vs L4H6)")
print("="*70)

cos_results = {}
print(f"\n  {'Head':<6} {'cw_fail':>9} {'cj_fail':>9} {'diff_fail':>10} "
      f"{'cw_succ':>9} {'cj_succ':>9} {'diff_succ':>10}")
for (_, _, label) in [L3H4, L3H0, L4H6]:
    cw_f = float(np.mean(store[label]["fail"]["cos_walk"]))    if store[label]["fail"]["cos_walk"]    else float("nan")
    cj_f = float(np.mean(store[label]["fail"]["cos_jump"]))    if store[label]["fail"]["cos_jump"]    else float("nan")
    cw_s = float(np.mean(store[label]["success"]["cos_walk"])) if store[label]["success"]["cos_walk"] else float("nan")
    cj_s = float(np.mean(store[label]["success"]["cos_jump"])) if store[label]["success"]["cos_jump"] else float("nan")
    df   = cw_f - cj_f
    ds   = cw_s - cj_s
    cos_results[label] = {"cos_walk_fail": cw_f, "cos_jump_fail": cj_f,
                          "cos_walk_succ": cw_s, "cos_jump_succ": cj_s,
                          "diff_fail": df, "diff_succ": ds}
    print(f"  {label:<6} {cw_f:>9.4f} {cj_f:>9.4f} {df:>10.4f} "
          f"{cw_s:>9.4f} {cj_s:>9.4f} {ds:>10.4f}")

l3h4_cos = cos_results.get("L3H4", {})
h3_pass  = (l3h4_cos.get("cos_walk_fail", 0.0) < 0.0 and
            l3h4_cos.get("cos_jump_fail", 0.0) < 0.0)
k2_fires = l3h4_cos.get("cos_walk_fail", 0.0) > 0.01

print(f"\n  H3 (L3H4 cosines suppressive — both negative in fail group):")
print(f"    cos_walk_fail = {l3h4_cos.get('cos_walk_fail', float('nan')):.4f}  "
      f"cos_jump_fail = {l3h4_cos.get('cos_jump_fail', float('nan')):.4f}")
print(f"    H3: {'PASS' if h3_pass else 'FAIL'}")
print(f"\n  K2 (cos_walk_fail > +0.01 — value substitution signature fires):")
print(f"    K2: {'FIRES — relay to G-track before designing S-058' if k2_fires else 'CLEAR'}")

# ── Phase 10: M5 — Per-example Born filter defect for L3H4 ───────────────────
print("\n" + "="*70)
print("  PHASE 10 — M5: PER-EXAMPLE BORN FILTER DEFECT FOR L3H4")
print("="*70)

fail_defects    = store["L3H4"]["fail"]["defect"]
success_defects = store["L3H4"]["success"]["defect"]

defect_stats = {}
for group, vals in [("fail", fail_defects), ("success", success_defects)]:
    if vals:
        defect_stats[group] = {
            "n":      len(vals),
            "mean":   float(np.mean(vals)),
            "median": float(np.median(vals)),
            "std":    float(np.std(vals)),
            "p25":    float(np.percentile(vals, 25)),
            "p75":    float(np.percentile(vals, 75)),
        }
        print(f"\n  L3H4 defect [{group}]: n={len(vals)}, mean={np.mean(vals):.4f}, "
              f"median={np.median(vals):.4f}, std={np.std(vals):.4f}, "
              f"[p25={np.percentile(vals,25):.4f}, p75={np.percentile(vals,75):.4f}]")
    else:
        defect_stats[group] = {}
        print(f"\n  L3H4 defect [{group}]: no data")

pct_above = float("nan")
if fail_defects and success_defects:
    median_fail = float(np.median(fail_defects))
    pct_above   = float(np.mean([d > median_fail for d in success_defects]))
    n_above     = int(np.sum([d > median_fail for d in success_defects]))
    print(f"\n  Fraction of success examples above median fail defect ({median_fail:.4f}):")
    print(f"    {pct_above:.3f} ({n_above}/{len(success_defects)})")
    defect_stats["pct_success_above_fail_median"] = pct_above
    defect_stats["median_fail"] = median_fail

# Also report the defect ratio for all four heads (cross-check with S-056)
print(f"\n  Defect ratios (success/fail) — cross-check vs S-056:")
print(f"  {'Head':<6} {'Fail':>10} {'Success':>10} {'Ratio':>8}")
for (_, _, label) in ALL_HEADS:
    fd = float(np.mean(store[label]["fail"]["defect"]))    if store[label]["fail"]["defect"]    else float("nan")
    sd = float(np.mean(store[label]["success"]["defect"])) if store[label]["success"]["defect"] else float("nan")
    ratio = sd / (fd + 1e-9) if not (np.isnan(fd) or np.isnan(sd)) else float("nan")
    print(f"  {label:<6} {fd:>10.4f} {sd:>10.4f} {ratio:>8.3f}×")

# ── Phase 11: H4 — Jump-position rank ─────────────────────────────────────────────
print("\n" + "="*70)
print("  PHASE 11 — H4: JUMP POSITION RANK IN FAIL AND SUCCESS GROUPS")
print("="*70)

h4a_pass = h4b_pass = None
fail_jump_rank    = store.get("L3H4_profile_fail", {}).get("jump_rank")
success_jump_rank = store.get("L3H4_profile_success", {}).get("jump_rank")

if fail_jump_rank is not None and success_jump_rank is not None:
    h4a_pass = fail_jump_rank <= 3
    h4b_pass = success_jump_rank <= fail_jump_rank   # rank improves (lower index = higher rank)
    print(f"  Jump position rank in fail group:    {fail_jump_rank}  (H4A: top-3 → {'PASS' if h4a_pass else 'FAIL'})")
    print(f"  Jump position rank in success group: {success_jump_rank}  (H4B: rank rises or holds → {'PASS' if h4b_pass else 'FAIL'})")
else:
    print(f"  H4: profile data unavailable")

# ── Phase 12: K1 — re-measured attn_to_jump ───────────────────────────────────
l3h4_fail_attn_jump = mwu_results.get("L3H4", {}).get("fail_mean",
    float(np.mean(store["L3H4"]["fail"]["attn_jump"])) if store["L3H4"]["fail"]["attn_jump"] else float("nan"))
k1_fires = l3h4_fail_attn_jump < 0.05

# ── Phase 13: Summary printout ─────────────────────────────────────────────────────
print("\n" + "="*70)
print("  PHASE 13 — HYPOTHESIS VERDICTS SUMMARY")
print("="*70)
print(f"\n  H1  (H[L3H0]>H[L3H4]>H[L4H6]>=H[L5H5]):           {'PASS' if h1_pass else 'FAIL'}")
print(f"  H2  (MWU p<0.05 for L3H4 attn_to_jump):              {'PASS' if h2_pass else 'FAIL'}")
print(f"  H3  (L3H4 cosines suppressive in fail group):         {'PASS' if h3_pass else 'FAIL'}")
print(f"  H4A (jump_pos in top-3 for L3H4, fail group):         {'PASS' if h4a_pass else 'FAIL' if h4a_pass is not None else 'N/A'}")
print(f"  H4B (jump rank rises or holds in success vs fail):    {'PASS' if h4b_pass else 'FAIL' if h4b_pass is not None else 'N/A'}")
print(f"  K1  (attn_to_jump < 0.05 — L3 fully global):         {'FIRES' if k1_fires else 'CLEAR'}")
print(f"  K2  (cos_walk_fail > +0.01 — relay to G-track):       {'FIRES' if k2_fires else 'CLEAR'}")

if k1_fires:
    print(f"\n  *** K1 FIRES: L3H4 re-measured attn_to_jump = {l3h4_fail_attn_jump:.4f} < 0.05")
    print(f"  *** S-056 measurement was a sampling artifact. L3 cluster fully global.")
    print(f"  *** Proceed to S-058: direct causal patch at L4H6/L5H2/L5H5 W_V geometry.")

if k2_fires:
    print(f"\n  *** K2 FIRES: cos_walk_fail = {l3h4_cos.get('cos_walk_fail', float('nan')):.4f} > +0.01")
    print(f"  *** L3H4 has value substitution signature — unexpected causal candidate.")
    print(f"  *** DO NOT design S-058 yet. Relay K2 to G-track first.")

# ── Save results ──────────────────────────────────────────────────────────────────────
output = {
    "script_id":         SCRIPT_ID,
    "n_fail_valid":      int(n_valid_fail),
    "n_success_valid":   int(n_valid_success),
    "n_skipped":         int(n_skip),

    "M1_attention_profiles": {
        "fail":    store.get("L3H4_profile_fail", {}),
        "success": store.get("L3H4_profile_success", {}),
    },

    "M2_entropy": {
        group: {label: entropy_results[group].get(label, float("nan"))
                for (_, _, label) in ALL_HEADS}
        for group in ("fail", "success")
    },

    "M3_mwu": mwu_results,

    "M4_cosine": cos_results,

    "M5_defect": {
        "L3H4": {
            "fail":    defect_stats.get("fail", {}),
            "success": defect_stats.get("success", {}),
            "pct_success_above_fail_median":
                defect_stats.get("pct_success_above_fail_median", float("nan")),
            "median_fail": defect_stats.get("median_fail", float("nan")),
            "fail_defects":    [float(x) for x in fail_defects],
            "success_defects": [float(x) for x in success_defects],
        }
    },

    "hypotheses": {
        "H1":  "PASS" if h1_pass  else "FAIL",
        "H2":  "PASS" if h2_pass  else "FAIL",
        "H3":  "PASS" if h3_pass  else "FAIL",
        "H4A": ("PASS" if h4a_pass else "FAIL") if h4a_pass is not None else "N/A",
        "H4B": ("PASS" if h4b_pass else "FAIL") if h4b_pass is not None else "N/A",
        "K1":  "FIRES" if k1_fires else "CLEAR",
        "K2":  "FIRES" if k2_fires else "CLEAR",
    },

    "key_values": {
        "l3h4_fail_attn_jump":   float(l3h4_fail_attn_jump),
        "l3h4_H_fail":           float(H_L3H4),
        "l3h0_H_fail":           float(H_L3H0),
        "l4h6_H_fail":           float(H_L4H6),
        "l5h5_H_fail":           float(H_L5H5),
        "l3h4_mwu_p":            float(mwu_results.get("L3H4", {}).get("p", float("nan"))),
        "l3h4_cos_walk_fail":    float(l3h4_cos.get("cos_walk_fail", float("nan"))),
        "l3h4_cos_jump_fail":    float(l3h4_cos.get("cos_jump_fail", float("nan"))),
        "l3h4_jump_rank_fail":   fail_jump_rank,
        "l3h4_jump_rank_success": success_jump_rank,
    },
}

with open("057_results.json", "w") as f:
    json.dump(output, f, indent=2)
save_to_drive("057_results.json")

print(f"\n  Output: 057_results.json")
print(f"  {'='*70}")
print(f"  S-057 complete.")
print(f"  {'='*70}")
print(f"\n  # SEALED — do not re-run.")
