#!/usr/bin/env python3
"""
S-055_REGIME_PROBE.py — Born Filter Layer Profile in T5-small Decoder
Applied Categorical Physics Workbench
Troy Teno | May 2026 | Open Access

Pre-registered hypotheses:
  workbench/proposals/S-055_REGIME_PROBE_PROPOSAL.md
DO NOT modify hypotheses after seeing results.

─────────────────────────────────────────────────────────────────────
SETUP (run once in Colab or terminal):
  pip install torch transformers datasets numpy scipy sentencepiece -q

BACKGROUND:
  G-054 mapped the full three-regime phase diagram for Born filter
  amplification:
    Regime 1 (slow-collapse): gradual decay, late modest peak (~2×)
    Regime 2 (sharp-collapse): mid-layer peak at specific depth (2–867×)
    Regime 3 (no-collapse): monotonic decay from layer 1
  S-051 found T5-small's layer 5 as the amplification layer. G-054
  predicts this is the sharp-collapse regime completing at layer 5.

DESIGN:
  Morphism: substitute "jump" → "walk" in encoder input.
  Measurement: per-layer L2 defect ‖h^l(jump) − h^l(walk)‖ at
  action slot divergence-point steps.
  Groups: fail (30 substituted_walk cases) and success (25 has_around
  correct cases), same as S-050b/S-051.

  H1: fail-case ratio profile peaks at layer 4 or 5 (not monotonic)
  H2: peak ratio (success/fail) > 2.0
  H3: jump-attending heads (L4H6, L5H2, L5H5) show larger defect in
      success than fail at their native layer

  K1: monotonic decay → no-collapse regime (Regime 3)
  K2: peak ratio > 50× → near super-amplification zone
─────────────────────────────────────────────────────────────────────
"""

import numpy as np
import json
import os
import random
from collections import defaultdict

SCRIPT_ID = "S-055_REGIME_PROBE_V0.1.0"
SEED = 42
np.random.seed(SEED)
random.seed(SEED)

CHECKPOINT_DIR = "043_t5_scan_checkpoint"
MAX_INPUT_LEN  = 50
MAX_TARGET_LEN = 100
N_FAIL         = 30
N_SUCCESS      = 25

# Jump-attending heads from S-051 (0-based layer block, 0-based head)
# L4H6 = block[3] head 6; L5H2 = block[4] head 2; L5H5 = block[4] head 5
JUMP_HEADS = [
    (3, 6, "L4H6"),
    (4, 2, "L5H2"),
    (4, 5, "L5H5"),
]

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
print(f"  Born filter regime probe: jump→walk morphism across decoder layers")
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

# ── Phase 2: Token IDs and architecture ───────────────────────────────────────
print("\n" + "="*70)
print("  PHASE 2 — ACTION TOKEN IDs AND ARCHITECTURE")
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
    """Substitute 'jump' → 'walk' token-by-token."""
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

# ── Phase 4: Divergence-point detection ───────────────────────────────────────
def find_div_points(gen_token_ids):
    """
    Find action slot divergence-point positions.
    Pattern: TURN_END, 27, 834, div_token (in seq_no_bos coordinates)
    Returns list of (step_idx, action_name) tuples.
    """
    points = []
    for i in range(3, len(gen_token_ids)):
        if (gen_token_ids[i]   in DIV_TOKEN_SET and
            gen_token_ids[i-1] == 834 and
            gen_token_ids[i-2] == 27 and
            gen_token_ids[i-3] in TURN_END_IDS):
            action_name = id_to_action.get(gen_token_ids[i], "UNKNOWN")
            points.append((i, action_name))
    return points

# ── Phase 5: Per-head contribution ────────────────────────────────────────────
def compute_head_contrib(l_block, h_idx, step_cross_attns, enc_hidden_final):
    """
    Compute cross-attention head contribution to residual stream.
    l_block: 0-based decoder block index (0 = layer 1, ..., 5 = layer 6)
    h_idx:   0-based head index
    step_cross_attns: tuple of n_dec_layers tensors [1, n_heads, 1, T_enc]
    enc_hidden_final: [1, T_enc, d_model]
    Returns: head_contrib [d_model] on CPU
    """
    enc_attn = model.decoder.block[l_block].layer[1].EncDecAttention

    with torch.no_grad():
        V_all = enc_hidden_final @ enc_attn.v.weight.T  # [1, T_enc, n_heads*d_kv]

    attn_w           = step_cross_attns[l_block]          # [1, n_heads, 1, T_enc]
    head_attn        = attn_w[0, h_idx, 0, :]             # [T_enc]
    head_V           = V_all[0, :, h_idx*d_kv:(h_idx+1)*d_kv]  # [T_enc, d_kv]
    head_val         = (head_attn.unsqueeze(0) @ head_V).squeeze(0)  # [d_kv]
    head_o_slice     = enc_attn.o.weight[:, h_idx*d_kv:(h_idx+1)*d_kv]  # [d_model, d_kv]
    return (head_o_slice @ head_val).cpu()                # [d_model]

# ── Phase 6: Main measurement loop ────────────────────────────────────────────
print("\n" + "="*70)
print("  PHASE 6 — PAIRED FORWARD PASSES (jump vs walk, per layer)")
print("="*70)
print(f"\n  Running {len(sub_walk_examples)} fail + {len(success_examples)} success pairs")
print(f"  (Each example = 2 generate passes + per-head attribution)\n")

N_LAYERS_TOTAL = n_dec_layers + 1  # 7: embedding input + 6 decoder layers

def measure_pair(jump_cmd, walk_cmd, label, idx, total):
    """
    Run both commands through T5, align div_points, compute per-layer
    and per-head BC defects.
    Returns: (layer_defects[7], head_defects[N_LAYERS, N_HEADS])
    or None if no aligned slots.
    """
    if idx % 5 == 0:
        print(f"  [{label}] {idx+1}/{total} ...")

    # ── Tokenize both commands ────────────────────────────────────────
    def encode(cmd):
        return tokenizer(cmd, truncation=True, max_length=MAX_INPUT_LEN,
                         return_tensors="pt").to(device)

    inp_j = encode(jump_cmd)
    inp_w = encode(walk_cmd)

    # ── Generate both (with hidden states + attentions) ───────────────
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

    # ── Encoder final hidden states for V computation ─────────────────
    enc_h_j = out_j.encoder_hidden_states[-1]   # [1, T_enc_j, d_model]
    enc_h_w = out_w.encoder_hidden_states[-1]   # [1, T_enc_w, d_model]

    # ── Find div points in both outputs ──────────────────────────────
    seq_j = out_j.sequences[0].tolist()[1:]   # strip BOS
    seq_w = out_w.sequences[0].tolist()[1:]

    pts_j = find_div_points(seq_j)
    pts_w = find_div_points(seq_w)

    n_slots = min(len(pts_j), len(pts_w))
    if n_slots == 0:
        return None

    n_steps_j = len(out_j.decoder_hidden_states)
    n_steps_w = len(out_w.decoder_hidden_states)

    slot_layer_defects = []   # [n_slots, N_LAYERS_TOTAL]
    slot_head_defects  = []   # [n_slots, n_dec_layers, n_heads]

    for k in range(n_slots):
        s_j = pts_j[k][0]
        s_w = pts_w[k][0]

        if s_j >= n_steps_j or s_w >= n_steps_w:
            continue

        # ── Per-layer full-state defect ───────────────────────────────
        # out.decoder_hidden_states[step] = tuple of N_LAYERS_TOTAL tensors [1,1,d_model]
        layer_defs = []
        for l in range(N_LAYERS_TOTAL):
            h_j = out_j.decoder_hidden_states[s_j][l][0, 0, :].cpu()
            h_w = out_w.decoder_hidden_states[s_w][l][0, 0, :].cpu()
            layer_defs.append((h_j - h_w).norm().item())
        slot_layer_defects.append(layer_defs)

        # ── Per-head cross-attention defect ───────────────────────────
        # out.cross_attentions[step] = tuple of n_dec_layers tensors [1,n_heads,1,T_enc]
        ca_j = out_j.cross_attentions[s_j]
        ca_w = out_w.cross_attentions[s_w]

        head_defs = np.zeros((n_dec_layers, n_heads))
        for l_block in range(n_dec_layers):
            for h_idx in range(n_heads):
                c_j = compute_head_contrib(l_block, h_idx, ca_j, enc_h_j)
                c_w = compute_head_contrib(l_block, h_idx, ca_w, enc_h_w)
                head_defs[l_block, h_idx] = (c_j - c_w).norm().item()
        slot_head_defects.append(head_defs)

    if not slot_layer_defects:
        return None

    return (
        np.mean(slot_layer_defects, axis=0),   # [N_LAYERS_TOTAL]
        np.mean(slot_head_defects,  axis=0),   # [n_dec_layers, n_heads]
    )

# Run fail group
print(f"\n  --- Fail group ---")
fail_layer_defects = []
fail_head_defects  = []

for idx, ex in enumerate(sub_walk_examples):
    walk_cmd = make_walk_cmd(ex["commands"])
    result   = measure_pair(ex["commands"], walk_cmd, "fail", idx, len(sub_walk_examples))
    if result is not None:
        fail_layer_defects.append(result[0])
        fail_head_defects.append(result[1])

# Run success group
print(f"\n  --- Success group ---")
success_layer_defects = []
success_head_defects  = []

for idx, ex in enumerate(success_examples):
    walk_cmd = make_walk_cmd(ex["commands"])
    result   = measure_pair(ex["commands"], walk_cmd, "success", idx, len(success_examples))
    if result is not None:
        success_layer_defects.append(result[0])
        success_head_defects.append(result[1])

print(f"\n  Valid fail pairs:    {len(fail_layer_defects)}")
print(f"  Valid success pairs: {len(success_layer_defects)}")

# ── Phase 7: Aggregate and evaluate ───────────────────────────────────────────
print("\n" + "="*70)
print("  PHASE 7 — AGGREGATE + HYPOTHESIS EVALUATION")
print("="*70)

mean_fail_l    = np.mean(fail_layer_defects,    axis=0)  # [N_LAYERS_TOTAL]
mean_success_l = np.mean(success_layer_defects, axis=0)
mean_fail_h    = np.mean(fail_head_defects,     axis=0)  # [n_dec_layers, n_heads]
mean_success_h = np.mean(success_head_defects,  axis=0)

# Ratio profile: success / fail at each layer
EPSILON = 1e-9
ratio_l = np.where(mean_fail_l > EPSILON,
                   mean_success_l / mean_fail_l,
                   np.zeros_like(mean_fail_l))

print(f"\n  Per-layer BC defect (morphism: jump→walk, L2 distance)")
print(f"  {'Layer':<8} {'Fail':>10} {'Success':>10} {'Ratio':>10}")
for l in range(N_LAYERS_TOTAL):
    lbl = "Embed" if l == 0 else f"L{l}"
    print(f"  {lbl:<8} {mean_fail_l[l]:>10.4f} {mean_success_l[l]:>10.4f} {ratio_l[l]:>10.3f}×")

# ── Regime identification ──────────────────────────────────────────────────────
# Exclude embedding layer (l=0) from peak search — focus on decoder layers (l=1..6)
decoder_ratio = ratio_l[1:]   # [6] corresponding to decoder layers 1..6

peak_dec_layer_0idx = int(np.argmax(decoder_ratio))  # 0-indexed in decoder_ratio
peak_dec_layer      = peak_dec_layer_0idx + 1         # 1-indexed (1..6)
peak_ratio          = float(decoder_ratio[peak_dec_layer_0idx])

# Monotonic check: is decoder_ratio non-increasing?
is_monotonic = all(decoder_ratio[i] >= decoder_ratio[i+1]
                   for i in range(len(decoder_ratio)-1))

# Post-peak sharpness: fraction of total post-peak drop that occurs in one step.
# Sharp-collapse (Regime 2): large drop in one step → sharpness > 0.5
# Slow-collapse (Regime 1):  gradual decay across layers → sharpness < 0.3
if peak_dec_layer_0idx < len(decoder_ratio) - 1:
    drop_first_step = float(decoder_ratio[peak_dec_layer_0idx] -
                            decoder_ratio[peak_dec_layer_0idx + 1])
    post_peak_min   = float(np.min(decoder_ratio[peak_dec_layer_0idx:]))
    drop_total      = float(decoder_ratio[peak_dec_layer_0idx]) - post_peak_min
    post_peak_sharpness = drop_first_step / drop_total if drop_total > EPSILON else float("nan")
else:
    post_peak_sharpness = float("nan")  # peak is at final layer

print(f"\n  Peak ratio layer:      L{peak_dec_layer}")
print(f"  Peak ratio:            {peak_ratio:.3f}×")
print(f"  Monotonic decay:       {is_monotonic}")
print(f"  Post-peak sharpness:   {post_peak_sharpness:.3f}"
      f"  (>0.5 = sharp-collapse; <0.3 = slow-collapse)"
      if not np.isnan(post_peak_sharpness) else "  Post-peak sharpness: n/a (peak at final layer)")

# ── Hypothesis evaluation ──────────────────────────────────────────────────────

# H1: peak at layer 4 or 5 (NOT monotonic, NOT peaked at L1 or L6)
h1_peak_at_mid = (peak_dec_layer in (4, 5))
h1_not_monotonic = not is_monotonic
h1_pass = h1_not_monotonic and h1_peak_at_mid

# H2: peak ratio > 2.0
h2_pass = peak_ratio > 2.0

# H3: jump-attending heads show larger success defect than fail defect
print(f"\n  Jump-attending head defects (from S-051):")
print(f"  {'Head':<8} {'Fail':>10} {'Success':>10} {'Succ>Fail':>10}")
h3_results = {}
for (l_block, h_idx, label) in JUMP_HEADS:
    fail_val    = float(mean_fail_h[l_block, h_idx])
    success_val = float(mean_success_h[l_block, h_idx])
    confirmed   = success_val > fail_val
    h3_results[label] = {
        "l_block": l_block, "h_idx": h_idx,
        "fail_defect": fail_val, "success_defect": success_val,
        "confirmed": confirmed,
    }
    print(f"  {label:<8} {fail_val:>10.4f} {success_val:>10.4f} {'YES' if confirmed else 'NO':>10}")

h3_pass    = all(v["confirmed"] for v in h3_results.values())
h3_partial = any(v["confirmed"] for v in h3_results.values())

# Kill conditions
k1_fires = is_monotonic
k2_fires = peak_ratio > 50.0

# Regime label
if k1_fires:
    regime = "NO_COLLAPSE — Regime 3 (K1 fires)"
elif k2_fires:
    regime = f"NEAR_SUPER_AMPLIFICATION — Regime 2 boundary (K2 fires, ratio={peak_ratio:.1f}×)"
elif h1_pass and h2_pass:
    sharp_str = (f", sharpness={post_peak_sharpness:.2f}"
                 if not np.isnan(post_peak_sharpness) else "")
    regime = (f"SHARP_COLLAPSE — Regime 2 confirmed (peak L{peak_dec_layer}, "
              f"{peak_ratio:.2f}×{sharp_str})")
elif h1_not_monotonic:
    regime = (f"MID_LAYER_PEAK ratio {peak_ratio:.2f}× < 2.0 — "
              f"possible Regime 1 slow-collapse "
              f"(sharpness={post_peak_sharpness:.2f})"
              if not np.isnan(post_peak_sharpness) else
              f"MID_LAYER_PEAK ratio {peak_ratio:.2f}× < 2.0 — possible Regime 1 slow-collapse")
else:
    regime = f"AMBIGUOUS — peak L{peak_dec_layer}, ratio={peak_ratio:.2f}×, monotonic={is_monotonic}"

# ── Print summary ──────────────────────────────────────────────────────────────
print(f"\n  ── HYPOTHESIS VERDICTS ─────────────────────────────────────────")
print(f"  H1 (mid-layer peak at L4 or L5, not monotonic):")
print(f"       not_monotonic={h1_not_monotonic}, peak_layer=L{peak_dec_layer}: {'PASS' if h1_pass else 'FAIL'}")
print(f"  H2 (peak ratio > 2.0): ratio={peak_ratio:.3f}×: {'PASS' if h2_pass else 'FAIL'}")
print(f"  H3 (jump heads success > fail): {'PASS' if h3_pass else ('PARTIAL' if h3_partial else 'FAIL')}")
print(f"       {', '.join(k + ('✓' if v['confirmed'] else '✗') for k,v in h3_results.items())}")
print(f"\n  K1 fires (monotonic → no-collapse regime):  {'YES' if k1_fires else 'NO'}")
print(f"  K2 fires (ratio > 50× → super-amplification zone): {'YES' if k2_fires else 'NO'}")
print(f"\n  REGIME IDENTIFICATION: {regime}")

# ── Full head matrix: top positive ratios ─────────────────────────────────────
print(f"\n  Top-5 heads by ratio (success_defect / fail_defect):")
head_ratios = {}
for l_block in range(n_dec_layers):
    for h_idx in range(n_heads):
        fail_v    = float(mean_fail_h[l_block, h_idx])
        success_v = float(mean_success_h[l_block, h_idx])
        r = success_v / fail_v if fail_v > EPSILON else 0.0
        head_ratios[f"L{l_block+1}H{h_idx}"] = {
            "l_block": l_block, "h_idx": h_idx,
            "fail": fail_v, "success": success_v, "ratio": r,
        }

sorted_heads = sorted(head_ratios.items(), key=lambda kv: -kv[1]["ratio"])
print(f"  {'Head':<8} {'Fail':>10} {'Success':>10} {'Ratio':>10}")
for label, vals in sorted_heads[:5]:
    print(f"  {label:<8} {vals['fail']:>10.4f} {vals['success']:>10.4f} {vals['ratio']:>10.3f}×")

# ── Save ───────────────────────────────────────────────────────────────────────
output = {
    "script_id":           SCRIPT_ID,
    "n_fail_examples":     int(len(fail_layer_defects)),
    "n_success_examples":  int(len(success_layer_defects)),
    "fail_layer_defects":     mean_fail_l.tolist(),
    "success_layer_defects":  mean_success_l.tolist(),
    "ratio_profile":          ratio_l.tolist(),
    "peak_dec_layer":         int(peak_dec_layer),
    "peak_ratio":             float(peak_ratio),
    "is_monotonic":           bool(is_monotonic),
    "post_peak_sharpness":    float(post_peak_sharpness) if not np.isnan(post_peak_sharpness) else None,
    "regime":                 regime,
    "jump_head_results":      h3_results,
    "all_head_ratios":        {k: v for k, v in sorted_heads},
    "fail_head_matrix":       mean_fail_h.tolist(),
    "success_head_matrix":    mean_success_h.tolist(),
    "hypotheses": {
        "H1": "PASS" if h1_pass else "FAIL",
        "H2": "PASS" if h2_pass else "FAIL",
        "H3": "PASS" if h3_pass else ("PARTIAL" if h3_partial else "FAIL"),
        "K1": "FIRES" if k1_fires else "CLEAR",
        "K2": "FIRES" if k2_fires else "CLEAR",
    },
}

with open("055_results.json", "w") as f:
    json.dump(output, f, indent=2)
save_to_drive("055_results.json")

print(f"\n  Output: 055_results.json")
print(f"  {'='*70}")
print(f"  S-055 complete.")
print(f"  {'='*70}")
print(f"\n  # SEALED — do not re-run.")
