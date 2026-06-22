#!/usr/bin/env python3
"""
S-050b_ACTION_SLOT_GEOMETRY_FIXED.py — Action-Slot Logit and Embedding Geometry
Applied Categorical Physics Workbench
Troy Teno | May 2026 | Open Access

Pre-registered hypotheses:
  workbench/proposals/S-050_ACTION_SLOT_GEOMETRY_PROPOSAL.md
DO NOT modify hypotheses after seeing results.

─────────────────────────────────────────────────────────────────────
BUG FIX FROM S-050:
  S-050 used ids[0] as proxy for all action tokens. Every action token
  (I_JUMP, I_WALK, I_RUN, I_LOOK, I_TURN_LEFT, I_TURN_RIGHT) is
  tokenized by T5 as a 4-subword sequence, ALL sharing the same first
  subword (token 27, ▁I). Using ids[0]=27 for everything meant:
    - All "action token embeddings" pointed to the same vector
    - TURN_IDS = {27} matched every occurrence of token 27 in the
      sequence → 492 phantom action slots instead of ~120
    - All pairwise distances = 0.0; all probabilities = 0.0

FIX: Use the DIVERGENCE-POINT token (index 2 in the 4-subword sequence):
    I_JUMP  → [27, 834, 683, 28468]   divergence token: 683
    I_WALK  → [27, 834, 12054, 22527]  divergence token: 12054
    I_RUN   → [27, 834, 448, 7443]    divergence token: 448
    I_LOOK  → [27, 834, 5017, 13458]  divergence token: 5017

  For TURN end detection, use the LAST subword of each TURN token
  (action slots begin with [27, 834, div_token] right after a complete
  TURN sequence):
    I_TURN_LEFT  last subword: 6245
    I_TURN_RIGHT last subword: 27262

WHAT THIS SCRIPT DOES:
  S-050 training-frequency table confirmed: I_JUMP appears 1,467 times
  (0.79%) vs I_WALK/I_RUN/I_LOOK each at 25,350 (13.58%). Training
  frequency rules out I_JUMP but CANNOT explain I_WALK vs I_RUN vs
  I_LOOK (all identical). The only remaining candidate is geometry:

    Troy's hypothesis: h_action at the divergence step is geometrically
    closer to embed(I_WALK divergence token) than to embed(I_RUN) or
    embed(I_LOOK) in the decoder output embedding space.

  Three instruments at the divergence point of each action slot:
    1. P(div_token) for each action type — the actual softmax decision
    2. Euclidean distance d(h_div, embed(div_token)) for each action
    3. Cosine similarity cos(h_div, embed(div_token)) for each action
    + Cross-attention weights (K1 confirmatory proof)

  Groups:
    Fail:    substituted_walk examples (I_WALK generated instead of I_JUMP)
    Success: has_around=1 + correct (I_JUMP correctly generated in cycles)
─────────────────────────────────────────────────────────────────────
SETUP (run once in Colab or terminal):
  pip install torch transformers datasets numpy scipy sentencepiece -q
"""

import numpy as np
import json
import os
import random
from collections import defaultdict

SCRIPT_ID = "S-050b_ACTION_SLOT_GEOMETRY_FIXED_V0.1.0"
SEED = 42
np.random.seed(SEED)
random.seed(SEED)

CHECKPOINT_DIR = "043_t5_scan_checkpoint"
MAX_INPUT_LEN  = 50
MAX_TARGET_LEN = 100
N_FAIL         = 30    # substituted_walk examples
N_SUCCESS      = 25    # has_around=1 + correct examples

# ── Google Drive ─────────────────────────────────────────────────────────────────────────────────────────
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
print(f"  Divergence-point geometry: I_WALK vs I_RUN vs I_LOOK at equal freq")
print(f"{'='*70}")

# ── Imports ─────────────────────────────────────────────────────────────────────────────────
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

# ── Phase 1: Load checkpoint ─────────────────────────────────────────────────────────────────────────────
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

# ── Phase 2: Action token tokenization and divergence points ──────────────────────────────────────────────────
print("\n" + "="*70)
print("  PHASE 2 — VERIFY TOKENIZATION & GET DIVERGENCE-POINT IDS")
print("="*70)

ALL_ACTION_NAMES = ["I_JUMP", "I_WALK", "I_RUN", "I_LOOK",
                    "I_TURN_LEFT", "I_TURN_RIGHT"]
core_actions = ["I_JUMP", "I_WALK", "I_RUN", "I_LOOK"]

print(f"\n  Full tokenization of action tokens:")
raw_ids = {}
for name in ALL_ACTION_NAMES:
    ids = tokenizer.encode(name, add_special_tokens=False)
    raw_ids[name] = ids
    print(f"  {name:<15} → {ids}")

# Verify shared prefix [27, 834] for core action tokens
print(f"\n  Shared prefix verification (expect [27, 834] for core actions):")
SHARED_PREFIX = [27, 834]
for name in core_actions:
    ids = raw_ids[name]
    prefix_ok = (len(ids) >= 3 and ids[0] == SHARED_PREFIX[0]
                 and ids[1] == SHARED_PREFIX[1])
    print(f"  {name:<15} prefix={ids[:2]}  {'OK' if prefix_ok else 'UNEXPECTED!'}")
    if not prefix_ok:
        print(f"  WARNING: {name} does not have expected prefix {SHARED_PREFIX}.")
        print(f"           Divergence-point logic may be incorrect. Inspect manually.")

# Divergence-point token IDs (index 2 in each action's subword sequence)
DIV_IDS = {}
for name in core_actions:
    ids = raw_ids[name]
    DIV_IDS[name] = ids[2]

print(f"\n  Divergence-point token IDs (index 2 in subword sequence):")
for name, div_id in DIV_IDS.items():
    div_tok = tokenizer.convert_ids_to_tokens([div_id])[0]
    print(f"  {name:<15} divergence token_id={div_id}  ({div_tok!r})")

# TURN last-subword IDs for action-slot detection
# Action slots follow a complete I_TURN_LEFT or I_TURN_RIGHT token sequence.
# We detect them by the last subword of the TURN token immediately preceding
# the [27, 834, div_token] triple.
TURN_LAST_IDS = {}
for name in ["I_TURN_LEFT", "I_TURN_RIGHT"]:
    ids = raw_ids[name]
    TURN_LAST_IDS[name] = ids[-1]
    print(f"  {name:<15} last subword_id={ids[-1]}  "
          f"({tokenizer.convert_ids_to_tokens([ids[-1]])[0]!r})")
TURN_END_IDS = set(TURN_LAST_IDS.values())
print(f"\n  TURN_END_IDS = {TURN_END_IDS}  (used for action-slot detection)")

# Embedding matrix (tied in T5): shape [vocab_size, d_model]
embed_matrix = model.shared.weight.detach()   # [V, 512]

# Divergence-point embeddings
div_embeds = {name: embed_matrix[div_id] for name, div_id in DIV_IDS.items()}

# Pairwise Euclidean distances between divergence-point embeddings
print(f"\n  Pairwise Euclidean distances (divergence-point embeddings):")
print(f"  {'':>10}", end="")
for b in core_actions:
    print(f"  {b:>10}", end="")
print()
for a in core_actions:
    print(f"  {a:>10}", end="")
    for b in core_actions:
        d = torch.dist(div_embeds[a], div_embeds[b]).item()
        print(f"  {d:>10.3f}", end="")
    print()

print(f"\n  (Non-zero distances confirm divergence-point embeddings are distinct.)")

# ── Phase 3: Load SCAN and classify ────────────────────────────────────────────────────────────────────────────
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

def is_jump_compound(ex):
    t = ex["commands"].split()
    return "jump" in t and len(t) > 1

test_jc = [ex for ex in test_raw if is_jump_compound(ex)]

# Count action token frequencies in training targets
print(f"\n  Action token frequencies in training targets:")
print(f"  (Re-confirmed from S-050: I_WALK/I_RUN/I_LOOK identical — only geometry")
print(f"   can explain I_WALK preference over I_RUN and I_LOOK.)")
freq = defaultdict(int)
total_tokens = 0
for ex in train_raw:
    for tok in ex["actions"].split():
        freq[tok] += 1
        total_tokens += 1
for name in core_actions:
    cnt = freq.get(name, 0)
    print(f"  {name:<15} {cnt:>8}  ({100*cnt/total_tokens:.2f}%)")

# Run inference to find groups
print(f"\n  Running inference on {len(test_jc)} jump-compound examples...")

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

# ── Phase 4: Divergence-point detection ─────────────────────────────────────────────────────────────────────────────
print("\n" + "="*70)
print("  PHASE 4 — DIVERGENCE-POINT DETECTION")
print("="*70)
print(f"""
  Every core action token shares subword prefix [27, 834].
  The decision point — where the model chooses WHICH action — is at
  subword index 2, the divergence token:
    I_JUMP: ...→ token 683   I_WALK: ...→ token 12054
    I_RUN:  ...→ token 448   I_LOOK: ...→ token 5017

  We detect action-slot divergence points in the generated subword
  sequence by the pattern:
    gen_ids[i-3] ∈ TURN_END_IDS   (last subword of a TURN token)
    gen_ids[i-2] == 27             (first shared subword of action token)
    gen_ids[i-1] == 834            (second shared subword of action token)
    gen_ids[i]   ∈ div_token_set   (the divergence token — which action?)

  At step i, we measure h_div (last decoder hidden state) and compute:
    P(683) vs P(12054) vs P(448) vs P(5017)  — the actual decision
    d(h_div, embed(div_token)) for each action — Troy's geometry test
""")

DIV_TOKEN_SET = set(DIV_IDS.values())

def find_div_points(gen_token_ids):
    """
    In the generated subword sequence (excluding BOS), find positions i where:
      gen_token_ids[i-3] ∈ TURN_END_IDS  — end of preceding TURN token
      gen_token_ids[i-2] == 27            — start of action token
      gen_token_ids[i-1] == 834           — second subword of action token
      gen_token_ids[i]   ∈ DIV_TOKEN_SET  — divergence point

    Returns list of (i, action_name, turn_direction).
    """
    id_to_action = {v: k for k, v in DIV_IDS.items()}
    turn_end_to_dir = {
        TURN_LAST_IDS["I_TURN_LEFT"]:  "left",
        TURN_LAST_IDS["I_TURN_RIGHT"]: "right",
    }
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

# ── Phase 5: Main geometry measurement ────────────────────────────────────────────────────────────────────────────
print("\n" + "="*70)
print("  PHASE 5 — GEOMETRY + LOGITS + ATTENTION AT DIVERGENCE POINTS")
print("="*70)
print(f"\n  Measuring for {len(sub_walk_examples)} fail and"
      f" {len(success_examples)} success examples...")
print(f"  (One example at a time — slow but necessary for attention capture)\n")

records_fail    = []
records_success = []

def measure_example(ex, group_label):
    """
    Generate with full attention+hidden state output.
    For each action-slot divergence point, record:
      - Cross-attention weights to all encoder positions (K1 confirmatory)
      - P(div_token) for each action type (the actual softmax decision)
      - Euclidean distance d(h_div, embed(div_token)) for each action
      - Cosine similarity cos(h_div, embed(div_token)) for each action
    """
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

    # gen_ids[0] is the pad/BOS token; decoder step t generates gen_ids[t+1]
    gen_ids = out.sequences[0].tolist()
    n_steps = len(out.cross_attentions)
    seq_no_bos = gen_ids[1:]   # skip BOS for slot detection

    div_points = find_div_points(seq_no_bos)

    slot_records = []
    for (div_pos_in_seq, action_name, turn_dir) in div_points:
        # div_pos_in_seq is index into seq_no_bos; decoder step index is same value
        step = div_pos_in_seq
        if step >= n_steps:
            continue

        # ── Cross-attention at divergence step ──────────────────────────────────────────────────────────────────
        step_attns = out.cross_attentions[step]
        # Each element: [1, n_heads, 1, T_enc]
        all_heads = np.concatenate(
            [layer_a[0, :, 0, :].cpu().numpy() for layer_a in step_attns],
            axis=0  # [n_layers*n_heads, T_enc]
        )
        mean_attn = all_heads.mean(0)   # [T_enc]

        attn_to_jump   = float(mean_attn[jump_pos])   if jump_pos   is not None else float("nan")
        attn_to_around = float(mean_attn[around_pos]) if around_pos is not None else float("nan")
        attn_entropy   = float(-np.sum((mean_attn + 1e-12) * np.log(mean_attn + 1e-12)))

        # ── Decoder hidden state at divergence step (last layer) ──────────────────────────────────────────────
        # out.decoder_hidden_states[step] = tuple of n_layers+1 tensors [1, 1, d_model]
        step_hidden = out.decoder_hidden_states[step]
        h = step_hidden[-1][0, 0, :]   # [d_model]

        # ── Logits and probabilities at divergence point ────────────────────────────────────────────────
        # logit(token) = h · embed(token)
        all_logits = (embed_matrix @ h).float()   # [vocab_size]
        probs = torch.softmax(all_logits, dim=0)

        # For each action, record P(div_token) — the decision probability
        action_div_probs  = {name: float(probs[div_id])
                             for name, div_id in DIV_IDS.items()}
        action_div_logits = {name: float(all_logits[div_id])
                             for name, div_id in DIV_IDS.items()}

        # ── Euclidean distance and cosine similarity ────────────────────────────────────────────────────────────────
        # Comparing h_div to each action's divergence-point embedding
        action_euc = {}
        action_cos = {}
        for name, div_id in DIV_IDS.items():
            e = div_embeds[name]
            action_euc[name] = float(torch.dist(h, e).item())
            action_cos[name] = float(F.cosine_similarity(
                h.unsqueeze(0), e.unsqueeze(0)).item())

        # Embedding norms (to diagnose logit vs geometry discrepancy)
        action_emb_norm = {name: float(div_embeds[name].norm().item())
                           for name in core_actions}
        h_norm = float(h.norm().item())

        slot_records.append({
            "step":            step,
            "div_pos_in_seq":  div_pos_in_seq,
            "action_generated": action_name,
            "turn_dir":        turn_dir,
            "attn_to_jump":    attn_to_jump,
            "attn_to_around":  attn_to_around,
            "attn_entropy":    attn_entropy,
            "h_norm":          h_norm,
            "div_probs":       action_div_probs,
            "div_logits":      action_div_logits,
            "euclidean":       action_euc,
            "cosine":          action_cos,
            "emb_norms":       action_emb_norm,
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

# ── Phase 6: Aggregate and report ────────────────────────────────────────────────────────────────────────────────
print("\n" + "="*70)
print("  PHASE 6 — AGGREGATE RESULTS")
print("="*70)

def collect_slots(records):
    """Flatten all slot measurements from a list of example records."""
    out = defaultdict(list)
    for rec in records:
        for slot in rec["slots"]:
            for key, val in slot.items():
                if isinstance(val, dict):
                    for subkey, subval in val.items():
                        out[f"{key}_{subkey}"].append(float(subval))
                elif isinstance(val, (int, float)):
                    out[key].append(float(val))
    return {k: np.array(v) for k, v in out.items()}

fail_data    = collect_slots(records_fail)
success_data = collect_slots(records_success)

n_fail_slots    = len(fail_data.get("step", []))
n_success_slots = len(success_data.get("step", []))
print(f"\n  Total divergence points measured: fail={n_fail_slots}, success={n_success_slots}")

# Report embedding norms (diagnostic — explains logit vs geometry discrepancy)
print(f"\n  ── Embedding norms (divergence-point tokens) ───────────────────────────────────────────────")
for name, div_id in DIV_IDS.items():
    norm = float(div_embeds[name].norm().item())
    print(f"  embed({name}) norm: {norm:.4f}  (div_token={div_id})")
print(f"\n  (If norms differ substantially, logit rank ≠ Euclidean rank.")
print(f"   High norm token can win by dot product even if Euclidean farther.)")

# H1: P(I_JUMP) suppressed at action slots in fail cases
print(f"\n  ── H1: P(div_token) at action-slot divergence points ─────────────────────────────────────")
for label, data in [("fail   (sub_walk)", fail_data), ("success (correct)", success_data)]:
    print(f"  {label}:")
    for name in core_actions:
        key = f"div_probs_{name}"
        if key in data and len(data[key]) > 0:
            vals = data[key]
            print(f"    P({name:<8}) mean={vals.mean():.5f}  "
                  f"median={np.median(vals):.5f}  max={vals.max():.5f}")
    print()

h1_jump_fail    = fail_data.get("div_probs_I_JUMP",    np.array([float("nan")]))
h1_jump_success = success_data.get("div_probs_I_JUMP", np.array([float("nan")]))
if len(h1_jump_fail) > 1 and len(h1_jump_success) > 1:
    stat, p = scipy_stats.mannwhitneyu(h1_jump_success, h1_jump_fail,
                                       alternative="greater")
    h1_pass = (np.nanmean(h1_jump_success) > np.nanmean(h1_jump_fail)) and (p < 0.05)
    print(f"  H1 (P(I_JUMP div) higher in success): U={stat:.0f}, p={p:.4f} — "
          f"{'PASS' if h1_pass else 'FAIL'}")
else:
    h1_pass = False
    print(f"  H1: insufficient data")

# H2: Geometry — Euclidean distances at divergence points (fail)
print(f"\n  ── H2: Euclidean d(h_div, embed(div_token)) ────────────────────────────────────────────")
print(f"  {'Action':<12}  {'Fail (mean)':>12}  {'Success (mean)':>14}")
euc_results = {}
for name in core_actions:
    key = f"euclidean_{name}"
    f_vals = fail_data.get(key,    np.array([float("nan")]))
    s_vals = success_data.get(key, np.array([float("nan")]))
    f_mean = float(np.nanmean(f_vals))
    s_mean = float(np.nanmean(s_vals))
    euc_results[name] = {"fail": f_mean, "success": s_mean}
    print(f"  {name:<12}  {f_mean:>12.4f}  {s_mean:>14.4f}")

fail_euc_rank = sorted(core_actions, key=lambda n: euc_results[n]["fail"])
print(f"\n  Euclidean rank (fail, closest first): {' < '.join(fail_euc_rank)}")
h2_pass = (fail_euc_rank[0] == "I_WALK")
print(f"  H2 (I_WALK geometrically closest in fail): {'PASS' if h2_pass else 'FAIL'}")
print(f"  Troy's geometry hypothesis: {'CONFIRMED' if h2_pass else 'CONTRADICTED'}")

# Among trained tokens only (I_WALK vs I_RUN vs I_LOOK — equal frequency)
trained_rank = [n for n in fail_euc_rank if n != "I_JUMP"]
print(f"\n  Among equal-frequency tokens (I_WALK, I_RUN, I_LOOK):")
print(f"  Euclidean rank (fail, closest first): {' < '.join(trained_rank)}")
h2_trained_pass = (trained_rank[0] == "I_WALK")
print(f"  I_WALK closest among equal-freq: {'YES — geometry explains substitution' if h2_trained_pass else 'NO — other mechanism'}")

# Cosine similarity
print(f"\n  ── Cosine similarity cos(h_div, embed(div_token)) ────────────────────────────────────────────")
print(f"  {'Action':<12}  {'Fail (mean)':>12}  {'Success (mean)':>14}")
cos_results = {}
for name in core_actions:
    key = f"cosine_{name}"
    f_vals = fail_data.get(key,    np.array([float("nan")]))
    s_vals = success_data.get(key, np.array([float("nan")]))
    f_mean = float(np.nanmean(f_vals))
    s_mean = float(np.nanmean(s_vals))
    cos_results[name] = {"fail": f_mean, "success": s_mean}
    print(f"  {name:<12}  {f_mean:>12.4f}  {s_mean:>14.4f}")

fail_cos_rank = sorted(core_actions, key=lambda n: -cos_results[n]["fail"])
print(f"\n  Cosine rank (fail, highest similarity first): {' > '.join(fail_cos_rank)}")
trained_cos_rank = [n for n in fail_cos_rank if n != "I_JUMP"]
print(f"  Among equal-freq tokens: {' > '.join(trained_cos_rank)}")

# Logit rank (for comparison — expected to match P() above)
print(f"\n  ── Logit rank at divergence point (fail) ─────────────────────────────────────────────────────────────────────────")
logit_means = {}
for name in core_actions:
    key = f"div_logits_{name}"
    vals = fail_data.get(key, np.array([float("nan")]))
    logit_means[name] = float(np.nanmean(vals))
    print(f"  {name:<15} mean logit: {logit_means[name]:.4f}")
fail_logit_rank = sorted(core_actions, key=lambda n: -logit_means[n])
print(f"\n  Logit rank (fail, highest first): {' > '.join(fail_logit_rank)}")

# H3: d(h_div, I_JUMP div token) larger in fail than success
h3_pass = euc_results["I_JUMP"]["fail"] > euc_results["I_JUMP"]["success"]
print(f"\n  ── H3: d(h_div, I_JUMP div token) larger in fail than success ──")
print(f"  Fail mean:    {euc_results['I_JUMP']['fail']:.4f}")
print(f"  Success mean: {euc_results['I_JUMP']['success']:.4f}")
print(f"  H3: {'PASS' if h3_pass else 'FAIL'}")

# H4: Cross-attention to 'jump' encoder position lower in fail (K1 expected)
print(f"\n  ── H4: Cross-attention to 'jump' encoder position ────────────────────────────────────────────────")
f_attn = fail_data.get("attn_to_jump",    np.array([float("nan")]))
s_attn = success_data.get("attn_to_jump", np.array([float("nan")]))
print(f"  Fail    attn_to_jump: mean={np.nanmean(f_attn):.5f}")
print(f"  Success attn_to_jump: mean={np.nanmean(s_attn):.5f}")
h4_pass = False
if not np.isnan(f_attn).all() and not np.isnan(s_attn).all():
    clean_s = s_attn[~np.isnan(s_attn)]
    clean_f = f_attn[~np.isnan(f_attn)]
    if len(clean_s) > 1 and len(clean_f) > 1:
        stat, p = scipy_stats.mannwhitneyu(clean_s, clean_f, alternative="greater")
        h4_pass = (np.nanmean(s_attn) > np.nanmean(f_attn)) and (p < 0.05)
        print(f"  H4 (success > fail attn_to_jump): p={p:.4f} — "
              f"{'PASS' if h4_pass else 'FAIL — K1 confirmed again'}")

k1_fires = not h4_pass
print(f"\n  K1 (routing identical at action slots): {'FIRES' if k1_fires else 'does not fire'}")

# Geometry vs logit discrepancy analysis
print(f"\n  ── Geometry/logit discrepancy analysis ─────────────────────────────────────────────────────────────────────")
print(f"  logit(token) = h · embed(token) = cos(h,e)·‖h‖·‖e‖")
print(f"  Euclidean distance and logit rank identically only when all ‖e‖ are equal.")
print(f"  If logit rank ≠ Euclidean rank, the discrepancy = embedding norm effect.")
for name in core_actions:
    enorm = float(div_embeds[name].norm().item())
    emean = logit_means[name]
    print(f"  {name:<15} ‖embed‖={enorm:.3f}  mean_logit={emean:.3f}")

logit_winner  = fail_logit_rank[0]
euc_winner    = fail_euc_rank[0]
if logit_winner == euc_winner:
    discrepancy_note = "Logit and Euclidean agree — embedding norm differences do not change the winner."
else:
    discrepancy_note = (f"Logit winner ({logit_winner}) ≠ Euclidean winner ({euc_winner}). "
                        f"K2 kill condition: embedding norm difference drives the selection, "
                        f"not spatial proximity.")
print(f"\n  {discrepancy_note}")

# ── Phase 7: Summary ────────────────────────────────────────────────────────────────────────────────────
print("\n" + "="*70)
print("  PHASE 7 — SUMMARY AND VERDICT")
print("="*70)

print(f"\n  ── HYPOTHESIS VERDICTS ──────────────────────────────────────────────────────────────────")
print(f"  H1: P(I_JUMP div) suppressed in fail cases        — {'PASS' if h1_pass else 'FAIL'}")
print(f"  H2: I_WALK div-embed closest in Euclidean (fail)  — {'PASS' if h2_pass else 'FAIL'}")
print(f"      Euclidean rank (fail): {' < '.join(fail_euc_rank)}")
print(f"      Cosine rank (fail):    {' > '.join(fail_cos_rank)}")
print(f"      Logit rank (fail):     {' > '.join(fail_logit_rank)}")
print(f"  H2 (trained tokens only, equal freq):             — {'PASS' if h2_trained_pass else 'FAIL'}")
print(f"      {' < '.join(trained_rank)}")
print(f"  H3: d(h_div, I_JUMP div) larger in fail           — {'PASS' if h3_pass else 'FAIL'}")
print(f"  H4: Attn to jump lower in fail (K1 probe)         — {'PASS' if h4_pass else 'FAIL'}")
print(f"  K1 (routing same — value geometry):               {'FIRES' if k1_fires else 'does not fire'}")

print(f"\n  ── GEOMETRY/FREQUENCY VERDICT ─────────────────────────────────────────────────────────────────────────")
print(f"  Training freq: I_WALK=I_RUN=I_LOOK ({freq.get('I_WALK',0)}) vs I_JUMP ({freq.get('I_JUMP',0)})")
print(f"                 Frequency rules out I_JUMP but CANNOT explain I_WALK>I_RUN>I_LOOK.")
print(f"  Geometric verdict: {discrepancy_note}")

if h2_trained_pass:
    verdict = (
        "GEOMETRY CONFIRMED (Troy's hypothesis): Among I_WALK, I_RUN, I_LOOK "
        "(equal training frequency), h_action at the divergence step is "
        "geometrically closest to embed(I_WALK divergence token) in Euclidean "
        "space. The substitution is determined by decoder embedding geometry, "
        "not training frequency. The 'around' cycle context biases h_action "
        "toward the I_WALK region of the output embedding space."
    )
elif logit_winner == "I_WALK" and not h2_trained_pass:
    verdict = (
        "GEOMETRY PARTIAL: I_WALK wins by logit but NOT by Euclidean distance. "
        "K2 kill condition fires: I_WALK's divergence-point embedding has larger "
        "norm than I_RUN/I_LOOK, amplifying its dot product independently of "
        "spatial proximity. The selection is driven by embedding norm asymmetry, "
        "not geometric closeness."
    )
else:
    verdict = (
        "GEOMETRY CONTRADICTED: I_WALK is neither the geometrically closest "
        "nor the logit-highest action token. Substitution mechanism unclear — "
        "inspect per-slot records for within-example patterns."
    )

print(f"\n  VERDICT: {verdict}")

# ── Save ───────────────────────────────────────────────────────────────────────────────────────────────────────────
output = {
    "script_id": SCRIPT_ID,
    "n_fail_slots":    int(n_fail_slots),
    "n_success_slots": int(n_success_slots),
    "div_ids": {name: int(div_id) for name, div_id in DIV_IDS.items()},
    "turn_end_ids": [int(x) for x in sorted(TURN_END_IDS)],
    "training_action_frequencies": {
        name: int(freq.get(name, 0)) for name in core_actions
    },
    "embedding_norms": {
        name: float(div_embeds[name].norm().item()) for name in core_actions
    },
    "euclidean_distances": {
        name: {"fail": float(euc_results[name]["fail"]),
               "success": float(euc_results[name]["success"])}
        for name in core_actions
    },
    "cosine_similarities": {
        name: {"fail": float(cos_results[name]["fail"]),
               "success": float(cos_results[name]["success"])}
        for name in core_actions
    },
    "logit_means_fail": {name: float(logit_means[name]) for name in core_actions},
    "euclidean_rank_fail": fail_euc_rank,
    "cosine_rank_fail":    fail_cos_rank,
    "logit_rank_fail":     fail_logit_rank,
    "action_div_probs_fail": {
        name: float(np.nanmean(fail_data.get(f"div_probs_{name}", [float("nan")])))
        for name in core_actions
    },
    "action_div_probs_success": {
        name: float(np.nanmean(success_data.get(f"div_probs_{name}", [float("nan")])))
        for name in core_actions
    },
    "verdict": {
        "h1_pass":         bool(h1_pass),
        "h2_pass":         bool(h2_pass),
        "h2_trained_pass": bool(h2_trained_pass),
        "h3_pass":         bool(h3_pass),
        "h4_pass":         bool(h4_pass),
        "k1_fires":        bool(k1_fires),
        "logit_euc_agree": bool(logit_winner == euc_winner),
        "description":     verdict,
    },
}

with open("050b_results.json", "w") as f:
    json.dump(output, f, indent=2)
save_to_drive("050b_results.json")

print(f"\n  Output: 050b_results.json")
print(f"  {'='*70}")
print(f"  S-050b complete.")
print(f"  {'='*70}")
