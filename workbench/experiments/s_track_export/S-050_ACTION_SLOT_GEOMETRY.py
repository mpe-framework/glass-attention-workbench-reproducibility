#!/usr/bin/env python3
"""
S-050_ACTION_SLOT_GEOMETRY.py — Action-Slot Logit and Embedding Geometry
Applied Categorical Physics Workbench
Troy Teno | May 2026 | Open Access

Pre-registered hypotheses:
  workbench/proposals/S-050_ACTION_SLOT_GEOMETRY_PROPOSAL.md
DO NOT modify hypotheses after seeing results.

─────────────────────────────────────────────────────────────────────
SETUP (run once in Colab or terminal):
  pip install torch transformers datasets numpy scipy sentencepiece -q

WHAT THIS SCRIPT DOES:
  S-049 established K1: decoder routing to 'around' is identical in fail vs
  success. The failure is at value/action-slot level. S-048 showed substituted_walk
  (26%): T5 generates (TURN I_WALK)×4 for "jump around" — correct structure,
  wrong primitive. The mystery: why I_WALK specifically, not I_RUN or I_LOOK?

  Two hypotheses:
    A. Training frequency: I_WALK is the most common action in 'around' contexts.
    B. Geometry: h_action (decoder hidden state at action slot) is geometrically
       closer to embed(I_WALK) than to embed(I_JUMP) in the output embedding space.

  This script collects all three instruments in one pass:
    1. Cross-attention weights at action slots (K1 confirmatory proof)
    2. Logit probabilities: P(I_JUMP), P(I_WALK), P(I_RUN), P(I_LOOK)
    3. Euclidean distance and cosine similarity from h_action to each action
       token embedding — Troy's geometry question

  Groups:
    Fail:    substituted_walk examples (T5 generates I_WALK instead of I_JUMP)
    Success: has_around=1 + correct (T5 correctly generates I_JUMP in cycles)
─────────────────────────────────────────────────────────────────────
"""

import numpy as np
import json
import os
import random
from collections import defaultdict

SCRIPT_ID = "S-050_ACTION_SLOT_GEOMETRY_V0.1.0"
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
print(f"  Action-slot geometry: why I_WALK and not I_JUMP?")
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

# ── Phase 2: Action token IDs and embeddings ──────────────────────────────────────────────────────────────────────────
print("\n" + "="*70)
print("  PHASE 2 — ACTION TOKEN EMBEDDINGS")
print("="*70)

ACTION_NAMES = ["I_JUMP", "I_WALK", "I_RUN", "I_LOOK",
                "I_TURN_LEFT", "I_TURN_RIGHT"]

action_ids = {}
for name in ACTION_NAMES:
    ids = tokenizer.encode(name, add_special_tokens=False)
    if len(ids) == 1:
        action_ids[name] = ids[0]
        print(f"  {name:<15} → token_id={ids[0]}")
    else:
        # T5 may split tokens; take the first subpiece as proxy
        action_ids[name] = ids[0]
        print(f"  {name:<15} → token_ids={ids} (using {ids[0]} as proxy)")

# Embedding matrix (tied in T5): shape [vocab_size, d_model]
embed_matrix = model.shared.weight.detach()   # [V, 512]

def get_embed(name):
    return embed_matrix[action_ids[name]]   # [512]

embeds = {name: get_embed(name) for name in ACTION_NAMES}

# Pairwise distances between action embeddings (for reference)
print(f"\n  Pairwise Euclidean distances between action token embeddings:")
core_actions = ["I_JUMP", "I_WALK", "I_RUN", "I_LOOK"]
print(f"  {'':>10}", end="")
for b in core_actions:
    print(f"  {b:>10}", end="")
print()
for a in core_actions:
    print(f"  {a:>10}", end="")
    for b in core_actions:
        d = torch.dist(embeds[a], embeds[b]).item()
        print(f"  {d:>10.3f}", end="")
    print()

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

# Count action token frequencies in training targets (H_freq hypothesis)
print(f"\n  Action token frequencies in training targets:")
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
            n_gold_j = gold.split().count("I_JUMP")
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

# ── Phase 4: Action-slot identification ─────────────────────────────────────────────────────────────────────────────
print("\n" + "="*70)
print("  PHASE 4 — DEFINE ACTION-SLOT DETECTION")
print("="*70)
print(f"""
  In 'X around Y' outputs, the pattern is (I_TURN_Y I_ACTION)×4.
  Action slots are at positions 1, 3, 5, 7 (0-indexed) within each cycle.
  We identify them by: prev token ∈ {{I_TURN_LEFT, I_TURN_RIGHT}}.
""")

TURN_IDS = {action_ids["I_TURN_LEFT"], action_ids["I_TURN_RIGHT"]}

def find_action_slots(token_id_sequence):
    """
    Return list of (position, turn_direction) for positions where prev token
    is I_TURN_LEFT or I_TURN_RIGHT — these are the action slots in around cycles.
    """
    slots = []
    for i in range(1, len(token_id_sequence)):
        if token_id_sequence[i-1] in TURN_IDS:
            turn_dir = ("left" if token_id_sequence[i-1] == action_ids["I_TURN_LEFT"]
                        else "right")
            slots.append((i, turn_dir))
    return slots

# ── Phase 5: Main geometry measurement ────────────────────────────────────────────────────────────────────────────
print("\n" + "="*70)
print("  PHASE 5 — GEOMETRY + LOGITS + ATTENTION AT ACTION SLOTS")
print("="*70)
print(f"\n  Measuring for {len(sub_walk_examples)} fail and"
      f" {len(success_examples)} success examples...")
print(f"  (Slow — one example at a time with attention + hidden states)\n")

# Store per-slot measurements
records_fail    = []   # list of dicts, one per action slot per example
records_success = []

ACTION_IDS_CORE = {name: action_ids[name] for name in core_actions}

def measure_example(ex, group_label):
    """
    Generate with output_attentions=True, output_hidden_states=True.
    For each decoder step that is an action slot, record:
      - cross-attention weights to all encoder positions
      - logits / probabilities for core action tokens
      - Euclidean distance from h_action to each action token embedding
      - cosine similarity from h_action to each action token embedding
    """
    cmd = ex["commands"]
    inp = tokenizer(cmd, truncation=True, max_length=MAX_INPUT_LEN,
                    return_tensors="pt").to(device)
    T_enc = inp["input_ids"].shape[1]

    # Locate "jump" and "around" encoder positions
    enc_tokens = tokenizer.convert_ids_to_tokens(inp["input_ids"][0])
    jump_pos  = next((i for i, t in enumerate(enc_tokens) if "jump"  in t.lower()), None)
    around_pos= next((i for i, t in enumerate(enc_tokens) if "around" in t.lower()), None)

    with torch.no_grad():
        out = model.generate(
            input_ids=inp["input_ids"],
            attention_mask=inp["attention_mask"],
            max_length=MAX_TARGET_LEN,
            output_attentions=True,
            output_hidden_states=True,
            return_dict_in_generate=True,
        )

    gen_ids = out.sequences[0].tolist()     # generated token IDs (includes BOS)
    n_steps = len(out.cross_attentions)

    # Find action slots in generated sequence
    # out.sequences includes BOS token at position 0; decoder step t corresponds to gen_ids[t+1]
    slots = find_action_slots(gen_ids[1:])  # skip BOS

    slot_records = []
    for (slot_pos, turn_dir) in slots:
        if slot_pos >= n_steps:
            continue

        # ── Cross-attention at this step ───────────────────────────────────────────────────────────────────────
        step_attns = out.cross_attentions[slot_pos]
        # step_attns: tuple of n_layers tensors [1, n_heads, 1, T_enc]
        all_heads = []
        for layer_attn in step_attns:
            a = layer_attn[0, :, 0, :].cpu().numpy()   # [n_heads, T_enc]
            all_heads.append(a)
        all_heads = np.concatenate(all_heads, axis=0)  # [n_layers*n_heads, T_enc]
        mean_attn = all_heads.mean(0)                  # [T_enc]

        attn_to_jump  = float(mean_attn[jump_pos])  if jump_pos  is not None else float("nan")
        attn_to_around= float(mean_attn[around_pos]) if around_pos is not None else float("nan")
        attn_entropy  = float(-np.sum((mean_attn+1e-12) * np.log(mean_attn+1e-12)))

        # ── Decoder hidden state at this step (last layer) ────────────────────────────────────────────────
        # out.decoder_hidden_states[step] = tuple of n_layers+1 tensors [1, 1, d_model]
        step_hidden = out.decoder_hidden_states[slot_pos]
        h = step_hidden[-1][0, 0, :]   # [d_model] — last decoder layer

        # ── Logits and probabilities ────────────────────────────────────────────────────────────────────────────────────
        # logit = h · embed(token)
        all_logits = (embed_matrix @ h).float()  # [vocab_size]
        probs = torch.softmax(all_logits, dim=0)

        action_probs  = {name: float(probs[tid])   for name, tid in ACTION_IDS_CORE.items()}
        action_logits = {name: float(all_logits[tid]) for name, tid in ACTION_IDS_CORE.items()}
        token_generated = gen_ids[slot_pos + 1]  # the actual generated token at this step

        # ── Euclidean distance and cosine similarity ────────────────────────────────────────────────────────────────
        h_norm = h / (h.norm() + 1e-9)
        action_euc  = {}
        action_cos  = {}
        for name, tid in ACTION_IDS_CORE.items():
            e = embeds[name]
            action_euc[name] = float(torch.dist(h, e).item())
            action_cos[name] = float(F.cosine_similarity(h.unsqueeze(0),
                                                          e.unsqueeze(0)).item())

        slot_records.append({
            "slot_pos":        slot_pos,
            "turn_dir":        turn_dir,
            "token_generated": int(token_generated),
            "token_name":      tokenizer.convert_ids_to_tokens([token_generated])[0],
            "attn_to_jump":    attn_to_jump,
            "attn_to_around":  attn_to_around,
            "attn_entropy":    attn_entropy,
            "probs":           action_probs,
            "logits":          action_logits,
            "euclidean":       action_euc,
            "cosine":          action_cos,
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
                        out[f"{key}_{subkey}"].append(subval)
                elif not isinstance(val, str):
                    out[key].append(val)
    return {k: np.array(v) for k, v in out.items()}

fail_data    = collect_slots(records_fail)
success_data = collect_slots(records_success)

n_fail_slots    = len(fail_data.get("slot_pos", []))
n_success_slots = len(success_data.get("slot_pos", []))
print(f"\n  Total action slots measured: fail={n_fail_slots}, success={n_success_slots}")

# H1: P(I_JUMP) near zero in fail cases
print(f"\n  ── H1: P(I_JUMP) at action slots ──────────────────────────────────────────────────────────────────")
for group, data, label in [
    (fail_data,    fail_data,    "fail   (sub_walk)"),
    (success_data, success_data, "success (correct)"),
]:
    for name in core_actions:
        key = f"probs_{name}"
        if key in data:
            vals = data[key]
            print(f"  {label}  P({name:<8}) mean={vals.mean():.5f}  "
                  f"median={np.median(vals):.5f}  max={vals.max():.5f}")
    print()

h1_jump_fail    = fail_data.get("probs_I_JUMP",    np.array([float("nan")]))
h1_jump_success = success_data.get("probs_I_JUMP", np.array([float("nan")]))
if len(h1_jump_fail) > 1 and len(h1_jump_success) > 1:
    stat, p = scipy_stats.mannwhitneyu(h1_jump_success, h1_jump_fail,
                                       alternative="greater")
    h1_pass = (h1_jump_success.mean() > h1_jump_fail.mean()) and (p < 0.05)
    print(f"  H1 (P(I_JUMP) higher in success): U={stat:.0f}, p={p:.4f} — "
          f"{'PASS' if h1_pass else 'FAIL'}")
else:
    h1_pass = False

# H2: Geometry — Euclidean distances at action slots (fail)
print(f"\n  ── H2: Euclidean distance h_action → action token embeddings ──")
print(f"  {'Action':<12}  {'Fail (mean)':>12}  {'Success (mean)':>14}  {'Fail closer?':>12}")
euc_results = {}
for name in core_actions:
    key = f"euclidean_{name}"
    f_vals = fail_data.get(key,    np.array([float("nan")]))
    s_vals = success_data.get(key, np.array([float("nan")]))
    f_mean = float(np.nanmean(f_vals))
    s_mean = float(np.nanmean(s_vals))
    euc_results[name] = {"fail": f_mean, "success": s_mean}
    print(f"  {name:<12}  {f_mean:>12.4f}  {s_mean:>14.4f}  "
          f"  {'yes' if f_mean < s_mean else 'no':>12}")

# Rank action tokens by Euclidean distance (fail group)
fail_euc_rank = sorted(core_actions,
                       key=lambda n: euc_results[n]["fail"])
print(f"\n  Euclidean rank (fail, closest first): {' < '.join(fail_euc_rank)}")

h2_pass = (fail_euc_rank[0] == "I_WALK")
print(f"  H2 (I_WALK closest in fail): {'PASS' if h2_pass else 'FAIL'}")
print(f"  Troy's geometry hypothesis: {'CONFIRMED' if h2_pass else 'CONTRADICTED'}")

# Cosine similarity at action slots
print(f"\n  ── Cosine similarity h_action → action token embeddings ──────")
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

fail_cos_rank = sorted(core_actions,
                       key=lambda n: -cos_results[n]["fail"])
print(f"\n  Cosine rank (fail, highest first): {' > '.join(fail_cos_rank)}")

# H3: d(h, I_JUMP) larger in fail than success
h3_pass = euc_results["I_JUMP"]["fail"] > euc_results["I_JUMP"]["success"]
print(f"\n  ── H3: d(h_action, I_JUMP) larger in fail than success ──────────")
print(f"  Fail mean:    {euc_results['I_JUMP']['fail']:.4f}")
print(f"  Success mean: {euc_results['I_JUMP']['success']:.4f}")
print(f"  H3: {'PASS' if h3_pass else 'FAIL'}")

# H4: Cross-attention to jump lower at action slots in fail
print(f"\n  ── H4: Cross-attention to 'jump' encoder position ───────────────")
f_attn = fail_data.get("attn_to_jump",    np.array([float("nan")]))
s_attn = success_data.get("attn_to_jump", np.array([float("nan")]))
print(f"  Fail    attn_to_jump: mean={np.nanmean(f_attn):.5f}")
print(f"  Success attn_to_jump: mean={np.nanmean(s_attn):.5f}")
if not np.isnan(f_attn).all() and not np.isnan(s_attn).all():
    stat, p = scipy_stats.mannwhitneyu(
        s_attn[~np.isnan(s_attn)], f_attn[~np.isnan(f_attn)],
        alternative="greater")
    h4_pass = (np.nanmean(s_attn) > np.nanmean(f_attn)) and (p < 0.05)
    print(f"  H4 (success > fail attn_to_jump): p={p:.4f} — "
          f"{'PASS' if h4_pass else 'FAIL — K1 confirmed again'}")
else:
    h4_pass = False

# K1 check for attention
k1_fires = not h4_pass
print(f"\n  K1 (routing identical at action slots): {'FIRES' if k1_fires else 'does not fire'}")

# ── Phase 7: Summary ────────────────────────────────────────────────────────────────────────────────────
print("\n" + "="*70)
print("  PHASE 7 — SUMMARY AND VERDICT")
print("="*70)

print(f"\n  ── HYPOTHESIS VERDICTS ──────────────────────────────────────────────────────────────────")
print(f"  H1: P(I_JUMP) suppressed in fail cases       — {'PASS' if h1_pass else 'FAIL'}")
print(f"  H2: I_WALK geometrically closest (Euclidean) — {'PASS' if h2_pass else 'FAIL'}")
print(f"      Fail rank: {' < '.join(fail_euc_rank)}")
print(f"      Cos rank:  {' > '.join(fail_cos_rank)}")
print(f"  H3: d(h, I_JUMP) larger in fail              — {'PASS' if h3_pass else 'FAIL'}")
print(f"  H4: Attn to jump lower in fail               — {'PASS' if h4_pass else 'FAIL'}")
print(f"  K1 (routing same = value geometry):          {'FIRES' if k1_fires else 'does not fire'}")

if h2_pass:
    verdict = ("GEOMETRY CONFIRMED: I_WALK is the geometrically nearest action "
               "token to h_action in fail cases. The softmax picks I_WALK because "
               "the action-slot hidden state lands closer to embed(I_WALK) than to "
               "embed(I_JUMP) in the decoder output embedding space.")
else:
    verdict = ("GEOMETRY INCONCLUSIVE: I_WALK is not the closest action token "
               "by Euclidean distance. Check cosine rank — logit (dot product) "
               "may explain the I_WALK selection through norm differences.")

print(f"\n  VERDICT: {verdict}")

# ── Save ───────────────────────────────────────────────────────────────────────────────────────────────────────────
output = {
    "script_id": SCRIPT_ID,
    "n_fail_slots": int(n_fail_slots),
    "n_success_slots": int(n_success_slots),
    "training_action_frequencies": {
        name: int(freq.get(name, 0)) for name in core_actions
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
    "euclidean_rank_fail": fail_euc_rank,
    "cosine_rank_fail":    fail_cos_rank,
    "action_probs_fail": {
        name: float(np.nanmean(fail_data.get(f"probs_{name}", [float("nan")])))
        for name in core_actions
    },
    "action_probs_success": {
        name: float(np.nanmean(success_data.get(f"probs_{name}", [float("nan")])))
        for name in core_actions
    },
    "verdict": {
        "h1_pass": bool(h1_pass),
        "h2_pass": bool(h2_pass),
        "h3_pass": bool(h3_pass),
        "h4_pass": bool(h4_pass),
        "k1_fires": bool(k1_fires),
        "description": verdict,
    },
}

with open("050_results.json", "w") as f:
    json.dump(output, f, indent=2)
save_to_drive("050_results.json")

print(f"\n  Output: 050_results.json")
print(f"  {'='*70}")
print(f"  S-050 complete.")
print(f"  {'='*70}")
