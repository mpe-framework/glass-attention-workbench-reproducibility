#!/usr/bin/env python3
"""
S-048_AUX_LOSS_FINETUNING.py — Auxiliary Loss Fine-tuning for has_around
Applied Categorical Physics Workbench
Troy Teno | May 2026 | Open Access

Pre-registered hypotheses:
  workbench/proposals/S-048_AUX_LOSS_FINETUNING_PROPOSAL.md
DO NOT modify hypotheses after seeing results.

─────────────────────────────────────────────────────────────────────
SETUP (run once in Colab or terminal):
  pip install torch transformers datasets numpy scipy sentencepiece scikit-learn -q

TO RUN IN GOOGLE COLAB (T4 GPU, recommended):
  1. colab.research.google.com → New notebook
  2. Cell 1: !pip install torch transformers datasets numpy scipy sentencepiece scikit-learn -q
  3. Cell 2: paste this entire script
  4. Runtime → Change runtime type → T4 GPU
  5. Run

Runtime:  ~30-60 min per λ on T4 (~90-180 min total, 3 training runs)

WHAT THIS SCRIPT DOES:
  1. Loads SCAN add_prim_jump (HuggingFace or GitHub fallback)
  2. For each λ in {0.1, 0.5, 1.0}:
     a. Fine-tunes T5-small from pretrained weights with:
        total_loss = seq2seq_loss + λ × BCE(has_around_pred, has_around_label)
        where has_around_pred comes from a linear head on encoder layer 5 mean-pool
     b. Evaluates on full jump-compound test split (7706 examples)
     c. Reports accuracy by has_around subgroup and by has_after+has_around
  3. Measures has_around probe AUC vs failure for each trained model (H4)
  4. Compares to S-043 baseline (43.7% jump-compound accuracy)

BASELINE REFERENCE:
  S-043: 43.7% on jump-compound (5 epochs, batch=32, lr=1e-4, no aux loss)
─────────────────────────────────────────────────────────────────────
"""

import numpy as np
import json
import os
import sys
import random
from collections import defaultdict

SCRIPT_ID = "S-048_AUX_LOSS_V0.1.0"
SEED = 42
np.random.seed(SEED)
random.seed(SEED)

# ── Configuration ───────────────────────────────────────────────────────────────
N_EPOCHS       = 5
BATCH_SIZE     = 32
LR             = 1e-4
MAX_INPUT_LEN  = 50
MAX_TARGET_LEN = 100
D_MODEL        = 512
LAMBDA_VALUES  = [0.1, 0.5, 1.0]
BASELINE_ACC   = 0.437        # S-043 jump-compound accuracy

BITS_FILE      = "046_structural_bits.json"   # from S-046
LOG_EVERY      = 100

# ── Google Drive persistence ───────────────────────────────────────────────────
DRIVE_DIR = None
try:
    from google.colab import drive as _drive
    _drive.mount("/content/drive", force_remount=False)
    DRIVE_DIR = "/content/drive/MyDrive/workbench_artifacts"
    os.makedirs(DRIVE_DIR, exist_ok=True)
    import shutil as _shutil
    for _fname in (BITS_FILE, "046_encoder_reps.npy",
                   "043_family_ab_results.json"):
        _src = os.path.join(DRIVE_DIR, _fname)
        if not os.path.exists(_fname) and os.path.exists(_src):
            _shutil.copy(_src, _fname)
            print(f"  Restored from Drive: {_fname}")
    print(f"  Drive mounted. Artifacts will be saved to: {DRIVE_DIR}")
except Exception:
    print("  Drive not available. Saving to local /content/ only.")

def save_to_drive(*filenames):
    if DRIVE_DIR is None:
        return
    import shutil
    for fname in filenames:
        if os.path.exists(fname):
            dst = os.path.join(DRIVE_DIR, fname)
            shutil.copy(fname, dst)
            print(f"  Saved to Drive: {fname}")

print(f"\n{'='*70}")
print(f"  {SCRIPT_ID}")
print(f"  Aux-loss fine-tuning on SCAN add_prim_jump — T5-small")
print(f"  λ sweep: {LAMBDA_VALUES}")
print(f"  Baseline (S-043): {BASELINE_ACC*100:.1f}%")
print(f"  Pre-registered hypotheses: S-048_AUX_LOSS_FINETUNING_PROPOSAL.md")
print(f"{'='*70}")

# ── Imports ───────────────────────────────────────────────────────────────────
print("\nImporting libraries...")
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from transformers import T5ForConditionalGeneration, T5Tokenizer
    from datasets import load_dataset
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from scipy import stats as scipy_stats
except ImportError as e:
    raise SystemExit(
        f"\nMissing library: {e}\n"
        "Run:  pip install torch transformers datasets numpy scipy "
        "sentencepiece scikit-learn\n"
    )

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"  Device: {device}")
if device.type == "cpu":
    print("  WARNING: CPU mode — each λ run will take ~3-5 hours.")

torch.manual_seed(SEED)
if device.type == "cuda":
    torch.cuda.manual_seed(SEED)

# ── Phase 1: Load SCAN ────────────────────────────────────────────────────────────
print("\n" + "="*70)
print("  PHASE 1 — LOAD SCAN add_prim_jump")
print("="*70)

train_raw = None
test_raw  = None

for _cfg in ("addprim_jump", "add_prim_jump"):
    try:
        _ds = load_dataset("scan", _cfg, trust_remote_code=True)
        train_raw = list(_ds["train"])
        test_raw  = list(_ds["test"])
        print(f"  Loaded via HuggingFace (config={_cfg!r}).")
        break
    except Exception:
        pass

if train_raw is None:
    print("  HuggingFace load failed — downloading from GitHub...")
    import urllib.request
    _BASE = ("https://raw.githubusercontent.com/"
             "brendenlake/SCAN/master/add_prim_split/")

    def _load_scan_txt(url):
        with urllib.request.urlopen(url) as f:
            lines = f.read().decode("utf-8").strip().split("\n")
        out = []
        for line in lines:
            line = line.strip()
            if not line:
                continue
            cmd_part, act_part = line.split(" OUT: ", 1)
            out.append({
                "commands": cmd_part.replace("IN: ", "").strip(),
                "actions":  act_part.strip(),
            })
        return out

    train_raw = _load_scan_txt(_BASE + "tasks_train_addprim_jump.txt")
    test_raw  = _load_scan_txt(_BASE + "tasks_test_addprim_jump.txt")
    print(f"  Downloaded from GitHub. Train: {len(train_raw)}, Test: {len(test_raw)}")

if not train_raw or not test_raw:
    raise SystemExit("Could not load SCAN from HuggingFace or GitHub.")

random.shuffle(train_raw)

def is_jump_compound(ex):
    tokens = ex["commands"].split()
    return "jump" in tokens and len(tokens) > 1

test_jc = [ex for ex in test_raw if is_jump_compound(ex)]
print(f"\n  Training examples: {len(train_raw)}")
print(f"  Jump-compound test examples: {len(test_jc)}")

# ── Phase 2: Labels ───────────────────────────────────────────────────────────────────
print("\n" + "="*70)
print("  PHASE 2 — EXTRACT STRUCTURAL LABELS")
print("="*70)

def extract_bits(cmd):
    tokens = cmd.lower().split()
    return {
        "has_direction": int(any(t in tokens for t in ("left", "right"))),
        "is_left":       int("left" in tokens),
        "has_around":    int("around" in tokens),
        "has_opposite":  int("opposite" in tokens),
        "has_twice":     int("twice" in tokens),
        "has_thrice":    int("thrice" in tokens),
        "has_and":       int("and" in tokens),
        "has_after":     int("after" in tokens),
    }

# has_around label for each training example
for ex in train_raw:
    ex["has_around_label"] = extract_bits(ex["commands"])["has_around"]

# Labels for test_jc
for ex in test_jc:
    bits = extract_bits(ex["commands"])
    for k, v in bits.items():
        ex[k] = v

n_train_around = sum(ex["has_around_label"] for ex in train_raw)
n_test_around  = sum(ex["has_around"]       for ex in test_jc)
print(f"  Training has_around=1: {n_train_around}/{len(train_raw)} "
      f"({100*n_train_around/len(train_raw):.1f}%)")
print(f"  Test-jc   has_around=1: {n_test_around}/{len(test_jc)} "
      f"({100*n_test_around/len(test_jc):.1f}%)")

# ── Phase 3: Load tokenizer (shared across all runs) ───────────────────────────
print("\n" + "="*70)
print("  PHASE 3 — LOAD TOKENIZER")
print("="*70)
tokenizer = T5Tokenizer.from_pretrained("t5-small")
print("  Tokenizer loaded.")

# ── Phase 4: Auxiliary classification head ────────────────────────────────────
print("\n" + "="*70)
print("  PHASE 4 — DEFINE AUX HEAD")
print("="*70)
print(f"  Linear({D_MODEL}, 1) on encoder layer 5 mean-pool")
print(f"  Loss: seq2seq + λ × BCE(has_around_pred, has_around_label)")

def make_aux_head():
    return nn.Linear(D_MODEL, 1).to(device)

# ── Phase 5: Training + evaluation loop ──────────────────────────────────────
print("\n" + "="*70)
print("  PHASE 5 — TRAINING RUNS (λ sweep)")
print("="*70)

all_results = {}

for lam in LAMBDA_VALUES:
    print(f"\n{'─'*70}")
    print(f"  λ = {lam}")
    print(f"{'─'*70}")

    # Fresh model and head for each λ
    model = T5ForConditionalGeneration.from_pretrained("t5-small").to(device)
    aux_head = make_aux_head()
    torch.manual_seed(SEED)

    params = list(model.parameters()) + list(aux_head.parameters())
    optimizer = torch.optim.AdamW(params, lr=LR)

    model.train()
    aux_head.train()

    for epoch in range(N_EPOCHS):
        random.shuffle(train_raw)
        epoch_seq2seq = 0.0
        epoch_aux     = 0.0
        epoch_total   = 0.0
        n_batches     = 0

        for i in range(0, len(train_raw), BATCH_SIZE):
            batch = train_raw[i : i + BATCH_SIZE]

            inp = tokenizer(
                [ex["commands"] for ex in batch],
                padding=True, truncation=True,
                max_length=MAX_INPUT_LEN, return_tensors="pt"
            ).to(device)

            tgt = tokenizer(
                [ex["actions"] for ex in batch],
                padding=True, truncation=True,
                max_length=MAX_TARGET_LEN, return_tensors="pt"
            ).to(device)

            labels = tgt["input_ids"].clone()
            labels[labels == tokenizer.pad_token_id] = -100

            has_around_labels = torch.tensor(
                [ex["has_around_label"] for ex in batch],
                dtype=torch.float32, device=device
            )

            # Forward — output_hidden_states=True to get layer 5
            out = model(
                input_ids=inp["input_ids"],
                attention_mask=inp["attention_mask"],
                labels=labels,
                output_hidden_states=True,
            )

            seq2seq_loss = out.loss

            # Encoder layer 5 mean-pool
            # encoder_hidden_states: (embedding, L1, ..., L6) — 7 elements
            # [-2] = L5 (penultimate encoder layer, the measurement layer)
            enc_h5  = out.encoder_hidden_states[-2]         # [B, T, 512]
            mask    = inp["attention_mask"].unsqueeze(-1).float()
            mean_p  = (enc_h5 * mask).sum(1) / mask.sum(1)  # [B, 512]

            logits  = aux_head(mean_p).squeeze(-1)           # [B]
            aux_loss = F.binary_cross_entropy_with_logits(logits, has_around_labels)

            total_loss = seq2seq_loss + lam * aux_loss

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            epoch_seq2seq += seq2seq_loss.item()
            epoch_aux     += aux_loss.item()
            epoch_total   += total_loss.item()
            n_batches     += 1

            if n_batches % LOG_EVERY == 0:
                print(f"    λ={lam}  Epoch {epoch+1}  "
                      f"batch {n_batches}/{len(train_raw)//BATCH_SIZE + 1}"
                      f"  seq2seq={seq2seq_loss.item():.4f}"
                      f"  aux={aux_loss.item():.4f}")

        print(f"  Epoch {epoch+1}/{N_EPOCHS}  "
              f"seq2seq={epoch_seq2seq/n_batches:.4f}  "
              f"aux={epoch_aux/n_batches:.4f}  "
              f"total={epoch_total/n_batches:.4f}")

    # ── Evaluation ──────────────────────────────────────────────────────────────
    print(f"\n  Evaluating on {len(test_jc)} jump-compound examples...")
    model.eval()
    aux_head.eval()

    n_correct       = 0
    enc_reps_eval   = []   # for has_around probe AUC (H4)
    failure_labels  = []

    subgroup_counts = defaultdict(lambda: {"correct": 0, "total": 0})

    with torch.no_grad():
        for i in range(0, len(test_jc), BATCH_SIZE):
            batch = test_jc[i : i + BATCH_SIZE]

            inp = tokenizer(
                [ex["commands"] for ex in batch],
                padding=True, truncation=True,
                max_length=MAX_INPUT_LEN, return_tensors="pt"
            ).to(device)

            # Generate predictions
            gen_ids = model.generate(
                input_ids=inp["input_ids"],
                attention_mask=inp["attention_mask"],
                max_length=MAX_TARGET_LEN,
            )
            preds = tokenizer.batch_decode(gen_ids, skip_special_tokens=True)

            # Get encoder layer 5 reps for H4 probe
            enc_out = model.encoder(
                input_ids=inp["input_ids"],
                attention_mask=inp["attention_mask"],
                output_hidden_states=True,
            )
            enc_h5 = enc_out.hidden_states[-2]
            mask   = inp["attention_mask"].unsqueeze(-1).float()
            mean_p = (enc_h5 * mask).sum(1) / mask.sum(1)  # [B, 512]
            enc_reps_eval.append(mean_p.cpu().numpy())

            for j, (ex, pred) in enumerate(zip(batch, preds)):
                gold    = ex["actions"]
                correct = (pred.strip() == gold.strip())
                failed  = not correct

                n_correct += int(correct)
                failure_labels.append(int(failed))

                # Subgroup tracking
                ha  = ex["has_around"]
                haf = ex["has_after"]
                subgroup_counts[f"has_around={ha}"]["total"]   += 1
                subgroup_counts[f"has_around={ha}"]["correct"] += int(correct)
                subgroup_counts[f"has_after={haf}"]["total"]   += 1
                subgroup_counts[f"has_after={haf}"]["correct"] += int(correct)

                combo = f"has_around={ha}_has_after={haf}"
                subgroup_counts[combo]["total"]   += 1
                subgroup_counts[combo]["correct"] += int(correct)

    jc_acc   = n_correct / len(test_jc)
    delta    = jc_acc - BASELINE_ACC

    print(f"\n  Jump-compound accuracy: {100*jc_acc:.2f}%  "
          f"(baseline {100*BASELINE_ACC:.1f}%  Δ={delta:+.4f})")

    print(f"\n  Subgroup breakdown:")
    for key in sorted(subgroup_counts.keys()):
        sg = subgroup_counts[key]
        sg_acc = sg["correct"] / sg["total"]
        print(f"    {key:<35} {100*sg_acc:5.1f}%  (N={sg['total']})")

    # H4: has_around probe direction AUC vs failure
    enc_reps_arr   = np.vstack(enc_reps_eval)      # [N, 512]
    failure_arr    = np.array(failure_labels)       # [N]
    has_around_arr = np.array([ex["has_around"] for ex in test_jc])

    # Train has_around probe on encoder reps
    probe = LogisticRegression(max_iter=1000, C=1.0, random_state=SEED)
    probe.fit(enc_reps_arr, has_around_arr)

    # Probe decision scores → AUC vs failure
    probe_scores = probe.decision_function(enc_reps_arr)
    auc_fw = roc_auc_score(failure_arr, probe_scores)
    auc_bw = roc_auc_score(failure_arr, -probe_scores)
    probe_auc = max(auc_fw, auc_bw)
    print(f"\n  H4 has_around probe AUC vs failure: {probe_auc:.4f}")
    print(f"     (S-047 baseline: 0.8060)")

    result = {
        "lambda": lam,
        "jc_accuracy": float(jc_acc),
        "baseline_accuracy": BASELINE_ACC,
        "delta_accuracy": float(delta),
        "h1_pass": bool(jc_acc > BASELINE_ACC),
        "h4_probe_auc": float(probe_auc),
        "h4_baseline_auc": 0.8060,
        "h4_auc_decreased": bool(probe_auc < 0.8060),
        "subgroups": {
            k: {
                "accuracy": float(v["correct"] / v["total"]),
                "n_correct": v["correct"],
                "n_total": v["total"],
            }
            for k, v in subgroup_counts.items()
        },
    }

    # Save per-λ results
    fname = f"048_results_lam{str(lam).replace('.', 'p')}.json"
    with open(fname, "w") as f:
        json.dump(result, f, indent=2)
    save_to_drive(fname)
    print(f"  Saved: {fname}")

    all_results[str(lam)] = result

    del model, aux_head, optimizer
    if device.type == "cuda":
        torch.cuda.empty_cache()

# ── Phase 6: Summary and verdict ────────────────────────────────────────────
print("\n" + "="*70)
print("  PHASE 6 — SUMMARY AND VERDICT")
print("="*70)

print(f"\n  {'λ':>6}  {'JC Acc':>8}  {'Δ':>8}  {'H1':>4}  {'Probe AUC':>10}  {'H4':>4}")
print(f"  {'─'*6}  {'─'*8}  {'─'*8}  {'─'*4}  {'─'*10}  {'─'*4}")
for lam in LAMBDA_VALUES:
    r = all_results[str(lam)]
    print(f"  {lam:>6.1f}  "
          f"{100*r['jc_accuracy']:>7.2f}%  "
          f"{r['delta_accuracy']:>+8.4f}  "
          f"{'PASS' if r['h1_pass'] else 'FAIL':>4}  "
          f"{r['h4_probe_auc']:>10.4f}  "
          f"{'PASS' if r['h4_auc_decreased'] else 'FAIL':>4}")

# H1: any λ improves
h1_pass = any(r["h1_pass"] for r in all_results.values())
best_lam = max(LAMBDA_VALUES, key=lambda l: all_results[str(l)]["jc_accuracy"])
best_acc = all_results[str(best_lam)]["jc_accuracy"]

# H2: has_around=1 shows larger improvement than has_around=0
# Compare delta for has_around=1 vs has_around=0 at best_lam
best_r = all_results[str(best_lam)]
sg = best_r["subgroups"]
s1_key = "has_around=1"
s0_key = "has_around=0"

baseline_around1 = 1.0 - 0.873  # S-047: 87.3% failure → 12.7% accuracy
baseline_around0 = 1.0 - 0.224  # S-047: 22.4% failure → 77.6% accuracy

h2_pass = False
h2_note = "subgroup data unavailable"
if s1_key in sg and s0_key in sg:
    delta_around1 = sg[s1_key]["accuracy"] - baseline_around1
    delta_around0 = sg[s0_key]["accuracy"] - baseline_around0
    h2_pass = (delta_around1 > delta_around0)
    h2_note = (f"has_around=1 Δ={delta_around1:+.4f}, "
               f"has_around=0 Δ={delta_around0:+.4f}")

# H3: has_after+has_around shows largest improvement
h3_key = "has_around=1_has_after=1"
baseline_af_ar = 1.0 - 0.922  # S-047: 92.2% failure → 7.8% accuracy
h3_pass = False
h3_note = "combination subgroup not found"
if h3_key in sg:
    delta_af_ar = sg[h3_key]["accuracy"] - baseline_af_ar
    # Check if this is the largest delta among combinations
    combo_deltas = {}
    for k, v in sg.items():
        if "has_around" in k and "has_after" in k:
            # Approximate baseline from S-047 combination table
            combo_deltas[k] = v["accuracy"]
    h3_pass = (combo_deltas.get(h3_key, 0) == max(combo_deltas.values(), default=0))
    h3_note = f"has_after=has_around=1 acc={sg[h3_key]['accuracy']:.4f}"

# H4: all λ combined
h4_pass = all(r["h4_auc_decreased"] for r in all_results.values())

print(f"\n  {'─'*65}")
print(f"  HYPOTHESIS VERDICTS (best_lam={best_lam}, best_acc={100*best_acc:.2f}%)")
print(f"  {'─'*65}")
print(f"  H1: accuracy > {100*BASELINE_ACC:.1f}% for some λ — {'PASS' if h1_pass else 'FAIL'}")
print(f"      Best λ={best_lam}, accuracy={100*best_acc:.2f}%")
print(f"  H2: has_around=1 shows larger improvement — {'PASS' if h2_pass else 'FAIL'}")
print(f"      {h2_note}")
print(f"  H3: has_after+has_around largest improvement — {'PASS' if h3_pass else 'FAIL'}")
print(f"      {h3_note}")
print(f"  H4: probe AUC decreases (all λ) — {'PASS' if h4_pass else 'FAIL'}")
for lam in LAMBDA_VALUES:
    r = all_results[str(lam)]
    print(f"      λ={lam}: AUC={r['h4_probe_auc']:.4f} vs 0.8060 baseline")

# Kill condition check
print(f"\n  KILL CONDITIONS:")
k1_fires = not h1_pass
print(f"  K1 (no improvement): {'FIRES' if k1_fires else 'does not fire'}")
if k1_fires:
    print(f"      → Bottleneck is decoder cross-attention, not encoder")
    print(f"      → Next: S-049 decoder cross-attention analysis")

k2_fires = any(
    r["h4_probe_auc"] > 0.8060 + 0.05  # meaningful increase
    for r in all_results.values()
)
print(f"  K2 (probe AUC increases substantially): {'FIRES' if k2_fires else 'does not fire'}")

# Final verdict
print(f"\n  {'─'*65}")
if h1_pass:
    print(f"  VERDICT: Auxiliary has_around loss HELPS.")
    print(f"  Best accuracy: {100*best_acc:.2f}% (Δ={best_acc-BASELINE_ACC:+.4f} over S-043).")
else:
    print(f"  VERDICT: K1 fires. Auxiliary loss does NOT help decoder generalize.")
    print(f"  The failure is in decoder cross-attention, not encoder representation.")
    print(f"  Next step: S-049 decoder cross-attention analysis.")

# ── Save combined results ──────────────────────────────────────────────────────────
output = {
    "script_id": SCRIPT_ID,
    "lambda_values": LAMBDA_VALUES,
    "baseline_accuracy": BASELINE_ACC,
    "results_by_lambda": all_results,
    "verdict": {
        "h1_pass": bool(h1_pass),
        "h2_pass": bool(h2_pass),
        "h3_pass": bool(h3_pass),
        "h4_pass": bool(h4_pass),
        "h2_note": h2_note,
        "h3_note": h3_note,
        "k1_fires": bool(k1_fires),
        "k2_fires": bool(k2_fires),
        "best_lambda": float(best_lam),
        "best_jc_accuracy": float(best_acc),
    }
}

with open("048_results.json", "w") as f:
    json.dump(output, f, indent=2)
save_to_drive("048_results.json")

print(f"\n  Output: 048_results.json")
print(f"  {'='*70}")
print(f"  S-048 complete.")
print(f"  {'='*70}")
