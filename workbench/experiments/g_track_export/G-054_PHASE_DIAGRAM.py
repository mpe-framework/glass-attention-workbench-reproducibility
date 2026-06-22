"""
G-054: Collapse Conditions — Mapping the Phase Transition
==========================================================
Applied Categorical Physics Workbench | Troy Teno | 2026 | PyTorch + NumPy

QUESTION
--------
What geometric condition in the embedding space causes the FIN specialist head to
collapse in the unambiguous geographic state (S3)? How much FIN signal in the bank
token is sufficient to prevent that collapse — and does suppressing FIN signal in
bank reliably produce the G-050 mid-layer peak?

MOTIVATION
----------
G-053 established that the mid-layer Born filter peak requires a two-component
mechanism:
  (1) The FIN specialist head must collapse in the geographic state (S3) — no FIN
      tokens to attend to → self-attention → near-zero defect by mid-layers.
  (2) The ambiguous (S1) and financial (S2) states must retain FIN signal so their
      defects remain large.

G-053's SEED=53 embeddings gave bank a strong FIN component even in S3, preventing
collapse. G-050's SEED=50 embeddings happened to give bank a weaker FIN projection
in S3, allowing collapse and producing the 2.58× peak at layer 5.

G-054 tests this explanation directly. The single variable: the FIN embedding weight
of the bank token. All other embeddings, weights, and architectures are held fixed.

DESIGN
------
- 12-layer BERT with semantic Q/K/V init (identical to G-049/G-050/G-053)
- SEED=54 for numpy embeddings and torch weights
- bank token constructed as:
    bank_raw = geo_bank + FIN_WEIGHT * fin_bank
    bank = bank_raw / ‖bank_raw‖
  Sweep FIN_WEIGHT ∈ {0.0, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0, 1.5, 2.0}
- At FIN_WEIGHT=0.0: bank is purely GEO — guaranteed no FIN signal in S3
- At FIN_WEIGHT=1.0: bank is equally GEO and FIN (baseline)
- torch.manual_seed held constant across sweep — only bank embedding changes
- Per-head Born filter at every layer (forward hooks, identical to G-050/G-053)
- Measurement: peak H1(FIN) ratio and peak layer vs. FIN_WEIGHT

PRE-REGISTERED HYPOTHESES (locked before run)
----------------------------------------------
H1 [Collapse at FIN_WEIGHT=0.0]: At FIN_WEIGHT=0.0, the H1(FIN) specialist head
    ratio peaks at layer > 1 (not layer 1) AND the peak ratio exceeds 2.0×.
    This confirms the mechanistic explanation: pure-GEO bank creates the collapse
    condition in S3, producing the mid-layer amplification.

H2 [Phase transition exists]: There is a FIN_WEIGHT_threshold ∈ (0.0, 1.0] above
    which the peak collapses to layer 1 (monotonic decay). The transition is not
    monotone across the full range — there is a detectable step or shoulder.

H3 [Threshold is low]: The phase transition threshold is below FIN_WEIGHT = 0.5.
    A small amount of FIN signal in bank is sufficient to prevent FIN-head collapse
    in S3 and kill the mid-layer peak.

K_transition [Full mapping]: Peak ratio and peak layer tracked continuously across
    all FIN_WEIGHT values. Both should decrease monotonically (or step-wise) from
    FIN_WEIGHT=0 to FIN_WEIGHT=2.0.

FALSIFICATION CRITERIA
-----------------------
H1 fails if FIN_WEIGHT=0.0 still produces monotonic decay (peak at layer 1) or
   peak ratio ≤ 2.0×.
H2 fails if peak is always at layer 1 regardless of FIN_WEIGHT (no transition at all).
H3 fails if threshold is at FIN_WEIGHT ≥ 0.5 (transition requires substantial FIN signal).
"""

import numpy as np
import torch
from transformers import BertConfig, BertModel
import json

print("=== G-054: Collapse Conditions — Mapping the Phase Transition ===")
print("Pre-registered hypotheses locked.\n")

# ── Shared constants (identical to G-049/G-050/G-053) ─────────────────────────
D_MODEL  = 64
N_HEADS  = 4
D_HEAD   = D_MODEL // N_HEADS   # 16
SEQ_LEN  = 8
N_LAYERS = 12
STRENGTH = 2.0
NOISE    = 0.02
CLUSTER_WEIGHT = 3.0
GEO_DIMS = slice(0,  16)
FIN_DIMS = slice(16, 32)
BANK_POS = 0
SWAP_POS = 1
SEED     = 54

rng = np.random.default_rng(SEED)
torch.manual_seed(SEED)

I16 = np.eye(D_HEAD, dtype=np.float32)

# ── Fixed base embeddings (everything except bank is held constant) ────────────
def make_cluster(n, dim_slice, std=0.05):
    vecs = rng.standard_normal((n, D_MODEL)) * std
    for v in vecs:
        v[dim_slice] += CLUSTER_WEIGHT * 0.5
        v /= np.linalg.norm(v)
    return vecs

geo_bank_base = rng.standard_normal(D_MODEL) * NOISE
geo_bank_base[GEO_DIMS] += CLUSTER_WEIGHT * 0.5
geo_bank_base /= np.linalg.norm(geo_bank_base)

fin_bank_base = rng.standard_normal(D_MODEL) * NOISE
fin_bank_base[FIN_DIMS] += CLUSTER_WEIGHT * 0.5
fin_bank_base /= np.linalg.norm(fin_bank_base)

current = rng.standard_normal(D_MODEL) * NOISE
current[GEO_DIMS] += CLUSTER_WEIGHT * 0.3
current[FIN_DIMS] += CLUSTER_WEIGHT * 0.3
current /= np.linalg.norm(current)

geo_tokens = make_cluster(3, GEO_DIMS)
fin_tokens = make_cluster(3, FIN_DIMS)

def make_bank(fin_weight):
    """Construct bank token with given FIN embedding weight. GEO weight = 1.0 always."""
    raw = geo_bank_base + fin_weight * fin_bank_base
    return raw / np.linalg.norm(raw)

def make_sequences(bank):
    S1 = np.vstack([bank,    current, geo_tokens[0], fin_tokens[0],
                    fin_tokens[1], geo_tokens[1], fin_tokens[2], geo_tokens[2]])
    S2 = np.vstack([bank,    current, fin_tokens[0], fin_tokens[1],
                    fin_tokens[2], fin_tokens[0], fin_tokens[1], fin_tokens[2]])
    S3 = np.vstack([bank,    current, geo_tokens[0], geo_tokens[1],
                    geo_tokens[2], geo_tokens[0], geo_tokens[1], geo_tokens[2]])
    S1_swap = S1.copy(); S1_swap[[BANK_POS, SWAP_POS]] = S1_swap[[SWAP_POS, BANK_POS]]
    S2_swap = S2.copy(); S2_swap[[BANK_POS, SWAP_POS]] = S2_swap[[SWAP_POS, BANK_POS]]
    S3_swap = S3.copy(); S3_swap[[BANK_POS, SWAP_POS]] = S3_swap[[SWAP_POS, BANK_POS]]
    return S1, S2, S3, S1_swap, S2_swap, S3_swap

# ── Build model once — torch seed fixed; only embeddings change ───────────────
def build_model():
    config = BertConfig(
        hidden_size=D_MODEL,
        num_hidden_layers=N_LAYERS,
        num_attention_heads=N_HEADS,
        intermediate_size=D_MODEL * 4,
        max_position_embeddings=SEQ_LEN + 2,
        vocab_size=30522,
    )
    config._attn_implementation = "eager"
    model = BertModel(config)
    model.eval()

    for layer_module in model.encoder.layer:
        attn = layer_module.attention.self
        Wq, Wk, Wv = attn.query.weight, attn.key.weight, attn.value.weight
        Wo = layer_module.attention.output.dense.weight

        with torch.no_grad():
            Wq.data[0:D_HEAD,         0:D_HEAD]         = STRENGTH * torch.tensor(I16)
            Wk.data[0:D_HEAD,         0:D_HEAD]         = STRENGTH * torch.tensor(I16)
            Wv.data[0:D_HEAD,         0:D_HEAD]         = torch.tensor(I16)
            Wq.data[D_HEAD:2*D_HEAD,  D_HEAD:2*D_HEAD]  = STRENGTH * torch.tensor(I16)
            Wk.data[D_HEAD:2*D_HEAD,  D_HEAD:2*D_HEAD]  = STRENGTH * torch.tensor(I16)
            Wv.data[D_HEAD:2*D_HEAD,  D_HEAD:2*D_HEAD]  = torch.tensor(I16)
            Wo.data[:, 0:D_HEAD]                         = 0.0
            Wo.data[:, D_HEAD:2*D_HEAD]                  = 0.0
            Wo.data[GEO_DIMS, 0:D_HEAD]                  = torch.tensor(I16)
            Wo.data[FIN_DIMS,  D_HEAD:2*D_HEAD]          = torch.tensor(I16)
    return model

torch.manual_seed(SEED)
model = build_model()

def to_tensor(arr):
    return torch.tensor(arr, dtype=torch.float32).unsqueeze(0)

# ── Per-head Born filter (same hook pattern as G-050/G-053) ──────────────────
def perhead_born_filter(seq_orig, seq_swap):
    """Returns dict: layer_idx → per-head defect norms at BANK_POS."""
    caps_orig, caps_swap = {}, {}

    def make_hook(store, li):
        def hook(m, inp, out):
            store[li] = out[0].detach().cpu().numpy()[0].reshape(SEQ_LEN, N_HEADS, D_HEAD)
        return hook

    hooks = [model.encoder.layer[i].attention.self.register_forward_hook(
             make_hook(caps_orig, i)) for i in range(N_LAYERS)]
    with torch.no_grad():
        model(inputs_embeds=to_tensor(seq_orig))
    for h in hooks: h.remove()

    hooks = [model.encoder.layer[i].attention.self.register_forward_hook(
             make_hook(caps_swap, i)) for i in range(N_LAYERS)]
    with torch.no_grad():
        model(inputs_embeds=to_tensor(seq_swap))
    for h in hooks: h.remove()

    result = {}
    for li in range(N_LAYERS):
        h_B = (caps_orig[li][BANK_POS] - caps_swap[li][BANK_POS]) / 2.0
        result[li] = np.linalg.norm(h_B, axis=1)  # [N_HEADS]
    return result

# ── FIN_WEIGHT sweep ──────────────────────────────────────────────────────────────
FIN_WEIGHTS = [0.0, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0, 1.5, 2.0]

print(f"{'FIN_WEIGHT':>12}  {'Peak layer':>11}  {'Peak % depth':>13}  "
      f"{'Peak ratio':>11}  {'S3 defect L5':>13}  {'Note':>20}")
print(f"{'─'*12}  {'─'*11}  {'─'*13}  {'─'*11}  {'─'*13}  {'─'*20}")

sweep_results = []

for fw in FIN_WEIGHTS:
    bank = make_bank(fw)
    S1, S2, S3, S1s, S2s, S3s = make_sequences(bank)

    r1 = perhead_born_filter(S1, S1s)
    r2 = perhead_born_filter(S2, S2s)
    r3 = perhead_born_filter(S3, S3s)

    # H1(FIN) ratio per layer
    ratios = []
    s3_fin_defects = []
    for li in range(N_LAYERS):
        d1 = float(r1[li][1])
        d2 = float(r2[li][1])
        d3 = float(r3[li][1])
        mean23 = (d2 + d3) / 2.0
        ratio = d1 / mean23 if mean23 > 1e-8 else float('nan')
        ratios.append(ratio)
        s3_fin_defects.append(d3)

    valid = [(i, r) for i, r in enumerate(ratios) if not np.isnan(r)]
    peak_idx, peak_ratio = max(valid, key=lambda x: x[1])
    peak_layer = peak_idx + 1
    peak_pct   = 100.0 * peak_layer / N_LAYERS
    s3_l5      = s3_fin_defects[4]  # layer 5 (index 4)

    note = ""
    if peak_layer == 1:
        note = "monotonic decay"
    elif peak_layer <= 4:
        note = "early peak"
    elif peak_layer <= 7:
        note = "mid-layer peak"
    else:
        note = "late peak"

    print(f"  {fw:>10.1f}  layer {peak_layer:2d}/{N_LAYERS}  {peak_pct:>11.1f}%  "
          f"  {peak_ratio:>9.4f}×  {s3_l5:>12.6f}  {note:>20}")

    sweep_results.append({
        'fin_weight': float(fw),
        'peak_layer': int(peak_layer),
        'peak_pct': float(peak_pct),
        'peak_ratio': float(peak_ratio),
        's3_fin_defect_L5': float(s3_l5),
        'all_ratios': [float(r) if not np.isnan(r) else None for r in ratios],
        'all_s3_fin_defects': [float(d) for d in s3_fin_defects],
        'note': note,
    })

# ── Detailed per-layer output for key FIN_WEIGHT values ───────────────────────
print()
key_weights = [0.0, 0.3, 1.0]
for fw in key_weights:
    res = next(r for r in sweep_results if r['fin_weight'] == fw)
    print(f"  FIN_WEIGHT={fw:.1f} — per-layer H1(FIN) ratio and S3 H1 defect:")
    print(f"  {'Layer':>6}  {'H1 ratio':>10}  {'S3 H1 defect':>13}  {'Peak':>5}")
    for li in range(N_LAYERS):
        ratio = res['all_ratios'][li]
        s3d   = res['all_s3_fin_defects'][li]
        marker = " ←" if li + 1 == res['peak_layer'] else ""
        ratio_str = f"{ratio:.4f}×" if ratio is not None else "    nan"
        print(f"  {li+1:>6}  {ratio_str:>10}  {s3d:>13.6f}  {marker}")
    print()

# ── Evaluate hypotheses ────────────────────────────────────────────────────────────
print("─" * 62)

# Find FIN_WEIGHT=0.0 result
res_0 = next(r for r in sweep_results if r['fin_weight'] == 0.0)
h1_pass = (res_0['peak_layer'] > 1) and (res_0['peak_ratio'] > 2.0)
print(f"\n[H1] FIN_WEIGHT=0.0 produces mid-layer peak > 2.0×:")
print(f"     Peak at layer {res_0['peak_layer']}/{N_LAYERS}  ratio={res_0['peak_ratio']:.4f}×")
print(f"     {'CONFIRMED' if h1_pass else 'NULL'} "
      f"(need peak_layer > 1 AND peak_ratio > 2.0×)")

# Find phase transition (first FIN_WEIGHT where peak collapses to layer 1)
transition_weight = None
prev_peak = None
for res in sweep_results:
    if prev_peak is not None and res['peak_layer'] == 1 and prev_peak > 1:
        transition_weight = res['fin_weight']
        break
    prev_peak = res['peak_layer']

h2_pass = transition_weight is not None
print(f"\n[H2] Phase transition exists (peak collapses to layer 1):")
if h2_pass:
    print(f"     Transition between FIN_WEIGHT={sweep_results[sweep_results.index(next(r for r in sweep_results if r['fin_weight'] == transition_weight))-1]['fin_weight']:.1f} "
          f"and {transition_weight:.1f}")
else:
    print(f"     Peak remains at layer 1 across all FIN_WEIGHT values")
print(f"     {'CONFIRMED' if h2_pass else 'NULL'}")

h3_pass = h2_pass and (transition_weight is not None) and (transition_weight < 0.5)
print(f"\n[H3] Transition threshold < 0.5 (small FIN signal suppresses collapse):")
if transition_weight is not None:
    print(f"     Transition at FIN_WEIGHT ≈ {transition_weight:.2f}")
print(f"     {'CONFIRMED' if h3_pass else 'NULL'} (threshold {'< 0.5' if h3_pass else '>= 0.5 or not found'})")

# K_transition summary
print(f"\n[K_transition] Peak ratio and peak layer vs. FIN_WEIGHT:")
print(f"  {'FIN_WEIGHT':>12}  {'Peak layer':>11}  {'Peak ratio':>11}")
for res in sweep_results:
    print(f"  {res['fin_weight']:>12.1f}  layer {res['peak_layer']:2d}/{N_LAYERS}  "
          f"  {res['peak_ratio']:>9.4f}×")

# ── Results summary ──────────────────────────────────────────────────────────────
print()
print("─" * 62)
print()
print("=== G-054 RESULTS ===\n")

n_confirmed = sum([h1_pass, h2_pass, h3_pass])
print(f"{n_confirmed}/3 hypotheses confirmed.\n")

print("=== HYPOTHESIS SUMMARY ===")
for label, result, detail in [
    ("H1", "CONFIRMED" if h1_pass else "NULL",
     f"FIN_WEIGHT=0.0: peak at layer {res_0['peak_layer']}, ratio={res_0['peak_ratio']:.4f}×"),
    ("H2", "CONFIRMED" if h2_pass else "NULL",
     f"Transition at FIN_WEIGHT={transition_weight}" if transition_weight else "No transition found"),
    ("H3", "CONFIRMED" if h3_pass else "NULL",
     f"Threshold = {transition_weight}" if transition_weight else "N/A"),
]:
    print(f"  {label}: {result} — {detail}")

print()
results_json = {
    "experiment": "G-054",
    "question": "What FIN embedding weight in bank suppresses the FIN-head collapse, and where is the phase transition?",
    "hypotheses": {
        "H1": {
            "result": "CONFIRMED" if h1_pass else "NULL",
            "fin_weight_0_peak_layer": res_0['peak_layer'],
            "fin_weight_0_peak_ratio": round(res_0['peak_ratio'], 4),
            "threshold_ratio": 2.0
        },
        "H2": {
            "result": "CONFIRMED" if h2_pass else "NULL",
            "transition_weight": transition_weight
        },
        "H3": {
            "result": "CONFIRMED" if h3_pass else "NULL",
            "threshold": transition_weight,
            "criterion": "< 0.5"
        }
    },
    "sweep": [
        {
            "fin_weight": r['fin_weight'],
            "peak_layer": r['peak_layer'],
            "peak_pct": r['peak_pct'],
            "peak_ratio": round(r['peak_ratio'], 4),
            "s3_fin_defect_L5": round(r['s3_fin_defect_L5'], 6),
            "note": r['note']
        }
        for r in sweep_results
    ]
}
print(json.dumps(results_json, indent=2))

print()
print("--- SEALED ---")
print("G-054 is sealed. Do not modify this script or re-run to improve outcomes.")
print("Results recorded in findings/FINDINGS.md as F-054.")
