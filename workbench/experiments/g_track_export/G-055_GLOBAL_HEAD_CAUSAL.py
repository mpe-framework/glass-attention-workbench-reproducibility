#!/usr/bin/env python3
"""
G-055 — Global-Head Causal vs. Diagnostic Distinction
Applied Categorical Physics Workbench | Troy Teno | May 2026
NumPy only. Sealed on first run.

Question: Does the global-engagement head at layer 3 causally set up conditions
for specialist-head collapse at layers 4-5, or is it a diagnostic reader of the
same encoder signal that independently drives the specialist heads?

Pre-registration: workbench/proposals/G055_GLOBAL_HEAD_CAUSAL_DIAGNOSTIC.md

H1 (Causal):    Ablating layer-3 output changes L4-6 Born filter ratio by >50%
H2 (Diagnostic):Ablating layer-3 output changes L4-6 Born filter ratio by <10%
H3:             Global head Born filter profile is monotonic through layer 3
H4:             Cross-state L3/L5 defect correlation; test whether ablation
                disrupts or preserves the correlation
"""

import numpy as np
from numpy.linalg import norm

RNG = np.random.default_rng(55)

# ── Dimensions ──────────────────────────────────────────────────────────────────────────────
D      = 16
D_GEO  = 8     # indices 0-7
D_FIN  = 8     # indices 8-15
N_TOK  = 6     # [bank, current, geo1, geo2, fin1, fin2]
N_LAY  = 6
SQRT_D = np.sqrt(D)

BANK_POS    = 0
CURRENT_POS = 1

# ── Cluster bases ─────────────────────────────────────────────────────────────────────────────
GEO_BASE = np.zeros(D); GEO_BASE[:D_GEO] = 1.0; GEO_BASE /= norm(GEO_BASE)
FIN_BASE = np.zeros(D); FIN_BASE[D_GEO:] = 1.0; FIN_BASE /= norm(FIN_BASE)
# (reassignment to suppress F401 warning for _g/_f)

def make_token(geo_w=1.0, fin_w=0.0, noise=0.05):
    v = geo_w * GEO_BASE + fin_w * FIN_BASE + RNG.normal(0, noise, D)
    return v / norm(v)

# Context tokens fixed for all sequences
CURRENT = make_token(geo_w=1.0, fin_w=0.0)
GEO1    = make_token(geo_w=1.0, fin_w=0.0)
GEO2    = make_token(geo_w=1.0, fin_w=0.0)
FIN1    = make_token(geo_w=0.0, fin_w=1.0)
FIN2    = make_token(geo_w=0.0, fin_w=1.0)
TOK_NAMES = ["bank", "current", "geo1", "geo2", "fin1", "fin2"]

def make_sequence(fin_weight_bank):
    """Build 6-token sequence; bank embedding varies, context is fixed."""
    bank = GEO_BASE + fin_weight_bank * FIN_BASE
    bank = bank / norm(bank)
    return np.stack([bank, CURRENT, GEO1, GEO2, FIN1, FIN2])  # (N_TOK, D)

def swap_tokens(seq, i=BANK_POS, j=CURRENT_POS):
    s = seq.copy(); s[i] = seq[j]; s[j] = seq[i]
    return s

# ── Q/K matrices per layer (V = I throughout) ──────────────────────────────────────────
def _near_id(scale=0.1):
    return scale * np.eye(D)

def _global():
    # Near-uniform attention: tiny uniform Q/K → all scores ≈ equal → softmax → 1/N
    return 0.005 * np.ones((D, D))

def _fin_spec(strength=4.0):
    Q = np.zeros((D, D))
    Q[D_GEO:, D_GEO:] = strength * np.eye(D_FIN)
    return Q

V_ID = np.eye(D)

# 0-indexed layer configs: (Q_mat, K_mat)
# Layer 0 → displayed as L1 (near-id)
# Layer 1 → displayed as L2 (near-id)
# Layer 2 → displayed as L3 (GLOBAL)
# Layer 3 → displayed as L4 (FIN-spec)
# Layer 4 → displayed as L5 (FIN-spec)
# Layer 5 → displayed as L6 (near-id)
LAYER_QK = [
    (_near_id(0.1), _near_id(0.1)),   # L1 near-id
    (_near_id(0.1), _near_id(0.1)),   # L2 near-id  ← control ablation target
    (_global(),     _global()    ),   # L3 GLOBAL   ← primary ablation target
    (_fin_spec(),   _fin_spec()  ),   # L4 FIN-spec
    (_fin_spec(),   _fin_spec()  ),   # L5 FIN-spec
    (_near_id(0.1), _near_id(0.1)),   # L6 near-id
]

LAY_LABELS = [
    "L0 (input)",
    "L1 (near-id)",
    "L2 (near-id)",
    "L3 (GLOBAL)",
    "L4 (FIN-spec)",
    "L5 (FIN-spec)",
    "L6 (near-id)",
]

# ── Attention + forward pass ────────────────────────────────────────────────────────────────────
def _head(X, Q_mat, K_mat):
    """Returns (head_output, attention_weights). V = I so values = X."""
    Q = X @ Q_mat.T                                    # (n, D)
    K = X @ K_mat.T                                    # (n, D)
    S = (Q @ K.T) / SQRT_D                             # (n, n)
    S -= S.max(axis=1, keepdims=True)
    W = np.exp(S); W /= W.sum(axis=1, keepdims=True)  # (n, n)
    return W @ X, W                                    # (n, D), (n, n)

def forward(seq, ablate_layer=None):
    """
    Run seq through N_LAY layers with residual connections.
    ablate_layer (0-indexed): zero that layer's head output.
    Returns: (hidden_states list of N_LAY+1, attn_weights list of N_LAY)
    hidden_states[0] = input embeddings; hidden_states[l+1] = after layer l.
    """
    X = seq.copy()
    hidden = [X.copy()]
    attn_ws = []
    for l in range(N_LAY):
        head_out, W = _head(X, *LAYER_QK[l])
        attn_ws.append(W)
        if ablate_layer is not None and l == ablate_layer:
            head_out = np.zeros_like(head_out)
        X = X + head_out
        hidden.append(X.copy())
    return hidden, attn_ws

def born_profile(seq, morphed, ablate_layer=None):
    """Born filter ‖h_B(bank)‖ at every hidden state index (0..N_LAY)."""
    h_o, _ = forward(seq, ablate_layer)
    h_m, _ = forward(morphed, ablate_layer)
    return np.array([norm(h_o[l][BANK_POS] - h_m[l][BANK_POS]) / 2.0
                     for l in range(N_LAY + 1)])

# ── States ──────────────────────────────────────────────────────────────────────────────────
FIN_SUCCESS = 0.5    # no-collapse regime (G-054 ≥ 0.5)
FIN_FAIL    = 0.05   # slow-collapse regime (G-054 ≤ 0.1)

seq_s  = make_sequence(FIN_SUCCESS);  morph_s = swap_tokens(seq_s)
seq_f  = make_sequence(FIN_FAIL);     morph_f = swap_tokens(seq_f)

def ratio(bf_s, bf_f):
    return np.where(bf_f > 1e-10, bf_s / bf_f, np.inf)

# ── Run ───────────────────────────────────────────────────────────────────────────────────────
print("=" * 72)
print("G-055 — Global-Head Causal vs. Diagnostic Distinction")
print("Applied Categorical Physics Workbench | Troy Teno | May 2026")
print("=" * 72)

s_fin = seq_s[BANK_POS, D_GEO:].sum()
f_fin = seq_f[BANK_POS, D_GEO:].sum()
print(f"\nBank FIN content  — S_success (FW={FIN_SUCCESS}): {s_fin:.4f}")
print(f"                  — S_fail    (FW={FIN_FAIL}):  {f_fin:.4f}")

# ── STEP 1: Baseline ─────────────────────────────────────────────────────────────────────────
bf_s_base = born_profile(seq_s, morph_s)
bf_f_base = born_profile(seq_f, morph_f)
r_base    = ratio(bf_s_base, bf_f_base)

print("\n" + "─" * 72)
print("STEP 1 — BASELINE Born Filter Profiles")
print("─" * 72)
print(f"{'Layer':<18} {'S_success':>10} {'S_fail':>10} {'Ratio S/F':>12}")
print("-" * 52)
for l in range(N_LAY + 1):
    print(f"{LAY_LABELS[l]:<18} {bf_s_base[l]:>10.4f} {bf_f_base[l]:>10.4f} {r_base[l]:>12.4f}")

# ── STEP 2: Layer-3 ablation (primary test) ──────────────────────────────────────────
bf_s_abl3 = born_profile(seq_s, morph_s, ablate_layer=2)
bf_f_abl3 = born_profile(seq_f, morph_f, ablate_layer=2)
r_abl3    = ratio(bf_s_abl3, bf_f_abl3)
dpct_abl3 = (r_abl3 - r_base) / np.where(r_base > 0, r_base, 1.0) * 100

print("\n" + "─" * 72)
print("STEP 2 — LAYER-3 (GLOBAL HEAD) ABLATED")
print("─" * 72)
print(f"{'Layer':<18} {'S_success':>10} {'S_fail':>10} {'Ratio S/F':>12} {'Δ ratio%':>10}")
print("-" * 62)
for l in range(N_LAY + 1):
    print(f"{LAY_LABELS[l]:<18} {bf_s_abl3[l]:>10.4f} {bf_f_abl3[l]:>10.4f} "
          f"{r_abl3[l]:>12.4f} {dpct_abl3[l]:>10.1f}%")

# ── STEP 3: Layer-2 ablation (control) ────────────────────────────────────────────
bf_s_abl2 = born_profile(seq_s, morph_s, ablate_layer=1)
bf_f_abl2 = born_profile(seq_f, morph_f, ablate_layer=1)
r_abl2    = ratio(bf_s_abl2, bf_f_abl2)
dpct_abl2 = (r_abl2 - r_base) / np.where(r_base > 0, r_base, 1.0) * 100

print("\n" + "─" * 72)
print("STEP 3 — LAYER-2 (NEAR-IDENTITY CONTROL) ABLATED")
print("─" * 72)
print(f"{'Layer':<18} {'S_success':>10} {'S_fail':>10} {'Ratio S/F':>12} {'Δ ratio%':>10}")
print("-" * 62)
for l in range(N_LAY + 1):
    print(f"{LAY_LABELS[l]:<18} {bf_s_abl2[l]:>10.4f} {bf_f_abl2[l]:>10.4f} "
          f"{r_abl2[l]:>12.4f} {dpct_abl2[l]:>10.1f}%")

# ── STEP 4: Attention weight diagnostics ─────────────────────────────────────────────
_, aw_s      = forward(seq_s)
_, aw_f      = forward(seq_f)
_, aw_s_abl3 = forward(seq_s, ablate_layer=2)
_, aw_f_abl3 = forward(seq_f, ablate_layer=2)

print("\n" + "─" * 72)
print("STEP 4 — ATTENTION: FROM BANK POSITION AT FIN-SPECIALIST LAYERS")
print("─" * 72)
for li, lname in [(3, "L4 (FIN-spec, code-idx 3)"), (4, "L5 (FIN-spec, code-idx 4)")]:
    print(f"\n  {lname}:")
    print(f"  {'→ token':<10} {'S_succ':>8} {'S_fail':>8} {'S_succ(abl3)':>14} {'S_fail(abl3)':>14}")
    print("  " + "-" * 56)
    for j, tn in enumerate(TOK_NAMES):
        ws  = aw_s[li][BANK_POS, j]
        wf  = aw_f[li][BANK_POS, j]
        wsa = aw_s_abl3[li][BANK_POS, j]
        wfa = aw_f_abl3[li][BANK_POS, j]
        print(f"  {tn:<10} {ws:>8.4f} {wf:>8.4f} {wsa:>14.4f} {wfa:>14.4f}")

# Global head attention weights (verify uniformity)
print(f"\n  L3 (GLOBAL, code-idx 2) — attention FROM bank (should be ≈1/6=0.167 for all):")
print(f"  {'→ token':<10} {'S_succ':>8} {'S_fail':>8}")
print("  " + "-" * 28)
_, aw_s_raw = forward(seq_s)
_, aw_f_raw = forward(seq_f)
for j, tn in enumerate(TOK_NAMES):
    ws = aw_s_raw[2][BANK_POS, j]
    wf = aw_f_raw[2][BANK_POS, j]
    print(f"  {tn:<10} {ws:>8.4f} {wf:>8.4f}")

# ── STEP 5: Cross-state correlation (H4) ─────────────────────────────────────────────
FIN_W_SWEEP = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 0.7]

l3_base, l5_base = [], []
l3_abl,  l5_abl  = [], []

for fw in FIN_W_SWEEP:
    sq = make_sequence(fw); mq = swap_tokens(sq)
    pb = born_profile(sq, mq)
    pa = born_profile(sq, mq, ablate_layer=2)
    l3_base.append(pb[3]); l5_base.append(pb[5])
    l3_abl.append(pa[3]);  l5_abl.append(pa[5])

l3b = np.array(l3_base); l5b = np.array(l5_base)
l3a = np.array(l3_abl);  l5a = np.array(l5_abl)

def pearson(a, b):
    ac = a - a.mean(); bc = b - b.mean()
    d = norm(ac) * norm(bc)
    return float(np.dot(ac, bc) / d) if d > 1e-10 else 0.0

r_corr_base = pearson(l3b, l5b)
r_corr_abl  = pearson(l3a, l5a)

print("\n" + "─" * 72)
print("STEP 5 — H4: CROSS-STATE CORRELATION (9 FIN_WEIGHT variants)")
print("─" * 72)
print(f"{'FIN_W':<8} {'L3 base':>9} {'L5 base':>9} {'L3 abl':>9} {'L5 abl':>9}")
print("-" * 40)
for i, fw in enumerate(FIN_W_SWEEP):
    print(f"{fw:<8.2f} {l3b[i]:>9.4f} {l5b[i]:>9.4f} {l3a[i]:>9.4f} {l5a[i]:>9.4f}")
print(f"\nPearson r(L3, L5) baseline: {r_corr_base:+.4f}")
print(f"Pearson r(L3, L5) ablated:  {r_corr_abl:+.4f}")
print(f"Δr = {r_corr_abl - r_corr_base:+.4f}")

# ── Verdicts ───────────────────────────────────────────────────────────────────────────────────────
# Specialist layers: hidden state indices 4, 5, 6 (after layers L4, L5, L6)
SPEC = [4, 5, 6]
max_pct_abl3 = max(abs(dpct_abl3[i]) for i in SPEC)
max_pct_ctrl = max(abs(dpct_abl2[i]) for i in SPEC)

# H3: ratio profile monotonic from L0 (idx 0) through L3 (idx 3)
ratios_early = [r_base[i] for i in range(4)]
mono_dec = all(ratios_early[i] >= ratios_early[i+1] for i in range(3))
mono_inc = all(ratios_early[i] <= ratios_early[i+1] for i in range(3))
h3 = mono_dec or mono_inc

h1 = max_pct_abl3 > 50
h2 = max_pct_abl3 < 10

# H4 interpretation
if h1:
    h4 = abs(r_corr_abl) < abs(r_corr_base) - 0.1
    h4_pred = "ablation drops correlation (causal mediation broken)"
elif h2:
    h4 = abs(r_corr_abl - r_corr_base) < 0.15
    h4_pred = "ablation preserves correlation (parallel readers)"
else:
    h4 = False
    h4_pred = "mixed — neither H1 nor H2 cleanly confirmed"

print("\n" + "=" * 72)
print("HYPOTHESIS VERDICTS")
print("=" * 72)

print(f"\nH1 (Causal — L4-6 ratio changes >50% after L3 ablation):")
print(f"  Max |Δ%| at specialist layers: {max_pct_abl3:.1f}%")
print(f"  → H1: {'CONFIRMED' if h1 else 'NULL'}")

print(f"\nH2 (Diagnostic — L4-6 ratio changes <10% after L3 ablation):")
print(f"  Max |Δ%| at specialist layers: {max_pct_abl3:.1f}%")
print(f"  → H2: {'CONFIRMED' if h2 else 'NULL'}")

print(f"\nH3 (Global head profile monotonic through L3):")
print(f"  Ratios at L0-L3: {[f'{r:.4f}' for r in ratios_early]}")
print(f"  Mono-decreasing: {mono_dec}  |  Mono-increasing: {mono_inc}")
print(f"  → H3: {'CONFIRMED' if h3 else 'NULL'}")

print(f"\nH4 (Cross-state correlation consistent with H1 or H2):")
print(f"  Prediction: {h4_pred}")
print(f"  r baseline={r_corr_base:+.4f}, r ablated={r_corr_abl:+.4f}, Δr={r_corr_abl - r_corr_base:+.4f}")
print(f"  → H4: {'CONSISTENT' if h4 else 'INCONSISTENT'}")

print(f"\nCONTROL (L2 ablation, near-identity — threshold <5%):")
print(f"  Max |Δ%| at specialist layers: {max_pct_ctrl:.1f}%")
print(f"  → Control clean: {'YES' if max_pct_ctrl < 5 else 'NO — architecture confound'}")

print("\n" + "=" * 72)
print("G-055 COMPLETE — SEALED")
print("=" * 72)
