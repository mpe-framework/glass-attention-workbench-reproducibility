"""
G-056: Suppressive Head Causal Test
Applied Categorical Physics Workbench | Troy Teno | May 2026

Pre-registration: workbench/proposals/G056_SUPPRESSIVE_HEAD_CAUSAL_TEST.md

Question: Does a suppressive head (V = I - alpha*P_FIN, actively removes FIN content
from residual) have causal downstream effect on specialist-head Born filter ratio,
unlike the neutral global head tested in G-055 (4% change = near-identity baseline)?

H1 (Causal):  ablation changes specialist-layer ratio by >10%
H2 (Diagnostic): ablation changes specialist-layer ratio by <10%
H3: suppressive head produces a downward kink in Born filter profile at layer 2
H4: ablation effect scales monotonically with alpha (no threshold)

SEAL after running. Do not re-run.
"""

import numpy as np

# ── reproducibility ───────────────────────────────────────────────────────────────────────────
SEED = 56
rng = np.random.default_rng(SEED)

# ── architecture ───────────────────────────────────────────────────────────────────────────
N_LAY = 6
D = 16
GEO_DIMS = slice(0, 8)
FIN_DIMS = slice(8, 16)
N_TOK = 6
BANK_POS = 0
SQRT_D = np.sqrt(D)

# FIN projection matrix
P_FIN = np.zeros((D, D))
P_FIN[FIN_DIMS, FIN_DIMS] = np.eye(8)

def make_V_suppressive(alpha):
    """V = I - alpha * P_FIN: reflects FIN component, leaves GEO unchanged."""
    return np.eye(D) - alpha * P_FIN

# ── input construction ──────────────────────────────────────────────────────────────────────────
GEO_BASE = np.zeros(D); GEO_BASE[GEO_DIMS] = 1.0 / np.sqrt(8)
FIN_BASE = np.zeros(D); FIN_BASE[FIN_DIMS] = 1.0 / np.sqrt(8)

def make_token(geo_w, fin_w, noise_scale=0.05):
    v = geo_w * GEO_BASE + fin_w * FIN_BASE
    v += rng.normal(0, noise_scale, D)
    return v / np.linalg.norm(v)

CURRENT = make_token(1.0, 0.0)
GEO1    = make_token(1.0, 0.0)
GEO2    = make_token(1.0, 0.0)
FIN1    = make_token(0.0, 1.0)
FIN2    = make_token(0.0, 1.0)

def make_sequence(fin_weight_bank):
    bank = GEO_BASE + fin_weight_bank * FIN_BASE
    bank = bank / np.linalg.norm(bank)
    return np.stack([bank, CURRENT, GEO1, GEO2, FIN1, FIN2])

SEQ_SUCCESS = make_sequence(0.5)   # no-collapse regime
SEQ_FAIL    = make_sequence(0.05)  # slow-collapse regime

def swap_morphism(seq):
    s = seq.copy()
    s[0], s[1] = seq[1].copy(), seq[0].copy()
    return s

# ── Q/K matrices ────────────────────────────────────────────────────────────────────────────
def make_layer_QK(alpha_supp):
    """
    Layer 0: near-identity
    Layer 1: near-identity (control ablation target)
    Layer 2: suppressive global (primary ablation target)
    Layer 3: FIN-specialist
    Layer 4: FIN-specialist
    Layer 5: near-identity
    """
    NEAR_ID  = 0.1 * np.eye(D)
    GLOBAL   = 0.005 * np.ones((D, D))
    FIN_SPEC = np.zeros((D, D))
    FIN_SPEC[FIN_DIMS, FIN_DIMS] = 4.0 * np.eye(8)

    qk = [NEAR_ID, NEAR_ID, GLOBAL, FIN_SPEC, FIN_SPEC, NEAR_ID]
    V_supp = make_V_suppressive(alpha_supp)

    # V matrices: suppressive only at layer 2
    V_mats = [np.eye(D)] * N_LAY
    V_mats[2] = V_supp

    return qk, V_mats

# ── forward pass ────────────────────────────────────────────────────────────────────────────
def _head(X, Q_mat, K_mat, V_mat):
    Q = X @ Q_mat.T
    K = X @ K_mat.T
    S = (Q @ K.T) / SQRT_D
    S -= S.max(axis=1, keepdims=True)
    W = np.exp(S); W /= W.sum(axis=1, keepdims=True)
    return W @ (X @ V_mat.T), W

def forward(seq, QK_list, V_list, ablate_layer=None):
    X = seq.copy()
    hidden = [X.copy()]
    attn_ws = []
    for l in range(N_LAY):
        head_out, W = _head(X, QK_list[l], QK_list[l], V_list[l])
        attn_ws.append(W)
        if ablate_layer is not None and l == ablate_layer:
            head_out = np.zeros_like(head_out)
        X = X + head_out
        hidden.append(X.copy())
    return hidden, attn_ws

def born_profile(seq, QK_list, V_list, ablate_layer=None):
    morphed = swap_morphism(seq)
    h_o, _ = forward(seq, QK_list, V_list, ablate_layer)
    h_m, _ = forward(morphed, QK_list, V_list, ablate_layer)
    return np.array([
        np.linalg.norm(h_o[l][BANK_POS] - h_m[l][BANK_POS]) / 2.0
        for l in range(N_LAY + 1)
    ])

def fin_content(vec):
    """FIN-dimension magnitude of a vector."""
    return np.linalg.norm(vec[FIN_DIMS])

# ═══════════════════════════════════════════════════════════════════════════════════
print("=" * 70)
print("  G-056_SUPPRESSIVE_HEAD_CAUSAL_TEST")
print("  H1/H2: is suppressive head causal or diagnostic?")
print("=" * 70)

# ── PRIMARY CONDITION: alpha = 2.0 ─────────────────────────────────────────────────────
ALPHA_PRIMARY = 2.0
QK_LIST, V_LIST = make_layer_QK(ALPHA_PRIMARY)

print(f"\n{'─'*70}")
print(f"  PRIMARY CONDITION: alpha = {ALPHA_PRIMARY}")
print(f"{'─'*70}")

bp_succ_base = born_profile(SEQ_SUCCESS, QK_LIST, V_LIST)
bp_fail_base = born_profile(SEQ_FAIL,    QK_LIST, V_LIST)
ratio_base = bp_succ_base / (bp_fail_base + 1e-12)

bp_succ_abl2 = born_profile(SEQ_SUCCESS, QK_LIST, V_LIST, ablate_layer=2)
bp_fail_abl2 = born_profile(SEQ_FAIL,    QK_LIST, V_LIST, ablate_layer=2)
ratio_abl2 = bp_succ_abl2 / (bp_fail_abl2 + 1e-12)

bp_succ_abl1 = born_profile(SEQ_SUCCESS, QK_LIST, V_LIST, ablate_layer=1)
bp_fail_abl1 = born_profile(SEQ_FAIL,    QK_LIST, V_LIST, ablate_layer=1)
ratio_abl1 = bp_succ_abl1 / (bp_fail_abl1 + 1e-12)

specialist_layers = [3, 4, 5]
mean_ratio_base = np.mean(ratio_base[specialist_layers])
mean_ratio_abl2 = np.mean(ratio_abl2[specialist_layers])
mean_ratio_abl1 = np.mean(ratio_abl1[specialist_layers])

pct_change_supp = 100 * abs(mean_ratio_abl2 - mean_ratio_base) / mean_ratio_base
pct_change_ctrl = 100 * abs(mean_ratio_abl1 - mean_ratio_base) / mean_ratio_base

print("\n  Baseline Born filter profile (S_success / S_fail ratio per layer):")
print("  Layer:  " + "  ".join(f"L{i}" for i in range(N_LAY + 1)))
print("  Ratio:  " + "  ".join(f"{r:.4f}" for r in ratio_base))

print(f"\n  Specialist-layer mean ratio (L3-L5):")
print(f"    Baseline:            {mean_ratio_base:.4f}x")
print(f"    After supp ablation: {mean_ratio_abl2:.4f}x  ({pct_change_supp:.1f}% change)")
print(f"    After ctrl ablation: {mean_ratio_abl1:.4f}x  ({pct_change_ctrl:.1f}% change)")

print(f"\n  H1 threshold: >10% change  |  H2 threshold: <10% change")
h1 = pct_change_supp > 10.0
h2 = pct_change_supp < 10.0
print(f"  H1 (suppressive head causal):    {'CONFIRMED' if h1 else 'NULL'}  ({pct_change_supp:.1f}%)")
print(f"  H2 (suppressive head diagnostic):{'CONFIRMED' if h2 else 'NULL'}  ({pct_change_supp:.1f}%)")
ctrl_clean = pct_change_ctrl < 5.0
print(f"  Control (layer-1 ablation):      {'CLEAN' if ctrl_clean else 'CONFOUND'}  ({pct_change_ctrl:.1f}%)")

# ── H3: Born filter profile shape ────────────────────────────────────────────────────────────
print(f"\n{'─'*70}")
print("  H3: Does suppressive head produce a downward kink at layer 2?")
print(f"{'─'*70}")
print(f"  Ratio at L0: {ratio_base[0]:.4f}")
print(f"  Ratio at L1: {ratio_base[1]:.4f}")
print(f"  Ratio at L2: {ratio_base[2]:.4f}  ← suppressive head layer")
print(f"  Ratio at L3: {ratio_base[3]:.4f}")
print(f"  Ratio at L4: {ratio_base[4]:.4f}")
print(f"  Ratio at L5: {ratio_base[5]:.4f}")
h3 = ratio_base[2] < ratio_base[1]
print(f"\n  H3 (downward kink at L2, ratio[L2] < ratio[L1]): {'CONFIRMED' if h3 else 'NULL'}")
print(f"  ratio[L2]={ratio_base[2]:.4f}  ratio[L1]={ratio_base[1]:.4f}")

# ── Residual FIN-content diagnostic ──────────────────────────────────────────────────────────
print(f"\n{'─'*70}")
print("  Residual FIN-content at bank position (before vs after suppressive head)")
print(f"{'─'*70}")

# Run forward to get hidden states at layers 1 and 2
h_succ_base, _ = forward(SEQ_SUCCESS, QK_LIST, V_LIST)
h_fail_base, _ = forward(SEQ_FAIL,    QK_LIST, V_LIST)
h_succ_abl2, _ = forward(SEQ_SUCCESS, QK_LIST, V_LIST, ablate_layer=2)
h_fail_abl2, _ = forward(SEQ_FAIL,    QK_LIST, V_LIST, ablate_layer=2)

fin_succ_L1 = fin_content(h_succ_base[2][BANK_POS])   # after layer 1, before layer 2
fin_fail_L1 = fin_content(h_fail_base[2][BANK_POS])
fin_succ_L2 = fin_content(h_succ_base[3][BANK_POS])   # after layer 2 (suppressive)
fin_fail_L2 = fin_content(h_fail_base[3][BANK_POS])
fin_succ_L2_abl = fin_content(h_succ_abl2[3][BANK_POS])  # after layer 2 ablated
fin_fail_L2_abl = fin_content(h_fail_abl2[3][BANK_POS])

print(f"  After L1 (before suppressive head):")
print(f"    FIN content [bank] — S_success: {fin_succ_L1:.4f}  S_fail: {fin_fail_L1:.4f}")
print(f"  After L2 (suppressive head active):")
print(f"    FIN content [bank] — S_success: {fin_succ_L2:.4f}  S_fail: {fin_fail_L2:.4f}")
print(f"  After L2 (suppressive head ablated):")
print(f"    FIN content [bank] — S_success: {fin_succ_L2_abl:.4f}  S_fail: {fin_fail_L2_abl:.4f}")

succ_delta = fin_succ_L1 - fin_succ_L2
fail_delta = fin_fail_L1 - fin_fail_L2
print(f"\n  FIN content removed by suppressive head:")
print(f"    S_success: {succ_delta:.4f}  |  S_fail: {fail_delta:.4f}")
print(f"  Suppresses S_success more than S_fail: {'YES' if succ_delta > fail_delta else 'NO'}")

# ── Attention at specialist layers (does ablation change fin1/fin2 routing?) ──────
print(f"\n{'─'*70}")
print("  Specialist-head attention (layers 3-4, bank position): fin1/fin2 routing")
print(f"{'─'*70}")
_, attn_base = forward(SEQ_SUCCESS, QK_LIST, V_LIST)
_, attn_abl2 = forward(SEQ_SUCCESS, QK_LIST, V_LIST, ablate_layer=2)
for l in [3, 4]:
    w_base = attn_base[l][BANK_POS]
    w_abl  = attn_abl2[l][BANK_POS]
    print(f"  L{l} [baseline]:  fin1={w_base[4]:.4f}  fin2={w_base[5]:.4f}  bank={w_base[0]:.4f}")
    print(f"  L{l} [ablated]:   fin1={w_abl[4]:.4f}  fin2={w_abl[5]:.4f}  bank={w_abl[0]:.4f}")

# ── H4: Alpha sweep ───────────────────────────────────────────────────────────────────────────
print(f"\n{'─'*70}")
print("  H4: Alpha sweep — does ablation effect scale monotonically with alpha?")
print(f"{'─'*70}")
alphas = [0.0, 0.5, 1.0, 1.5, 2.0]
sweep_results = []
print(f"  {'alpha':>6}  {'base_ratio':>12}  {'abl_ratio':>12}  {'pct_change':>12}")
for a in alphas:
    qk_a, v_a = make_layer_QK(a)
    bp_s = born_profile(SEQ_SUCCESS, qk_a, v_a)
    bp_f = born_profile(SEQ_FAIL,    qk_a, v_a)
    r_base = np.mean((bp_s / (bp_f + 1e-12))[specialist_layers])

    bp_s_abl = born_profile(SEQ_SUCCESS, qk_a, v_a, ablate_layer=2)
    bp_f_abl = born_profile(SEQ_FAIL,    qk_a, v_a, ablate_layer=2)
    r_abl = np.mean((bp_s_abl / (bp_f_abl + 1e-12))[specialist_layers])

    pct = 100 * abs(r_abl - r_base) / r_base
    sweep_results.append(pct)
    print(f"  {a:>6.1f}  {r_base:>12.4f}  {r_abl:>12.4f}  {pct:>11.2f}%")

# Check monotonicity of pct_change across alphas
monotone = all(sweep_results[i] <= sweep_results[i+1]
               for i in range(len(sweep_results)-1))
print(f"\n  H4 (monotonically increasing pct_change with alpha): {'CONFIRMED' if monotone else 'NULL'}")
if not monotone:
    print("  Non-monotonic — threshold or peak behavior detected.")

# ── Final verdicts ───────────────────────────────────────────────────────────────────────────────
print(f"\n{'=' * 70}")
print("  HYPOTHESIS VERDICTS SUMMARY")
print(f"{'=' * 70}")
print(f"  H1 (supp head causal, >10% ablation change):     {'CONFIRMED' if h1 else 'NULL'}  ({pct_change_supp:.1f}%)")
print(f"  H2 (supp head diagnostic, <10% ablation change): {'CONFIRMED' if h2 else 'NULL'}  ({pct_change_supp:.1f}%)")
print(f"  H3 (downward kink at L2):                        {'CONFIRMED' if h3 else 'NULL'}")
print(f"  H4 (monotonic scaling with alpha):               {'CONFIRMED' if monotone else 'NULL'}")
print(f"  Control (layer-1 near-identity):                 {'CLEAN' if ctrl_clean else 'CONFOUND'}  ({pct_change_ctrl:.1f}%)")
print()

# S-058 relay
if h2:
    print("  S-058 RELAY: suppressive heads are DIAGNOSTIC.")
    print("  S-058 focuses on L4H6/L5H2/L5H5 only. L3H4 is not a repair target.")
else:
    print("  S-058 RELAY: suppressive heads are CAUSAL.")
    print("  S-058 design must include L3H4 in the intervention. Consult sandbox_017.")

print()
print("  # SEALED — do not re-run.")
print("=" * 70)
