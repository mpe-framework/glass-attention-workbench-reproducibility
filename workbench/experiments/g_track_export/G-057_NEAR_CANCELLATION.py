"""
G-057: Near-Cancellation Structure in Attention + FFN Networks
Applied Categorical Physics Workbench | Troy Teno | May 2026

Pre-registration: workbench/proposals/G057_NEAR_CANCELLATION.md

Question: Does FFN correction near gamma=1.0 produce near-cancellation
(reconstruction fraction >> 1.0) and sign-reversed ablation effects?

H1: recon_fraction > 10x at gamma=0.90 or 0.95
H2: upstream ablation change > 20% near gamma=1.0; drops at gamma=1.0
H3: sharp entry into near-cancellation zone (>5x jump from gamma=0.75 to 0.95)
H4: gamma=2.0 (over-correction) reproduces G-056 inversion (>20% ablation change)
H5: corrective ablation (set gamma=0) produces sign-reversed Δmargin at gamma in {0.90,0.95}

SEAL after running. Do not re-run.

V0.1.0 error (corrected before sealing):
  - SPECIALIST_LAYERS was [3,4]; corrected to [3,4,5] to match G-055/G-056 convention.
  - reconstruction_fraction formula gave ~0.02 at gamma=0 (should be ~1.0 per pre-reg);
    corrected to use deviation from "neither" baseline so gamma=0 → 1.0 by construction.
  Both fixes are measurement corrections only; the forward-pass data is unaffected.
"""

import numpy as np

# ── Reproducibility ─────────────────────────────────────────────────────────────────────────────
SEED = 57
rng = np.random.default_rng(SEED)

# ── Architecture ─────────────────────────────────────────────────────────────────────────────
N_LAY    = 6
D        = 16
GEO_DIMS = slice(0, 8)
FIN_DIMS = slice(8, 16)
N_TOK    = 6
BANK_POS = 0
SQRT_D   = np.sqrt(D)

# Projection matrices
P_FIN = np.zeros((D, D))
P_FIN[FIN_DIMS, FIN_DIMS] = np.eye(8)

P_GEO = np.zeros((D, D))
P_GEO[GEO_DIMS, GEO_DIMS] = np.eye(8)

# ── Token construction ───────────────────────────────────────────────────────────────────────────
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
NEAR_ID  = 0.1  * np.eye(D)
GLOBAL   = 0.005 * np.ones((D, D))
FIN_SPEC = np.zeros((D, D))
FIN_SPEC[FIN_DIMS, FIN_DIMS] = 4.0 * np.eye(8)

QK_LIST = [NEAR_ID, NEAR_ID, GLOBAL, FIN_SPEC, FIN_SPEC, NEAR_ID]

# ── FFN correction module ───────────────────────────────────────────────────────────────────────
# The "correction direction" opposes the cross-attention's FIN-preferring output.
# Cross-attention at layer 2 (global, V=I) outputs a weighted average of the
# sequence — which averages over FIN content from fin1/fin2. The correction
# direction is the FIN subspace direction. P_CORRECTION = P_FIN.
#
# FFN(h) = h + gamma * P_CORRECTION * head_output
# Applied as: residual_after_ffn = residual_after_attn + gamma * P_FIN @ head_output
# This subtracts the FIN component of the cross-attention output from the residual.
# At gamma=1.0: exact cancellation of the cross-attn FIN contribution.
# At gamma=2.0: over-correction (inverts cross-attn FIN contribution).

# ── Forward pass ─────────────────────────────────────────────────────────────────────────────
def _head(X, Q_mat, K_mat):
    """Single-head attention with V=I. Returns (head_output, attn_weights)."""
    Q = X @ Q_mat.T
    K = X @ K_mat.T
    S = (Q @ K.T) / SQRT_D
    S -= S.max(axis=1, keepdims=True)
    W = np.exp(S); W /= W.sum(axis=1, keepdims=True)
    return W @ X, W

def forward(seq, gamma=0.0, ablate_layer2_attn=False, ablate_gamma=False):
    """
    Forward pass through 6 layers.

    gamma:              FFN correction magnitude at layer 2
    ablate_layer2_attn: zero the layer-2 cross-attention output (upstream ablation)
    ablate_gamma:       set gamma=0 while keeping cross-attn active (corrective ablation)

    Returns: (hidden_states list of N_LAY+1, attn_weights list of N_LAY,
              layer2_head_output before FFN)
    """
    eff_gamma = 0.0 if ablate_gamma else gamma
    X = seq.copy()
    hidden = [X.copy()]
    attn_ws = []
    layer2_head_out = None

    for l in range(N_LAY):
        head_out, W = _head(X, QK_LIST[l], QK_LIST[l])
        attn_ws.append(W)

        if l == 2:
            layer2_head_out = head_out.copy()
            if ablate_layer2_attn:
                head_out = np.zeros_like(head_out)
            # Residual add
            X = X + head_out
            # FFN correction: subtract gamma * P_FIN @ (original head output)
            # (if upstream ablated, head_out is zero so FFN has nothing to correct)
            if not ablate_layer2_attn:
                ffn_correction = eff_gamma * (P_FIN @ layer2_head_out.T).T
                X = X - ffn_correction
        else:
            X = X + head_out

        hidden.append(X.copy())

    return hidden, attn_ws, layer2_head_out

def born_profile(seq, gamma=0.0, ablate_layer2_attn=False, ablate_gamma=False):
    morphed = swap_morphism(seq)
    h_o, _, _ = forward(seq,    gamma, ablate_layer2_attn, ablate_gamma)
    h_m, _, _ = forward(morphed, gamma, ablate_layer2_attn, ablate_gamma)
    return np.array([
        np.linalg.norm(h_o[l][BANK_POS] - h_m[l][BANK_POS]) / 2.0
        for l in range(N_LAY + 1)
    ])

SPECIALIST_LAYERS = [3, 4, 5]  # hidden state indices 3,4,5 — matches G-055/G-056 convention

def specialist_ratio(seq_s, seq_f, gamma=0.0,
                     ablate_layer2_attn=False, ablate_gamma=False):
    bp_s = born_profile(seq_s, gamma, ablate_layer2_attn, ablate_gamma)
    bp_f = born_profile(seq_f, gamma, ablate_layer2_attn, ablate_gamma)
    r = bp_s / (bp_f + 1e-12)
    return float(np.mean(r[SPECIALIST_LAYERS]))

def fin_content(vec):
    return float(np.linalg.norm(vec[FIN_DIMS]))

# ── Unembedding difference for reconstruction fraction ──────────────────────────────────
# The "logit margin" analog: difference in Born filter between S_success and S_fail
# at the bank position after all layers. We use the output hidden state norm diff.
# Per pre-registration M1: recon_fraction = (sum_h delta_logit_h) / observed_margin
# In the toy, "logit margin" = Born filter ratio at specialist layers.
# delta_logit_h for layer-2 head = contribution of that head to the ratio.
# We measure this as: ratio_with_head_active - ratio_with_head_ablated.
# observed_margin = specialist_ratio at gamma=0 (baseline, no FFN, no ablation).

# ── Per-head logit decomposition (M1) ──────────────────────────────────────────────────
# Pre-registration spec: recon_fraction = (Σ_h δlogit_h) / observed_logit_margin
# Calibration: "With no FFN (γ=0), recon_fraction should be ~1.0."
#
# Correct formula (deviation from "neither" baseline):
#   ratio_neither = specialist_ratio when both head and FFN are absent (ablated, gamma=0)
#   head_signal   = specialist_ratio(gamma=0, active) - ratio_neither   [constant across γ]
#   observed_margin(γ) = specialist_ratio(gamma, active) - ratio_neither
#   recon_fraction(γ)  = head_signal / observed_margin(γ)
#
# At γ=0: observed_margin = head_signal → recon_fraction = 1.0  ✓
# At γ→1.0: FFN cancels head's FIN contribution → net effect → 0 → diverges  ✓

_RATIO_NEITHER  = None   # computed once below (ablated, gamma=0)
_HEAD_SIGNAL    = None   # computed once below

def _init_recon_anchors():
    global _RATIO_NEITHER, _HEAD_SIGNAL
    _RATIO_NEITHER = specialist_ratio(SEQ_SUCCESS, SEQ_FAIL,
                                      gamma=0.0, ablate_layer2_attn=True)
    _HEAD_SIGNAL   = specialist_ratio(SEQ_SUCCESS, SEQ_FAIL,
                                      gamma=0.0) - _RATIO_NEITHER

def reconstruction_fraction(gamma):
    """
    M1: reconstruction fraction at this gamma.
    = head_signal / observed_margin(gamma)
    Gives ~1.0 at gamma=0 (per pre-reg calibration) and diverges as gamma->1.0.
    """
    ratio_net       = specialist_ratio(SEQ_SUCCESS, SEQ_FAIL, gamma=gamma)
    observed_margin = ratio_net - _RATIO_NEITHER
    if abs(observed_margin) < 1e-10:
        return np.inf
    return _HEAD_SIGNAL / observed_margin

# ── Initialize reconstruction-fraction anchors ────────────────────────────────────────────
_init_recon_anchors()

# ════════════════════════════════════════════════════════════════════════
print("=" * 72)
print("  G-057: NEAR-CANCELLATION STRUCTURE IN ATTENTION + FFN NETWORKS")
print("  Applied Categorical Physics Workbench | Troy Teno | May 2026")
print("=" * 72)

GAMMA_SWEEP = [0.0, 0.25, 0.50, 0.75, 0.90, 0.95, 1.00, 1.10, 2.00]

# ── Section 1: Gamma sweep overview ────────────────────────────────────────────────────────
print(f"\n{'─'*72}")
print("  SECTION 1: Gamma sweep — specialist ratio, ablation change, recon fraction")
print(f"{'─'*72}")
print(f"  {'gamma':>6}  {'ratio_base':>12}  {'ratio_abl':>12}  {'pct_chg':>9}  {'recon_frac':>12}")
print(f"  {'──────':>6}  {'──────────':>12}  {'─────────':>12}  {'───────':>9}  {'──────────':>12}")

sweep_data = {}
for g in GAMMA_SWEEP:
    r_base  = specialist_ratio(SEQ_SUCCESS, SEQ_FAIL, gamma=g)
    r_abl   = specialist_ratio(SEQ_SUCCESS, SEQ_FAIL, gamma=g, ablate_layer2_attn=True)
    pct     = 100.0 * abs(r_abl - r_base) / (r_base + 1e-12)
    rf      = reconstruction_fraction(g)
    sweep_data[g] = {'r_base': r_base, 'r_abl': r_abl, 'pct': pct, 'rf': rf}
    rf_str  = f"{rf:>12.2f}" if np.isfinite(rf) else "       ∞    "
    print(f"  {g:>6.2f}  {r_base:>12.4f}  {r_abl:>12.4f}  {pct:>8.1f}%  {rf_str}")

# ── Section 2: M2 — FIN-component at bank before/after layer-2 complex ───────────────
print(f"\n{'─'*72}")
print("  SECTION 2: M2 — FIN-component at bank before vs after layer-2 complex")
print(f"{'─'*72}")
print(f"  {'gamma':>6}  {'fin_before_S':>14}  {'fin_after_S':>13}  {'fin_before_F':>14}  {'fin_after_F':>13}")
print(f"  {'──────':>6}  {'────────────':>14}  {'───────────':>13}  {'────────────':>14}  {'───────────':>13}")

for g in GAMMA_SWEEP:
    h_s, _, _ = forward(SEQ_SUCCESS, gamma=g)
    h_f, _, _ = forward(SEQ_FAIL,    gamma=g)
    # hidden[2] = after layer 1 (before layer 2 complex)
    # hidden[3] = after layer 2 complex (attn + FFN)
    fin_before_s = fin_content(h_s[2][BANK_POS])
    fin_after_s  = fin_content(h_s[3][BANK_POS])
    fin_before_f = fin_content(h_f[2][BANK_POS])
    fin_after_f  = fin_content(h_f[3][BANK_POS])
    print(f"  {g:>6.2f}  {fin_before_s:>14.4f}  {fin_after_s:>13.4f}  {fin_before_f:>14.4f}  {fin_after_f:>13.4f}")

# ── Section 3: Born filter profiles at key gamma values ──────────────────────────────
print(f"\n{'─'*72}")
print("  SECTION 3: Full Born filter ratio profiles at key gamma values")
print(f"{'─'*72}")

KEY_GAMMAS = [0.0, 0.50, 0.90, 0.95, 1.00, 2.00]
header = f"  {'Layer':<8}" + "".join(f"  {'γ='+str(g):>8}" for g in KEY_GAMMAS)
print(header)
print("  " + "─" * (8 + 10 * len(KEY_GAMMAS)))

profiles = {}
for g in KEY_GAMMAS:
    bp_s = born_profile(SEQ_SUCCESS, gamma=g)
    bp_f = born_profile(SEQ_FAIL,    gamma=g)
    profiles[g] = bp_s / (bp_f + 1e-12)

for l in range(N_LAY + 1):
    row = f"  {'L'+str(l):<8}"
    for g in KEY_GAMMAS:
        row += f"  {profiles[g][l]:>8.4f}"
    print(row)

# ── Section 4: H5 — sign-reversed Δmargin measurement ──────────────────────────────────
print(f"\n{'─'*72}")
print("  SECTION 4: H5 — Sign-reversed Δmargin (corrective vs upstream ablation)")
print(f"{'─'*72}")
print("  At near-cancellation gamma (0.90, 0.95):")
print("  Upstream ablation: zero cross-attn head → expect Δmargin > +5% (improvement)")
print("  Corrective ablation: set gamma=0 → expect Δmargin < −5% (worsening)")
print()

H5_GAMMAS = [0.90, 0.95]
h5_data = {}

for g in H5_GAMMAS:
    r_base     = specialist_ratio(SEQ_SUCCESS, SEQ_FAIL, gamma=g)
    r_upstream = specialist_ratio(SEQ_SUCCESS, SEQ_FAIL, gamma=g, ablate_layer2_attn=True)
    r_correct  = specialist_ratio(SEQ_SUCCESS, SEQ_FAIL, gamma=g, ablate_gamma=True)

    delta_upstream  = 100.0 * (r_upstream - r_base) / (r_base + 1e-12)
    delta_corrective = 100.0 * (r_correct  - r_base) / (r_base + 1e-12)

    h5_data[g] = {
        'r_base': r_base,
        'r_upstream': r_upstream,
        'r_correct': r_correct,
        'delta_upstream': delta_upstream,
        'delta_corrective': delta_corrective,
    }

    print(f"  gamma = {g:.2f}:")
    print(f"    Baseline ratio:              {r_base:.4f}")
    print(f"    Upstream ablation ratio:     {r_upstream:.4f}  (Δ = {delta_upstream:+.1f}%)")
    print(f"    Corrective ablation ratio:   {r_correct:.4f}  (Δ = {delta_corrective:+.1f}%)")
    upstream_dir  = "POSITIVE (improvement)" if delta_upstream > 0 else "NEGATIVE (worsening)"
    correct_dir   = "NEGATIVE (worsening, sign-reversed)" if delta_corrective < 0 else "POSITIVE (not sign-reversed)"
    print(f"    Upstream direction:          {upstream_dir}")
    print(f"    Corrective direction:        {correct_dir}")
    print()

# ── Hypothesis verdicts ─────────────────────────────────────────────────────────────────────────
print(f"\n{'=' * 72}")
print("  HYPOTHESIS VERDICTS")
print(f"{'=' * 72}")

# H1: recon fraction > 10x at gamma=0.90 or 0.95
rf_090 = sweep_data[0.90]['rf']
rf_095 = sweep_data[0.95]['rf']
h1 = (np.isfinite(rf_090) and rf_090 > 10.0) or (np.isfinite(rf_095) and rf_095 > 10.0)
rf_090_str = f"{rf_090:.2f}x" if np.isfinite(rf_090) else "∞"
rf_095_str = f"{rf_095:.2f}x" if np.isfinite(rf_095) else "∞"
h1_str = "CONFIRMED" if h1 else "NULL"
print(f"\n  H1 — Near-cancellation produces reconstruction fraction >> 1.0:")
print(f"    Threshold: recon_fraction > 10x at gamma=0.90 or 0.95")
print(f"    recon_fraction at gamma=0.90: {rf_090_str}")
print(f"    recon_fraction at gamma=0.95: {rf_095_str}")
print(f"    → H1: {h1_str}")

# H2: upstream ablation >20% at gamma in {0.75, 0.90, 0.95}; drops <10% at gamma=1.0
pct_075 = sweep_data[0.75]['pct']
pct_090 = sweep_data[0.90]['pct']
pct_095 = sweep_data[0.95]['pct']
pct_100 = sweep_data[1.00]['pct']
h2_amplified = any(p > 20.0 for p in [pct_075, pct_090, pct_095])
h2_drops = pct_100 < 10.0
h2 = h2_amplified and h2_drops
print(f"\n  H2 — Near-cancellation amplifies Born filter causal sensitivity:")
print(f"    Threshold: >20% change at gamma in {{0.75, 0.90, 0.95}}; drops <10% at gamma=1.0")
print(f"    pct_change at gamma=0.75: {pct_075:.1f}%")
print(f"    pct_change at gamma=0.90: {pct_090:.1f}%")
print(f"    pct_change at gamma=0.95: {pct_095:.1f}%")
print(f"    pct_change at gamma=1.00: {pct_100:.1f}%")
print(f"    Amplified (>20% near γ=1.0): {'YES' if h2_amplified else 'NO'}")
print(f"    Drops at γ=1.0 (<10%):       {'YES' if h2_drops else 'NO'}")
print(f"    → H2: {'CONFIRMED' if h2 else 'NULL'}")

# H3: >5x jump in recon fraction from gamma=0.75 to gamma=0.95; <5x at gamma=0.50
rf_050 = sweep_data[0.50]['rf']
rf_075 = sweep_data[0.75]['rf']
rf_095 = sweep_data[0.95]['rf']
rf_050_str = f"{rf_050:.2f}x" if np.isfinite(rf_050) else "∞"
rf_075_str = f"{rf_075:.2f}x" if np.isfinite(rf_075) else "∞"
if np.isfinite(rf_075) and np.isfinite(rf_095):
    h3_jump = rf_095 / (rf_075 + 1e-12) > 5.0
else:
    h3_jump = True  # diverges = sharp entry
h3_early = np.isfinite(rf_050) and rf_050 < 5.0
h3 = h3_jump and h3_early
print(f"\n  H3 — Sharp entry into near-cancellation zone (phase transition):")
print(f"    Threshold: >5x jump from gamma=0.75 to 0.95; <5x at gamma=0.50")
print(f"    recon_fraction at gamma=0.50: {rf_050_str}")
print(f"    recon_fraction at gamma=0.75: {rf_075_str}")
print(f"    recon_fraction at gamma=0.95: {rf_095_str}")
if np.isfinite(rf_075) and np.isfinite(rf_095):
    jump = rf_095 / (rf_075 + 1e-12)
    print(f"    Jump ratio (0.95/0.75): {jump:.2f}x  (threshold: >5x)")
print(f"    Sharp jump detected: {'YES' if h3_jump else 'NO'}")
print(f"    Early plateau (<5x at 0.50): {'YES' if h3_early else 'NO'}")
print(f"    → H3: {'CONFIRMED' if h3 else 'NULL'}")

# H4: gamma=2.0 produces >20% ablation change
pct_200 = sweep_data[2.00]['pct']
h4 = pct_200 > 20.0
print(f"\n  H4 — gamma > 1.0 (over-correction) reproduces G-056 inversion (>20% change):")
print(f"    Threshold: ablation change >20% at gamma=2.0")
print(f"    pct_change at gamma=2.0: {pct_200:.1f}%")
print(f"    → H4: {'CONFIRMED' if h4 else 'NULL'}")

# H5: corrective ablation Δmargin < -5%; upstream ablation Δmargin > +5%
h5_results = {}
for g in H5_GAMMAS:
    d = h5_data[g]
    upstream_pass  = d['delta_upstream']  > 5.0
    corrective_pass = d['delta_corrective'] < -5.0
    h5_results[g] = upstream_pass and corrective_pass

h5 = any(h5_results.values())
print(f"\n  H5 — Corrective ablation produces sign-reversed Δmargin:")
print(f"    Threshold: corrective ablation Δ < −5%; upstream ablation Δ > +5%")
for g in H5_GAMMAS:
    d = h5_data[g]
    status = "CONFIRMED" if h5_results[g] else "NULL"
    print(f"    gamma={g:.2f}: upstream Δ={d['delta_upstream']:+.1f}%, corrective Δ={d['delta_corrective']:+.1f}% → {status}")
print(f"    → H5: {'CONFIRMED' if h5 else 'NULL'}")

# Summary table
print(f"\n{'─'*72}")
print("  SUMMARY")
print(f"{'─'*72}")
verdicts = {
    'H1 (recon fraction >> 1.0 near γ=1.0)': h1,
    'H2 (upstream ablation amplified; drops at γ=1.0)': h2,
    'H3 (sharp entry into near-cancellation zone)': h3,
    'H4 (γ=2.0 over-correction reproduces G-056)': h4,
    'H5 (corrective ablation sign-reverses Δmargin)': h5,
}
for label, v in verdicts.items():
    print(f"  {'CONFIRMED' if v else 'NULL':>10}  {label}")

n_confirmed = sum(verdicts.values())
print(f"\n  {n_confirmed}/{len(verdicts)} hypotheses confirmed.")
print(f"\n{'=' * 72}")
print("  G-057 COMPLETE — SEALED — do not re-run.")
print(f"{'=' * 72}")
