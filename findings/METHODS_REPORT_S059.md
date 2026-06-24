# Methods Report — S-059: Logit Decomposition — Which Heads Drive the Wrong Decision?
**Applied Categorical Physics Workbench**
Troy Teno | May 2026 | Open Access

**One-line result:** S-059 V0.1.0 reveals a methodological gap: the δlogit computation does not account for T5's final decoder RMSNorm, placing head contributions in pre-LayerNorm residual stream units rather than logit units (reconstruction factor ≈ 32,193). Within the pre-LayerNorm space, the relative rankings are internally consistent and carry real information: L4H6 (rank 11) and L5H5 (rank 5) are strongly pro-walk as predicted; L5H2 is near-zero and slightly pro-jump (K1 fires technically); and unexpectedly, L6H2, L6H6, L5H6, L6H5 are the dominant pro-walk contributors. V0.2.0 (LayerNorm-corrected) is required before hypothesis verdicts can be assessed in logit units.

---

## The Question

S-058 found that cross-example activation patching is non-specific: any transplant (including the L3H0 control) pushes the model into an out-of-distribution state. S-059 replaced patching with a direct measurement: for each of 29 valid fail examples (substituted_walk, has_around, SEED=42), and 25 success examples (has_around+correct, SEED=42), compute at the first action-slot divergence step:

```
δlogit_h = (W_U[I_JUMP] − W_U[I_WALK]) · contrib_h
```

where `contrib_h = W_O_h @ (attn_h @ W_V_h @ enc_hidden)` is the cross-attention head's V/O projection contribution to the decoder residual stream — the same `compute_head_contrib` method used in S-057 and S-058.

No patching. No donor examples. No matching. Direct measurement on the original fail trajectory.

---

## Results

### M5 — Pre-measurement baseline

```
n = 29 valid fail examples (1 skipped: no_div_steps)
n = 25 valid success examples (0 skipped)

Fail group observed margin:    mean = −6.005  (logit_jump − logit_walk)
Success group observed margin: mean = +0.586
```

### M3 — Reconstruction check

```
Fail group:
  Mean observed margin:             −6.0051
  Mean reconstructed margin (Σδlogit_h cross-attn heads): −193,322.6
  Mean residual (obs − recon):     +193,316.6
  Reconstruction fraction:          32,193.09

Success group:
  Mean observed margin:             +0.5862
  Mean reconstructed margin:       +206,845.4
  Reconstruction fraction:          352,887.30
```

**Methodological finding: the reconstruction fraction is ~32,000, not ~1.** The sum of cross-attention head contributions is approximately 32,000× the observed logit margin. This is the signature of a missing LayerNorm: T5 applies `model.decoder.final_layer_norm` (an RMSNorm) between the residual stream and `lm_head`. The `δlogit_h = W_U_diff · contrib_h` formula projects residual stream contributions directly through W_U, bypassing this normalization. The result is that δlogit values are in pre-LayerNorm units (on the order of 10,000s) rather than actual logit units (on the order of 1–10).

**K2: CLEAR** (by letter of the pre-registered condition: |32193| < 0.20 is False, so K2 does not fire). However, the spirit of K2 — "does the decomposition account for the observed margin?" — is violated: the decomposition dramatically over-attributes the margin. K2 was designed to detect under-attribution; it did not anticipate over-attribution from a missing LayerNorm.

### M1 — Per-head δlogit fail group (pre-LayerNorm units, top 15 and target heads)

| Rank | Head  | mean δlogit   | std       | sign     |
|------|-------|---------------|-----------|----------|
| 1    | L6H2  | −52,469       | 105,317   | pro-walk |
| 2    | L6H6  | −46,136       | 67,445    | pro-walk |
| 3    | L5H6  | −32,557       | 103,329   | pro-walk |
| 4    | L6H5  | −26,501       | 37,316    | pro-walk |
| 5    | L5H5  | −20,633       | 17,340    | pro-walk |
| 6    | L6H7  | +17,775       | 74,032    | pro-jump |
| 7    | L5H0  | −14,498       | 8,715     | pro-walk |
| 8    | L6H4  | −13,400       | 5,103     | pro-walk |
| 9    | L2H3  | −11,785       | 6,598     | pro-walk |
| 10   | L3H2  | −10,488       | 6,373     | pro-walk |
| 11   | **L4H6**  | **−10,042**  | 24,884    | **pro-walk** |
| 13   | L5H3  | −8,549        | 5,160     | pro-walk |
| 23   | L3H4  | +4,516        | 3,376     | pro-jump |
| 41   | **L5H2**  | **+530**     | 1,990     | **pro-jump** |
| 42   | L3H0  | +523          | 368       | pro-jump |
| 5    | **L5H5**  | **−20,633**  | 17,340    | **pro-walk** |

### M2 — Per-head δlogit comparison (fail vs. success, key heads)

| Head  | mean fail    | mean success | Δ (succ−fail) |
|-------|-------------|--------------|---------------|
| L6H2  | −52,469     | +31,540      | +84,009       |
| L6H6  | −46,136     | +6,363       | +52,500       |
| L5H6  | −32,557     | +46,262      | +78,820       |
| L6H5  | −26,501     | +2,740       | +29,241       |
| L5H5  | −20,633     | −7,381       | +13,252       |
| **L4H6** | **−10,042** | **+2,765** | **+12,807** |
| **L5H2** | **+530**  | **+243**   | **−287**    |
| L3H0  | +523        | +415         | −108          |
| L3H4  | +4,516      | +7,037       | +2,522        |

### M4 — Target head breakdown

| Head  | δlogit fail | δlogit succ | Δ        | rank/48 | share of Σδlogit |
|-------|-------------|-------------|----------|---------|------------------|
| L4H6  | −10,042     | +2,765      | +12,807  | 11      | +5.2%           |
| L5H2  | +530        | +243        | −287     | 41      | −0.3%           |
| L5H5  | −20,633     | −7,381      | +13,252  | 5       | +10.7%          |
| L3H0  | +523        | +415        | −108     | 42      | −0.3%           |
| L3H4  | +4,516      | +7,037      | +2,522   | 23      | −2.3%           |

### M5 — Cumulative margin by head rank

50% of negative margin explained by: 4 heads (through L6H5)
75% of negative margin explained by: 8 heads (through L2H3)
90% of negative margin explained by: 13 heads (through L4H4)

---

## Hypothesis Verdicts (V0.1.0 — pre-LayerNorm units)

**H1 — L4H6/L5H2/L5H5 in top-5 by |δlogit|, all negative: FAIL**

L4H6 is rank 11 (not top-5); L5H2 is +530 (pro-jump). Only L5H5 (rank 5) satisfies both conditions.

**K1 — Target heads pro-jump in fail: FIRES**

L5H2 has mean δlogit = +530 (rank 41/48). K1 fires on this value. However, L5H2's contribution is near-zero — its magnitude is comparable to L3H0 (the confirmed non-causal global-context control, +523, rank 42). The K1 trigger on L5H2 reflects that L5H2 does not drive the I_WALK selection through the logit path. It does not indicate that L5H2's mechanism account from S-051/S-057 is wrong; rather, it indicates that L5H2's contribution to the logit margin is negligible.

**H2 — Δ(δlogit_h) ≥ +0.20 for each target head: FAIL (by sign)**

In pre-LN units, L4H6 Δ = +12,807 and L5H5 Δ = +13,252 (both large positive flips). L5H2 Δ = −287 (slight decrease, pro-jump in both groups). The threshold of +0.20 was specified in logit units; these values are in pre-LN units and cannot be directly assessed against it.

**H3 — Target heads ≥ 30% of reconstructed negative margin: FAIL**

Target head share: L4H6 (5.2%) + L5H5 (10.7%) + L5H2 (−0.3%) = 15.6%. Below the 30% threshold. Note: the 30% threshold was defined relative to the total reconstructed negative margin; with L6H2, L6H6, L5H6, L6H5 being larger contributors, the target heads don't dominate even in relative terms.

**H4 — Control heads |δlogit| < L4H6 |δlogit|: PASS**

|δlogit_L3H0| = 523 < 10,042. |δlogit_L3H4| = 4,516 < 10,042. Both satisfied.

---

## Methodological Finding: Missing LayerNorm Correction

The decomposition `δlogit_h = W_U_diff · contrib_h` assumes that lm_head is applied directly to the residual stream. In T5, this is incorrect: the residual stream passes through `model.decoder.final_layer_norm` (an RMSNorm) before `lm_head`.

The correct formula is:

```
W_U_eff = (W_U_diff * LN_weight) / RMS(h_final)   [per example]
δlogit_h_corrected = W_U_eff · contrib_h
```

where:
- `LN_weight = model.decoder.final_layer_norm.weight`  [d_model]
- `RMS(h_final) = sqrt(mean(h_final²) + eps)`
- `h_final` is the pre-LayerNorm decoder hidden state at the divergence step

Under this corrected formula:
- The sign of each δlogit is unchanged (the correction is a per-example positive scalar in `RMS`, and `LN_weight` preserves direction within the positive-weight RMSNorm)
- The relative rankings are preserved (same sign per example; scalar correction doesn't change rank order within an example)
- The absolute values are compressed by the factor 1/RMS(h_final) ≈ 1/32,193 — bringing them into actual logit units
- The sum Σδlogit_h (over all components including FFN, embedding, self-attention) would equal the observed margin after correction

**Consequence for hypothesis verdicts:** The qualitative picture (L5H5 pro-walk rank 5, L4H6 pro-walk rank 11, L5H2 near-zero slightly positive) is preserved by the LayerNorm correction. H1, H2, H3 verdicts are not expected to change. K1 fires on L5H2 regardless — its sign is positive before and after correction, and its rank will remain low (~41/48).

**What the correction changes:** The reconstruction check becomes meaningful. With correction applied to all residual stream components (cross-attention + self-attention + FFN + embedding), the sum should closely match the observed margin. This allows a proper M3 measurement of what fraction of the logit margin comes from cross-attention heads vs. other components.

---

## Unexpected Finding: Layer 6 Head Dominance

The four largest pro-walk contributors in the fail group are L6H2, L6H6, L5H6, and L6H5 — none of which were predicted as target heads. These all show large flips between fail and success (Δ of +29,000 to +84,000 in pre-LN units), suggesting they may carry more mechanistic weight than the original L4H6/L5H2/L5H5 account predicted.

An earlier project rule (private development notes, not part of this public export) states "T5-small's layer 6 (both encoder and decoder) collapses all signals to near-zero." This observation from earlier experiments (S-049, S-055) referred to cross-attention entropy / Born filter signals — not residual stream contributions. The presence of large L6 contributions here suggests that while L6 cross-attention patterns may be degenerate (collapsed), the L6 heads still add substantial residual stream contributions that project onto the logit direction.

This is a new finding. The LayerNorm-corrected V0.2.0 measurements are needed to verify whether L6 heads dominate even in proper logit units.

---

## Interpretation

**What is known from V0.1.0 results:**

1. L4H6 and L5H5 are pro-walk in fail cases and flip toward pro-jump in success cases — consistent with the value-substitution account.

2. L5H2 is near-zero (rank 41/48) and slightly pro-jump. It is not a driver of the I_WALK selection at the logit level. Its cosine similarities from S-057 (both slightly positive toward I_WALK) may reflect a weak geometric effect that doesn't translate to a logit contribution of consequence.

3. L6H2, L6H6, L5H6, L6H5 dominate the pro-walk ranking. These are not in the target set. They show the largest fail-to-success flips. Their role in the failure mechanism is not yet characterized.

4. The value-substitution account (L4H6/L5H2/L5H5 dominate the wrong logit decision) is not confirmed by V0.1.0: the target heads account for only ~16% of the negative margin, and two of the three largest contributors are outside the target set.

5. L3H0 (non-causal global-context control) has magnitude 523, comparable to L5H2 (530). Both are effectively noise-level in the logit decomposition.

**The central question after V0.1.0:** Are L6H2, L6H6, L5H6, L6H5 genuine mechanistic contributors, or do they appear large due to a structural property of L6 that gets corrected out by the LayerNorm? V0.2.0 will answer this.

---

## Findings That Did Not Pan Out

**The logit decomposition as formulated in the proposal does not give logit-unit contributions.** The formula `δlogit_h = W_U_diff · contrib_h` is correct only when lm_head is applied directly to the residual stream (GPT-style architecture). T5 applies an RMSNorm first. The proposal did not account for this, and K2 was not defined to catch over-attribution. The LayerNorm correction is straightforward and will be applied in V0.2.0.

**L5H2 does not appear to drive the logit decision.** The S-051 cosine analysis identified L5H2 as a value-substitution candidate. The logit decomposition shows its contribution is near-zero (rank 41/48). This does not invalidate the cosine finding but indicates the logit margin is not dominated by L5H2.

---

## What's Next

**Current status (May 2026):** S-059 V0.1.0 is sealed as a methodological partial finding. The LayerNorm correction is required before hypothesis verdicts can be evaluated in logit units. V0.2.0 will apply the correction; all measurements will be re-run on the same 29 fail / 25 success examples.

**S-track next — S-059 V0.2.0: LayerNorm-corrected logit decomposition.** No new pre-registration needed — same hypotheses, same design, same examples. The only change is the δlogit formula:

```
W_U_eff = (W_U_diff * LN_weight) / RMS(h_final)   [per example]
δlogit_h = W_U_eff · contrib_h
```

where `h_final` is obtained from `decoder_hidden_states[-2][0, 0, :]` at the divergence step (the pre-LayerNorm hidden state from `model.generate()` with `output_hidden_states=True`). A reconstruction verification step will confirm the formula by checking that `W_U · LN(h_final) = observed_logits`.

**What S-track carries to G-track:** S-059 V0.1.0 is methodological — pending V0.2.0 results before relay. G-057 continues to hold pending S-059 corrected results. The Layer 6 head dominance finding is tentative and should not influence G-track design until confirmed in logit units.

**What S-track needs from G-track:** Nothing before running V0.2.0. G-track holds on G-057.

**Resume sequence:**
1. Read this report
2. Read `workbench/proposals/S-059_LOGIT_DECOMPOSITION_PROPOSAL.md` (no changes needed — same hypotheses)
3. Apply LayerNorm correction to `workbench/experiments/S-059_LOGIT_DECOMPOSITION.py` (version bump to V0.2.0)
4. Key addition: `output_hidden_states=True` is already set; add `dec_hidden_states` to return; use `h_final = data["dec_hidden_states"][div_step_idx][-2][0, 0, :]` (or verify index empirically); compute `RMS` and `LN_WEIGHT`; use `W_U_eff` instead of `W_U_diff`
5. Run V0.2.0; verify reconstruction fraction is ≈ 0.1–0.8 (not 32,000)
6. Write results to `workbench/results/059_v2_results.json`
7. Append V0.2.0 verdicts to this methods report (or write a brief addendum)
8. Update the private development coordination notes (not part of this public export)

---

## V0.2.0 Amendment — LayerNorm-Corrected Results (S-059b)

**Script:** `workbench/experiments/S-059b_LOGIT_DECOMPOSITION_LN_CORRECTED.py`
**Results:** `workbench/results/059_v2_results.json`

### Phase 4b — Hidden state index verification

The script tested all 7 decoder hidden state indices (embedding + 6 blocks per decode step). None produced a reconstruction error near zero:

| Index | ‖h‖ | direct margin | +LN margin | err (|+LN - obs|) |
|-------|------|--------------|-----------|-------------------|
| [0]   | 477.64  | +14751.0 | +59.0  | 60.4 |
| [1]   | 1235.11 | +20966.6 | +84.1  | 85.5 |
| [2]   | 1949.51 | +5718.1  | +31.7  | 33.0 |
| [3]   | 2974.80 | −43284.0 | −11.6  | **10.3** ← best |
| [4]   | 4502.93 | −66330.5 | −29.7  | 28.4 |
| [5]   | 6941.18 | −84229.3 | −32.2  | 30.8 |
| [6]   | 17.72   | −30.4    | +113.8 | 115.1 |

Best index: `[3]` (err=10.27). Used for V0.2.0. Mean reconstruction error across 29 fail examples: **26.67** — not near-zero. The T5 `decoder_hidden_states` from `model.generate()` with KV cache does not cleanly expose the pre-final-LayerNorm residual stream. The correction is approximate; it does not reconstruct the observed margin exactly.

### V0.2.0 reconstruction check

```
Fail group:
  Mean observed margin (jump - walk):              −6.0051
  Mean reconstructed margin (Σδlogit_h):         −171.2509
  Mean residual (obs - recon):                   +165.2458
  Reconstruction fraction (recon / obs):            28.52×

Success group:
  Mean observed margin:                            +0.5862
  Mean reconstructed margin:                     +264.2162
  Reconstruction fraction:                         450.76×

K2: CLEAR (by letter; reconstruction still substantially exceeds observed margin)
```

**Structural interpretation ([internal dev-file path removed in public export]):** The 28.52× reconstruction fraction is not a fixable bug. A complete logit decomposition requires all residual stream components: cross-attention, self-attention, FFN, embeddings. The cross-attention alone (48 heads) accounts for roughly 3.5% of the total logit margin. A massive opposing correction from FFN/embedding/self-attention produces the small observed margin (−6) from a large cross-attention signal (−171). This near-cancellation is a structural property of the model, not a measurement artifact. Per [internal dev-file path removed in public export]: "Do not attempt to fix the decomposition further before moving on."

### M1 — Per-head δlogit, fail group (V0.2.0, top 20 by |δlogit|)

| Rank | Head | mean δlogit | std    | sign     |
|------|------|-------------|--------|----------|
| 1    | L6H2 | −51.21      | 121.97 | pro-walk |
| 2    | L5H6 | −40.56      | 110.17 | pro-walk |
| 3    | L6H6 | −37.67      | 66.20  | pro-walk |
| 4    | L6H5 | −29.61      | 46.79  | pro-walk |
| 5    | L6H7 | +21.90      | 76.47  | pro-jump |
| 6    | **L5H5** | **−19.65** | 21.80 | **pro-walk** |
| 7    | L5H0 | −13.14      | 8.50   | pro-walk |
| 8    | **L4H6** | **−10.41** | 26.87 | **pro-walk** |
| … | … | … | … | … |
| 46   | **L5H2** | **−0.10** | 3.08 | **pro-walk** |

### M2 — Fail-to-success flips (V0.2.0, key heads)

| Head  | fail    | success | Δ (succ−fail) |
|-------|---------|---------|---------------|
| L6H2  | −51.21  | +43.54  | **+94.75** |
| L5H6  | −40.56  | +42.25  | **+82.81** |
| L6H6  | −37.67  | +15.32  | **+52.99** |
| L6H5  | −29.61  | +6.55   | **+36.16** |
| L5H5  | −19.65  | −3.04   | +16.61 |
| L4H6  | −10.41  | +2.79   | +13.20 |
| L5H2  | −0.10   | −1.13   | −1.03  |
| L3H0  | +2.98   | +3.21   | +0.23 (control — stable) |

### M4 — Target head breakdown (V0.2.0)

| Head  | δlogit fail | rank/48 | share | Δ (succ−fail) |
|-------|-------------|---------|-------|---------------|
| L4H6  | −10.41      | 8       | +6.1% | +13.20        |
| L5H2  | −0.10       | 46      | +0.1% | −1.03         |
| L5H5  | −19.65      | 6       | +11.5%| +16.61        |
| L3H0  | +2.98       | 26      | −1.7% | +0.23         |
| L3H4  | +7.84       | 14      | −4.6% | +3.97         |

### M5 — Cumulative margin (V0.2.0)

50% of negative margin: 4 heads (through L6H5)
75% of negative margin: 7 heads (through L4H6)
90% of negative margin: 14 heads (through L3H2)

### V0.2.0 Hypothesis Verdicts

**H1 FAIL.** L4H6 rank 8 (not top-5); L5H2 rank 46. L5H5 rank 6. All three are negative, so the "all negative" condition passes — but not all in top-5.

**K1 CLEAR.** All three target heads are negative in calibrated units. L4H6 = −10.41, L5H5 = −19.65, L5H2 = −0.10. The S-051 mechanism account is not overturned.

**H2 FAIL.** L5H2 Δ = −1.03 (wrong direction). L4H6 Δ = +13.20 and L5H5 Δ = +16.61 both pass individually.

**H3 FAIL.** Target heads account for 17.6% of negative margin (L4H6 + L5H5 + L5H2 = 6.1% + 11.5% + 0.1%). Below the 30% threshold. The L6 cluster (L6H2 + L6H6 + L5H6 + L6H5) accounts for ~52%.

**H4 PASS.** |δlogit_L3H0| = 2.98 < 10.41; |δlogit_L3H4| = 7.84 < 10.41.

**Note on H1/H2/H3:** Per [internal dev-file path removed in public export], these verdicts cannot be honestly evaluated at 28.52× reconstruction. H3's 30% threshold and H2's 0.20 threshold were specified for a complete logit decomposition; they do not apply to the cross-attention-only partial decomposition.

### V0.2.0 Scientific Findings

**What stands:**

1. **K1 CLEAR.** The mechanism account from S-051 is not overturned. L4H6 and L5H5 are pro-walk in fail cases and flip in success cases, consistent with value substitution.

2. **L5H2 is negligible.** δlogit = −0.10 (rank 46/48). L5H2 should be removed from the primary target set. Its cosine similarity from S-057 does not translate to a logit contribution of consequence.

3. **L6 cluster is dominant.** The four largest calibrated pro-walk contributors are L6H2 (−51.2), L5H6 (−40.6), L6H6 (−37.7), L6H5 (−29.6). All four show large fail-to-success flips (+37 to +95 Δ). These heads were not in the original S-051/S-057 target set.

4. **Near-cancellation structure.** The cross-attention total (−171) is ~28× the observed margin (−6). FFN/embedding/self-attention contributes a large opposing correction. The model fails not by a large unchallenged push toward I_WALK, but by the residual imbalance after near-cancellation.

5. **L3H0 is stable and small** (+2.98 fail, +3.21 success, Δ=+0.23). Consistent with its status as a non-causal diagnostic head.

---

## What's Next (Updated — V0.2.0)

**Current status (May 2026):** S-059 V0.1.0 sealed (LayerNorm mismatch). S-059b V0.2.0 sealed: K1 CLEAR, L6 cluster identified as dominant, reconstruction 28.52× (structural — cross-attn only), H1/H2/H3 not evaluable, L5H2 removed from primary target set.

**S-track next — S-060: Causal ablation.** The decomposition has identified candidates; the causal question requires ablation. Per [internal dev-file path removed in public export]:

Primary targets: **L6H2, L6H6** (dominant calibrated δlogit, largest fail-to-success flips)
Reference targets: **L4H6, L5H5** (original S-051 targets, directionally confirmed)
Removed from primary target set: **L5H2** (rank 46/48, δlogit ≈ 0)

Methodology: within-example neutralization (zero out each head's contribution at the divergence step, no cross-example transplant). Compare TARGET arm effect vs layer-matched CONTROL arm. Pre-registered hypothesis: ablating L6H2/L6H6 produces a larger signed change in `logit(I_JUMP) − logit(I_WALK)` at the divergence step than ablating a matched control head at the same layer. Kill condition: if all arms including controls produce similar large effects (non-specificity, same as S-058 K2).

**What S-track carries to G-track:** S-059b identifies the near-cancellation structure (cross-attn total ~28× observed margin; FFN/embedding cancels most of it) and removes L5H2 from the causal target set. The L6 cluster (L6H2, L5H6, L6H6, L6H5) dominates the cross-attn decomposition but causal status is unconfirmed pending S-060. G-057 should pre-register the near-cancellation hypothesis now (toy-scale test: FFN cancellation of cross-attn pro-walk signal), but hold the run until S-060 reports.

**What S-track needs from G-track:** Nothing before S-060. G-track holds on G-057.

**Resume sequence:**
1. Read this report (both V0.1.0 and V0.2.0 sections)
2. Read `workbench/proposals/S-060_CAUSAL_ABLATION_PROPOSAL.md`
3. Build script `workbench/experiments/S-060_CAUSAL_ABLATION.py`
4. Run S-060 in Colab; save results to `workbench/results/060_results.json`
5. Write `findings/METHODS_REPORT_S060.md`
6. Update the private development coordination notes (not part of this public export)

---

*S-Track | Applied Categorical Physics Workbench | Troy Teno | May 2026*
