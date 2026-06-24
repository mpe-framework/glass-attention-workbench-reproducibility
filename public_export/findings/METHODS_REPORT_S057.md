# Methods Report — S-057: L3H4 Characterization
**Applied Categorical Physics Workbench**
Troy Teno | May 2026 | Open Access

> **Historical sealed report.** This report was sealed at the S-057 stage of the
> experiment sequence. At sealing time, the repair target list included L5H2 alongside
> L4H6 and L5H5. Subsequent experiments (S-059b, S-060, S-061) revised the causal picture:
> L5H2 was dropped as a repair target; the final rank-1 W_V repair (S-061) targets L4H6
> and L5H5 only. The content of this report is preserved exactly as sealed; the
> "What's Next" section reflects the state of knowledge at sealing time.
> One redaction applied: `[retired parallax-lever calibration claim removed in public export
> — see RETIREMENTS_AND_METHOD_LESSONS.md]`

**One-line result:** L3H4 is a primitive-sensitive suppressive reader, not a value-substitution head — it attends reliably to the `jump` encoder position (MWU p=0.0007) and its jump-position rank rises from 2nd to 1st in success cases, but its output contribution is suppressive (both cosines negative); the entropy ordering hypothesis fails in a methodologically informative direction, showing that attention entropy does not track semantic specialization in this decoder.

---

## The Question

S-056 characterized the L3 head cluster and found K1 fires for L3H0: not a jump-attending head. G-055 confirmed L3H0 is diagnostic, not causal. L3H4 is the only L3 head that cleared the K1 threshold in S-056 (attn_jump_fail = 0.1018), making it the remaining uncharacterized L3 candidate. S-057 asks: is L3H4 a genuine partial primitive reader, or is the above-threshold attn_to_jump a floor effect from mildly focused global attention?

---

## Design

**Groups:** 29 valid fail pairs (30 selected; 1 excluded for divergence-point alignment failure), 25 success pairs. SEED=42.

**Heads measured:** Primary — L3H4. Comparison — L3H0 (confirmed global), L4H6 (confirmed specialist), L5H5 (second specialist anchor, added per sandbox_015).

**Morphism:** Same paired forward pass as S-055 and S-056 — each example run twice (jump command and matched walk command) through model.generate(); cross-attention weights and hidden states collected at action-slot divergence steps.

**Five measurements:**
- M1: Full attention distribution for L3H4 across all encoder positions; top-5 positions and jump-position rank reported separately for fail and success groups
- M2: Shannon entropy of the cross-attention distribution at action-slot divergence steps for L3H4, L3H0, L4H6, L5H5
- M3: Per-example Mann-Whitney U test on attn_to_jump for L3H4 (and L3H0, L4H6 for comparison)
- M4: Cosine attribution — head contribution via V/O projection against embed(I_WALK div) and embed(I_JUMP div)
- M5: Per-example Born filter defect distribution for L3H4

---

## Results

### M1 — Full attention distribution for L3H4

```
Fail group (T_enc = 8):
  Rank 1: pos=7  attn=0.2351  (EOS/padding position)
  Rank 2: pos=4  attn=0.0716  ← jump encoder position (mean 0.1018 per M3)
  Rank 3: pos=5  attn=0.0437
  Rank 4: pos=6  attn=0.0364
  Rank 5: pos=0  attn=0.0263
  Jump position rank: 2nd of 8

Success group (T_enc = 6):
  Rank 1: pos=0  attn=0.1504  ← jump encoder position (mean 0.2418 per M3)
  Rank 2: pos=5  attn=0.1114
  Rank 3: pos=4  attn=0.0534
  Rank 4: pos=1  attn=0.0432
  Rank 5: pos=3  attn=0.0293
  Jump position rank: 1st of 6
```

The jump encoder position rises from 2nd to 1st between fail and success groups.

### M2 — Shannon entropy comparison (fail group)

```
Head    H (nats)   Character
L3H0    0.3887     Most concentrated (single-position focused)
L5H5    0.9627     Moderate concentration
L3H4    1.0566     Broad distribution
L4H6    1.3032     Most diffuse
```

The ordering is inverted relative to the pre-registered prediction. L3H0, the confirmed global-context head (S-056 K1), is the most concentrated head in the dataset. L4H6, the confirmed jump-attending specialist, is the most diffuse. Success-group entropies show the same ordering with slightly higher values across all four heads.

### M3 — Per-example Mann-Whitney U test on attn_to_jump

```
Head    fail mean   success mean    U       p        r
L3H4    0.1018      0.2418          167.0   0.0007   0.539
L3H0    0.0089      0.0425          89.0    0.0000   0.754
L4H6    0.2838      0.4057          220.0   0.0138   0.393
```

L3H4's attn_to_jump difference is statistically reliable at the per-example level (p=0.0007, r=0.539, medium-to-strong effect). The 2.4× group ratio from S-056 is not a sampling artifact.

### M4 — Cosine attribution

```
Head    cos_walk_fail  cos_jump_fail  diff_fail   cos_walk_succ  cos_jump_succ  diff_succ
L3H4       -0.0436        -0.0257      -0.0178       -0.0438        -0.0095       -0.0343
L3H0       -0.0883        -0.0838      -0.0046       -0.0712        -0.0718        0.0006
L4H6       +0.0771        +0.0476      +0.0295       +0.0361        +0.0607       -0.0246
```

L3H4 and L3H0 are both suppressive (negative cosines for both I_WALK and I_JUMP in both groups). L4H6 shows the value-substitution signature: positive cosines in fail cases, flip in success cases. L3H4 does not show this pattern.

### M5 — Per-example Born filter defect for L3H4

```
           Fail (n=29)          Success (n=25)
Mean:       41.2931              92.0977
Median:     33.6798              88.5159
Std:        21.0174              53.5494
25th pct:   28.2251              56.0086
75th pct:   49.0909             110.3701
```

Fraction of success examples above median fail defect (33.68): **84% (21/25)**

Defect ratios for all four heads (cross-check vs S-056):

```
Head    Fail        Success     Ratio
L3H4    41.2931     92.0977     2.230×   ← matches S-056 exactly
L3H0    14.5251     56.1650     3.867×
L4H6   224.0696    352.4178     1.573×
L5H5   555.4495    682.9574     1.230×
```

The cross-check against S-056 is exact — the instrument is stable.

---

## Hypothesis Verdicts

**H1 (H[L3H0] > H[L3H4] > H[L4H6] >= H[L5H5]): FAIL**

The measured ordering in the fail group is L3H0(0.3887) < L5H5(0.9627) < L3H4(1.0566) < L4H6(1.3032) — opposite to the prediction on both ends. The most concentrated head is L3H0 (the confirmed global-context head), and the most diffuse is L4H6 (the confirmed specialist). Entropy does not track the global-to-specialist axis in this decoder. The methodological lesson: concentration of attention (low entropy) reflects how many encoder positions a head distributes weight across, not how semantically specific that distribution is. A specialist head can attend broadly across multiple positions while still routing most weight toward the semantically relevant one; a non-specialist head can concentrate on a single irrelevant position and appear maximally focused.

**H2 (MWU p < 0.05 for L3H4 attn_to_jump): PASS**

MWU U=167, p=0.0007, r=0.539. L3H4 reliably attends more to the jump encoder position in success cases than in fail cases at the per-example level. The 2.4× group ratio from S-056 is not a sampling artifact.

**H3 (L3H4 cosines suppressive — both negative in fail group): PASS**

cos_walk_fail = −0.0436, cos_jump_fail = −0.0257. Both negative. K2 clear. L3H4 does not carry the value-substitution signature present in L4H6 (positive cosines, correct fail→success flip). L3H4 suppresses both I_WALK and I_JUMP output directions — it is not directly committing to either output token.

**H4A (jump_pos in top 3 for L3H4, fail group): PASS**

Jump ranks 2nd among 8 encoder positions (attn=0.1018, behind EOS/padding at pos=7 with attn=0.2351). Jump is among the top semantically attended positions, as the top position is structural.

**H4B (jump rank rises in success group): PASS**

Jump rises from 2nd to 1st in success cases (attn=0.2418). The head attends more strongly to the jump encoder position when the model will succeed. This is consistent with the head reading or filtering primitive identity, not merely responding to encoder-level salience noise.

**K1 (attn_to_jump < 0.05): CLEAR**

Re-measured attn_to_jump = 0.1018. S-056 measurement confirmed; not a sampling artifact.

**K2 (cos_walk_fail > +0.01): CLEAR**

cos_walk_fail = −0.0436. No value-substitution signature. Proceed without G-track relay.

---

## Interpretation

L3H4 is a **primitive-sensitive suppressive reader**: it reliably tracks the primitive identity of the action token (attending more to the jump encoder position in success cases than in fail cases), but its output contribution suppresses both I_WALK and I_JUMP output directions rather than biasing toward either.

This places L3H4 in a distinct role from both confirmed head types identified so far:
- L3H0 (global-context): low attn_to_jump, diagnostic, not primitive-sensitive
- L4H6/L5H2/L5H5 (value-substitution): high attn_to_jump, positive cosines in fail cases, directly biases toward I_WALK

L3H4 is between these: moderate attn_to_jump (above K1 threshold), real per-example signal (MWU p=0.0007), jump rank rises with success — but suppressive cosines rule out a direct causal role in the I_WALK substitution.

The H1 failure is independently important. Entropy is not a proxy for semantic specialization. L3H0, the confirmed global-context head, is the most concentrated (0.3887 nats — weight on a single non-semantic position). L4H6, the most reliable jump reader, is the most diffuse (1.3032 nats — attention spread across multiple positions). Concentration and semantic focus are orthogonal properties in this architecture. Future experiments should not use entropy alone to classify head roles.

**What is not yet established:** L3H4 has not been ablated with a matched control. The sandbox_015 caution holds: suppressive cosines rule out direct value-substitution participation, but do not rule out a modulatory or upstream role. The appropriate next move is not to ablate L3H4 in isolation but to proceed to the S-058 causal patch at L4H6/L5H2/L5H5 — the known value-substitution heads — and reserve any L3H4 ablation for a later experiment with matched controls.

---

## Implication for the Other Track

The S-057 result leaves the failure chain unchanged at the causal level: L4H6, L5H2, and L5H5 remain the primary repair targets. L3H4's primitive sensitivity may be relevant at the representation level — it may help mark whether the model is in a success or fail trajectory — but it does not drive the wrong token selection.

G-track has completed the phase diagram (G-054) and the global-context ablation (G-055). The next G-track experiment (G-056) can take as input: the value-substitution mechanism in T5-small is causally localized to L4H6/L5H2/L5H5 via W_V miscalibration; no earlier layer carries a direct value-substitution contribution. The natural G-056 question is whether a targeted W_V correction at those heads restores the correct output direction, and what geometric conditions make that correction possible.

---

## What's Next

**Status as of S-057 seal (May 2026):** The L3 cluster is fully characterized and closed. L3H0 is a global-context head (diagnostic, not causal — confirmed by both S-056 and G-055 ablation). L3H4 is a primitive-sensitive suppressive reader (attends reliably to the jump encoder position, but suppresses both output directions — no value-substitution role). The failure chain is causally localized to three decoder heads: **L4H6, L5H2, and L5H5**, all exhibiting the value-substitution signature (positive cosines toward I_WALK in fail cases, flip in success). These are the only remaining causal repair targets.

**S-track next experiment — S-058:** Causal patch at the value geometry of L4H6, L5H2, and L5H5. The core question: does a targeted correction to W_V at these heads restore the correct output direction (I_JUMP instead of I_WALK) for compound jump inputs? Pre-register before running — hypotheses locked before data. The ablation must follow the G-055 standard: matched control head, scale-preserving test, action-slot-local readout. Do not ablate L3H4 in this experiment; its role is peripheral and a matched-control ablation would require a separate experiment.

**What S-track is carrying to G-track:** The causal picture is complete enough to hand off a precise question. The value-substitution mechanism is localized to L4H6/L5H2/L5H5 via W_V miscalibration; no L3 head carries a direct value-substitution contribution. The entropy-specialization decoupling (H1 failure) is a methodological finding that applies directly to G-056 head classification — do not use entropy alone to categorize head roles. The S-track defect ratios for reference: L3H4=2.23×, L3H0=3.87×, L4H6=1.57×, L5H5=1.23×.

**What S-track needs from G-track (for S-058 design):** G-056 is investigating the geometric conditions under which a targeted W_V correction is sufficient to restore the correct output direction. S-058 needs two things from G-056 before finalizing the correction magnitude: (1) [retired parallax-lever calibration claim removed in public export — see RETIREMENTS_AND_METHOD_LESSONS.md]; (2) confirmation of the regime boundary constraints from G-054 — whether T5-small's operating point is in the sharp-collapse regime and whether a small W_V correction can push it across the boundary. If G-056 is not yet sealed when S-058 is ready to run, S-058 can proceed with an unconstrained correction sweep and use G-056's results to interpret the magnitude post-hoc.

**Resuming after a gap:** Read this report and the COORDINATION.md in the private development repository (not part of this public export). S-058 proposal goes in `workbench/proposals/S-058_*.md` before any script is written. Results save to `workbench/results/058_results.json`. Methods report goes in `findings/METHODS_REPORT_S058.md`.

---

*S-Track | Applied Categorical Physics Workbench | Troy Teno | May 2026*
