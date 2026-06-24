# Retirements and Method Lessons

**Glass Attention Workbench — Reproducibility Export**

This document records experiments that were retired from the evidentiary chain and experiments
that returned methodological findings rather than hypothesis-level results. All are preserved
in the record as lessons, not as failed science. Pre-registered kill conditions are documented
in the corresponding methods reports.

---

## Retired Experiments

### G-052: Parallax Lever Calibration — RETIRED

G-052 computed a norm asymmetry formula ("logit gap = N·cos(θ), R²=1.000000") claiming
that a 500× norm advantage is "essentially absolute" and that the formula reproduces
T5-small's exact production logits (WALK: +146.12, JUMP: −46.00).

**Retirement reason:** The calibration is excluded from the public evidentiary chain because:
1. The R²=1.0 result is a mathematical identity (linear dot product formula), not an
   empirical finding.
2. The claim that the toy geometry "reproduces" T5-small's exact logits is an artifact of
   parameter fitting — the toy geometry parameters (norm=520, angle=73.7°) were selected to
   match the S-track logits, not derived independently.
3. References to the "exact analytical boundary" and safety applications derived from this
   calibration are not part of the paper's evidentiary chain.

**What is retained:** The norm asymmetry phenomenon itself (I_WALK norm > I_JUMP norm as a
training-frequency artifact) is a valid observation reported in S-051. The parallax metaphor
is used descriptively in the paper but the G-052 formula is not cited as a quantitative
predictor of repair magnitude.

**In this repository:** `findings/METHODS_NOTE_G052.md` is replaced by a stub.
All references to "the G-052 parallax formula predicts the exact angle correction needed"
in other methods reports are replaced with:
`[retired parallax-lever calibration claim removed in public export — see RETIREMENTS_AND_METHOD_LESSONS.md]`

---

### G-039 through G-051: Pre-Paper Development Experiments — EXCLUDED

These experiments were conducted during the development phase of the G-track prior to the
paper's experimental design being finalized. They are not part of the public evidentiary chain.
Methods reports for these experiments are not included in this export.

---

## Method Lessons

The following experiments returned methodological findings rather than hypothesis-level results.
Their methods reports are included in this export as they document the experimental record.

### S-058: Activation Patching — Method Lesson (K2 Confound)

All six patch arms (including the known-diagnostic global head L3H0) produced 100% failure
flip with statistically indistinguishable margins (~+1.69 to +1.94). The intervention was
not head-specific. The confound: donor-context injection at the action-slot divergence step
injects success-context signal broadly across the representation, regardless of which head
is patched. Within-example hook-based patching is invalidated for head-specific causal
measurement at this step.

**Consequence:** This experiment redirected the repair approach to weight-level intervention
(S-061), which is the primary repair experiment.

### S-059 V0.1.0: Logit Decomposition (Pre-RMSNorm) — Method Lesson (Methodological Error)

The decomposition was computed on pre-RMSNorm activations, producing a reconstruction fraction
of ≈32,193× instead of ≈1.0. All hypothesis verdicts from this run are invalid. The corrected
run S-059b (post-LayerNorm activations) produces reconstruction fraction 28.5×, confirming
genuine near-cancellation structure. All values cited in the paper are S-059b figures.

### S-060: Causal Ablation — Method Lesson (Hook Disruption)

Hook-based within-example subtraction disrupted RMSNorm rescaling. L6H2 and L6H6 showed
sign-reversed Δmargin (ablating these heads worsened the margin), identifying them as
compensatory rather than causal heads. L4H6 and L5H5 showed positive specificity
(+0.13 and +0.23 respectively on non-OOD examples). This established the compensation
structure and directed S-061 to intervene at L4H6 and L5H5 via weight modification,
not hook subtraction.

---

## Attribution Note

References to a collaborator (Sargsyan) who is not an author of the paper appear in some
development-era methods reports. These attributions are removed in the public export with
the marker:
`[Sargsyan attribution removed in public export — see RETIREMENTS_AND_METHOD_LESSONS.md]`

---

*Glass Attention Workbench · mpe-framework/glass-attention-workbench-reproducibility · 2026*
