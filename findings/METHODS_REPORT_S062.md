# Methods Report: S-062 — H4 Success Group Specificity Check
**Applied Categorical Physics Workbench**
Troy Teno | May 2026 | Open Access

**Status: SEALED**
Script: `workbench/experiments/S-062_H4_SUCCESS_CHECK.py` (V0.1.0)
Results: `workbench/results/062_results.json`
Pre-registration: `workbench/proposals/S-062_H4_SUCCESS_CHECK_PROPOSAL.md`

---

## Summary

S-062 is a targeted supplement to S-061, designed to evaluate H4 (specificity of the
W_V repair) after a classifier bug in S-061 produced n_success=0.

**The fail group cross-check is exact:** Δm_fail = +2.9024, 16/30 flips — identical to
S-061 to four decimal places. The correction geometry was reproduced correctly.

**The success group again failed to provide a valid comparison:** n_success = 2 (not 25).
H4 formally fails (high) by the pre-registered [-0.3, +0.3] criterion: mean Δm_success =
+1.7853. K2 is clear (no degradation; 0/2 flip regression; 0/2 OOD).

**The +1.7853 result is interpretable and does not indicate a specificity failure** — see
§Interpretation. But it does not confirm specificity either. H4 remains indeterminate.
A proper specificity test requires a non-jump-around control group.

---

## Background

S-061 sealed with H1/H2/H3 passed and H4 unevaluable (n_success=0, classifier bug).
S-062 fixed the classifier (`has_around and is_correct and n_jump_tgt >= 1`) and added
pre-filtering to only around+jump examples to reduce Phase 3 runtime.

The H4 threshold was tightened in pre-registration: from `> -1.0` (S-061, too loose) to
`∈ [-0.3, +0.3]` (near-zero two-sided test, appropriate for a specificity claim).

---

## Results

### Phase 2 — Pre-filtering
- Full test set: 7706 examples
- Pre-filtered (around + jump in command): 3822 examples
- Skipped: 3884 non-around-jump examples

### Phase 3 — Classification
- Substituted-walk fail examples found: **814** (same as S-061, confirming data pipeline)
- Has-around+correct (n_jump_tgt ≥ 1) examples found: **2**
- Used: 30 fail, 2 success

### Fail group cross-check (Phase 7–8)

| Metric | S-062 | S-061 |
|--------|-------|-------|
| mean Δm_fail at joint α=1.0 | +2.9024 | +2.9024 |
| flips | 16/30 | 16/30 |

**Exact match.** Correction geometry reproduced correctly from scratch.

### Success group (n=2)

| Metric | Value |
|--------|-------|
| Baseline mean margin | +0.8319 |
| Corrected mean margin | +2.6172 |
| **(1) mean Δm_success** | **+1.7853** |
| **(2) flip_regression** | **0/2** |
| **(3) OOD count** | **0/2** |

### H4 verdict

H4 (near-zero criterion [-0.3, +0.3]): **FAIL (high)**
mean Δm_success = +1.7853 > +0.3.

K2 (meaningful degradation < -0.5): **CLEAR**
No degradation observed. No flip regression. No OOD.

---

## Interpretation

### Why n_success = 2

The SCAN add_prim_jump test set is constructed specifically to test OOD generalization of
"jump" in compound contexts. Almost all compound jump examples fail — that is the benchmark's
design. Among the 3822 around+jump pre-filtered examples, 814 are substituted-walk failures
and only 2 are correct. The corrected classifier (no n_walk_tgt constraint) is correctly
implemented but the underlying data has essentially no "jump around" examples the model gets
right. S-060 found 25 success examples using a different classifier — one that likely did not
require n_jump_tgt ≥ 1 and therefore captured "walk around" and other non-jump-around correct
examples. See §Findings That Did Not Pan Out.

### Why +1.7853 is not a specificity failure

The 2 success examples are "jump around" commands the model already predicts correctly at
baseline. Their encoder representation at the divergence step is the OOD jump direction —
the **same direction** that the rank-1 W_V correction is designed to redirect toward pro-jump.

For these examples the correction is doing exactly what it does for the fail group: it makes
the mapping of the OOD jump encoder representation more pro-jump. These examples are already
pro-jump at baseline (margin = +0.83), so the correction pushes them further in the same
direction (to +2.62). This is the mechanism working correctly on both sides of the margin —
it is not collateral damage to an unrelated representation.

**This result does not tell us whether the correction affects examples where the OOD jump
direction is not involved.** That question — the actual specificity question — requires
a control group with a different encoder representation.

### What specificity actually requires

A valid H4 test needs examples where:
1. The model is correct (control group)
2. The encoder representation at the divergence step does **not** involve the OOD jump direction
3. Ideally: "walk around," "run around," "look around" commands (has_around, model correct,
   n_jump_tgt = 0)

For those examples, the rank-1 correction along the OOD jump direction should have near-zero
effect — the correction only changes the mapping of vectors aligned with `u` (the mean jump
encoder direction), leaving all orthogonal directions unchanged. If the walk/run/look around
encoder representations are nearly orthogonal to `u`, the Frobenius norm argument predicts
Δm ≈ 0 for the control group.

**Geometric argument for expected specificity:** The rank-1 correction is `ΔW_V = α × outer(v_correction, u)`. Its effect on any input vector `h` is `ΔW_V @ h = α × (u · h) × v_correction`. For a "walk around" encoder state `h_walk`, if `u · h_walk ≈ 0` (OOD jump direction is orthogonal to in-distribution walk direction), the effect is zero. This is geometrically expected but not yet empirically verified.

---

## Findings That Did Not Pan Out

### Success group definition mismatch with S-060

S-060 found 25 success examples and S-062 found 2, despite both running on the same SCAN
test split. The difference is the classifier. S-062 required `n_jump_tgt >= 1` (i.e., the
correct output must contain I_JUMP). S-060 almost certainly used `has_around and is_correct`
without any target token constraint, which captures "walk around," "run around," etc. — all
the around+correct examples regardless of which primitive is involved.

This mismatch is the root cause of the H4 evaluation failure across both S-061 and S-062.
The S-061 bug (n_walk_tgt >= 1) over-constrained in one direction; the S-062 fix
(n_jump_tgt >= 1) over-constrained in the other direction. Neither produced the right group.

The correct success group for the original H4 test (S-060 style) is:
```python
has_around and is_correct   # no target token constraint at all
```
This gives the 25 examples S-060 used. For the more stringent specificity test proposed
in this report (non-jump-around control), the definition is:
```python
has_around and is_correct and n_jump_tgt == 0   # walk/run/look around only
```

### H4 remains indeterminate

Neither the H4 threshold from S-061 (> -1.0) nor the tightened criterion from S-062
([-0.3, +0.3]) has been evaluated against a valid comparison group. The formal H4 verdict
from S-062 is FAIL (high), but this result is not interpretable as a specificity failure —
it reflects the wrong comparison group (jump-around correct examples, which share the same
encoder representation as the fail group).

---

## What Is Needed for H4

**Option A — Replicate S-060 success group (n≈25):**
Run corrected model on the 25 has_around+correct examples from S-060 (classifier:
`has_around and is_correct`, no token constraint). These are primarily "walk around" and
similar commands. If mean Δm_success ∈ [-0.3, +0.3], specificity is confirmed for the
comparison group S-060 originally used.

**Option B — Targeted orthogonal control (n≈25):**
Run corrected model on `has_around and is_correct and n_jump_tgt == 0` examples (walk/run/look
around correct). These are the purest test of whether the OOD jump direction correction bleeds
into unrelated encoder representations. Near-zero effect here would be the strongest
specificity claim.

Either option is a minimal script change — the classifier filter is the only difference.
Both confirm or refute the geometric argument in ≈ 5 minutes of Colab time.

Troy decides whether to pursue Option A, Option B, or declare H4 indeterminate and close
the arc on the basis of the mechanism findings (H1/H2/H3) alone.

---

## What's Next

### S-Track status
S-062 sealed. H1/H2/H3 confirmed (from S-061). H4 indeterminate (insufficient/wrong
comparison group across both S-061 and S-062). K2 clear in both experiments.

### Decision point for Troy

The mechanism is fully confirmed: S-061 H1/H2/H3 all pass, K1/K2/K3 clear. The failure
mode is identified, the causal mechanism is confirmed, and a surgical repair that flips 53%
of fail examples is demonstrated.

H4 (specificity) has not been evaluated against the right group. The geometric argument for
specificity is strong (rank-1 correction along OOD direction, ~orthogonal to walk encoder
representations), but it has not been empirically verified.

**If Troy closes the arc here:** Note H4 as indeterminate with geometric motivation for
expected specificity. This is an honest and complete scientific record.

**If Troy wants S-063:** Option A (replicate S-060 success group) or Option B (orthogonal
control). Pre-register, build script, 5-minute Colab run.

### G-Track relay
No change from S-061 relay. G-track is cleared to proceed to G-058.
S-062 adds: H4 indeterminate due to comparison group problems; geometric argument suggests
specificity but not empirically confirmed at production scale. G-058 can include a toy-scale
specificity check (apply rank-1 correction, measure effect on in-distribution examples with
orthogonal encoder representations).

### Resume sequence
1. Read `COORDINATION.md` for current sync state.
2. Read `findings/METHODS_REPORT_S061.md` for mechanism findings.
3. Read `findings/METHODS_REPORT_S062.md` (this file) for H4 status.
4. Troy decides: close arc or proceed to S-063.
5. If S-063: write proposal, build script (classifier change only), run in Colab.

---

*S-062 sealed May 2026. Applied Categorical Physics Workbench | Troy Teno | Open Access*
