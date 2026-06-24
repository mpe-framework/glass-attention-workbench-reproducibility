# Methods Report — G-055: Global-Head Causal vs. Diagnostic Distinction

**Applied Categorical Physics Workbench | Troy Teno | May 2026**
**Status:** Sealed | **Script:** `workbench/experiments/G-055_GLOBAL_HEAD_CAUSAL_DIAGNOSTIC.py`
**One-line result:** H2 confirmed — the global-engagement head is diagnostic, not causal; ablating it changes specialist-layer Born filter ratio by 4%, equal to ablating a near-identity head.

---

## The Question

S-056 found that decoder head L3H0 in T5-small shows the highest Born filter
success/fail ratio (3.867×) of any measured head — larger than the identified failure
heads at layers 4–5 (L4H6, L5H2, L5H5). The direction is inverted from expectation:
success cases produce higher Born filter defect than failure cases at L3H0. L3H0 also
engages 4–5× more with relevant encoder positions in success than fail. It is a
global-engagement head (diffuse attention, no specialist routing) and sits *before* the
failure heads in the network (layer 3 < layers 4–5).

Two interpretations are both consistent with S-056:

**Resolution A — Causal:** L3H0 actively routes encoder information into the residual
stream in success cases. In failure cases it cannot find the signal (the encoder hasn't
encoded it), so the routing is absent. Layers 4–5 depend on what L3H0 deposits — without
that deposit, the specialist heads resort to fallback value substitution (I_WALK). L3H0's
sensitivity is mechanistically upstream of the failure.

**Resolution B — Diagnostic:** The encoder encodes success and failure states
differently. L3H0 reads that encoding — its high Born filter ratio means it sees the
morphism acting strongly on a real signal. But layers 4–5 also read the encoder directly
via their own cross-attention. Both L3H0 and L4–5 are parallel readers of the same
upstream encoder state. Ablating L3H0 would not change L4–5 behavior.

G-055 tests this distinction at toy scale with a controlled ablation. The question
matters for intervention design: if causal (A), L3H0 is a new repair target beyond
S-051; if diagnostic (B), S-051's mechanism is complete and L3H0 is a probe.

---

## Design

**Architecture:** 6-layer toy transformer (NumPy only), single head per layer, D=16
(8 GEO dims, 8 FIN dims), 6-token sequences.

**Head roles:**

| Layer | Role | Q/K initialization | Code index |
|-------|------|--------------------|------------|
| 1 | Near-identity | 0.1 × I | 0 |
| 2 | Near-identity (control ablation target) | 0.1 × I | 1 |
| 3 | Global (primary ablation target) | 0.005 × ones(D,D) | 2 |
| 4 | FIN-specialist | 4.0 × I on FIN dims, 0 on GEO dims | 3 |
| 5 | FIN-specialist | same | 4 |
| 6 | Near-identity | 0.1 × I | 5 |

All layers use V = I (identity value projection). The global head's near-zero Q/K
ensures attention weights ≈ 1/6 per token pair regardless of content — verified in
Step 4 (observed weights: 0.1666–0.1668 for all tokens in both states). The
FIN-specialist Q/K projects strongly onto FIN dimensions, creating high attention
weight to FIN-rich tokens.

**Input states:**
- **S_success:** bank = normalize(GEO_BASE + 0.5 × FIN_BASE). FIN_WEIGHT=0.5 places
  bank in the G-054 no-collapse regime. FIN-specialist heads find clear FIN signal in bank.
- **S_fail:** bank = normalize(GEO_BASE + 0.05 × FIN_BASE). FIN_WEIGHT=0.05 places
  bank in the slow-collapse regime. FIN-specialist heads have minimal FIN signal from bank.

Both states use the same context tokens: current (pure GEO), geo1, geo2 (pure GEO),
fin1, fin2 (pure FIN). Tokens are constructed with small noise (σ=0.05) and fixed by
seed (SEED=55).

**Morphism:** Swap bank ↔ current (positions 0 and 1). Born filter at bank position:
‖h(seq)[0] − h(swapped)[0]‖ / 2.

**Ablation procedure:**
1. Baseline: run forward pass normally, record Born filter profile at all layers.
2. Layer-3 ablation: zero the global head's output before adding to residual. Re-run
   layers 4–6 and measure Born filter.
3. Layer-2 control: zero layer-2's (near-identity) output. If this changes specialist
   behavior by >5%, flag as architecture confound.

**H4 sweep:** 9 FIN_WEIGHT variants (0.05 to 0.70) — compute Pearson r between
L3-defect and L5-defect in baseline and ablated conditions.

---

## Results

### Baseline Born filter profile

The ratio S_success/S_fail at the specialist layers is approximately **1.87×** across
all layers — modest but consistent. The profile is nearly flat from L0 through L6: both
states grow proportionally as information propagates. There is no mid-layer peak (the
ratio stays within 1.86–1.87 across all layers). This is consistent with G-054's
no-collapse regime for S_success and the slow-collapse regime for S_fail — in the
slow-collapse regime the denominator does not approach zero at mid-layers, so no
super-amplification occurs.

### H1 — NULL (Causal hypothesis rejected)

Ablating layer 3 changes the specialist-layer Born filter ratio by **4.0%** (layers
4–6: 1.87× → 1.94×). Threshold was >50%. The global head is not causally load-bearing.

### H2 — CONFIRMED (Diagnostic hypothesis confirmed)

4.0% < 10% threshold. The global head's output does not meaningfully determine what
the specialist heads at layers 4–5 see. Removing it changes the ratio by less than
the pre-registered diagnostic threshold.

### H3 — CONFIRMED (Monotonic profile through layer 3)

Ratio at L0–L3: [1.8651, 1.8660, 1.8669, 1.8669] — monotonically increasing, no
mid-layer amplification. The global head contributes small upward drift, not a
peak. Consistent with G-054: global heads without specialist collapse conditions
produce monotonic profiles.

### H4 — CONSISTENT (Correlation preserved after ablation)

Pearson r(L3 defect, L5 defect) baseline: **+1.0000**. After ablation: **+0.9999**.
Δr = −0.0001. The correlation is essentially unchanged. This is the diagnostic
signature: L3 and L5 both track the bank token's FIN content directly. Removing L3
as a mediator does not break the correlation because there is no mediation — both
layers read the same input signal in parallel.

### Control — CLEAN

Ablating layer 2 (near-identity) produces the same **4.0%** change as ablating layer 3
(global). This equivalence is itself a finding (see below). Control threshold (5%) met.

### Attention diagnostics

The specialist heads (L4, L5) attend predominantly to **fin1** (≈56–63%) and
**fin2** (≈37–44%) from the bank position, not to bank itself (<0.01). This pattern
holds in both S_success and S_fail, with or without global-head ablation. The
Born filter at the bank position is therefore driven by the **direct embedding difference**
between bank and current propagating through the residual stream — not by
specialist-head routing of the bank token's FIN content. The specialist heads read from
the dedicated FIN tokens (fin1, fin2), which are identical between S_success and S_fail.
What differs is the bank token's own FIN content in the residual, and this difference
propagates independently of what layer 3 does.

---

## Interpretation

**The global head and a near-identity head are downstream-equivalent.** Ablating either
produces the same 4.0% change at specialist layers. The global head's uniform mixing —
spreading all token information to all positions — adds nothing that the near-identity
heads don't already contribute. From the specialist heads' perspective, the residual
stream after layer 3 carries the same information it would carry after any other
near-identity layer.

**The 1.87× ratio is embedding-driven, not routing-driven.** The FIN-specialist heads
attend almost entirely to fin1 and fin2 (the dedicated FIN tokens in the sequence),
regardless of state or ablation. The Born filter at the bank position reflects how much
bank's own FIN embedding differs from current's (pure GEO), propagated through
subsequent layers as a residual difference. In S_success, bank has substantial FIN
content (FIN_WEIGHT=0.5); in S_fail, bank has minimal FIN content (FIN_WEIGHT=0.05).
The specialist heads do not create this difference — they simply pass it along.

**Why the correlation is r=+1.0000.** Both L3 and L5 defects are measuring the same
underlying quantity: how much the bank-to-current difference has propagated through
the residual stream at that depth. Since each layer adds only small contributions and
all layers observe the same initial embedding difference, the correlation is essentially
perfect. Ablating layer 3 removes one small contributor without disrupting the
underlying signal.

**The G-054 regime prediction was correct but incomplete.** G-054 mapped three regimes
by collapse conditions. G-055 shows that the global head — even when placed at an
architecturally significant position (before the specialist heads) — does not generate
or amplify those collapse conditions. The regime is determined by the bank token's
embedding FIN content, not by what the global head routes into the residual. A global
head cannot shift the model from one G-054 regime to another.

---

## Implication for the Other Track

**S-track implication:** L3H0 in T5-small is diagnostic. The complete failure mechanism
is as identified in S-051: norm asymmetry (S-050b), layer-5 amplification (S-051), and
value substitution at L4H6/L5H2/L5H5. L3H0's 3.867× Born filter ratio reflects that
it observes the encoder's success/fail encoding directly — it is a sensitive indicator,
not a mediator. An intervention targeting L3H0 (e.g., patching its output) should not
change the failure rate measurably. S-057, if designed, should focus on the identified
mechanism sites (L4–5 value geometry, embedding norm), not on L3H0.

The S-track result L3H4 (attn_jump_fail=0.1018, above the K1 threshold) remains an
open characterization question — G-055 does not address specialist heads with genuine
jump-attention, only global-engagement heads without it.

---

## What G-055 Does Not Prove

G-055 tests the causal/diagnostic distinction for a global head at toy scale with
designed embeddings. Several questions remain:

- Whether T5-small's L3H0 is truly diagnostic requires a direct intervention in the
  production model (patch L3H0's output and measure failure rate). G-055 provides
  strong prior evidence that it will be, but does not directly test T5-small.

- Whether a global head placed at a *different* architectural position (e.g., after the
  specialist heads rather than before) would be causal is not tested. G-055 tests only
  the pre-specialist placement.

- Whether a global head with *learned* Q/K weights (from training) could become
  genuinely causal is an open question. G-055 uses a fixed-initialization global head.

- Whether a *suppressive* head (V projects away from the target semantic direction,
  producing negative cosines to both action tokens) is also diagnostic, or whether
  active suppression of the residual creates causal downstream effects. S-057 (below)
  identified exactly this head type in T5-small. G-056 must answer this.

---

## What's Next

### State of both tracks as of this report (May 2026)

**G-track is sealed through G-055 and is waiting.** There is no new G-track experiment
to run until S-058 produces results. G-track is one experiment ahead of S-track in
the feedback loop, which is within the acceptable synchrony window.

**S-track is sealed through S-057.** S-057 was the direct S-track response to this
report's finding. It characterized L3H4 — the only other L3 decoder head above the
K1 attention threshold in S-055. Results from S-057 are summarized immediately below
because they close the loop this report opened, and because understanding them is
required to understand what comes next.

---

### S-057 result summary (the S-track response to G-055)

S-057 (script: `S-057_L3H4_CHARACTERIZATION_V0.1.0.py`, sealed May 2026) ran 29 fail
and 25 success examples through T5-small and measured L3H4 alongside L3H0, L4H6, and
L5H5. Key findings:

**H1 NULL — entropy ordering inverted.** The expected ordering (global head = highest
entropy, specialists = lower) was wrong. In fail cases, L3H0 has the LOWEST entropy of
the four heads (0.387 nats). L3H4=1.057, L5H5=0.963, L4H6=1.303. In success cases,
L3H0's entropy doubles to 0.779 — suggesting it concentrates narrowly on one encoder
position in fail and spreads in success. Entropy does not organize heads by role.

**H2 CONFIRMED — L3H4 significantly attends to jump position.** attn_to_jump: fail=0.1018
(rank 2 of 8 encoder positions), success=0.2418 (rank 1 of 6). MWU p=0.0007, r=0.539.
L3H4 is not fully global — it has partial jump specificity that is measurably stronger in
success than in fail.

**H3 CONFIRMED — L3H4 cosines are suppressive.** cos_walk_fail=−0.044, cos_jump_fail=−0.026.
Both negative. L3H4's output at the action slot projects away from both I_WALK and I_JUMP
directions — it is neither routing nor substituting. Contrast with L4H6: cos_walk_fail=+0.077
(strongly positive = I_WALK substitution).

**H4 CONFIRMED — jump rank rises from fail to success** (rank 2 → rank 1).

**K1 CLEAR — L3H4 is not fully global.** attn_to_jump=0.1018 exceeds the 0.05 threshold.

**K2 CLEAR — no value substitution at L3H4.** cos_walk_fail=−0.044 is not >+0.01.
The value substitution signature (positive I_WALK cosine) does NOT fire at L3H4.

Born filter defect ratios (success/fail): L3H0=3.867×, L3H4=2.230×, L4H6=1.573×,
L5H5=1.230×. Diagnostic/passive heads (L3H0, L3H4) have HIGHER ratios than failure
heads (L4H6, L5H5) — the failure heads are locked into I_WALK substitution and show
a smaller signal difference. The diagnostic heads observe the encoder signal directly,
tracking the full success/fail difference.

**Head taxonomy established:**

| Head | Character | attn_jump fail | cos_walk fail | Born ratio |
|------|-----------|---------------|---------------|------------|
| L3H0 | Global, passive observer | 0.009 | −0.088 | 3.867× |
| L3H4 | Suppressive, partial jump specificity | 0.1018 | −0.044 | 2.230× |
| L4H6 | Value substitution (I_WALK) | 0.284 | +0.077 | 1.573× |
| L5H5 | Value substitution (I_WALK) | — | — | 1.230× |

Both L3 heads are now characterized. Neither is a repair target. The sign reversal in
cos_walk_fail — from negative at L3H4 (−0.044) to positive at L4H6 (+0.077) — is the
boundary between diagnostic and causal. Heads that project away from I_WALK are
observers; heads that project toward I_WALK are the failure mechanism.

---

### What must be carried to the S-track thread

The following should be pasted to the S-track agent at the start of its next session.
Quote precisely — do not summarize beyond what is written here.

1. **S-057 is sealed. The suppressive head (L3H4) has been characterized.** It is not
   a repair target. The sign on cos_walk_fail distinguishes diagnostic heads (negative)
   from causal heads (positive). Both L3 heads are confirmed as observers.

2. **The complete causal failure chain is:** encoder marks jump OOD → L4H6/L5H2/L5H5
   attend to jump position at p=0.5–0.7 → W_V returns I_WALK direction (value
   substitution) → norm asymmetry (‖I_WALK‖=520) amplifies I_WALK in logit competition
   → layer-5 amplification (sharp-collapse regime, G-054) → failure output. This chain
   was complete as of S-051 and has not changed.

3. **S-058 is drafted but NOT run until G-056 reports back.** Sandbox_017 (supervisor)
   requires this sequencing: if G-056 finds that suppressive early heads are causal,
   then L3H4 must be included in the S-058 intervention design. If G-056 finds suppressive
   heads are diagnostic, S-058 focuses cleanly on L4H6/L5H2/L5H5. Pre-register S-058
   now, but gate the run on G-056 result.

   S-058 design: activation patching first, weight patching second. Replace the
   value-output tensors of L4H6/L5H2/L5H5 at the action-slot divergence step with
   tensors from a structurally matched success example. Matched pair criteria (from
   [internal dev-file path removed in public export]): same command structure, same `around` operator, same action count,
   same direction pattern where possible, same action-slot location. Only the primitive
   differs (jump vs. trained primitive).

   **S-058 pre-registered primary hypothesis ([internal dev-file path removed in public export] formulation):**
   After activation patching L4H6/L5H2/L5H5, `P(I_JUMP) > P(I_WALK)` at the
   action-slot divergence point in at least 50% of patched fail examples.
   This is a per-example token-competition criterion, not aggregate accuracy.

   Required controls: matched control head (similar scale, no predicted value-substitution
   role), scale-preserving control, action-slot-local readout, per-example reporting.

4. **S-058 is gated on G-056.** S-track should draft the S-058 proposal in parallel
   but must wait for G-056 results before finalizing and running.

---

### What G-track is doing next (G-056)

G-056 will test the suppressive head at toy scale — the counterpart to G-055's neutral
global head test. G-055 showed that a head with V = I (neutral, identity value projection)
is diagnostic. S-057 identified a head with negative cosines to both action directions
(suppressive). The open question: does a suppressive head — one whose V matrix actively
projects away from the target semantic direction — create a causal downstream effect,
unlike the neutral head?

**Pre-registered G-056 design (ready to run, pending Troy's go-ahead):**

Same 6-layer NumPy toy as G-055 (D=16, N_TOK=6, S_success FIN_WEIGHT=0.5,
S_fail FIN_WEIGHT=0.05). Replace layer 3's V = I with V = I − 2·P_FIN (projection
matrix that subtracts the FIN component from its input). Q/K remain globally uniform
(same as G-055 global head). Ablation test: zero suppressive head output from residual,
remeasure Born filter at layers 4–6. Control: layer-2 near-identity ablation (same as
G-055). Compare ablation effect to G-055's 4.0% baseline.

Hypotheses:
- **H1 (Causal):** Ablating the suppressive head changes specialist-layer Born filter
  ratio by >10%. The active removal of FIN content from the residual is load-bearing.
- **H2 (Diagnostic):** Ablating the suppressive head changes Born filter ratio by <10%,
  consistent with G-055. Even suppressive routing is downstream-equivalent.
- **H3:** The suppressive head's Born filter profile shows a mid-layer dip (not peak),
  because the suppression subtracts signal from one state and not the other.

H1 and H2 are mutually exclusive. If H1, suppression is a new mechanism that changes
the conclusion from G-055. If H2, the G-055 result generalizes: early-layer heads of
any V-routing type are downstream-equivalent at specialist layers, and the causal chain
is fully embedding-determined.

---

### Information each track needs before proceeding

**G-track needs nothing from S-track.** G-056 is fully self-contained at toy scale.
Runs immediately.

**S-track must wait for G-056 before running S-058.** This is the [internal dev-file path removed in public export] sequencing
requirement. S-track drafts the S-058 proposal now. The run gate is G-056's H1/H2
verdict: if H1 (suppressive head causal), revise S-058 to include L3H4 in the
intervention; if H2 (suppressive head diagnostic), run S-058 as designed.

**After G-056 seals:** relay H1/H2 verdict to S-track via the private development coordination notes (not part of this public export). S-track
then finalizes and runs S-058.

**After S-058 seals:** relay the per-example patching result to G-track. If
`P(I_JUMP) > P(I_WALK)` in ≥50% of patched fail examples, the mechanism is confirmed
causal and the intervention is sufficient at the head level. The workbench moves from
mechanism identification to mechanism repair — the paper-worthy result.

---

*Methods report updated after S-057 to reflect complete track status.*
*G-track | Applied Categorical Physics Workbench | Troy Teno | May 2026*
