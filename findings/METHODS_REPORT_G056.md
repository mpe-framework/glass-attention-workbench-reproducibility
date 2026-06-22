# Methods Report — G-056: Suppressive Head Causal Test

**Applied Categorical Physics Workbench | Troy Teno | May 2026**
**Status:** Sealed | **Script:** `workbench/experiments/G-056_SUPPRESSIVE_HEAD_CAUSAL_TEST.py`
**One-line result:** H1 confirmed — a suppressive head (V = I − α·P_FIN) is causal above
the FIN-annihilation threshold (α=1.0), producing 42% specialist-layer ratio change vs.
G-055's 4% neutral baseline; the mechanism is FIN inversion, not suppression.

---

## The Question

G-055 tested a neutral global head (V = I, Q/K uniform) and found it diagnostic: 4%
ablation change, equal to a near-identity control. S-057 then identified L3H4 in T5-small
as a *suppressive* head — partial jump specificity, but negative cosines to both I_WALK
and I_JUMP. G-055 does not answer whether a suppressive head is also diagnostic, or
whether its active subtraction of FIN content from the residual creates causal downstream
effects. Sandbox_017 required G-056 to run before S-058 is finalized.

---

## Design

**Architecture:** Same 6-layer NumPy toy as G-055 (D=16, 8 GEO + 8 FIN dims, 6 tokens,
SEED=56). Layer 2 (suppressive global head): Q/K = 0.005 × ones(D,D) (globally uniform,
same as G-055 global head). V = I − α·P_FIN (reflects FIN component; GEO unchanged).

- α=0.0: neutral global head (G-055 reference)
- α=1.0: FIN-annihilation (maps FIN component to zero)
- α=2.0: FIN-inversion (maps FIN component to its negative; primary condition)

Sweep: α ∈ {0.0, 0.5, 1.0, 1.5, 2.0}.

**Input states, morphism, ablation procedure:** Identical to G-055.
**Control:** Layer-1 near-identity ablation.
**Residual diagnostic:** Record FIN-component magnitude at bank position before and
after layer-2, in both states, to directly measure what the suppressive head removes.

---

## Results

### H1 — CONFIRMED (Suppressive head is causal at α=2.0)

Ablating the suppressive head changes specialist-layer Born filter ratio by **42.0%**
(5.06× baseline → 2.93× ablated). The threshold is >10%. H1 confirmed.

Control ablation (layer-1 near-identity): **0.1%** change. The 42% effect is not
a residual-removal artifact — it is specific to what the suppressive head was doing.

### H2 — NULL

42.0% >> 10% threshold. H2 rejected.

### H3 — NULL (No downward kink at layer 2)

Born filter ratio profile with suppressive head active:

| L0 | L1 | L2 | L3 | L4 | L5 | L6 |
|----|----|----|----|----|----|----|----|  
| 2.894 | 2.895 | 2.897 | 2.897 | 4.216 | 8.058 | 8.092 |

Layer 2 ratio (2.897) is slightly above layer 1 (2.895) — no downward kink. The
suppressive head does not visibly reduce the Born filter ratio at its own layer.
The effect appears downstream: ratio jumps from 2.9 at L3 to 4.2 at L4 and 8.1 at L5.

When ablated, the profile flattens entirely to ~2.93 at all layers — the specialist
layers produce the same modest ratio as the early layers.

### H4 — NULL (Non-monotonic alpha sweep; threshold detected)

| α | Baseline ratio | Ablated ratio | % change |
|---|---------------|---------------|----------|
| 0.0 | 2.90 | 2.93 | 1.2% |
| 0.5 | 2.91 | 2.93 | 1.0% |
| 1.0 | 2.93 | 2.93 | **0.0%** |
| 1.5 | 3.15 | 2.93 | 6.7% |
| 2.0 | 5.06 | 2.93 | **42.0%** |

The change is NOT monotonic. At α=1.0, the ablation effect is exactly 0.0%. The effect
grows only for α > 1.0. The pre-registered monotonicity hypothesis is null.

---

## Interpretation

### The threshold is the FIN-annihilation point

At α=1.0, V = I − P_FIN maps FIN-component to zero (annihilates it). The head's output
carries no FIN information. Ablating it makes no difference — the specialist layers
produce the same ratio with or without it. This is a precise null: the head has no causal
effect exactly when it removes FIN content but does not invert it.

At α < 1.0 (partial suppression), the head partially reduces FIN content. Ablating
it restores a small amount of FIN content, but not enough to change specialist-head
behaviour materially. These are the *diagnostic* suppression regimes.

At α > 1.0 (FIN inversion), the head maps FIN content to its negative. Now bank's
residual carries **anti-FIN** signal — large magnitude on FIN dimensions, but negative.
This is mechanistically different from suppression. The FIN-specialist heads have Q/K
weighted strongly on FIN dimensions; anti-FIN bank is now salient to those heads because
it has high FIN-dimension magnitude regardless of sign.

### The causal mechanism is re-routing, not signal removal

With the suppressive head active at α=2.0:

| | Specialist-head attention to fin1 | fin2 | bank |
|--|---|---|---|
| Baseline (suppressive head active) | 0.176 | 0.179 | 0.168 |
| Ablated (suppressive head removed) | 0.504 | 0.481 | 0.013 |

When the suppressive head is removed, specialist heads route strongly to fin1 and fin2
(~0.50 each) and ignore bank (<0.01) — exactly the G-055 pattern. When the suppressive
head is active, specialist heads spread attention more uniformly, including bank (~0.17).

The suppressive head at α=2.0 inverts bank's FIN content so severely that bank becomes
salient to FIN-specialist Q/K. The specialist heads re-route toward bank, reading its
inverted FIN signal. This propagates an inverted, amplified signal through layers 4–5,
producing the 8× Born filter ratio at layer 5.

**The suppressive head is causal not by removing a signal from the chain, but by
creating a new (inverted) signal that the specialist heads then amplify.**

### Residual FIN content at bank

| | Before layer 2 | After layer 2 (active) | Removed |
|--|---|---|---|
| S_success | 1.708 | 0.051 | 1.657 |
| S_fail | 1.114 | 0.306 | 0.808 |

The suppressive head removes 2× more FIN content from S_success than from S_fail
(1.657 vs 0.808). After suppression, S_fail bank has more FIN content than S_success
bank — the success/fail ordering at bank is inverted. This inversion is what the
specialist heads then amplify.

### Connection to G-054 regime structure

G-054 mapped three regimes by FIN content at bank. G-056 shows the suppressive head
can shift bank's effective FIN content from regime to regime: at α=2.0, S_success bank
goes from FIN_WEIGHT≈0.5 (no-collapse) to FIN_WEIGHT near zero (slow-collapse). The
suppressive head performs a regime-shift operation on the downstream processing, which
is why it is causal. The neutral head (α=0) does not shift regimes — it is diagnostic.

---

## Implication for the Other Track

**The relay to S-track is nuanced, not a simple "suppressive = causal."**

G-056 shows suppressive heads are causal *above* the FIN-annihilation threshold. Below
it (including the neutral case α=0 = G-055), they are diagnostic. The boundary is
the point where FIN content stops being reduced and starts being inverted — not merely reduced.

L3H4 in T5-small shows mild suppressive cosines: cos_walk_fail = −0.044,
cos_jump_fail = −0.026. These are small negative values — consistent with partial
suppression in the α < 1.0 regime, which G-056 shows is still diagnostic (0.0%–1.2%
ablation change). L3H4 has NOT been directly ablated; G-056 cannot confirm its regime
without that measurement.

**S-058 design consequences:**

1. The primary target remains L4H6/L5H2/L5H5 activation patching — this is unaffected
   by G-056 because those heads are value-substitution heads, not suppressive heads.

2. S-058 should include an L3H4 control ablation alongside the primary patch. If
   ablating L3H4 changes `P(I_JUMP) vs P(I_WALK)` at the action slot, L3H4 may be
   operating in the causal (inversion) regime. If it produces no change, it is in the
   diagnostic (suppression) regime. This is sandbox_017's requirement: L3H4 must be
   tested before it is ruled out as a repair target.

3. The G-056 mechanism predicts that if L3H4 is causal, the signature would be:
   ablating L3H4 *reduces* the model's tendency toward I_WALK (because the inverted
   signal driving specialist re-routing is removed). The causal suppressor hurts the
   model by creating false salience, not by blocking correct signal.

---

## What's Next

### State of both tracks as of this report (May 2026)

**G-track is sealed through G-056.** Both the neutral-head (G-055) and suppressive-head
(G-056) causal tests are complete. The G-track findings now cover the full range of
early-layer head types: neutral/global = diagnostic; mild suppression (α < 1.0) =
diagnostic; FIN-inversion (α > 1.0) = causal via re-routing mechanism.

**S-track is sealed through S-057. S-058 is drafted but not run.** Per sandbox_017,
S-058 was gated on G-056. The gate is now open.

---

### Carry to S-track thread (paste verbatim)

1. **G-056 sealed. H1 confirmed: suppressive heads are causal, but only above the
   FIN-annihilation threshold (α=1.0 in toy).** Below the threshold, suppressive heads
   are diagnostic — same conclusion as G-055's neutral head. The boundary is where
   FIN content stops being reduced and starts being inverted.

2. **L3H4's cosines (cos_walk_fail=−0.044, cos_jump_fail=−0.026) are small negatives,
   consistent with the diagnostic (α < 1.0) regime.** G-056 cannot confirm L3H4's
   regime without direct ablation. S-058 must include an L3H4 control ablation.

3. **S-058 primary experiment:** activation patching at L4H6/L5H2/L5H5. Unchanged.
   Primary hypothesis (sandbox_017): `P(I_JUMP) > P(I_WALK)` at action-slot divergence
   in ≥50% of patched fail examples.

4. **S-058 required control (new from G-056):** Ablate L3H4 in isolation (zero its
   head output at the action-slot divergence steps, with matched control head). If
   ablating L3H4 changes `P(I_JUMP) vs P(I_WALK)` measurably, L3H4 is in the causal
   (inversion) regime and should be added to the repair target set. If it produces
   no change, L3H4 is confirmed diagnostic.

5. **Mechanism signature to look for:** If L3H4 is causal via the G-056 inversion
   mechanism, ablating it should *reduce* I_WALK probability (removing the false
   anti-JUMP salience it creates). If it is diagnostic, ablating it changes nothing.
   Direction matters for distinguishing the two cases.

---

### What each track needs

**S-track needs G-056's verdict to finalize S-058.** That verdict is now available:
suppressive heads above the inversion threshold are causal; L3H4's mild cosines suggest
it is probably below the threshold, but L3H4 ablation is required to confirm.
**S-track can now finalize and run S-058.** All gates are open.

**G-track needs nothing from S-track.** G-track is fully caught up with S-track.
G-056 is the answer to the question S-057 raised. G-057, if any, will be motivated by
S-058's result.

---

### After S-058 seals

Relay to G-track via COORDINATION.md:
- Did activation patching at L4H6/L5H2/L5H5 flip `P(I_JUMP) > P(I_WALK)` in ≥50%
  of fail examples? If yes: mechanism confirmed and repaired. Paper-worthy.
- Did L3H4 control ablation change the result? If yes: L3H4 is in the causal regime
  and the failure chain is longer than S-051 identified. G-057 would then characterize
  the full suppressive-inversion mechanism at production scale.
- If patching succeeded and L3H4 ablation had no effect: S-051's three-head mechanism
  is confirmed as the complete repair target. Research arc is closed at the mechanism level.

The paper-worthy experiment is S-058. If activation patching at three heads restores
correct output, the workbench will have moved from mechanism identification to
mechanism repair. That is the result this program has been building toward.

---

### UPDATE — S-058 Sealed (May 2026)

S-058 ran. The result was not what the "After S-058 seals" section anticipated.

**Actual S-058 result: K2 fired.** All six arms — including Arm C (L3H0, the known-diagnostic
global head, which has no causal role in the failure) — produced 100% flip rate with
statistically indistinguishable margins (~+1.69 to +1.94). The intervention is not head-specific.
The confound is representational: patching any activation from a success donor at the
action-slot divergence step injects success-context signal broadly across the representation,
regardless of which head is patched. Donor representations at that step are globally different
from fail representations in a way that the targeted patch can only partially contain.

**What this changes for each track:**

- **Mechanism identification stands.** L4H6/L5H2/L5H5 value substitution (S-051), norm
  asymmetry (G-052/S-050b), and layer-5 amplification (G-054) are all intact. The K2 confound
  is about the repair *method*, not the mechanism *identification*.

- **L3H4 status is still unresolved.** Arm D (L3H4 neutralization) also produced 100% flip
  (delta=+7.71). Because Arm D used a zero-output intervention rather than donor substitution,
  it is mechanistically distinct from Arms A–C but shares the same delta magnitude as the
  confounded arms — making it impossible to attribute the effect to L3H4's specific causal
  role without a cleaner ablation design. L3H4 is neither confirmed diagnostic nor confirmed
  causal by S-058.

**Current state of both tracks after S-058:**

**G-track:** Sealed through G-056. G-057 candidate: W_V correction geometry at toy scale —
use the norm-amplification mechanism formula (G-052) to compute the minimum W_V rotation needed to flip
cosine alignment from I_WALK to I_JUMP at the production norm ratio (520:424). This would
formalize the theoretical correction target and directly inform S-059 design. No G-track
action required until S-059 results are available or Troy brings a new question.

**S-track:** Sealed through S-058. Next experiment (S-059 candidate): targeted W_V correction
operating within fail trajectories — modify W_V at L4H6/L5H2/L5H5 directly, without donor
transplant, to redirect value routing. The G-052 formula predicts the exact angle correction
needed. If S-059 succeeds (targeted W_V correction flips I_JUMP > I_WALK within fail
trajectories), the program will have moved from mechanism identification to mechanism repair.

**What each track needs:**
- S-track needs to design S-059 (targeted W_V correction). It can do so now; no G-track gate.
- G-track needs S-059 results to know whether a G-057 W_V geometry experiment is warranted
  at toy scale, or whether S-059's empirical W_V correction is sufficient to close the
  research arc. G-track is waiting on S-059.

---

*Methods report written at sealing time. Not revised after data.*
*UPDATE section added after S-058 sealed.*
*G-track | Applied Categorical Physics Workbench | Troy Teno | May 2026*
