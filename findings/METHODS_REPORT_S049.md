# Methods Report — S-049: Decoder Cross-Attention Analysis

**Applied Categorical Physics Workbench | Troy Teno | May 2026**
*Status: Sealed. One-line result: K1 fires — decoder routing to the "around" encoder position is statistically identical in fail and success cases (p=0.92); the failure is in value projection, not attention routing; 26% of has_around failures are substituted_walk — correct structure, wrong primitive.*

---

## The Question

S-048 confirmed that the failure is in the decoder, not the encoder. But "decoder failure" covers a wide space. Two major possibilities:

1. **Routing failure:** The decoder's cross-attention does not attend correctly to the "around" encoder position, so it does not know to apply the four-cycle identity morphism. If this is the failure, fixing the attention routing should fix the output.

2. **Value failure:** The decoder routes correctly to "around" but gets the wrong value back — either because the OOD jump representation confounds what the decoder reads, or because the decoder's value projection maps the "around" signal to the wrong output token.

S-049 tests which of these is the case by capturing the decoder's cross-attention weights during generation and measuring where the decoder is looking when it generates the wrong token.

A secondary question motivated by prior intuition: does T5 miscount the number of repetitions? "Around" should generate 4 cycles; "opposite" generates 2. If T5 interprets "around" as 180° (2 cycles) rather than 360° (4 cycles), we would see the wrong cycle count even in failure cases. S-049 tests this directly.

---

## Design

**Model:** S-043 checkpoint. Full generation with attention capture: `model.generate(output_attentions=True)`.

**Groups:** 
- Group A: has_around=1 + fail (N=50)
- Group B: has_around=1 + success (N=25)
- Group C: has_around=0 + success (N=50, control)

**Measurement:** Per-step cross-attention weights at each decoder layer. "Around" encoder position identified by character-offset mapping from input text to tokenization. Action-slot steps identified by pattern matching.

**Phase 3b:** Failure mode analysis of 50 Group A examples. For each, the output sequence was classified into failure categories: substituted_walk, wrong_structure (confound — "around" on non-jump primitive), no_jump, half_count_2of4, three_quarter_3of4, short_count, over_count, other.

---

## Results

**H1 (lower attention to "around" in fail vs success): FAIL — K1 fires.**
```
A (fail): mean attn to "around" = 0.10582
B (success): mean attn to "around" = 0.10258
Mann-Whitney p = 0.9208
```
The decoder routes through "around" equally well in fail and success cases. K1 fires: the failure is not in routing.

**H2 (periodic attention rhythm for "around" cycle in success): FAIL.** A=+0.047, B=−0.025, p=0.98. No periodic cross-attention rhythm distinguishing fail from success.

**H3 (higher entropy in fail): CONFIRMED.** Cross-attention entropy is higher for Group A (fail) than Group C (control): A=1.4338 vs C=1.3571, p=0.0017. Failing cases show more diffuse decoder attention.

**H4 (attention divergence peaks at decoder layer 4): CONFIRMED.** Maximum attention divergence (A vs B) at layer 4. Layers 1–3: small differences; layer 4: peak (Δ=−0.017); layers 5–6: diminish.

**180° hypothesis (half_count_2of4): ZERO cases.** T5 never generates exactly 2 cycles for "around" commands. The null is clean.

**Substituted_walk failure mode:** 13/50 (26%) of has_around=1 fail examples are substituted_walk — T5 generates (I_TURN_LEFT I_WALK) × 4 when the correct output is (I_TURN_LEFT I_JUMP) × 4. Correct structure, correct cycle count, wrong primitive.

**Full failure mode distribution (N=50 Group A):**
```
wrong_structure (confound)    14  (28.0%)
substituted_walk              13  (26.0%)
no_jump                        8  (16.0%)
three_quarter_3of4             3   (6.0%)
short_count (various)         ~8  (16.0%)
over_count / other            ~4   (8.0%)
half_count_2of4                0   (0.0%)
```

---

## Interpretation

The identity morphism insight: every "X around Y" command is the identity transformation in SCAN's position × orientation space. "Jump around left" = (I_TURN_LEFT I_JUMP) × 4. After 4 quarter-turns, you return to the starting position facing the same direction. The substituted_walk output (I_TURN_LEFT I_WALK) × 4 is also identity — T5 generated a physically correct identity sequence. It is wrong only by the SCAN evaluator's exact-token-match criterion, not by environment semantics.

T5 learned "around = identity, implemented as 4× turn-action cycles." When the OOD primitive appears inside that identity structure, the decoder correctly composes the cycle count but cannot fill the action slot with the OOD token. It defaults to the trained action-slot value: I_WALK. This is value substitution within a correctly-composed structure.

The 180° null is clean and important. T5 counts correctly. It substitutes, it does not miscount. This rules out a whole class of explanations involving cycle-count representation.

---

## Implication for the Other Track

S-049's substituted_walk finding defines the specific target for the geometry track: the failure is not a routing failure (K1), not a counting failure (0% half_count_2of4), and not a structural misunderstanding (the cycle structure is correct). The failure is at the specific token-generation step where the action token inside the cycle is selected — and the wrong token is selected despite correct attention routing.

The G-track's value projection analysis (W_V) is the natural next target: what property of the value weight matrix causes the decoder, when attending to the correct encoder position, to return a value that points toward the wrong output token? This is the question S-051 answers at the head level; the geometry track should answer it at the weight-geometry level.

---

*Applied Categorical Physics Workbench | Troy Teno | May 2026 | All work open access*
*Numbers match FINDINGS.md F-049. Do not revise post-hoc.*
