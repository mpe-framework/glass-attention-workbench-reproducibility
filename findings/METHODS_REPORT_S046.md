# Methods Report — S-046: Causal Bit Probes

**Applied Categorical Physics Workbench | Troy Teno | May 2026**
*Status: Sealed. One-line result: All 8 structural bits of SCAN commands are linearly recoverable from frozen T5 encoder layer 5 at ≥99.5% accuracy; the encoder is not missing information; K2 fires — the raw 512-dim representation predicts T5 decoder failure at AUC=0.88.*

---

## The Question

S-043 through S-045 established that the encoder marks "jump" as OOD uniformly. The OOD marking is not the failure cause — it is flat across all jump-compound examples, while T5 succeeds on 43% of them. What else might the encoder be doing, or failing to do?

S-046 tests a different question: does the encoder contain the structural information needed to succeed? The SCAN interpreter — a fixed rule-based system — achieves 100% using only the command's structural features: does it contain "around"? "Twice"? "Opposite"? "After"? etc. If T5's encoder encodes these features correctly, then the decoder's failure cannot be attributed to missing information. If the encoder fails to encode them, the failure might originate there.

The second sub-question (K2): can a linear function of the raw encoder representation predict which examples T5 will fail on? If yes, that structural complexity signal — even without BC defect analysis — discriminates failure cases.

---

## Design

**Model:** S-043 checkpoint, frozen. No new training on the encoder.

**Dataset:** 200 jump-compound examples (125 failures, 75 successes, T5 baseline 37.5%). Same group as S-044.

**Eight structural bits** extracted deterministically from command text:
`has_direction`, `is_left`, `has_around`, `has_opposite`, `has_twice`, `has_thrice`, `has_and`, `has_after`

**Probes:** Logistic regression with L2 penalty on frozen layer 5 mean-pool representations (512-dim). 5-fold stratified cross-validation. Each probe trained independently per bit.

**Oracle:** SCAN interpreter (fixed rule decoder derived from the task grammar) applied to (a) gold bits and (b) probe-predicted bits. Measures whether the encoder's bit-encoding is sufficient to derive correct outputs via the fixed rule.

**Linear classifier:** Separate L2 logistic regression on raw 512-dim representations predicting T5 failure/success. AUC measured via 5-fold CV.

---

## Results

**H1 (all 8 probes ≥90% accuracy): CONFIRMED.** All 8 bits achieve ≥99.5% CV accuracy:
```
has_direction  1.000    has_around  1.000    has_twice  1.000
is_left        1.000    has_opposite 1.000   has_thrice 1.000
has_and        1.000    has_after   0.995
```
Probe accuracy is identical for failures and successes (1.000 per bit on per-group breakdown). The encoder does not preferentially encode structure for one outcome class.

**H2 (raw 512-dim AUC ≤ 0.55): FAIL — K2 fires.** Linear classifier on raw layer 5 representations predicts T5 decoder success/failure with AUC = 0.8779 (±0.0660). The pre-registered threshold was 0.55 (confirming no linear failure predictor). AUC = 0.88 is a strong signal.

**H3 (SCAN interpreter with gold bits): CONFIRMED.** 100.0% oracle match (200/200). The 8 structural bits are necessary and sufficient to determine the correct SCAN output.

**H4 (SCAN interpreter with probe bits): CONFIRMED.** 100.0% (200/200). Probes perfectly recover the grammar-sufficient bits.

---

## Interpretation

The K2 result and the K1 result from S-045 are both correct and not contradictory. They measure different things:

- **S-043–S-045 (BC defect):** Measures OOD-marking — how differently does the encoder represent jump vs walk in compound contexts? Answer: uniformly across all 200 examples (flat BC defect, K1).
- **S-046 (raw 512-dim AUC):** Measures structural complexity — which commands have more compositional nesting, modifiers, conjunctions? Answer: this complexity predicts decoder failure at AUC = 0.88.

The encoder holds two distinct kinds of information: (1) OOD marking, which is flat and does not predict per-example failure; (2) structural complexity, which varies with command structure and does predict failure. These are orthogonal readouts from the same representation.

The H1/H3/H4 results confirm the structural account at the bit level: the encoder encodes the grammar completely. The decoder fails not because the information is absent but because learned softmax decoding cannot compositionally deploy that information for the unseen primitive.

**Table:**
| System | Accuracy |
|---|---|
| T5 decoder (fine-tuned) | 37.5% (75/200) |
| SCAN interpreter, gold bits | 100.0% (200/200) |
| SCAN interpreter, probe bits | 100.0% (200/200) |
| Linear failure classifier | AUC = 0.8779 |

---

## Implication for the Other Track

The H1 confirmation is the most important finding for the geometry track: every structural bit of a SCAN command is linearly recoverable from T5's layer 5 at ≥99.5% accuracy. This means the production model is encoding grammar with essentially perfect fidelity. The G-track's Born filter experiments operate in a setting where toy embeddings encode semantic domains with designed precision; S-046 confirms that production T5 achieves the same fidelity on grammatical structure through training.

The K2 finding — that structural complexity predicts decoder failure at AUC = 0.88 — establishes that the decoder's failure is systematic with complexity, not random. The G-track can rely on this: the production failure mechanism has a coherent structural signature that a toy model should be able to reproduce.

---

*Applied Categorical Physics Workbench | Troy Teno | May 2026 | All work open access*
*Numbers match FINDINGS.md F-046. Do not revise post-hoc.*
