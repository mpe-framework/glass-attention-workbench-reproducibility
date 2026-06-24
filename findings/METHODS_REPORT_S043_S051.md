# Methods Report: SCAN Track S-043 through S-051
**Applied Categorical Physics Workbench**
Troy Teno | May 2026 | Open Access

*Written for handoff to the geometry track. Plain language throughout.
Precise references to sealed findings and formulations are maintained in the private development findings log (not part of this public export).*

---

## What We Were Trying to Understand

T5-small, a 60-million parameter language model, was fine-tuned to translate English
commands into action sequences. The training data deliberately excluded one primitive:
"jump" in compound contexts. "Jump left" was trained. "Jump around left" was not.
This is the SCAN add_prim_jump benchmark — a clean, guaranteed compositional failure.

The question driving the S-track: *why exactly does T5 fail?* Not in the abstract
sense that the softmax attention architecture predicts ([Sargsyan attribution removed in public export — see RETIREMENTS_AND_METHOD_LESSONS.md] — compositional generalization failure is a structural consequence),
but in the concrete mechanistic sense. Which part of the model fails? What does it
do instead? Why does it do that specific wrong thing rather than some other wrong thing?

The experiments ran from S-043 through S-051, each one narrowing the answer.

---

## The System

**Model:** T5-small. Six encoder layers, six decoder layers, eight attention heads per
layer, 512-dimensional hidden states, 64-dimensional key/value projections per head.
Trained on the English C4 corpus, then fine-tuned on SCAN.

**Task:** Translate commands like "jump around left thrice after walk right twice" into
action sequences like "I_TURN_LEFT I_JUMP I_TURN_LEFT I_JUMP I_TURN_LEFT I_JUMP
I_TURN_LEFT I_JUMP I_TURN_LEFT I_JUMP I_TURN_LEFT I_JUMP I_TURN_LEFT I_JUMP
I_TURN_LEFT I_JUMP I_WALK I_TURN_RIGHT I_WALK I_TURN_RIGHT I_WALK."

**The failure rate:** 56.8% of compound jump test examples fail. The SCAN interpreter
(a simple rule-based system using the same grammar) gets 100%.

**Measurement layer:** T5's final encoder layer (layer 6) collapses all signals to
near-zero — it was discovered early that every measurement had to be taken at encoder
layer 5, the penultimate layer. This turns out to be a decoder property too, as S-051
confirmed.

---

## The Experimental Chain

### S-043 — Does the encoder know "jump" is different?

**What we did:** Ran T5's encoder on commands containing "jump" vs commands containing
"walk", "run", "look" in otherwise identical structures. Measured how differently the
encoder represents these tokens using BC (Beck-Chevalley) defect — a measure of how
much the model's internal representation fails to commute with a structural substitution.

**What we found:** Yes, categorically. The encoder consistently represents "jump" as
different from trained primitives at layer 5. The signal is strong and statistically
certain (p=0.0000). The encoder knows "jump" is OOD (out-of-distribution).

**Why it matters:** The encoder has the information. The failure isn't here.

---

### S-044 and S-045 — Does the encoder's OOD signal grade difficulty?

**What we did:** Within the failing examples (jump compounds that T5 gets wrong), asked
whether the BC defect was higher for "harder" examples. Tried both mean-pool (averaging
the signal across the whole command) and per-token (measuring at the jump token's
position specifically).

**What we found:** No. The encoder marks all jump-compound examples with elevated BC
defect uniformly — it doesn't grade difficulty within the failing category. The examples
T5 gets right don't have better encoder representations than the ones it gets wrong.
This is K1 firing for the first time: the encoder does its job uniformly, the decoder
determines the outcome.

A useful side finding: mean-pool BC defect was correlated with how many jump tokens
were in the command (more jumps = higher signal) rather than with actual difficulty.
Per-token measurement removed that confound and confirmed the same null result.

---

### S-046 — Does the encoder contain the grammar bits needed to succeed?

**What we did:** Extracted eight structural yes/no features from each command (does it
contain "around"? "opposite"? "twice"? etc.) and trained linear probes on the encoder's
frozen representations to predict these features.

**What we found:** All eight features are linearly readable from the encoder at ≥99.5%
accuracy. A fixed rule decoder (the SCAN interpreter) using these probe-predicted bits
achieves 100% correct output. The encoder contains everything needed.

**Why it matters:** The failure is definitively in the decoder, not because information
is missing.

---

### S-047 — Which structural feature drives failure most?

**What we did:** Measured, for each of the eight structural bits, how strongly having
that feature present correlates with T5 failing.

**What we found:** "has_around" dominates. Commands containing "around" fail 87.3% of
the time. Commands without "around" fail only 22.4%. No other feature comes close.

"Around" means the command requires four repetitions of a turn-action cycle —
"jump around left" = (turn left, then jump) repeated four times. The insight here
was categorical: every "X around Y" command is the identity transformation in position
and orientation space. You turn four quarter-turns and end up exactly where you started,
facing the same direction. T5 learned this structure correctly. What it can't do is
fill the "X" slot with the unseen primitive.

---

### S-048 — Can we fix the encoder to fix the decoder?

**What we did:** Re-fine-tuned T5 with an additional training signal forcing the encoder
to represent "has_around" more cleanly. Tried three different strengths of this signal.

**What we found:** Accuracy got worse at every strength tested (43.7% → 33.7% → 15.0%
→ 10.7%). Making the encoder better at encoding "around" made the decoder worse at
decoding it. K1 fires again: the bottleneck is in the decoder.

This experiment also confirmed the probe AUC went up as accuracy went down — the encoder
was representing "around" ever more cleanly, and the decoder was ever less able to use
it. More signal in the encoder does not help when the decoder cannot consume it.

---

### S-049 — Is the decoder's failure in how it routes attention, or in what it does with what it reads?

**What we did:** Ran T5's generation with full attention capture. For the failing examples,
found where the decoder was attending when it generated the wrong token. Specifically: at
the action slots inside the "around" cycle, what was the decoder looking at in the encoder?

**What we found:** The decoder routes correctly. Its cross-attention to the "around"
encoder position is statistically identical in failing and succeeding cases (p=0.92).
K1 fires again. The failure is not in routing — it's in what the decoder does with what
it reads there.

**The failure mode:** 26% of failing has_around cases are "substituted_walk" — T5
generates (I_TURN_LEFT I_WALK)×4 for "jump around left." Correct structure, correct
number of cycles, wrong action token inside each cycle. T5 has perfectly learned "around
= four-cycle identity morphism." It just fills the slot with the trained default (I_WALK)
instead of the OOD primitive (I_JUMP).

**The 180-degree null:** The hypothesis that T5 might interpret "around" as 180° (two
cycles) rather than 360° (four cycles) was tested. Zero examples showed this pattern.
The null result is clean and informative — T5 counts correctly. It substitutes, not
miscounts.

---

### S-050b — Among I_WALK, I_RUN, and I_LOOK (equal training frequency), why I_WALK specifically?

**What we did:** Measured, at the exact subword step where the decoder chooses which
action token to generate, three things: the probability it assigns to each action token,
the Euclidean distance from the decoder's hidden state to each action token's embedding,
and the cosine similarity (directional alignment) between the hidden state and each
embedding.

**What we found:** I_WALK is chosen for two compounding reasons.

First, I_JUMP is suppressed by training frequency: it appeared 17× less often than
I_WALK/I_RUN/I_LOOK during training (1,467 vs 25,350 occurrences). The softmax
reflects this — P(I_JUMP) ≈ 0.002 at fail action slots.

Second, among the equal-frequency tokens, I_WALK wins through the output embedding
geometry. The dot product formula is: logit = (cosine of angle between hidden state and
token embedding) × (length of hidden state) × (length of token embedding). I_WALK's
divergence-point embedding has the largest length (norm=520) of any action token, and
the decoder's hidden state also points slightly toward I_WALK's direction (cosine=+0.024,
the only positive cosine among the four action tokens in fail cases). Both factors
multiply together into logit=+146 for I_WALK vs logit=−46 for I_RUN, producing
P(I_WALK)=0.91.

The norm-amplification mechanism: the embedding length acts as a lever arm that
amplifies the tiny directional preference into near-certain selection.

---

### S-051 — Causal confirmation: is the norm really the mechanism? Where is it built? Which heads build it?

**What we did:** Three interventions in one pass.

**Normalization (Option A):** Re-ran the probability calculation using token embeddings
scaled to unit length (norm removed). Measured what happened to P(I_WALK).

**Layer tracking (Option B):** At each of the six decoder layers, measured how much the
hidden state was directionally aligned (cosine) with each action token's embedding.

**Head attribution (Option C):** For each of the 48 cross-attention heads (6 layers × 8
heads), computed how much that head's output contribution was pulling the hidden state
toward I_WALK vs I_JUMP.

**What we found:**

*Option A — the norm is causally load-bearing.* P(I_WALK) dropped from 0.91 to 0.32
when embedding lengths were equalized. Without the lever arm, the distribution spread
out to near-uniform (~0.25 each token). I_WALK still edged ahead (the directional bias
is real), but the near-certainty was gone. Crucially: P(I_JUMP) in *success* cases also
dropped from 0.81 to 0.28 under the same normalization. The same lever arm amplifies
whichever token the hidden state is pointing toward — in fail cases that's I_WALK, in
success cases that's I_JUMP. Both groups are equally norm-dominated.

*Option B — layer 5 is the fork.* At the embedding input (layer 0), the hidden state
has essentially no alignment with I_WALK (cosine = +0.001). The alignment grows through
layers 1–4 and peaks dramatically at layer 5: cosine = +0.231 in fail cases. In success
cases, I_JUMP follows the same pattern and peaks at layer 5 at +0.201. Then layer 6
collapses both — I_WALK drops from +0.231 to +0.024, I_JUMP from +0.201 to +0.008.
These collapsed residuals are what the embedding norm then amplifies. Layer 5 is where
the model commits. Layer 6 compresses the commitment. The norm converts the compressed
residual into a probability.

This matches the encoder: encoder layer 5 was the measurement layer throughout the
entire S-track. The decoder shares the same architectural property — layer 5 carries
the signal, layer 6 squashes it.

*Option C — the jump-attending heads are the mechanism.* The heads that contribute most
to the I_WALK directional bias (L4H6, L5H2, L5H5) attend strongly to the JUMP encoder
position — attention weights of 0.5–0.7 on "jump" — not to "around." These heads are
reading the word "jump" in the input correctly. But the VALUE they get back from reading
it points toward I_WALK rather than I_JUMP.

This is the value substitution found in S-049 made concrete and anatomized. The encoder
has marked "jump" as OOD. The decoder reads the OOD jump value through its value
projection. That projection maps the OOD jump representation onto the I_WALK direction
in the output space — not because the model is confused about what "jump" means
syntactically, but because the OOD value of "jump" was never seen in the "around"
context during training, so its value projection has not learned to point toward I_JUMP.

In success cases, the same heads shift: I_JUMP contribution increases, I_WALK
contribution decreases. The head is not broken — its output direction varies with the
encoder's representation of the jump token, and in the small fraction of cases where
T5 succeeds, something about that context produces a slightly better-aligned jump value.

There is one "around" head (L3H6) that shows the expected fail/success flip — its
output points more toward I_WALK in fail cases and more toward I_JUMP in success cases.
But it is not the dominant contributor. The dominant mechanism runs through the jump
encoder token, not the around encoder token.

---

## The Complete Mechanism

Every piece is now in place. Here is the full chain in plain language:

1. **The encoder marks "jump" as OOD.** T5's encoder layer 5 represents "jump" in
   compound contexts differently from "walk", "run", "look". The marking is categorical
   — every jump-compound example gets the same elevated OOD signal, regardless of whether
   T5 will fail or succeed on it. (S-043)

2. **The encoder contains the grammar perfectly.** All eight structural features of any
   SCAN command are linearly recoverable from the encoder at ≥99.5% accuracy. The
   information needed to succeed is there. The decoder's failure is not caused by missing
   information. (S-046)

3. **The dominant failure trigger is the "around" structure.** "Around" requires four
   repetitions of a turn-action cycle. Among all structural features, "has_around"
   predicts failure with 87% accuracy. Improving the encoder's representation of "around"
   makes things worse. The bottleneck is in the decoder. (S-047, S-048)

4. **The decoder routes correctly but substitutes the wrong value.** When T5 generates
   the action token inside an "around" cycle, its cross-attention correctly attends to the
   "around" encoder position. The failure is not in where it looks — it's in what it gets
   back. (S-049)

5. **What it gets back is I_WALK.** The dominant failure mode (26% of has_around cases)
   is substituted_walk: T5 generates (I_TURN I_WALK)×4 — the correct identity-morphism
   structure with the wrong primitive inside it. (S-049)

6. **I_WALK wins for two compounding reasons.** Training frequency suppresses I_JUMP
   (17× less common). Among equal-frequency tokens, I_WALK's embedding has the largest
   length (norm) and the decoder's hidden state at the action slot is the only one with
   positive cosine alignment toward I_WALK. Both multiply together into a near-certain
   probability. (S-050b)

7. **The norm is causally load-bearing; layer 5 builds the directional bias; the
   jump-attending heads are the site.** Removing the norm lever collapses P(I_WALK) from
   0.91 to 0.32. The directional bias builds from essentially zero at the decoder input
   and peaks at layer 5. The cross-attention heads (L4H6, L5H2, L5H5) that attend most
   strongly to the jump encoder position produce outputs that point toward I_WALK — they
   are reading the OOD jump value and getting the wrong direction back. (S-051)

---

## What This Means

**For the structural account:** The failure is confirmed as categorical and architectural,
not a capacity limitation. The SCAN interpreter solves every case with a fixed rule.
T5 at 60M parameters and 100% training accuracy on seen forms still fails 56.8% of
compound jump test cases. [Sargsyan attribution removed in public export — see RETIREMENTS_AND_METHOD_LESSONS.md], and compositional
generalization failure is the structural consequence.

**For the geometry track (G-track):** The complete mechanism gives the G-track three
concrete targets:

*Target 1 — The norm asymmetry.* At toy scale: design a vocabulary where two tokens
have equal training frequency but one has a larger embedding norm from pretraining.
Confirm that the large-norm token is preferentially selected even when the model's
hidden state is not directionally closer to it. This validates the norm-amplification
mechanism at controllable scale.

*Target 2 — Layer 5 as the amplification layer.* At toy scale: which architectural
property of T5 makes layer 5 the amplification layer? The Born filter applied per-head
at layer 5 of the decoder should show maximum signal — mirroring G-050's finding that
per-head at the peak layer reveals specialist heads that mean-pool across layers misses.

*Target 3 — The jump-attending head as the failure site.* The heads L4H6, L5H2, L5H5
attend to the OOD token and produce I_WALK-aligned outputs. At toy scale: design a
head that reads an OOD token value and measure whether its output aligns with the
"trained default" direction vs the "correct OOD" direction. The Born filter applied to
this head's OUTPUT CONTRIBUTION (not the full hidden state) should show the largest
defect signal in the workbench so far. This is the per-head Born filter design from
G-050 applied to the decoder cross-attention rather than the encoder self-attention.

The S-track has run out of mechanistic depth to drill into with the current approach.
The production-scale mechanism is fully characterized. The G-track now has specific
heads, specific layers, and specific embedding properties to replicate and theorize
at toy scale.

---

## Handoff Note to G-Track

The S-track found the OOD failure mechanism at head-level granularity. The geometry
track's job is to explain WHY those heads produce those outputs — which is a question
about the weight geometry of W_V and W_O, and about how pretraining shapes the
embedding norms and decoder column directions.

The key question for G-051 onward: in a model where you control the weights, can you
reproduce the L4H6 pattern — a head that attends to an OOD token and produces output
aligned with the trained default rather than the correct OOD output? If yes, what
property of W_V or W_O causes it? That property, once identified in the toy model,
can be looked for directly in T5's weights — making the mechanism not just described
but geometrically explained.

---

*Applied Categorical Physics Workbench | Troy Teno | May 2026 | All work open access*
*Locked formulations and track-status coordination are maintained in the private development repository (not part of this public export).*
