# Methods Note — G-053: Layer-Peak Amplification

**Applied Categorical Physics Workbench | Troy Teno | May 2026**
*Companion to findings/FINDINGS.md F-053. 1/3 confirmed — null result with mechanistic explanation.*

> **Historical sealed note.** This note was sealed at the G-053 stage and is preserved as
> written; methods notes are not revised after sealing. Its forward-looking "Real-World
> Implications" and any language extrapolating this controlled toy model to T5-small or to
> production models — including the norm-asymmetry "parallax lever" framing and statements
> about what T5-small's geometry "should" look like — predate the paper's final G-track
> boundary. The G-track is a controlled analogy, not a production or T5-small claim; treat
> such language as sealed historical interpretation, not a current paper claim. See
> `README.md` and `RETIREMENTS_AND_METHOD_LESSONS.md` for the paper's bounded G-track scope.

---

## The Question We Came In With

G-050 found that the per-head Born filter signal for specialist attention heads peaks at
layer 5 in a 12-layer transformer, then collapses to near zero. This was surprising —
the standard expectation is that later layers produce richer, more context-sensitive
representations. Why would the signal be *strongest* at mid-layers and not the final layer?

The S-track found the same layer independently: T5-small's decoder layer 5 is where the
parallax lever (the norm asymmetry that produces the +146/−46 logit gap) becomes
decisive. Two independent experimental tracks agreed on layer 5, from different angles,
measuring different things. That convergence was worth testing directly.

G-053 asked: is the layer-5 peak a general property of multi-layer transformers with
semantic specialist heads? Does it consistently occur at roughly 40% of total depth
regardless of how many layers the model has?

The answer was no — and understanding why not is the most valuable output of the
experiment.

---

## What the Data Showed

With a different random seed (SEED=53 vs G-050's SEED=50), the H1(FIN) specialist head
ratio at every layer was:

- Layer 1: 0.58× (this is the peak)
- Layer 2: 0.15×
- Layer 6: effectively zero

Across all four architecture depths (4, 6, 8, 12 layers), the peak was always at layer 1.
The signal monotonically decays from the first layer. There is no mid-layer accumulation
phase, no peak, no amplification — just steady decay as depth increases.

The 12-layer model in G-053 is architecturally identical to G-050. The only difference is
the seed, which controls the random noise added to the cluster-constructed embeddings. And
yet the results are completely different: 0.08× at layer 1 in G-053 vs 2.58× at layer 5
in G-050. Something in the embedding geometry is decisive.

---

## Why the Seeds Produce Different Results

This requires understanding what drove G-050's layer-5 peak in the first place.

In G-050, the key mechanism was a *collapse* in State 3 (geographic-only context). State 3
contains geographic tokens (river, grass) and the "bank" token at position 0. The FIN
specialist head (Head 1) is trained to attend to financial tokens. In State 3, there are
no financial tokens — only geographic ones and "bank." So what does the FIN head do?

With G-050's embeddings (SEED=50), "bank" in the S3 sequence had a relatively weak FIN
component. The FIN head found no good financial token to attend to, fell back to attending
to "bank" itself (self-attention), and over multiple layers converged to a stable,
self-referential representation. By layer 5, State 3's FIN-head defect had collapsed to
0.09 (from 2.50 at layer 1). Meanwhile, States 1 and 2 retained substantial FIN signal
because they had actual financial tokens to attend to. The ratio (S1 defect / mean(S2, S3)
defect) exploded because the denominator's S3 term had collapsed to near zero.

With G-053's embeddings (SEED=53), "bank" retains a strong FIN embedding component —
strong enough that the FIN head does not collapse to self-attention in State 3. Instead,
the FIN head attends to "bank" with high weight, which is a real FIN-signal token. The
S3 defect does not collapse. No collapse means no ratio amplification. The ratio stays
low and decays monotonically as the signal mixes into the residual stream across layers.

The layer-5 peak is not a property of 12-layer depth. It is a signature of a specific
geometric condition: **the FIN head must collapse in the unambiguous geographic state but
not in the ambiguous or financial states.** When that condition is met, the ratio
amplifies. When it is not met — when every state provides FIN signal to the FIN head —
the ratio decays monotonically from layer 1.

---

## What This Tells Us About G-050

G-050 was a genuine finding. It was not wrong. But the explanation for *why* the peak was
at layer 5 was underspecified. The finding was: "specialist heads peak at mid-layers and
collapse by the final layer." The missing piece was: "the mid-layer peak requires that one
or more states produce specialist-head collapse."

G-053 supplies the missing piece. The two-component mechanism is:

1. At least one unambiguous state must cause the relevant specialist head to fall back to
   self-attention — "no signal to attend to in this state." This requires that the
   unambiguous state lacks the relevant semantic features.

2. The ambiguous state and the other unambiguous state must retain the relevant semantic
   signal so their defects remain large.

When both conditions are met, the ratio amplifies at the depth where the collapse becomes
complete (layer 5 in G-050). When condition 1 fails (as in G-053), there is no
amplification, and the default behavior — monotonic decay — is what you see.

---

## The Monotonic Decay Pattern Explained

Why does the signal always decay as depth increases, regardless of whether the collapse
condition is met?

Each transformer layer consists of attention followed by a feed-forward network (FFN) and
layer normalization. The residual stream carries contributions from all heads mixed
together. As depth increases:

- The FFN layers perform nonlinear transformations that mix information across the
  embedding dimensions, partially erasing the per-head structure.
- The layer normalization re-centers and rescales the hidden state, reducing the
  geometric spread that the Born filter measures.
- The random heads (H2, H3) produce large, noisy defects that dominate the residual
  stream at early layers. As those large defects decay faster than the specialist heads'
  smaller signals, the signal-to-noise ratio can momentarily improve — but only if the
  specialist heads have a collapse condition that keeps their denominator small.

Without the collapse, the signal just gets mixed in and smoothed out. The Born filter
measures how sensitive the head's output is to the swap morphism. After enough layers of
FFN mixing, every head's output looks approximately the same for similar inputs. The
sensitivity goes to zero.

---

## The Measurement Principle Revision

G-053 revises one aspect of the G-050 measurement principle. G-050 concluded: "measure
per-head at mid-layers (5-6), not full-state at final layer." This is correct *when* the
collapse condition is met. When the collapse condition is not met, layer 1 is the peak.

The revised principle: the correct measurement layer is *wherever the collapse-driven ratio
peak occurs*, not a fixed depth fraction. To find that layer, you need to first verify
that the geometric conditions for specialist-head collapse are present in at least one
unambiguous state. If they are not, the Born filter signal is maximal at layer 1 and the
per-layer profile is diagnostic (monotonic decay = no collapse; mid-layer peak = collapse
in at least one state).

This has an implication for applied interpretability. If you are trying to use the Born
filter to find the "best" layer to probe a trained model, you cannot assume mid-layers are
best. You need to profile the per-layer defect and look for either: (a) a mid-layer peak,
which indicates the model has learned collapse-inducing state distinctions, or (b)
monotonic decay from layer 1, which means the model lacks that structure and layer 1 is
your best bet.

---

## Real-World Implications

**The collapse condition is a trainability marker.** A model whose Born filter ratio peaks
at mid-layers has learned to sharply distinguish between contexts that do and do not
activate a given feature dimension. The collapse (one state activates the head, others
don't) is evidence of clean semantic specialization. A model whose ratio decays monotonically
has not learned this distinction — all states look similar to the specialist head. The
monotonic vs. peak profile is a one-pass diagnostic for whether the model has learned
state-selective head activation.

**Norm temperature and the collapse condition interact.** G-052 showed that the parallax
lever is driven by norm asymmetry. G-053 shows that the amplification layer (where the
lever is decisive) depends on when the collapse occurs. If you equalize output embedding
norms (the S-track intervention), you eliminate the parallax lever but not the underlying
attentional collapse. If you additionally correct the collapse condition (making the FIN
head attend correctly in all states), you address the representational source of the
failure, not just its amplification. The two interventions target different parts of the
causal chain.

**Layer profiling before probing.** Any interpretability study that commits to a fixed layer
for extracting representations (e.g., "we use the final hidden state" or "we use layer 6
of 12") is implicitly assuming a specific failure mode or success condition. G-053 shows
that the correct layer is context-dependent and can only be found by profiling the per-head
defect across all layers first. A single fixed layer will miss the signal when the peak
is elsewhere.

---

## What Comes Next: G-054

The G-053 null result points directly to the next experiment. We now know exactly what
geometric condition produces the mid-layer peak: the bank token's FIN component must be
weak enough in the geographic state that the FIN head collapses to self-attention.

G-054 will vary this condition systematically. We will build a range of bank tokens with
FIN embedding components ranging from 0.0 (no FIN signal at all — guaranteed collapse)
to 3.0 (full FIN signal — no collapse). For each bank FIN component strength, we will
profile the per-layer H1(FIN) ratio and measure:
- The layer at which the peak occurs (if any)
- The peak ratio
- The depth of the S3 FIN-head defect at that layer

This maps the phase transition between "monotonic decay" (no collapse, G-053 regime) and
"mid-layer peak" (collapse, G-050 regime). Finding the transition point tells us exactly
how much FIN embedding ambiguity in the geographic state is required to produce the
layer-5 amplification — and suggests, by analogy in the controlled model, what embedding
geometry in T5-small might look like if its layer-5 amplification is driven by the same
mechanism.

---

*Methods notes are written at sealing time and not revised.*
*G-track | Applied Categorical Physics Workbench | Troy Teno | 2026*
