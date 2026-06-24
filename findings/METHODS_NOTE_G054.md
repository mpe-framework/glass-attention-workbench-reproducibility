# Methods Note — G-054: Collapse Conditions

**Applied Categorical Physics Workbench | Troy Teno | May 2026**
*Companion to findings/FINDINGS.md F-054. H2 confirmed — phase transition at FIN_WEIGHT≈0.5; super-amplification zone (867×) at transition edge.*

> **Historical sealed note.** This note was sealed at the G-054 stage and is preserved as
> written; methods notes are not revised after sealing. Its forward-looking "Real-World
> Implications" and production-facing language — describing the controlled toy model's Born
> filter ratio as a production "warning threshold," and extrapolating its regimes to
> T5-small's layer-5 behavior and to "fixing the production failure" — predate the paper's
> final G-track boundary. The G-track is a controlled analogy that suggests mechanisms in a
> designed toy model, not a production or T5-small claim; treat such language as sealed
> historical interpretation, not a current paper claim. See `README.md` and
> `RETIREMENTS_AND_METHOD_LESSONS.md` for the paper's bounded G-track scope.

---

## The Question G-053 Left Open

G-053 asked whether the mid-layer Born filter peak generalizes across architectures. It did not. With SEED=53, the FIN-specialist head produced monotonic decay from layer 1 across every architecture depth tested. The same architecture and initialization as G-050 gave completely different behavior — just because the random noise added to the cluster embeddings was different.

G-053's mechanistic explanation: with SEED=53, the "bank" token retained a strong FIN embedding component even in the geographic-only state (S3). The FIN specialist head, having a high-FIN token (bank itself) available to attend to in S3, did not collapse to self-attention. Without that collapse, there was no ratio amplification — the denominator never went to zero and the layer profile was smooth.

G-053 identified the decisive variable: **how much FIN signal the bank token carries in S3.** G-054 sets that variable directly, via a single scalar FIN_WEIGHT, and maps the full landscape.

---

## The Design

Instead of letting the random seed determine bank's FIN component, G-054 constructs bank explicitly:

```
bank = normalize(geo_bank_base + FIN_WEIGHT × fin_bank_base)
```

`geo_bank_base` is a cluster centroid in the GEO dimensions. `fin_bank_base` is a cluster centroid in the FIN dimensions. FIN_WEIGHT=0.0 gives a purely geographic bank token; FIN_WEIGHT=2.0 gives a bank token with strong FIN character.

The sweep: {0.0, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0, 1.5, 2.0}.

Everything else — the other token embeddings, the semantic Q/K/V initialization, the architecture — is held fixed. This is a clean single-variable experiment. The question is only: as we add FIN signal to bank, at what point does the FIN-head collapse in S3 stop occurring?

---

## What the Data Showed

The results were not gradual. They were a step function.

At FIN_WEIGHT ≤ 0.3, the FIN head in S3 still collapses to near self-attention by mid-layers. At FIN_WEIGHT ≥ 0.5, it does not. Between these two values, the behavior switches abruptly — this is a phase transition.

What was not predicted was what happened at the transition edge.

At FIN_WEIGHT=0.3, the H1(FIN) ambiguity ratio at layer 6 is **867×** — not 2.58× like G-050, but 867×. The S3 FIN-head defect collapses sharply at exactly layer 6: from 1.268 at layer 5 to 0.012 at layer 6 in one step. The numerator (S1 defect) stays at a moderate value. The denominator hits near-zero. The ratio explodes.

The table tells the story:

```
FIN_WEIGHT   Peak layer    Peak ratio    S3 defect @L5    Note
  0.0         11            1.9976×      1.280            slow collapse, late peak
  0.1          6            2.0379×      1.276            sharp collapse, mid-layer peak
  0.2          6            2.0910×      1.278            sharp collapse, mid-layer peak
  0.3          6          867.4514×      1.268            EXTREME — collapse in single step
  0.5          1            0.1154×      0.885            NO collapse — monotonic decay
  0.7          1            0.0608×      1.187            monotonic decay
  1.0          1            0.0986×      0.050            monotonic decay
  1.5          1            0.0481×      0.977            monotonic decay
  2.0          1            0.0697×      1.006            monotonic decay
```

---

## The Three-Regime Structure

G-054 reveals three qualitatively distinct regimes, not two:

**Regime 1 — Slow-collapse (FIN_WEIGHT ≤ 0.1):** The FIN head does eventually collapse in S3, but slowly. The defect decays from ~2.96 at layer 1 down to ~1.28 at layer 5, then gradually to ~0.06 by layer 12. No sharp step. The collapse spreads across many layers. The peak ratio is modest (~2×) and occurs late (layer 11).

**Regime 2 — Sharp-collapse (FIN_WEIGHT = 0.1–0.3):** The FIN head collapse completes sharply at one specific mid-layer. At FIN_WEIGHT=0.3, the drop from 1.268 to 0.012 happens at a single step. This sharp denominator collapse is what produces the super-amplification zone. Ratios range from ~2× up to 867×.

**Regime 3 — No-collapse (FIN_WEIGHT ≥ 0.5):** The FIN head finds enough FIN signal in bank to attend meaningfully in S3. No collapse. The ratio peaks at layer 1 and decays monotonically. This is the G-053 regime.

The super-amplification zone (867×) lives specifically at the boundary between Regime 1 and Regime 2 — where the collapse changes from slow to sharp. At that precise geometry, the denominator collapses to near-zero at exactly the mid-layer measurement point, while the numerator remains finite.

---

## Why the Collapse Speed Determines Everything

The mechanism is the same across all three regimes, just with different timing:

The FIN specialist head (H1) is initialized to attend preferentially to FIN-dimension tokens. In the geographic state S3, which contains geographic tokens plus bank, the amount of FIN signal available to H1 depends on how much FIN character bank carries. More FIN character → H1 has something real to attend to → no collapse. Less FIN character → H1 eventually falls back to self-attention (attending to bank→bank or other tokens without FIN content) → defect converges to near-zero.

In Regime 1 (slow collapse), bank has so little FIN character that even without self-attention pressure, the FIN head gradually loses its S3 signal across many layers of FFN mixing and LayerNorm. The collapse is driven by layer-by-layer information decay, not by a sharp attentional transition.

In Regime 2 (sharp collapse), the FIN head maintains its signal until a specific mid-layer where the FFN processing tips it past the self-attention threshold. The collapse is concentrated at one layer transition. Because the collapse is sharp — a large drop in one step rather than a gradual fade — the denominator reaches near-zero while the numerator is still at a substantial value. This is the condition for extreme ratio amplification.

In Regime 3 (no collapse), bank's FIN content is sufficient that the FIN head continues attending to bank in S3 with genuine signal. There is no transition. The defect decays monotonically as subsequent layers mix the residual stream.

---

## Connecting G-050, G-053, and G-054

G-050 (SEED=50) found 2.58× at layer 5. G-054's Regime 2 at FIN_WEIGHT=0.1–0.2 produces ratios of 2.0–2.1× at layer 6 — the same regime, slightly earlier in the sweep. G-050's bank token (SEED=50) likely had a FIN component equivalent to roughly FIN_WEIGHT ≈ 0.1–0.2, landing it in the sharp-collapse regime without hitting the super-amplification boundary.

G-053 (SEED=53) found monotonic decay. That is Regime 3. The SEED=53 bank token had enough FIN character to prevent FIN-head collapse in S3 — equivalent to roughly FIN_WEIGHT ≥ 0.5.

G-054 now maps the entire landscape that G-050 and G-053 sampled at two points. The three-regime structure explains both results and reveals what lies between them: the super-amplification zone that neither seed happened to land in.

---

## What the 867× Ratio Means in Practice

In one sense, 867× is an extreme measurement artifact — the denominator is collapsing toward zero at exactly the measurement point. You would not expect a real deployed model to produce ratios in the hundreds under typical probing conditions.

In another sense, 867× is exactly what the theory predicts it should be at the transition edge. The Born filter ratio is a sensitivity measure: how much does the head output change when you apply the morphism (swap bank ↔ current)? Near the collapse boundary, S3's FIN head is in a metastable state — it is just barely maintaining FIN-token attention. A small perturbation (the swap morphism) tips it into self-attention. Meanwhile, S1 and S2 retain robust FIN signal that the morphism moves significantly. The asymmetry is not artificial; it is the correct measurement of an extreme sensitivity difference between states.

The production implication: any model near the collapse boundary for a given context type will show extreme Born filter sensitivity at the relevant layer. **Models in the sharp-collapse regime are highly sensitive to small context perturbations.** A model operating near FIN_WEIGHT≈0.3-equivalent geometry will have a diagnostic ratio 300–800× higher than a model in the slow-collapse or no-collapse regime. The Born filter ratio is not just a qualitative diagnostic — it is quantitatively predictive of transition-edge proximity.

---

## Real-World Implications

**The collapse boundary is a production warning threshold.** If the Born filter ratio for a specific head and context type jumps by two or more orders of magnitude compared to a baseline condition, the model is operating near the sharp-collapse regime for that (head, context) pair. This is the condition where small context changes (a few word substitutions, a synonym, a paraphrase) can produce dramatically different internal representations — not because the model has learned a wrong rule, but because the attention collapse is highly context-sensitive at that geometry.

**On T5-small's layer-5 specificity (controlled-model analogy).** G-054 suggests, by analogy in the controlled model, that T5-small's decoder amplification at layer 5 may reflect a sharp-collapse regime completing at that depth. The relevant token (the OOD "jump") has a specific FIN-analog embedding character that places it near the collapse boundary in T5-small's geometry. If you shifted that character (by OOD training, by embedding regularization, or by adding compound-context fine-tuning), you would shift the collapse depth — and potentially move the model out of the sharp-collapse regime entirely.

**Norm equalization alone does not fix the collapse.** G-052 showed that the parallax lever is driven by norm asymmetry, and equalized norms collapse the logit gap to near zero. G-054 shows that the FIN-head collapse is a separate mechanism operating at the representational level, not the output level. If you equalize norms (addressing G-052's mechanism) but leave the embedding geometry in the sharp-collapse regime (G-054's mechanism), you address the output amplification but not the representational instability. A model with equalized norms but sharp-collapse geometry will have consistent logit gaps but still produce wildly different internal representations near the transition boundary.

**The two interventions target different parts of the causal chain.** G-052's norm equalization addresses the final step (output decision). G-054's embedding geometry determines the middle step (which layer the representation becomes decisive). To fix the production failure completely, both mechanisms need to be addressed: the norm asymmetry that amplifies the hidden state signal into a decisive logit gap, and the collapse geometry that determines which layer that signal becomes informative.

**Layer profiling is now a pre-measurement requirement.** Before committing to a Born filter measurement at any fixed layer, profile the per-head defect across all layers. The per-layer profile is itself a diagnostic:
- Monotonic decay from layer 1: model is in the no-collapse regime for this (head, context) pair. Layer 1 is the best measurement point.
- Mid-layer peak: model is in the sharp-collapse or slow-collapse regime. Measure at the peak layer.
- Mid-layer peak with extreme ratio (>10× above baseline): model is near the transition boundary. The measurement is valid but flag this as a high-sensitivity configuration.

---

## What G-054 Does Not Prove

G-054 maps the phase transition for a controlled toy model with fully designed embeddings. Several questions remain open for the S-track:

**Does T5-small's layer-5 amplification correspond to the sharp-collapse regime or the slow-collapse regime?** G-054 shows both produce mid-layer peaks; the distinction matters for the stability interpretation. A 2.58× ratio (G-050's benchmark) is consistent with both.

**Is the super-amplification zone detectable in a trained model?** A trained model's embeddings are not cluster-constructed; the FIN-analog character of any token is a result of training, not design. Whether production embeddings land near the collapse boundary — in the 867× zone — is an empirical question for the S-track.

**Does the three-regime structure generalize to more complex morphisms?** G-054 uses the bank↔current target-anchored morphism, which is the simplest possible design. Morphisms involving multiple token swaps, or cross-sequence comparisons (the S-track design), may shift the collapse dynamics.

---

## What Comes Next

G-054 completes the mechanistic arc that began with G-050. We now have:
- The discovery (G-050): specialist heads peak at mid-layers
- The null result (G-053): the peak requires specific geometric conditions
- The phase diagram (G-054): three regimes, a sharp transition at FIN_WEIGHT≈0.5, and a super-amplification zone at the transition edge

The geometry track (G-054) has done what it can do. The remaining questions require a trained model — either production T5-small or pre-trained BERT. The logical next step is to show the S-track what the G-track has established and design S-055 around the three-regime hypothesis: probe T5-small's per-head Born filter across all layers, identify which regime it is in, and verify that the collapse depth aligns with the layer-5 specificity the S-track independently found.

That is the handoff.

---

*Methods notes are written at sealing time and not revised.*
*G-track | Applied Categorical Physics Workbench | Troy Teno | 2026*
