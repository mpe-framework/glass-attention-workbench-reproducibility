# Methods Report — G-054: Collapse Conditions

**Applied Categorical Physics Workbench | Troy Teno | May 2026**
**Status:** Sealed | **Script:** `workbench/experiments/G-054_COLLAPSE_CONDITIONS.py`
**One-line result:** Phase diagram complete — three regimes (slow-collapse ~2×, sharp-collapse up to 867×, no-collapse); sharp transition at FIN_WEIGHT≈0.5; super-amplification at the transition edge.

*Note: findings/METHODS_NOTE_G054.md contains an extended plain-language explanation of this experiment.*

---

## The Question

G-053 identified the decisive variable explaining the difference between G-050's mid-layer peak (2.58× at layer 5) and G-053's monotonic decay: the bank token's FIN embedding weight. With SEED=50, bank had a weak FIN component in S3, so the FIN specialist head collapsed to self-attention — producing a ratio spike. With SEED=53, bank retained strong FIN character, so the FIN head did not collapse — producing monotonic decay.

What is the full landscape? At exactly what FIN_WEIGHT does the collapse condition switch from active to inactive? Is the transition gradual or sharp? And what happens at the transition boundary itself?

---

## Setup

- **Architecture:** 12-layer BERT, semantic Q/K/V init, SEED=54
- **Variable:** Bank token FIN embedding weight (FIN_WEIGHT ∈ {0.0, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0, 1.5, 2.0})
- **Construction:** `bank = normalize(geo_bank + FIN_WEIGHT × fin_bank)`. All other embeddings fixed.
- **Morphism:** swap `bank` ↔ `current` (target-anchored)
- **Measurement:** Per-head Born filter H1(FIN) ratio at every layer via forward hooks

---

## Pre-Registered Hypotheses

**H1:** FIN_WEIGHT=0.0 produces peak ratio > 2.0× (slow-collapse with substantial amplification).

**H2:** Phase transition between FIN_WEIGHT=0.3 (mid-layer peak) and FIN_WEIGHT=0.5 (monotonic decay).

**H3:** Transition threshold is at FIN_WEIGHT < 0.5 × CLUSTER_WEIGHT.

---

## Key Measurements

### Per-FIN_WEIGHT summary

| FIN_WEIGHT | Peak layer | Peak ratio  | S3 H1 defect @L5 | Note                          |
|------------|------------|-------------|------------------|-------------------------------|
| 0.0        | 11/12      | 1.9976×     | 1.280477         | slow collapse, late peak      |
| 0.1        | 6/12       | 2.0379×     | 1.275949         | mid-layer peak                |
| 0.2        | 6/12       | 2.0910×     | 1.278340         | mid-layer peak                |
| 0.3        | 6/12       | 867.4514×   | 1.267525         | MID-LAYER PEAK (extreme)      |
| 0.5        | 1/12       | 0.1154×     | 0.885401         | monotonic decay               |
| 0.7        | 1/12       | 0.0608×     | 1.186912         | monotonic decay               |
| 1.0        | 1/12       | 0.0986×     | 0.050360         | monotonic decay               |
| 1.5        | 1/12       | 0.0481×     | 0.977127         | monotonic decay               |
| 2.0        | 1/12       | 0.0697×     | 1.005914         | monotonic decay               |

### Per-layer H1(FIN) ratio at FIN_WEIGHT=0.3 (peak condition)

| Layer | Ratio    | S3 H1 defect | Note                            |
|-------|----------|--------------|----------------------------------|
| 1     | 1.62×    | 2.752        |                                  |
| 5     | 8.79×    | 1.268        |                                  |
| 6     | 867.45×  | 0.012        | sharp collapse to near-zero      |
| 7     | 307.32×  | 0.028        |                                  |
| 12    | 127.46×  | 0.026        |                                  |

---

## Results

| Hypothesis | Verdict |
|------------|---------|
| H1: FIN_WEIGHT=0.0 produces peak ratio > 2.0× | **NULL** (ratio = 1.9976×, missed threshold by 0.0024) |
| H2: Phase transition between FIN_WEIGHT=0.3 and FIN_WEIGHT=0.5 | **CONFIRMED** (sharp step from layer-6 mid-layer peak at FIN_WEIGHT=0.3 to layer-1 monotonic decay at FIN_WEIGHT=0.5) |
| H3: Transition threshold at FIN_WEIGHT < 0.5 × CLUSTER_WEIGHT | **NULL** (boundary at FIN_WEIGHT≈0.5, not below 0.5 × CLUSTER_WEIGHT) |

---

## Interpretation

Three regimes emerge — not predicted, discovered:

**Slow-collapse regime (FIN_WEIGHT ≤ 0.1):** The FIN head eventually collapses in S3, but over many layers. The S3 defect decays gradually rather than dropping sharply. Peak occurs at late layers (layer 11), peak ratio ~2×.

**Sharp-collapse regime (FIN_WEIGHT 0.1–0.3):** The FIN head collapse completes at a specific mid-layer in a single step. At FIN_WEIGHT=0.3, the S3 defect drops from 1.268 at layer 5 to 0.012 at layer 6 — a near-zero denominator while the S1 numerator remains finite. Ratios range from ~2× up to 867×. This is the super-amplification zone.

**No-collapse regime (FIN_WEIGHT ≥ 0.5):** Bank carries sufficient FIN signal that the FIN head finds real FIN content to attend to in S3. No collapse occurs. Ratio peaks at layer 1 and decays monotonically — the G-053 regime.

The collapse speed determines both the peak layer and the peak magnitude. A sharp, single-step denominator collapse produces the largest ratio. A gradual collapse produces a modest late-layer peak. No collapse produces monotonic decay.

G-050 (SEED=50, 2.58× at layer 5) corresponds to the sharp-collapse regime at moderate FIN_WEIGHT (~0.1–0.2). G-053 (SEED=53, monotonic decay) corresponds to the no-collapse regime (bank's FIN component above threshold ~0.5). G-054 maps the full landscape between and beyond both.

---

## Null / Confound Note

H1's 2.0× threshold was missed by 0.0024 (measured ratio = 1.9976×). The slow-collapse regime does produce amplification, but the peak lands below the pre-registered threshold. This is an effect-size near-miss, not a qualitative failure: FIN_WEIGHT=0.0 does produce a late-layer peak, just below the benchmark. The three-regime discovery is the more informative output of this experiment.

---

## Canonical Finding

> "The phase transition between mid-layer peak and monotonic decay is sharp, occurring between FIN_WEIGHT=0.3 and FIN_WEIGHT=0.5." — Script G-054

> "At the transition edge (FIN_WEIGHT=0.3), the H1 ratio reaches 867× — three orders of magnitude above baseline. Sharpest collapse = largest ratio." — Script G-054

> "Three regimes: slow-collapse (late peak, modest ratio), sharp-collapse (mid-layer peak, extreme ratio), no-collapse (monotonic decay, ratio < 1×)." — Script G-054

> "The collapse speed — how quickly the FIN head reaches self-attention in S3 — determines both the peak layer and the peak magnitude." — Script G-054

---

## What This Does Not Prove

G-054 maps the phase transition for a controlled toy model with fully designed embeddings. It does not establish which regime T5-small occupies — that requires production probing (S-track). A 2.58× ratio (G-050's benchmark) is consistent with both slow-collapse and the lower range of sharp-collapse, so regime identification for T5-small requires a direct per-layer profile. The three-regime structure is also characterized via the bank↔current morphism; morphisms involving multiple token swaps or cross-sequence comparisons may shift the collapse dynamics. The super-amplification zone (867×) has not been observed in a trained model and may require production embedding geometry to land at the precise transition boundary.

---

## Links

- **Script:** `workbench/experiments/G-054_COLLAPSE_CONDITIONS.py`
- **Saved output:** Output artifact not found in repo
- **FINDINGS.md entry:** F-054
- **Companion note:** findings/METHODS_NOTE_G054.md
- **Experiment index:** EXPERIMENT_INDEX.md
- **Related:** METHODS_REPORT_G050.md, METHODS_REPORT_G053.md
