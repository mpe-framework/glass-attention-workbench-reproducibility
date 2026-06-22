# Experiment Index

**Glass Attention Workbench — Navigation Guide**
*All sealed experiments, in order. Click any link to go directly to the file on GitHub.*

**Base repo:** `https://github.com/mpe-framework/glass-attention-workbench`

---

## How to Use This Index

Each entry links to every associated file: experiment script, pre-registration proposal,
methods report, and the relevant section of the canonical findings record. Not every
experiment has all four — the "Resources" column shows what exists.

For the synthesis across all experiments, read [`WHAT_WE_FOUND.md`](./WHAT_WE_FOUND.md)
at the root of main.

---

## Foundation Experiments (branch: main)

| ID | One-line result | Resources |
|----|-----------------|------------|
| 038 | PPMI geometry is load-bearing — random embeddings → 0% compositional accuracy, PPMI+SVD → 100% | [code](https://github.com/mpe-framework/glass-attention-workbench/blob/main/workbench[...]|
| 039 | Split information: V0.6.0 rerun — 100% holdout accuracy across six PLC classes; WRITE_SETPOINT 46.6% standard eval only — shortcut learning diagnosis | [code](https://github.com/mpe-framework/glass-a[...]|
| 040 | Causal bit recovery: four linear probes on frozen 039 representations + fixed symbolic decoder. V0.7.0 rerun: gold-bit decode 100%; verb_bit and actor_id_bit not linearly recoverable — information present but partly non-linearly encoded | [code](https://github.com/mpe-framework/glass-attention-workbench/blob/main/workbench/exper[...]|

---

## G-Track Experiments (branch: G-Track)

*Geometry / toy scale. NumPy-first, CPU, fully inspectable.*
*Canonical findings:* [`findings/FINDINGS.md`](https://github.com/mpe-framework/glass-attention-workbench/blob/G-Track/findings/FINDINGS.md)

| ID | One-line result | Resources |
|----|-----------------|------------|
| G-042 | Scrambled token ("bnak") gets 9% self-attention; context absorbs defect via Softmax — post-attention cosine(bank, bnak) = 0.9987 | [code](https://github.com/mpe-framework/glass-attenti[...]|
| G-043 | W_O moves to a new geometric position 2.33 units from the head centroid — not a convex combination | [code](https://github.com/mpe-framework/glass-attention-workbench/blob/G-Track/work[...]|
| G-044 | Untrained weights cannot distinguish structured from scrambled input — both converge to the same fixed point | [code](https://github.com/mpe-framework/glass-attention-workbench/blob/G-[...]|
| G-045 | Born filter identity confirmed (2‖h_B‖ = δ_BC to float precision); defect field is local — peaks at swap positions, not at the target token | [code](https://github.com/mpe-framewo[...]|
| G-046 | Born filter separates ambiguous from unambiguous context at 1.44× with zero training; attention entropy flat across all states — Born filter strictly more sensitive | [code](https://g[...]|
| G-047 | Within-sequence morphism: per-token = mean-pool in bidirectional attention; cross-sequence is the correct instrument for the S-track design | [code](https://github.com/mpe-framework/glas[...]|
| G-048 | Random BERT weights collapse Born filter ratio to ~1.0; architecture depth without trained weights adds nothing | [code](https://github.com/mpe-framework/glass-attention-workbench/blob/G[...]|
| G-049 | Semantic Q/K/V initialization amplifies ratio 1.08× → 1.69× across 12 layers; attention entropy NOT flat with semantic weights | [code](https://github.com/mpe-framework/glass-attenti[...]|
| G-050 | Per-head Born filter: genuine 2.58× at layer 5 for FIN-specialist head; full-state layer-12 result was seed-dependent noise | [code](https://github.com/mpe-framework/glass-attention-wor[...]|
| G-051 | W_V miscalibration: 133× separation between miscalibrated (‖h_B‖=0.011) and correctly-calibrated (‖h_B‖=1.490) OOD heads | [code](https://github.com/mpe-framework/glass-attentio[...]|
| G-052 | Norm asymmetry reproduces S-track logits to float precision; 259× amplification from norm asymmetry alone | [code](https://github.com/mpe-framework/glass-attent[...]|
| G-053 | Mid-layer peak is not a general property of depth — default is monotonic decay; peak requires specialist-head collapse in at least one state | [code](https://github.com/mpe-framework/g[...]|
| G-054 | Phase diagram complete: three regimes (slow-collapse ~2×, sharp-collapse up to 867×, no-collapse); sharp transition at FIN_WEIGHT≈0.5 | [code](https://github.com/mpe-framework/glass-[...]|
| G-055 | H2 confirmed: global head diagnostic not causal; 4% ablation = near-identity baseline; specialist heads attend to FIN tokens; L3H0 is a probe; S-051 mechanism complete | [code](https://github.com/mpe-framework/glass-attention-workbench/blob/G-Track/workbench/experiments/G-055_GLOBAL_HEAD_CAUSAL_DIAGNOSTIC.py) · [methods](https://github.com/mpe-framework/glass-attention-workbench/blob/main/findings/METHODS_REPORT_G055.md) |
| G-056 | H1 confirmed — suppressive head (V = I − α·P_FIN) is causal above FIN-annihilation threshold (α=1.0): 42% ablation change vs 4% neutral baseline; mechanism is FIN inversion via specialist re-routing, not suppression | [code](https://github.com/mpe-framework/glass-attention-workbench/blob/G-Track/workbench/experiments/G-056_SUPPRESSIVE_HEAD_CAUSAL_TEST.py) · [methods](https://github.com/mpe-framework/glass-attention-workbench/blob/main/findings/METHODS_REPORT_G056.md) |
| G-057 | FFN over-correction (γ=2.0) reproduces G-056 FIN-inversion (51.3% ablation change, H4 confirmed); near-cancellation divergence structurally confirmed (recon_frac 1.0→∞ across γ=0→1.0); FIN-inversion mechanism generalizes from V-matrix to FFN | [code](https://github.com/mpe-framework/glass-attention-workbench/blob/G-Track/workbench/experiments/G-057_NEAR_CANCELLATION.py) · [methods](https://github.com/mpe-framework/glass-attention-workbench/blob/main/findings/METHODS_REPORT_G057.md) |

---

## S-Track Experiments (branch: S-Track)

*SCAN / production scale. T5-small fine-tuned on SCAN add_prim_jump.*
*Canonical findings:* [`FINDINGS.md`](https://github.com/mpe-framework/glass-attention-workbench/blob/S-Track/FINDINGS.md)

| ID | One-line result | Resources |
|----|-----------------|------------|
| S-041 | Confounded by rain/snow pair — see S-043 for corrected design | [code](https://github.com/mpe-framework/glass-attention-workbench/blob/S-Track/workbench/experiments/041_BC_MEASURE.py) [...]|
| S-042 | Confounded — see F-042 in FINDINGS.md | [code](https://github.com/mpe-framework/glass-attention-workbench/blob/S-Track/workbench/experiments/042_HEAD_LOCALIZATION.py) · [findings](htt[...]|
| S-043 | T5 encoder marks "jump" as categorically different from trained primitives (p=0.0000) — the encoder knows; the failure is downstream | [code](https://github.com/mpe-framework/glass-att[...]|
| S-044 | Mean-pool BC defect does not grade difficulty within the failing category — within-group signal reverses; failure is categorical (encoder) and instance-determined (decoder) | [code](ht[...]|
| S-045 | Per-token BC defect at jump position also flat — K1 fires: encoder marks all jump-compound examples uniformly; failure is in the decoder | [code](https://github.com/mpe-framework/glass[...]|
| S-046 | All 8 structural bits recoverable from frozen layer 5 at ≥99.5%; K2 fires: raw representation predicts failure at AUC=0.88 (structural complexity) | [code](https://github.com/mpe-frame[...]|
| S-047 | "has_around" dominates failure (AUC=0.806, lift=+0.648); commands with "around" fail 87% of the time | [code](https://github.com/mpe-framework/glass-attention-workbench/blob/S-Track/work[...]|
| S-048 | Improving encoder's "around" representation via aux loss makes accuracy worse (43.7%→15.0%) — bottleneck is in the decoder | [code](https://github.com/mpe-framework/glass-attention-w[...]|
| S-049 | Decoder routing intact (p=0.92); value substitution confirmed — 26% of has_around failures are substituted_walk: correct structure, wrong primitive | [code](https://github.com/mpe-fram[...]|
| S-050 | Buggy run — tokenization error (proxy_id=27 for all action tokens); superseded by S-050b | [code](https://github.com/mpe-framework/glass-attention-workbench/blob/S-Track/workbench/expe[...]|
| S-050b | Embedding norm asymmetry confirmed — ‖I_WALK‖≈520; two-stage selection: routing correct, norms determine which token wins | [code](https://github.com/mpe-framework/glass-attenti[...]|
| S-051 | Full mechanism: norm causally necessary (P(I_WALK) 0.91→0.32 after equalization); layer 5 is the fork; L4H6/L5H2/L5H5 drive value substitution | [code](https://github.com/mpe-framework[...]|
| S-055 | Per-head Born filter profile across T5 decoder layers; L3H0 shows highest ratio (3.867×, inverted direction success>fail); sharp-collapse regime consistent | [code](https://github.com/m[...]|
| S-056 | L3H0 is not a jump-attending head (attn=0.0089) — global-context head; reference heads L4H6/L5H2/L5H5 confirm S-051; causal question referred to G-055 | [code](https://github.com/mpe-framework/glass-attention-workbench/blob/S-Track/workbench/experiments/S-056_L3_HEAD_CLUSTER.py) · [methods](https://github.com/mpe-framework/glass-attention-workbench/blob/main/findings/METHODS_REPORT_S056.md) |
| S-057 | L3H4 is a suppressive reader — reliably attends to jump position (p=0.0007, r=0.539) but output cosines negative to both I_WALK and I_JUMP; entropy ordering inverted (L3H0 most concentrated, not most diffuse); suppressive head type established | [code](https://github.com/mpe-framework/glass-attention-workbench/blob/S-Track/workbench/experiments/S-057_L3H4_CHARACTERIZATION.py) · [methods](https://github.com/mpe-framework/glass-attention-workbench/blob/main/findings/METHODS_REPORT_S057.md) |
| S-058 | K2 fires — activation patching from success donors flips 100% of fail examples regardless of which head is patched; confound is representational context injection, not head-specific; donor transplant method invalidated; W_V weight-level repair required | [findings](https://github.com/mpe-framework/glass-attention-workbench/blob/main/DCRP/sandbox_019_S-058_methodological_finding_and_S-059_direction.md) · [methods](https://github.com/mpe-framework/glass-attention-workbench/blob/main/findings/METHODS_REPORT_S058.md) |
| S-059 | Methodological finding only — LayerNorm unit mismatch: decomposition computed in pre-RMSNorm units gives reconstruction fraction 32,193× instead of ~1.0; all H verdicts invalid; near-cancellation structure cannot be confirmed from this run; superseded by S-059b | [findings](https://github.com/mpe-framework/glass-attention-workbench/blob/main/DCRP/sandbox_020_S-059_layernorm_mismatch_and_near_cancellation.md) · [methods](https://github.com/mpe-framework/glass-attention-workbench/blob/main/findings/METHODS_REPORT_S059.md) |
| S-059b | Genuine near-cancellation confirmed — reconstruction fraction 28.5× (cross-attn total −171.25 logit units vs observed margin −6.0); L4H6/L5H5 pro-walk directionally confirmed; L6H2/L6H6 dominant in decomposition (further causal test required) | [findings](https://github.com/mpe-framework/glass-attention-workbench/blob/main/DCRP/sandbox_021_S-059b_results_and_S-060_direction.md) · [methods](https://github.com/mpe-framework/glass-attention-workbench/blob/main/findings/METHODS_REPORT_S059.md) |
| S-060 | L6H2/L6H6 are compensatory heads (ablating worsens margin — sign-reversed Δmargin); L4H6 specificity +0.13, L5H5 +0.23 on non-OOD examples; near-cancellation causally confirmed; no single ablation sufficient; W_V weight-level repair is the principled next step | [findings](https://github.com/mpe-framework/glass-attention-workbench/blob/main/DCRP/sandbox_022_S-060_compensatory_L6_and_weak_L4_L5_signal.md) |
| S-061 | Rank-1 W_V repair at L4H6/L5H5 flips 16/30 fail examples (+2.90 logit margin improvement, α=1.0); H1/H2/H3 all PASS; H4 indeterminate (n=2; non-jump-around control not run); mechanistic arc closed for selected fail group | [findings](https://github.com/mpe-framework/glass-attention-workbench/blob/main/DCRP/sandbox_024_S-061_fully_sealed_mechanism_confirmed_end_to_end.md) |
| S-062 | H4 success-group cross-check: fail-group replication exact (Δm=+2.9024, 16/30 flips match); 0/2 flip regression on success group (n=2); repair non-degrading on available sample; H4 specificity indeterminate — broader specificity (non-jump-around commands) remains a recommended follow-up | [findings](https://github.com/mpe-framework/glass-attention-workbench/blob/main/DCRP/sandbox_024_S-061_fully_sealed_mechanism_confirmed_end_to_end.md) |

---

## Key Documents

| Document | Link | Purpose |
|----------|------|--------|
| `WHAT_WE_FOUND.md` | [main](https://github.com/mpe-framework/glass-attention-workbench/blob/main/WHAT_WE_FOUND.md) | Synthesis narrative — inception through G-055/S-056 |
| `WHAT_WE_FOUND_VOL_II.md` | [main](https://github.com/mpe-framework/glass-attention-workbench/blob/main/WHAT_WE_FOUND_VOL_II.md) | Synthesis narrative continuation — G-056/G-057, S-057 through S-062 |
| `findings/FINDINGS.md` (G-Track) | [G-Track](https://github.com/mpe-framework/glass-attention-workbench/blob/G-Track/findings/FINDINGS.md) | Locked per-hypothesis results — G-track canonical record |
| `FINDINGS.md` (S-Track) | [S-Track](https://github.com/mpe-framework/glass-attention-workbench/blob/S-Track/FINDINGS.md) | Locked per-hypothesis results — S-track canonical record |
| `COORDINATION.md` | [G-Track](https://github.com/mpe-framework/glass-attention-workbench/blob/G-Track/COORDINATION.md) · [S-Track](https://github.com/mpe-framework/glass-attention-workbench/blob/S-Track/COORDINATION.md) | Cross-track state, action items, relay findings |
| `findings/LEXICON.md` | [G-Track](https://github.com/mpe-framework/glass-attention-workbench/blob/G-Track/findings/LEXICON.md) | Definitions, framework connections, conventions (G-track) |
| `Lexicon of the Glass Attention Workbench.md` | [main](https://github.com/mpe-framework/glass-attention-workbench/blob/main/Lexicon%20of%20the%20Glass%20Attention%20Workbench.md) | Public-facing glossary — full vocabulary through G-057/S-062 |
| Omnibus methods (G-Track) | [G042–G051](https://github.com/mpe-framework/glass-attention-workbench/blob/main/findings/METHODS_REPORT_G042_G051.md) | Narrative covering G-042 through G-051 |
| Per-experiment methods (G-Track) | [G054](https://github.com/mpe-framework/glass-attention-workbench/blob/main/findings/METHODS_REPORT_G054.md) · [G055](https://github.com/mpe-framework/glass-attention-workbench/blob/main/findings/METHODS_REPORT_G055.md) · [G056](https://github.com/mpe-framework/glass-attention-workbench/blob/main/findings/METHODS_REPORT_G056.md) · [G057](https://github.com/mpe-framework/glass-attention-workbench/blob/main/findings/METHODS_REPORT_G057.md) | Individual methods reports — G-track experiments |
| Omnibus methods (S-Track) | [S043–S051](https://github.com/mpe-framework/glass-attention-workbench/blob/main/findings/METHODS_REPORT_S043_S051.md) | Original omnibus (individual reports supersede for S-055+) |
| Per-experiment methods (S-Track) | [S046](https://github.com/mpe-framework/glass-attention-workbench/blob/main/findings/METHODS_REPORT_S046.md) · [S047](https://github.com/mpe-framework/glass-attention-workbench/blob/main/findings/METHODS_REPORT_S047.md) · [S049](https://github.com/mpe-framework/glass-attention-workbench/blob/main/findings/METHODS_REPORT_S049.md) · [S057](https://github.com/mpe-framework/glass-attention-workbench/blob/main/findings/METHODS_REPORT_S057.md) · [S059](https://github.com/mpe-framework/glass-attention-workbench/blob/main/findings/METHODS_REPORT_S059.md) | Individual methods reports — S-track experiments |
| DCRP Sandboxes | [main/DCRP/](https://github.com/mpe-framework/glass-attention-workbench/tree/main/DCRP) | Superintendent synthesis nodes, S-058 through S-062 results |

---

*Applied Categorical Physics Workbench | Troy Teno | May 2026 | Open Access*
*Sealed through G-057 (G-track) and S-062 (S-track). Mechanistic arc closed: locate → explain → repair.*
