# Canonical Findings Index

**Applied Categorical Physics Workbench**  
Troy Teno | 2026 | Open Access

This file is the public canonical findings index and synthesis map for the workbench.

Per-experiment findings are sealed in their linked scripts, methods reports, branch records, saved outputs, and DCRP records. This index is not a replacement for those underlying records.

For navigation, see:

- [`EXPERIMENT_INDEX.md`](./EXPERIMENT_INDEX.md) — experiment-by-experiment links
- [`REPRODUCIBILITY.md`](./REPRODUCIBILITY.md) — reproduction and audit guidance
- [`WHAT_WE_FOUND.md`](./WHAT_WE_FOUND.md) — synthesis narrative, Volume I
- [`WHAT_WE_FOUND_VOL_II.md`](./WHAT_WE_FOUND_VOL_II.md) — synthesis narrative, Volume II
- [`DCRP/`](./DCRP) — cross-agent audit and synthesis records

---

## Public Boundary

The workbench reports a narrow, bounded mechanistic case study:

> one model, one checkpoint, one benchmark split, one failure mode, one traced mechanism, one targeted partial repair.

It does not claim a general theory of all transformer failures, validation at GPT-scale, proof for natural language, or universal repair specificity.

---

## Repository-Level Finding

A selected `jump` → `walk` substitution failure in T5-small fine-tuned on SCAN `add_prim_jump` was traced through encoder representation, decoder cross-attention, value-projection geometry, near-cancellation, and targeted weight-level repair.

The strongest public claim is:

> A targeted rank-1 edit to decoder cross-attention `W_V` matrices at L4H6 and L5H5 partially repairs a selected `jump` → `walk` substitution failure in a T5-small model fine-tuned on SCAN `add_prim_jump`, supporting a decoder value-projection account of the failure.

The repair is partial, selected, and bounded.

---

## Foundation Findings

### F-038 — PPMI Geometry is Load-Bearing

*Script: `038_PPMI_GEOMETRY.py` | Status: Sealed*

Random embeddings produce 0% holdout accuracy on cross-domain compositional test cases. PPMI+SVD embeddings produce 100%.

The corpus geometry is not decorative. It is the structural precondition for the attention head to learn meaningful routing.

> "PPMI geometry is load-bearing, not decorative." — F-038

---

### F-034 — Orthogonal Initialization Breaks the Lazy Attractor

*Script: 034 series | Status: Sealed*

Two-head architectures collapsed to a single attractor under standard initialization: one head dominated and the other became inactive.

Three mutually orthogonal heads broke this attractor and produced the first dual-wire specialization without supervision.

---

### F-037 — Glass Head Role Assignments Confirmed by Linear Probes

*Script: `037_FULL_COMPOSITIONAL_AUDIT.py` | Status: Sealed*

The six-class PLC taxonomy produced the first formally confirmed glass-head role assignments via linear probes on frozen head outputs.

| Head | Role |
|------|------|
| H1 | Context / subject locus |
| H2 | Verb locus |
| H3 | Command class / composition |

H3 reads from the intermediate representation, not directly from the raw token sequence. This is a structural constraint of the architecture.

---

### F-039 — Split Information: Shortcut Learning Diagnosis

*Script: `039_SPLIT_INFORMATION.py` | Status: Sealed*

Two instances were trained on strictly partitioned vocabularies:

- Instance A: actors and verbs
- Instance B: contexts
- vocabulary overlap: zero

V0.6.0 Colab rerun: Layer 3 reached 100% holdout accuracy across all six PLC classes.
WRITE_SETPOINT accuracy was 46.6% under standard eval only.

The diagnosis was that the information was present in the frozen representations, but the objective did not require Layer 3 to reconstruct the latent Boolean program in all evaluation conditions. Gradient descent found a shortcut in standard eval; the full holdout accuracy required the split-information architecture.

> "I(inter_A ; ctx) = 0 and I(inter_B ; actor) = 0 — enforced, not inferred." — F-039

> "Attention heads optimize the loss they are given." — F-039 diagnosis

---

### F-040 — Causal Bit Recovery: Information Was Present

*Script: `040_CAUSAL_BIT_RECOVERY.py` | Status: Sealed*

F-040 tested the F-039 diagnosis directly.

Instead of asking a learned Layer 3 classifier to map the joint representation directly to a six-way command class, F-040 used:

1. four linear probes on frozen F-039 representations;
2. a fixed symbolic decoder derived from the task generator;
3. a comparison between gold-bit decoding and probe-bit decoding.

The four causal bits were:

| Bit | Source | Definition |
|-----|--------|------------|
| `verb_bit` | Instance A | `caught=0`, `lifted=1` |
| `actor_dom_bit` | Instance A | bird actors=0, machine actors=1 |
| `actor_id_bit` | Instance A | canonical actor=0, inverted actor=1 |
| `ctx_dom_bit` | Instance B | nature=0, industrial=1 |

F-040 closed the foundation diagnosis: the split representations contained the needed causal information; the F-039 failure was a shortcut-learning/objective failure, not an absence-of-information failure.

V0.7.0 Colab rerun: gold-bit decode reached 100%; `verb_bit` and `actor_id_bit` were not linearly recoverable from frozen representations, showing that the needed information was present but partly non-linearly encoded.

---

## Branch-Level Records

From 041 onward, the workbench branches into S-track and G-track records.

- **S-track:** production-scale T5-small / SCAN `add_prim_jump` mechanism record, S-041 through S-062. See [`FINDINGS.md` on S-Track](https://github.com/mpe-framework/glass-attention-workbench/blob/S-Track/FINDINGS.md).
- **G-track:** controlled geometry / toy-scale mechanism record, G-042 through G-057. See [`findings/FINDINGS.md` on G-Track](https://github.com/mpe-framework/glass-attention-workbench/blob/G-Track/findings/FINDINGS.md).
- **Main index:** see [`EXPERIMENT_INDEX.md`](./EXPERIMENT_INDEX.md) for scripts, methods reports, findings entries, and DCRP records.

The old pending main-branch F-041 entry has been removed. F-041 belongs to the later branch-level lineage.

---

## Sealed Mechanism Arc

The current public mechanism arc is:

```text
S-043 → S-047 → S-049 → S-050b → S-051 → S-059b → S-060 → S-061 → S-062
```

Supporting G-track arc:

```text
G-055 → G-056 → G-057
```

### What the arc establishes

1. **The encoder marks `jump` as different.**  
   The model does not simply fail because it cannot see or represent `jump`.

2. **The failure is concentrated around compound `jump`, especially `around`.**  
   The failure is structured, not random noise.

3. **Decoder routing is often intact.**  
   The decoder frequently attends to the relevant source position; the problem is downstream of mere attention location.

4. **The value-projection mechanism is implicated.**  
   L4H6 and L5H5 write value geometry that favors `I_WALK` over `I_JUMP` in the selected fail group.

5. **Large late-layer heads can be compensatory rather than causal.**  
   L6H2 and L6H6 appear large in decomposition but behave as compensatory heads under causal tests.

6. **The failure lives in a near-cancellation regime.**  
   The final wrong answer is a small residual imbalance after opposing components nearly cancel.

7. **A targeted rank-1 `W_V` repair moves the model in the predicted direction.**  
   S-061 repairs the selected fail group partially and monotonically.

---

## S-061 / S-062 Repair Summary

S-061 applied a targeted rank-1 correction to the `W_V` matrices of L4H6 and L5H5.

At `α=1.0`, the joint repair:

- improved the sampled fail-group margin by `+2.9024` logits;
- flipped 16/30 sampled failures;
- did not claim universal repair.

S-062 reproduced the fail-group result exactly and checked the corrected success sample.

The success-group check was very small:

- corrected success sample: `n=2`;
- flip regression: 0/2;
- no OOD collapse observed;
- margin moved in the positive direction.

The honest public claim is:

> The repair improved the selected failure group and did not degrade the tiny available corrected jump-around success sample. Broader specificity against unrelated around-commands remains a recommended follow-up.

---

## Findings That Did Not Pan Out

Failures and confounds remain part of the record.

- **029C:** behavior improved while interpretability degraded. More capable did not mean more interpretable under the standard objective.
- **S-041 / S-042:** the BERT permutation-morphism approach was underpowered/confounded by lexical asymmetry, especially the `rain`/`snow` pair.
- **S-058:** donor activation patching flipped outputs by contaminating the residual stream, not by cleanly repairing the mechanism.
- **S-059:** initial logit decomposition used the wrong normalization units and was superseded by S-059b.
- **S-060:** naive head ablation in a near-cancellation regime gave sign-reversed effects, teaching that large contribution is not identical to causal blame.

These are not discarded runs. They define the instrument boundary.

---

## Key Formulations

1. "PPMI geometry is load-bearing, not decorative." — F-038
2. "Attention heads optimize the loss they are given." — F-039
3. "Behavior improved while interpretability degraded." — 029C
4. "I(inter_A ; ctx) = 0 and I(inter_B ; actor) = 0 — enforced, not inferred." — F-039
5. "The model does not simply fail to see `jump`. It reads the relevant source and writes the wrong value from it." — public mechanism summary
6. "Large participation is not the same as causal blame." — S-060 / near-cancellation lesson

---

## Evidence Map

For the full record, read in this order:

1. [`README.md`](./README.md) — public front door
2. [`EXPERIMENT_INDEX.md`](./EXPERIMENT_INDEX.md) — full experiment navigation
3. [`REPRODUCIBILITY.md`](./REPRODUCIBILITY.md) — reproduction and audit path
4. [`WHAT_WE_FOUND.md`](./WHAT_WE_FOUND.md) — synthesis, Volume I
5. [`WHAT_WE_FOUND_VOL_II.md`](./WHAT_WE_FOUND_VOL_II.md) — synthesis, Volume II
6. Branch findings:
   - [`S-Track/FINDINGS.md`](https://github.com/mpe-framework/glass-attention-workbench/blob/S-Track/FINDINGS.md)
   - [`G-Track/findings/FINDINGS.md`](https://github.com/mpe-framework/glass-attention-workbench/blob/G-Track/findings/FINDINGS.md)
7. DCRP synthesis records, especially [`sandbox_024_S-061_fully_sealed_mechanism_confirmed_end_to_end.md`](./DCRP/sandbox_024_S-061_fully_sealed_mechanism_confirmed_end_to_end.md)

---

## Literature / Context Anchors

These references frame the workbench. They are not substitutes for the experiments reported above.

- **Lo, Sadrzadeh & Mansfield** — language-model contextuality and Bell/contextuality framing.
- **Geiger et al.** — causal abstraction and distributed alignment framing for mechanistic interpretability.
- **Pacela et al.** — motivation for moving beyond static probing toward direct geometric/mechanistic measurement.

---

*Applied Categorical Physics Workbench | Troy Teno | 2026 | Open Access*