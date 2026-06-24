# Glass Attention Workbench — Reproducibility Export

**Applied Categorical Physics Workbench · Troy Teno · 2026**

This repository is a curated public export of the Glass Attention Workbench: a mechanistic
interpretability study of T5-small on the SCAN `add_prim_jump` compositional generalization
benchmark. It contains sealed experiment results, experiment scripts, and the paper source
bundle.

---

## What This Repository Contains

| Directory | Contents |
|---|---|
| `findings/` | Sealed methods reports for all public experiments (S-043 through S-062; G-054 through G-057) |
| `workbench/experiments/s_track_export/` | S-track experiment scripts (S-043 through S-057) |
| `workbench/experiments/g_track_export/` | G-track experiment scripts (G-054 through G-057) |
| `paper/` | LaTeX source, bibliography, figure scripts, and figure CSVs |

See `MANIFEST.md` for a complete file list.

---

## Primary Finding

T5-small fails on compositional jump commands (56.8% failure rate on SCAN `add_prim_jump`)
via a **near-cancellation mechanism**: cross-attention contributes −171.25 logit units while
the FFN contributes +165.25, leaving a net margin of −6.0 units (28.5× reconstruction fraction).

The primary repair is a **rank-1 correction to W_V at L4H6 and L5H5** (experiment S-061):
- Joint Δmargin: **+2.902** (fail→success direction)
- Flip rate: **16/30** examples (α=1.0 scaling)
- Repair specificity confirmed on n=2 jump-around success examples (S-062)

Two-track convergence: the G-track (NumPy toy, PPMI–SVD embeddings) independently
reproduced the same failure-mode structure — value-direction miscalibration,
near-cancellation, and compensatory over-correction — under fully controlled conditions
with no shared weights or architecture.

---

## Retired Experiments

Activation patching (S-058) was retired when a K2 confound was detected: all patch arms
produced 100% flip regardless of which head was patched, indicating that donor-context
injection was the causal agent, not head-specific signal. The rank-1 W_V repair (S-061)
is the valid repair experiment.

See `RETIREMENTS_AND_METHOD_LESSONS.md` for the full list of retired experiments and
method lessons.

---

## Reproducing the Experiments

See `REPRODUCIBILITY.md` for detailed setup instructions.

**Quick start (G-track, NumPy-only, no GPU required):**
```bash
cd workbench/experiments/g_track_export
python G-054_PHASE_DIAGRAM.py
python G-055_GLOBAL_HEAD_ABLATION.py
python G-056_SUPPRESSIVE_HEAD_CAUSAL_TEST.py
python G-057_NEAR_CANCELLATION.py
```

**Quick start (S-track, requires T5-small checkpoint + SCAN):**
```bash
cd workbench/experiments/s_track_export
python S-043_FAILURE_AUDIT.py
# See REPRODUCIBILITY.md for the full sequence
```

---

## Paper

LaTeX source and figure CSVs are in `paper/`. The arXiv submission bundle is documented
in `paper/ARXIV_BUNDLE_MANIFEST.md`.

---

*Glass Attention Workbench · mpe-framework/glass-attention-workbench-reproducibility · 2026*
