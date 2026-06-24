# Public Repository Manifest

**Glass Attention Workbench — Reproducibility Export**  
Last updated: 2026-06-24

---

## Root Files

| File | Description |
|------|-------------|
| `README.md` | Repository overview and quick-start |
| `MANIFEST.md` | This file |
| `RETIREMENTS_AND_METHOD_LESSONS.md` | Retired experiments and method lessons |
| `REPRODUCIBILITY.md` | Detailed reproduction instructions |
| `requirements.txt` | Python dependencies |

---

## Redaction Markers

Three types of redaction markers appear in `findings/` files:

| Marker | Meaning |
|--------|---------|
| `[Sargsyan attribution removed in public export — see RETIREMENTS_AND_METHOD_LESSONS.md]` | Attribution to a non-author collaborator removed |
| `[retired parallax-lever calibration claim removed in public export — see RETIREMENTS_AND_METHOD_LESSONS.md]` | G-052 formula reference removed (see retirement rationale) |
| `[internal dev-file path removed in public export]` | Internal path not present in this repository |

---

## Findings

### Included

| File | Description | Redactions |
|------|-------------|------------|
| `findings/METHODS_NOTE_G052.md` | Stub — original note retired from evidentiary chain | Full replacement |
| `findings/METHODS_NOTE_G053.md` | G-053 methods note: layer-5 amplification | None |
| `findings/METHODS_NOTE_G054.md` | G-054 methods note: phase diagram | None |
| `findings/METHODS_REPORT_G054.md` | G-054: Phase diagram (three-regime collapse) | None |
| `findings/METHODS_REPORT_G055.md` | G-055: Global head causal diagnostic | None |
| `findings/METHODS_REPORT_G056.md` | G-056: Suppressive head causal test | Parallax (×2) |
| `findings/METHODS_REPORT_G057.md` | G-057: Near-cancellation demonstration | Parallax (×3) |
| `findings/METHODS_REPORT_S043.md` | S-043: Baseline failure audit | None |
| `findings/METHODS_REPORT_S043_S051.md` | S-043 through S-051 omnibus | None |
| `findings/METHODS_REPORT_S044.md` | S-044 | None |
| `findings/METHODS_REPORT_S045.md` | S-045 | None |
| `findings/METHODS_REPORT_S046.md` | S-046 | None |
| `findings/METHODS_REPORT_S047.md` | S-047 | None |
| `findings/METHODS_REPORT_S048.md` | S-048 | None |
| `findings/METHODS_REPORT_S049.md` | S-049 | None |
| `findings/METHODS_REPORT_S050.md` | S-050 | None |
| `findings/METHODS_REPORT_S050b.md` | S-050b (corrected tokenization run) | None |
| `findings/METHODS_REPORT_S051.md` | S-051: Norm causal | None |
| `findings/METHODS_REPORT_S055.md` | S-055: L3 cluster | None |
| `findings/METHODS_REPORT_S056.md` | S-056: L3 head classification | None |
| `findings/METHODS_REPORT_S057.md` | S-057: L3H4 characterization | Parallax (×1) |
| `findings/METHODS_REPORT_S058.md` | S-058: Activation patching (method lesson) | None |
| `findings/METHODS_REPORT_S059.md` | S-059: Logit decomp V0.1.0 (method lesson) | None |
| `findings/METHODS_REPORT_S059b.md` | S-059b: Logit decomp corrected (primary near-cancellation) | None |
| `findings/METHODS_REPORT_S060.md` | S-060: Causal ablation (method lesson) | None |
| `findings/METHODS_REPORT_S061.md` | S-061: Rank-1 W_V repair (PRIMARY REPAIR) | None |
| `findings/METHODS_REPORT_S062.md` | S-062: Repair specificity check | None |

### Excluded

| File | Reason |
|------|--------|
| `findings/METHODS_REPORT_G039.md` — `G051.md` | Pre-paper development; not in evidentiary chain |
| `findings/METHODS_REPORT_G042_G051.md` | Omnibus for excluded development experiments |
| `findings/METHODS_REPORT_G053.md` | Superseded by G-054 phase diagram |
| `findings/FINDINGS.md` | Internal summary document |
| `findings/EXPERIMENT_INDEX.md` | Internal index |
| `findings/PRIOR_ART.md` | Internal reference document |

---

## Experiment Scripts

### G-track (`workbench/experiments/g_track_export/`)

| File | Experiment |
|------|-----------|
| `G-054_PHASE_DIAGRAM.py` | G-054: Phase diagram |
| `G-055_GLOBAL_HEAD_ABLATION.py` | G-055: Global head causal diagnostic |
| `G-056_SUPPRESSIVE_HEAD_CAUSAL_TEST.py` | G-056: Suppressive head causal test |
| `G-057_NEAR_CANCELLATION.py` | G-057: Near-cancellation demonstration |

### S-track (`workbench/experiments/s_track_export/`)

| File | Experiment |
|------|-----------|
| `S-043_FAILURE_AUDIT.py` | S-043: Baseline failure audit |
| `S-044_ENCODER_OOD.py` | S-044 |
| `S-045_DECODER_ATTENTION.py` | S-045 |
| `S-046_CROSS_ATTENTION.py` | S-046 |
| `S-047_HEAD_ABLATION.py` | S-047 |
| `S-048_VALUE_PROJECTION.py` | S-048 |
| `S-049_WV_CHARACTERIZATION.py` | S-049 |
| `S-050b_WV_CORRECTION.py` | S-050b |
| `S-051_NORM_CAUSAL.py` | S-051 |
| `S-055_L3_CLUSTER.py` | S-055 |
| `S-056_L3_HEAD_CLASSIFICATION.py` | S-056 |
| `S-057_L3H4_CHARACTERIZATION.py` | S-057 |

S-058 through S-062: Colab notebooks only — no scripts included; see `findings/` for sealed results.

---

## Paper

| Path | Description |
|------|-------------|
| `paper/main.tex` | Main LaTeX source |
| `paper/main.bbl` | Bibliography (processed) |
| `paper/bib/references.bib` | BibTeX source |
| `paper/sections/00_abstract.tex` | §0 |
| `paper/sections/01_introduction.tex` | §1 |
| `paper/sections/02_related_work.tex` | §2 |
| `paper/sections/03_theoretical_framework.tex` | §3 |
| `paper/sections/04_experimental_setup.tex` | §4 |
| `paper/sections/05_results.tex` | §5 |
| `paper/sections/06_two_track_convergence.tex` | §6 |
| `paper/sections/07_repair_and_validation.tex` | §7 |
| `paper/sections/08_discussion.tex` | §8 |
| `paper/sections/09_conclusion.tex` | §9 |
| `paper/figures/fig_born_filter_trace.pdf` | Fig. 1 |
| `paper/figures/fig_cross_attn_heatmap.pdf` | Fig. 2 |
| `paper/figures/fig_phase_diagram.pdf` | Fig. 3 |
| `paper/figures/fig_two_track.pdf` | Fig. 4 |
| `paper/figures/fig_repair_margin.pdf` | Fig. 5 |
| `paper/figures/fig_logit_decomp.pdf` | Fig. 6 |
| `paper/figures/fig_cancellation_structure.pdf` | Fig. 7 |
| `paper/data/` | CSV files used by figure scripts |
| `paper/ARXIV_BUNDLE_MANIFEST.md` | arXiv submission file list |

---

*Glass Attention Workbench · mpe-framework/glass-attention-workbench-reproducibility · 2026*
