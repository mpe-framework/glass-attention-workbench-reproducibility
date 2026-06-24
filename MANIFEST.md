# Reproducibility Archive Manifest

**Paper:** Value Projection Failure in a Near-Cancellation Regime: A Mechanistic Account of Compositional Generalization Failure in T5-small  
**Source private repo:** `mpe-framework/glass-attention-workbench`  
**Source commit:** `2c9f0432410cf69325d956cbaa24c4d6a7803b06` (post-PR #47)  
**Export method:** Whitelist copy. No git history transferred.

---

## What Is Included

### Paper source (`paper/`)

| Path | Description |
|------|-------------|
| `paper/main.tex` | Root LaTeX document |
| `paper/main.bbl` | Compiled bibliography |
| `paper/ARXIV_BUNDLE_MANIFEST.md` | arXiv submission manifest |
| `paper/bib/` | Reference database |
| `paper/sections/00_abstract.tex` – `paper/sections/09_discussion.tex` | All 10 paper sections |
| `paper/data/fig01_failure_rates.csv` – `paper/data/fig07_two_track_near_cancellation.csv` | Source data for all 7 figures |
| `paper/scripts/make_fig01_failure_rates.py` – `paper/scripts/make_fig07_two_track_near_cancellation.py` | Figure-generation scripts |

### Figures (`paper/figures/`)

7 figures in both PDF and PNG format (fig01–fig07).

### G-track methods findings (`findings/`)

| File | Description |
|------|-------------|
| `findings/METHODS_NOTE_G052.md` | G-052 norm-asymmetry / parallax-lever note — retired calibration (redacted) |
| `findings/METHODS_NOTE_G053.md` | G-053 layer-peak amplification — null result with mechanistic explanation |
| `findings/METHODS_NOTE_G054.md` | G-054 collapse conditions / phase diagram — phase transition and super-amplification zone |
| `findings/METHODS_REPORT_G054.md` | G-054 full methods report |
| `findings/METHODS_REPORT_G055.md` | G-055 full methods report (redacted) |
| `findings/METHODS_REPORT_G056.md` | G-056 full methods report (redacted) |
| `findings/METHODS_REPORT_G057.md` | G-057 full methods report (redacted) |

### G-track experiment scripts (`workbench/experiments/g_track_export/`)

| File | Description |
|------|-------------|
| `G-054_PHASE_DIAGRAM.py` | Phase diagram sweep over norm ratio and angle |
| `G-055_GLOBAL_HEAD_CAUSAL.py` | Causal intervention on global-routing head |
| `G-056_SUPPRESSIVE_HEAD_CAUSAL.py` | Causal intervention on suppressive head |
| `G-057_NEAR_CANCELLATION.py` | Two-track near-cancellation sweep |

### S-track experiment scripts (`workbench/experiments/s_track_export/`)

17 scripts: S-045 through S-062 (including S-050b and S-059b as corrected replacements for S-050 and S-059). Formatted for Colab-compatible execution.

### Environment and meta

| File | Description |
|------|-------------|
| `requirements.txt` | Python dependency pinning |
| `README.md` | Repository overview |
| `MANIFEST.md` | This file |
| `RETIREMENTS_AND_METHOD_LESSONS.md` | Retired claims and method lessons |
| `REPRODUCIBILITY.md` | Reproduction guide |
| `LICENSE` | MIT license |

---

## What Is Not Included

| Excluded path | Reason |
|---------------|--------|
| `FINDINGS.md` | Private running findings log |
| `EXPERIMENT_INDEX.md` | Private experiment index |
| `PRIOR_ART.md` | Private prior-art tracking |
| `paper/sections/10_limitations.tex` | Draft section not included in submitted paper |
| `paper/sections/11_future_work.tex` | Draft section not included in submitted paper |
| `paper/notes/` | Private working notes directory |
| `findings/METHODS_REPORT_G052.md` | Contains retired R²=1.0 parallax lever calibration claim; excluded in full — see `RETIREMENTS_AND_METHOD_LESSONS.md` |
| `findings/METHODS_REPORT_G039.md` – `findings/METHODS_REPORT_G051.md` | Pre-G-054 intermediate methods reports (superseded) |
| `findings/METHODS_REPORT_G042_G051.md` | Combined pre-G-054 report (superseded) |
| `findings/METHODS_REPORT_G053.md` | Intermediate report; findings superseded by G-054 report |

---

## Redaction Rules

Three classes of content were removed from exported findings files before publication:

**1. Sargsyan attribution.** Any mention of "Sargsyan's theorem," "softmax attention is not a monoidal functor," or related categorical phrasing is replaced with:

> `[Sargsyan attribution removed in public export — see RETIREMENTS_AND_METHOD_LESSONS.md]`

**2. Retired parallax-lever calibration claim.** Any reference to the retired G-052 parallax-lever calibration (its logit-gap formula and toy-geometry fit) is replaced with:

> `[retired parallax-lever calibration claim removed in public export — see RETIREMENTS_AND_METHOD_LESSONS.md]`

**3. Internal dev-file paths.** References to internal sandbox notebook paths (`sandbox_0XX`) are replaced with:

> `[internal dev-file path removed in public export]`
