# Reproducibility Guide

This archive is self-contained: all figures can be regenerated from the included CSV
data, and all G-track and S-track experiment scripts are included for inspection and
re-execution.

---

## Requirements

```bash
pip install -r requirements.txt
```

Tested with Python 3.10. The **S-track** scripts require `torch`, `transformers`,
`datasets`, and `scikit-learn` (exact versions in `requirements.txt`). The **G-track**
scripts are controlled toy models — mostly pure NumPy on CPU (G-054 additionally builds a
small `transformers` toy model from config); they require **no** T5-small checkpoint and
**no** SCAN dataset. Figure-generation scripts require only `matplotlib` and `numpy` (CSV
reading uses the Python standard library).

---

## Figure Regeneration

All 7 paper figures can be regenerated from the archived CSV data without model access.

```bash
cd paper/scripts
python make_fig01_failure_rates.py
python make_fig02_decoder_routing_summary.py
python make_fig03_value_direction_evidence.py
python make_fig04_near_cancellation.py
python make_fig05_phase_diagram.py
python make_fig06_dose_response.py
python make_fig07_two_track_near_cancellation.py
```

Each script reads from `paper/data/figXX_*.csv` and writes to `paper/figures/`. The
pre-generated PDFs and PNGs are already included in `paper/figures/` and can be used
directly without re-running the scripts.

---

## G-Track Experiment Scripts

The four core G-track experiments are in `workbench/experiments/g_track_export/`:

| Script | Experiment |
|--------|-----------|
| `G-054_PHASE_DIAGRAM.py` | Phase diagram sweep over norm ratio and angle |
| `G-055_GLOBAL_HEAD_CAUSAL.py` | Causal intervention on global-routing head |
| `G-056_SUPPRESSIVE_HEAD_CAUSAL.py` | Causal intervention on suppressive head |
| `G-057_NEAR_CANCELLATION.py` | Two-track near-cancellation sweep |

These are self-contained controlled toy models: each constructs its own small in-process
model (pure NumPy, except G-054 which also builds a small `transformers` model from
config). They do **not** load T5-small and do **not** use the SCAN dataset. Expected
runtime: seconds to a few minutes per script on CPU; no GPU required.

```bash
python workbench/experiments/g_track_export/G-054_PHASE_DIAGRAM.py
```

---

## S-Track Experiment Scripts

17 S-track scripts are in `workbench/experiments/s_track_export/`, covering S-045
through S-062 (with S-050b and S-059b as corrected replacements for S-050 and S-059).
These scripts were formatted for Colab-compatible execution and can also be run locally.

Scripts S-058 through S-062 cover the causal geometry analysis, value-direction
intervention, and H4 characterization that underpin §6–§7 of the paper:

| Script | Content |
|--------|---------|
| `S-058_CAUSAL_PATCH_VALUE_GEOMETRY.py` | Activation patching on value geometry (see caveats in `RETIREMENTS_AND_METHOD_LESSONS.md`) |
| `S-059_LOGIT_DECOMPOSITION.py` | Logit decomposition V0.1.0 (retired; see `RETIREMENTS_AND_METHOD_LESSONS.md`) |
| `S-059b_LOGIT_DECOMPOSITION_LN_CORRECTED.py` | Corrected logit decomposition with LayerNorm |
| `S-060_CAUSAL_ABLATION.py` | Causal ablation of attention heads |
| `S-061_WV_GEOMETRY_INTERVENTION.py` | W_V geometry and projection intervention |
| `S-062_H4_SUCCESS_CHECK.py` | H4 success condition characterization |

---

## Paper Compilation

```bash
cd paper
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
```

The compiled `.bbl` file is already included. All 10 paper sections (00–09) are present
in `paper/sections/`. The bibliography source is in `paper/bib/`.

---

## G-Track Methods Notes

The `findings/` directory contains 7 methods notes and reports documenting the G-track
experimental arc (G-052–G-057). These are research logs, not polished prose. They record
what was run, what was found, and how the next experiment was designed.

Three classes of content are redacted from these files before publication. See
`MANIFEST.md` for the full redaction rules.

`findings/METHODS_REPORT_G052.md` is excluded entirely from this archive; see
`RETIREMENTS_AND_METHOD_LESSONS.md` item 4 for the reason.

---

## What Is Not Re-Executable from This Archive

- **Model weights** are not archived. All scripts download T5-small from Hugging Face
  (`google/t5-small`) at runtime.

- **SCAN dataset** is not archived. All scripts download it via `datasets` at runtime.

- **Random seeds** were not fixed in some early S-track scripts. Results are expected to
  be stable across seeds at the precision reported in the paper, but exact floating-point
  reproduction may differ.

- **Private working notes** (`paper/notes/`, `FINDINGS.md`, `EXPERIMENT_INDEX.md`,
  `PRIOR_ART.md`) are not included. They contain intermediate analysis that did not
  survive into the final paper.
