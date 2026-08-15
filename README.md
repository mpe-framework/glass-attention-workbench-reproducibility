# Value Projection Failure in a Near-Cancellation Regime

Reproducibility export for a mechanistic interpretability study of value-projection failure
in T5-small fine-tuned on the SCAN `add_prim_jump` compositional-generalization split.

**arXiv:** [link once live]

---

## What the paper studies

T5-small fine-tuned on SCAN `add_prim_jump` fails on 56.8% of compound-jump test commands
(4376/7706), generating `I_WALK` where the target requires `I_JUMP`.

The study traces this failure through three stages:

**Locate.** Two decoder cross-attention heads, L4H6 and L5H5, attend strongly to the `jump`
encoder token but contribute in the `I_WALK` direction at the action slot. Cross-attention
routing is not the primary discriminator (p=0.9208, S-049); the failure is in what the decoder
reads back, not where it looks.

**Explain.** Cross-attention and FFN contributions nearly cancel: cross-attention pushes −171.25
logit units toward `I_WALK`; FFN/embedding correction pushes +165.25 back, leaving a net margin
of −6.005 (28.52× reconstruction fraction, S-059b). Output embedding norm asymmetry
(‖I_WALK‖ ≈ 520 vs ‖I_JUMP‖ ≈ 424) amplifies this small pro-walk residual to P(I_WALK) = 0.914.

**Repair.** A rank-1 modification to W_V at L4H6 and L5H5 — redirecting the mapping of the
`jump` encoder representation from a pro-walk to a pro-jump value direction — shifts the mean
fail-group logit margin by +2.902 units and flips 16/30 sampled failures at α=1.0,
using ~8% of each head's W_V Frobenius norm (S-061). The dose-response is monotonically linear.

A parallel controlled geometry track (G-track) independently demonstrates the same structural
phenomena — near-cancellation, value-direction substitution, and causal/diagnostic head
distinction — in a transparent NumPy toy model where all parameters are designed.

---

## Strongest bounded claim

A rank-1 modification to W_V at L4H6 and L5H5 jointly improves the fail-group logit margin
at every tested α value and flips 16/30 sampled failures at α=1.0 (S-061).
The dose-response is monotonically linear. The mean fail-group margin remains pro-walk after
the intervention; the repair is partial and has been tested on one sampled group.
This is a claim about this model at this checkpoint on this task.
It does not extend to other models, checkpoints, or tasks.

See `RETIREMENTS_AND_METHOD_LESSONS.md` for retired claims and method lessons, including
S-058 (activation patching confound), S-059 V0.1.0 (pre-LayerNorm error), and the
parallax lever retirement.

---

## Repository structure

```
paper/
  main.tex, main.bbl, bib/references.bib
  sections/   — 10 LaTeX section files (00_abstract through 09_discussion)
  figures/    — 7 PDFs + 7 PNGs
  data/       — 7 CSVs (one per figure; single source of truth for plotted values)
  scripts/    — 7 Python scripts to regenerate figures from CSVs
  ARXIV_BUNDLE_MANIFEST.md
findings/
  METHODS_REPORT_S043_S051.md through METHODS_REPORT_S062.md  — 9 sealed S-track reports
  METHODS_NOTE_G052.md, METHODS_NOTE_G053.md, METHODS_NOTE_G054.md
  METHODS_REPORT_G054.md through METHODS_REPORT_G057.md  — 7 G-track reports
workbench/
  experiments/   — scripts for G-054–G-057 and S-045–S-062
REPRODUCIBILITY.md
RETIREMENTS_AND_METHOD_LESSONS.md
requirements.txt
LICENSE
.gitignore
```

---

## How to regenerate figures

Every paper figure regenerates from a committed CSV — no experiment re-execution needed:

```bash
pip install -r requirements.txt
cd paper/scripts
python make_fig01_failure_rates.py
python make_fig02_decoder_routing_summary.py
python make_fig03_value_direction_evidence.py
python make_fig04_near_cancellation.py
python make_fig05_phase_diagram.py
python make_fig06_dose_response.py
python make_fig07_two_track_near_cancellation.py
```

Each script reads from `../data/<name>.csv` and writes to `../figures/<name>.pdf/.png`.

## How to re-run experiments

G-track (G-054–G-057): Pure NumPy, CPU only, no checkpoint needed.

```bash
cd workbench/experiments/g_track_export
python G-054_PHASE_DIAGRAM.py
python G-055_GLOBAL_HEAD_CAUSAL.py
python G-056_SUPPRESSIVE_HEAD_CAUSAL.py
python G-057_NEAR_CANCELLATION.py
```

S-track (S-045–S-062): Requires the T5-small SCAN checkpoint (~300MB, not included). See `REPRODUCIBILITY.md` for setup instructions. S-058–S-062 are exported as Colab-oriented scripts; canonical numerical results are in the corresponding sealed methods reports in `findings/`.

## Limitations

* All results are on one T5-small model at one checkpoint on one SCAN split. Generalization to other models or tasks is not claimed.
* The rank-1 W_V repair (S-061) improves 16/30 sampled failures; 14 do not flip. Broad specificity (non-`jump` `around` commands) was not tested.
* Several intermediate approaches were retired mid-study. See `RETIREMENTS_AND_METHOD_LESSONS.md`.

This public repository is a curated export of a larger private lab archive. See `MANIFEST.md` for what is included, what is excluded, and the redaction rules applied to the sealed methods reports.

## Status
This is a sealed reproducibility archive for the paper. Issues and discussions are disabled.
For questions, see the paper or contact the authors via [email/etc].
