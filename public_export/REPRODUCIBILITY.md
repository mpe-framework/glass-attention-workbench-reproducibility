# Reproducibility Guide

**Glass Attention Workbench — Reproducibility Export**

---

## Environment Setup

```bash
pip install -r requirements.txt
```

The SCAN dataset is downloaded automatically by the S-track scripts via HuggingFace `datasets`.

For S-track experiments (S-043 through S-057), a T5-small checkpoint fine-tuned on SCAN
`add_prim_jump` must be provided. See paper §4 for training details. Set `CHECKPOINT_PATH`
at the top of each script.

G-track experiments (G-054 through G-057) are **NumPy-only** and do not require a GPU,
T5-small checkpoint, or SCAN dataset.

---

## G-track Experiments (NumPy-only, no GPU required)

G-track experiments are self-contained NumPy simulations using controlled toy models with
explicit PPMI–SVD embeddings. They do not load T5-small or SCAN data.

```bash
cd workbench/experiments/g_track_export
python G-054_PHASE_DIAGRAM.py
python G-055_GLOBAL_HEAD_ABLATION.py
python G-056_SUPPRESSIVE_HEAD_CAUSAL_TEST.py
python G-057_NEAR_CANCELLATION.py
```

Expected runtimes: each script completes in under 60 seconds on CPU.

### G-054: Phase Diagram (SEED=54)

Sweeps `FIN_WEIGHT` to produce the three-regime collapse structure (slow-collapse,
sharp-collapse, no-collapse). Reference result: `findings/METHODS_REPORT_G054.md`.

### G-055: Global Head Causal Diagnostic (SEED=55)

Confirms a global attention head is not causally responsible for the failure (4% ablation
change). Near-identity baseline established. Reference result: `findings/METHODS_REPORT_G055.md`.

### G-056: Suppressive Head Causal Test (SEED=56)

FIN-inversion mechanism confirmed above the FIN-annihilation threshold (α=1.0). 42% ablation
change versus 4% baseline at α=2.0. Reference result: `findings/METHODS_REPORT_G056.md`.

### G-057: Near-Cancellation Demonstration (SEED=57)

Gamma sweep confirms reconstruction fraction diverges as γ→1.0. FFN over-correction at
γ=2.0 reproduces the G-056 FIN-inversion result (51.3% ablation change).
Reference result: `findings/METHODS_REPORT_G057.md`.

---

## S-track Experiments (requires T5-small checkpoint + SCAN)

S-track experiments require:
- Python ≥ 3.9
- PyTorch ≥ 2.0
- HuggingFace `transformers`, `datasets`, `sentencepiece`
- T5-small checkpoint fine-tuned on SCAN `add_prim_jump`

Expected runtimes: 5–30 minutes per script on GPU; longer on CPU.

Scripts for S-043 through S-057 are in `workbench/experiments/s_track_export/`:

```bash
cd workbench/experiments/s_track_export
python S-043_FAILURE_AUDIT.py        # Baseline failure audit
python S-044_ENCODER_OOD.py
python S-045_DECODER_ATTENTION.py
python S-046_CROSS_ATTENTION.py
python S-047_HEAD_ABLATION.py
python S-048_VALUE_PROJECTION.py
python S-049_WV_CHARACTERIZATION.py
python S-050b_WV_CORRECTION.py
python S-051_NORM_CAUSAL.py
python S-055_L3_CLUSTER.py
python S-056_L3_HEAD_CLASSIFICATION.py
python S-057_L3H4_CHARACTERIZATION.py
```

See each script's header for `CHECKPOINT_PATH` and output file locations.

### Note on S-058 through S-062

Experiments S-058 through S-062 were conducted in Google Colab notebooks. No scripts are
included for these experiments. The canonical numerical results are the sealed methods
reports in `findings/`.

---

## Seeded Reproducibility

All experiments use fixed seeds (SEED=42 for S-track; SEED=54–57 for G-track). Exact
numerical reproduction depends on:
- For G-track: NumPy version (tested with NumPy ≥ 1.24)
- For S-track: PyTorch version, CUDA version, and checkpoint weights

Small floating-point differences may appear across hardware configurations; the sign and
direction of all reported effects should be stable.

---

## Figures

Figure scripts are in `paper/figures/`. Each script reads from committed CSV files in
`paper/data/` and does not require re-running experiments.

```bash
cd paper/figures
python fig_born_filter_trace.py
python fig_phase_diagram.py
python fig_two_track.py
python fig_repair_margin.py
python fig_logit_decomp.py
python fig_cancellation_structure.py
python fig_cross_attn_heatmap.py
```

---

*Glass Attention Workbench · mpe-framework/glass-attention-workbench-reproducibility · 2026*
