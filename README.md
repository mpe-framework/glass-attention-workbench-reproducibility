# Value Projection Failure in a Near-Cancellation Regime

Reproducibility export for a mechanistic interpretability study of value-projection failure in a T5 model fine-tuned on the SCAN add-prim-jump compositional-generalization split.

## What the paper studies

A fine-tuned T5 model that achieves near-perfect accuracy on SCAN add-prim-jump fails on a specific subset of novel jump-compound commands: it generates `I_WALK` where the gold target requires `I_JUMP`. The study traces this failure to three decoder attention heads (L4H6, L5H2, L5H5) that function as value-substitution heads — they attend to the jump encoder position but output residual-stream contributions aligned with `I_WALK` rather than `I_JUMP`. Causal patching of these heads' value contributions from matched success trajectories redirects the logit decision toward `I_JUMP` in the majority of patched fail examples.

## Strongest bounded claim

Activation patching at L4H6 + L5H2 + L5H5 simultaneously is causally sufficient to flip the action-slot logit decision from `I_WALK` toward `I_JUMP` in over 50% of matched fail examples (H1 and H2 passed). This is a causal claim about this model on this task under this patching procedure. It does not extend to other models or tasks. See `RETIREMENTS_AND_METHOD_LESSONS.md` for retired claims and method lessons.

## What is included

```
workbench/
  experiments/
    g_track_export/
      G-056_SUPPRESSIVE_HEAD_CAUSAL.py
    s_track_export/
      S-045_PER_TOKEN_BC.py
      S-057_L3H4_CHARACTERIZATION.py
      S-058_CAUSAL_PATCH_VALUE_GEOMETRY.py
findings/
  METHODS_REPORT_G055.md
  METHODS_REPORT_G056.md
  METHODS_REPORT_S057.md
  METHODS_REPORT_S059.md
  METHODS_REPORT_S061.md
REPRODUCIBILITY.md
RETIREMENTS_AND_METHOD_LESSONS.md
requirements.txt
```

## How to read this repo
1. Read `REPRODUCIBILITY.md` for a guided path through the experiments in order.
2. Each experiment script is sealed (do not re-run without reading the corresponding proposal). Scripts print structured output: measurement values, kill-condition checks, hypothesis verdicts.
3. Methods reports in `findings/` document the pre-registered hypotheses, measurement outcomes, and verdicts for each experiment.

## How to regenerate figures
Figures are produced inline during experiment runs. See the relevant `METHODS_REPORT_*.md` for which script and phase produces each figure. All scripts require the S-043 fine-tuned T5 checkpoint; see `REPRODUCIBILITY.md` for setup.

## How to inspect the methods reports
The `findings/` directory contains one methods report per key experiment. Each report records: the pre-registered hypotheses, the measurements taken (M1 through M6 or equivalent), kill-condition outcomes (K1, K2, K3), and hypothesis verdicts (PASS/FAIL).

## Limitations
- All results are on a single T5-small model fine-tuned on SCAN add-prim-jump. Generalization to other models or tasks is not claimed.
- Several intermediate hypotheses were retired mid-study. See `RETIREMENTS_AND_METHOD_LESSONS.md` for the full list and the reasons for each retirement.
- The parallax-lever calibration claim (G-052) was retired before the S-track causal work began and does not appear in the findings reported here.
