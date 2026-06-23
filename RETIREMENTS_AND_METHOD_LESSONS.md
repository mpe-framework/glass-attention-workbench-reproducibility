# Retirements and Method Lessons

This document records claims and methods that were explored during the project but are
**not** part of the submitted findings. It is included in the reproducibility archive
to document negative results and methodological dead-ends that influenced the paper's
experimental design.

---

## 1. S-059 V0.1.0: Logit Decomposition without LayerNorm Correction

**Status:** Retired. Superseded by S-059b.

**What happened:** The initial logit decomposition script (S-059 V0.1.0) decomposed
residual-stream contributions to output logits without accounting for the final LayerNorm.
This produces numerically plausible-looking attribution values but conflates the LayerNorm
rescaling with the underlying direction geometry.

**Lesson:** Logit decomposition in T5 must be applied to the LayerNorm-normalized
residual stream, not the raw residual. S-059b corrects this and is the authoritative
decomposition script in this archive.

---

## 2. S-058: Activation Patching Confound

**Status:** Retired as a causal attribution method. Script retained for reference.

**What happened:** Activation patching on intermediate representations produced large
apparent causal effects, but these were partially driven by distributional shift: patching
in activations from a different sequence type changes the activation distribution, not
only the computation. The head-ablation paradigm used in S-060 and S-061 avoids this
confound.

**Lesson:** Activation patching overstates causal strength when the patch source and
target differ in sequence structure. Use ablation-to-mean or targeted projection removal
instead. S-058 is included in the archive as an exported script but its causal estimates
should not be read as authoritative.

---

## 3. S-050 V0.1.0: Tokenization/Classifier Bug in Action Slot Geometry

**Status:** Retired. Superseded by S-050b.

**What happened:** The initial action-slot geometry script (S-050) used the wrong
tokenization when constructing classifier representations for multi-token action words.
This produced misleading clustering results for slots covering more than one token.

**Lesson:** Action slots spanning multiple tokens must aggregate per-token classifier
representations before computing geometry. S-050b corrects this and is the authoritative
slot geometry script in this archive.

---

## 4. G-052: Parallax Lever Calibration — Retired Claim

**Status:** Claim retired. The G-052 experimental script and phase-diagram machinery are
sound; the *calibration formula* is not.

**What happened:** G-052 identified that the dominant token's attention weight collapses
when its angle to the hidden state exceeds a threshold that depends on the norm ratio N.
An initial calibration suggested an exact closed-form formula relating the logit gap to
N and the angle θ, giving a perfect fit on a small number of hand-selected examples.

**Why it is retired:** The formula was over-fit to training examples. The correct
characterization — that collapse occurs at high norm ratio and large angular separation —
is preserved in the paper's phase diagram (§8). The specific closed-form calibration
equation is not.

**What this affects in the archive:** `findings/METHODS_REPORT_G052.md` is excluded
entirely from this archive. References to the formula in other findings files are
replaced with the `[retired parallax-lever calibration claim removed in public export]`
redaction marker. `findings/METHODS_NOTE_G052.md` is included with the parallax-lever
formula passage removed; the remaining phase-diagram observations are sound and are
reflected in the paper.
