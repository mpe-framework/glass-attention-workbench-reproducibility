# Methods Report — S-047: Per-Bit Failure Decomposition

**Applied Categorical Physics Workbench | Troy Teno | May 2026**
*Status: Sealed. One-line result: "has_around" dominates failure prediction with lift=+0.648 and AUC=0.806; commands with "around" fail 87.3% of the time; the pre-registered dominant feature (has_after) showed essentially no individual predictive power (lift=+0.083).*

---

## The Question

S-046 confirmed that all 8 structural bits are linearly recoverable from T5's encoder, and that the raw encoder representation predicts failure at AUC=0.88. But which structural features actually drive failure? The K2 result from S-046 says the encoder encodes complexity that predicts failure — but "complexity" was measured as a composite. S-047 asks which individual structural bit is the strongest predictor of T5 getting the command wrong.

This matters for mechanistic understanding. If "has_around" is the dominant driver, the failure is about the four-cycle identity morphism — T5 has to repeat (TURN, ACTION) four times, filling the action slot with an OOD primitive each time. If "has_after" is the dominant driver, it is about argument-order reversal — a different compositional challenge. The pre-registration favored has_after (argument reversal) based on prior intuition about which compositional operation was hardest.

---

## Design

**Model:** S-043 checkpoint. No new model loading — analysis runs on precomputed encoder representations from S-046.

**Dataset:** 200 jump-compound examples. 111 failures (55.5% failure rate — slight sampling variation from S-046's 125/200 due to different example ordering). All 8 structural bits available per example.

**Three phases:**
1. Per-bit failure rates with chi-square significance tests
2. Per-probe-direction failure AUC (project 512-dim representation onto each probe weight vector, treating the projected scalar as a failure predictor)
3. Complexity score analysis: how does total number of active bits (0–8) correlate with failure rate?

---

## Results

**H1 (has_around achieves highest AUC, ≥0.75): CONFIRMED.** has_around probe direction achieves AUC = 0.806. Chi-square p = 0.0000. Per-bit failure rates:
```
has_around: fail% = 87.3% (has_around=1) vs 22.4% (has_around=0)   lift = +0.648
```

**H2 (complexity score correlates with failure): CONFIRMED.** Spearman r = 0.258, p = 0.0002. Score 4 → 39%, score 5 → 58%, score 6 → 68%, score 7 → 100%. The trend is real despite non-monotone behavior at score=2 (40%).

**H3 (has_after is dominant failure feature): FAIL.** has_after lift = +0.083, AUC = 0.572, p = 0.2968 — barely distinguishable from noise. The pre-registered prediction was wrong.

**H4 (has_after AND has_around → high failure): CONFIRMED.** has_after=1 AND has_around=1: 92.2% failure rate (N=51).

**Full per-bit breakdown:**
```
Bit             Fail%(=0)  Fail%(=1)   Lift    AUC    chi2 p
has_around         22.4%      87.3%  +0.648  0.806  0.0000  ← dominant
has_thrice         48.4%      61.7%  +0.133  0.610  0.0811
is_left            48.3%      58.5%  +0.102  0.558  0.2472
has_after          51.5%      59.8%  +0.083  0.572  0.2968
has_and            59.2%      52.0%  -0.072  0.535  0.3761
has_twice          60.9%      51.3%  -0.096  0.553  0.2264
has_opposite       67.7%      44.2%  -0.235  0.682  0.0014  ← negative predictor
has_direction      83.3%      54.6%  -0.287  0.514  0.3291
```

---

## Interpretation

"Around" means the command requires four repetitions of a turn-action cycle. "Jump around left" = (I_TURN_LEFT I_JUMP) × 4. Every "X around Y" command is the identity transformation in position-and-orientation space — you execute four quarter-turns and return to the starting position facing the same direction. T5 has learned this as a four-cycle identity morphism. What it cannot do is fill the action slot with the OOD primitive (I_JUMP) — it substitutes the trained default (I_WALK) instead.

"has_opposite" being a significant negative predictor (lift = −0.235, AUC = 0.682, p = 0.0014) is not a causal finding about "opposite" — it is a confound through mutual exclusivity with "around." Commands with "opposite" rarely have "around," so has_opposite=1 is largely a proxy for has_around=0. The low-failure group masquerades as an "opposite" effect.

The H3 failure is recorded as a real null. The pre-registration predicted argument-order reversal (has_after) as the dominant structural failure driver. The data said it is the four-cycle counting mechanism (has_around). Both are compositional challenges; the four-cycle identity morphism with an OOD primitive is the harder one for T5's decoder.

---

## Implication for the Other Track

The has_around result is the most important S-track finding for the geometry track to absorb: the production failure is not about arbitrary OOD compositional generalization — it is about the specific structure of repeating an (TURN, ACTION) cycle four times with an OOD token in the action slot. The geometry track's toy models should include a "counting cycle with OOD token" analog. The G-045 identity morphism finding (every "X around Y" is the identity natural transformation) is the categorical frame for this.

The has_opposite negative prediction is also relevant: in production data, structural features are not independent. A toy model that treats each feature as orthogonal will miss confound effects of this kind. This is a structural limitation of controlled toy experiments relative to the production data.

---

*Applied Categorical Physics Workbench | Troy Teno | May 2026 | All work open access*
*Numbers match FINDINGS.md F-047. Do not revise post-hoc.*
