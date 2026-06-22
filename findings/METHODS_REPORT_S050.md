# Methods Report — S-050 / S-050b: Action-Slot Geometry

**Applied Categorical Physics Workbench | Troy Teno | May 2026**
*Status: Sealed (S-050b). One-line result: K2 fires — I_WALK is Euclidean-furthest, not closest; in the norm-dominated regime, Euclidean distance is invalid; cosine proximity confirms I_WALK is the only action token with positive hidden-state alignment in fail cases; two-stage selection mechanism established (frequency suppresses I_JUMP; norm+cosine select I_WALK among equal-frequency tokens).*

*Note: S-050 (V0.1.0) was a buggy run. A tokenization error mapped all action tokens to the shared first subword (proxy token ID=27), making all four action embeddings identical. The training frequency table was not affected by the bug and is re-confirmed here. S-050b (this finding) corrects the tokenization and is the canonical result. The two experiments share this report.*

---

## The Question

S-049 identified substituted_walk as the dominant failure mode: T5 generates (I_TURN_LEFT I_WALK) × 4 for "jump around left" — correct structure, wrong primitive. Among the four action tokens (I_JUMP, I_WALK, I_RUN, I_LOOK), why specifically I_WALK?

At the step where the action token is selected (the "divergence point" — the 3rd subword of each action token, where they diverge from the shared prefix [27, 834, div_token]), two candidate explanations exist:

1. **Frequency:** I_JUMP appeared 17× less often than the others in training (1,467 occurrences vs 25,350 for I_WALK, I_RUN, I_LOOK). This suppresses I_JUMP via softmax frequency effects, but it does not explain why I_WALK wins over I_RUN and I_LOOK (equal frequency).

2. **Geometry:** The decoder's hidden state at the divergence step is geometrically closer to I_WALK's output embedding. Troy pre-registered this as H2 (Euclidean proximity).

S-050b measures both by capturing the decoder's hidden state at the divergence point and comparing distances and cosines to each action token's divergence-point embedding.

---

## Design

**Model:** S-043 checkpoint. Full generation with hidden-state capture.

**Groups:** N=30 substituted_walk fail examples (has_around=1, T5 generates I_WALK instead of I_JUMP at action slots, n_jump=0, n_walk≥1), SEED=42. N=25 has_around=1 + correct success examples, SEED=42.

**Divergence-point identification:** Each action token tokenizes as 4 subwords sharing prefix [27, 834]. The model chooses which action at subword index 2. Correct divergence-point IDs (corrected in S-050b): I_JUMP=683, I_WALK=12054, I_RUN=448, I_LOOK=5017. TURN_END_IDS={6245, 27262} (last subwords of I_TURN_LEFT and I_TURN_RIGHT).

Action slot detection: pattern [TURN_END, 27, 834, div_token] in the generated sequence. N=227 fail divergence slots, N=144 success slots measured.

**Measurements:**
1. Token probabilities at divergence step (P(I_JUMP), P(I_WALK), P(I_RUN), P(I_LOOK))
2. Euclidean distance from decoder hidden state to each divergence-point embedding
3. Cosine similarity between decoder hidden state and each divergence-point embedding
4. Embedding norms for each divergence-point embedding

---

## Results

**H1 (I_JUMP suppressed at fail slots): PASS.**
```
Fail:    P(I_JUMP)=0.00174  P(I_WALK)=0.91455  P(I_RUN)=0.01762  P(I_LOOK)=0.06608
Success: P(I_JUMP)=0.80599  P(I_WALK)=0.00489  P(I_RUN)=0.13194  P(I_LOOK)=0.05557
MWU p=0.0000
```

**H2 (Euclidean proximity — I_WALK closest): FAIL — K2 fires.**
```
Euclidean distance to divergence-point embeddings (mean over fail slots):
I_RUN = 407.66   I_JUMP = 423.92   I_LOOK = 493.97   I_WALK = 520.12
```
I_WALK is the **furthest** action token from the decoder hidden state in Euclidean space, not the closest.

**H2 (cosine proximity — I_WALK directionally closest): PASS.**
```
cos(h_div, embed(div_token)) in fail cases:
I_WALK = +0.024   I_LOOK = −0.005   I_JUMP = −0.005   I_RUN = −0.009
```
I_WALK has the **only positive** cosine alignment at fail action slots. In success cases, I_JUMP moves to the positive position (+0.008) and I_WALK goes negative (−0.002).

**Logit computation (dominant mechanism):**
```
I_WALK: mean logit = +146.35  (cos=+0.024, ‖embed‖=520)
I_JUMP: mean logit =  −23.07  (cos=−0.005, ‖embed‖=424)
I_LOOK: mean logit =  −31.15  (cos=−0.005, ‖embed‖=494)
I_RUN:  mean logit =  −45.52  (cos=−0.009, ‖embed‖=407)
```
logit = cos(h, e) · ‖h‖ · ‖e‖. The ~192-unit I_WALK advantage collapses the softmax to P(I_WALK)=0.91.

**Embedding norms (divergence-point):**
```
‖embed(I_WALK div)‖ = 520.25   (largest)
‖embed(I_LOOK div)‖ = 493.75
‖embed(I_JUMP div)‖ = 423.68
‖embed(I_RUN  div)‖ = 407.36   (smallest)
```

**Training frequency (re-confirmed from S-050):**
```
I_JUMP  = 1,467 target occurrences  (0.79%)
I_WALK  = 25,350                     (13.58%)
I_RUN   = 25,350                     (13.58%)
I_LOOK  = 25,350                     (13.58%)
```

---

## Interpretation

**Why Euclidean fails:** The decoder hidden state norm at the divergence step is ‖h_div‖ ≈ 12 (inferred from logit = cos·‖h‖·‖e‖ with observed values). The embedding norms are ~400–520. Since ‖h_div‖ << ‖embed‖, Euclidean distance ‖h_div − embed‖ ≈ ‖embed‖. The Euclidean winner is just the smallest-norm embedding (I_RUN = 407), not the token whose embedding direction is closest to h_div. In this norm-dominated regime, Euclidean distance is not a valid proximity instrument. Cosine is the correct measure.

[retired parallax-lever calibration claim removed in public export — see RETIREMENTS_AND_METHOD_LESSONS.md]

**Two-stage mechanism:**
1. Frequency suppresses I_JUMP to P≈0.002 (17× less common in training)
2. Among equal-frequency tokens, largest norm (I_WALK=520) × only positive cosine (+0.024) produces logit=+146 vs I_RUN logit=−46

Both stages are necessary. Frequency alone explains why not I_JUMP. Geometry explains why I_WALK wins over I_RUN and I_LOOK.

---

## Implication for the Other Track

The norm-amplification mechanism (‖I_WALK‖=520 vs ‖I_JUMP‖=424 producing a 192-unit logit gap) is the S-track's most concrete mechanistic finding At toy scale: design a vocabulary where two tokens have equal training frequency but one has a larger embedding norm from the training statistics. Confirm that the large-norm token is preferentially selected even when the model's hidden state is not directionally closer to it. This tests whether the parallax mechanism is a general property of frequency-driven training or specific to T5-small's fine-tuning.

[retired parallax-lever calibration claim removed in public export — see RETIREMENTS_AND_METHOD_LESSONS.md]

---

*Applied Categorical Physics Workbench | Troy Teno | May 2026 | All work open access*
*Numbers match FINDINGS.md F-050. Do not revise post-hoc.*
