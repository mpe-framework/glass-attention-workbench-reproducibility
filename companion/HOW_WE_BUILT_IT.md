# How We Built It
*Attention by hand, and what it showed us about a failure in T5-small*

**Troy Teno | November 2026**
*Companion document — Not Peer Reviewed. This is a guided walk-through of the archive, not a new result. Every number here is quoted from a sealed methods report in `../findings/`, or printed by `glass_head.py` when you run it.*

---

## I. Why Build It By Hand

Most people meet attention as one line of library code. You call it, a tensor comes back, and whatever happened inside is somebody else's business.

That works for building systems. It does not work for understanding them. If you want to know why a model fails, you have to be able to point at the place where it fails, and you cannot point at something you have never seen.

So this project started by building attention from nothing. Word vectors from counted text. Query, key and value projections. A softmax. A backward pass derived on paper and checked by brute force. No framework, only NumPy. The instrument was slow and small, and it was entirely visible.

I am a maintenance engineer, not a machine-learning researcher. I came to this by reading, and I built the instrument because I wanted to see the thing I was reading about. This document is the path I took, cleaned up so that you can take it in an afternoon.

The path has two halves. The first half is `glass_head.py`, a single file in this folder, which builds one attention head by hand and shows every number. The second half is the study in this archive, which used the same ideas to locate and partly repair a specific failure in a real model.

---

## II. Attention Is a Few Matrix Products and a Softmax

Run the script first.

```bash
pip install numpy
python glass_head.py
```

It finishes in a few seconds. Read along below.

**Words become vectors.** The script writes a tiny corpus of sentences about birds and machines, counts which words appear near which, and turns the counts into vectors with PPMI and SVD. PPMI scores each pair of words by how much more often they appear together than chance would predict. SVD then compresses those scores into a few directions. No one labels anything. Birds end up nearer to birds and machines nearer to machines, because birds keep bird company in the text. The geometry comes from the corpus.

**The task forces the head to look.** Each input is five words: *the SUBJECT verb the OBJECT*. The answer is whether the subject is a bird or a machine. The object is chosen at random, so half the time it disagrees with the subject. That is deliberate. A head that averages every word equally sees a bird and a machine and cancels them out. The script trains exactly that control, a head with attention frozen to uniform, and it scores at or below chance on the sentences where subject and object disagree. Averaging cannot do this task. Looking can.

**Order has to be supplied.** Attention by itself sees a set of vectors, not a sequence. *The heron watched the truck* and *the truck watched the heron* look identical to it. So every word vector gets a one-hot position code appended. This is the reason transformers carry positional encodings, and the toy makes the reason concrete: without positions, the task cannot be solved at all.

**The forward pass.** Every position produces three vectors: a query (what it is looking for), a key (what it offers to be found by) and a value (what it hands over if chosen).

```
Q = X W_Q      K = X W_K      V = X W_V
S = Q Kᵀ / √d                  scores: how well each query matches each key
A = softmax(S)                 each row becomes a probability distribution
Z = A V                        each position receives a weighted mix of values
```

The answer is read out at position 0, through one more matrix, `W_O`. Position 0 must learn to put its attention on position 1, where the subject is. Nothing tells it to.

**The backward pass.** This is the step most tutorials skip. For one row of attention weights `a = softmax(s)` and an upstream gradient `g`, the gradient into the scores is

```
dL/ds_k = a_k · ( g_k − Σ_j g_j a_j )
```

In plain language: a score's gradient is its weight times how much better than average its value turned out to be. That single line is what lets attention patterns move. Without it you can train the values and the readout, but the head keeps looking wherever it looked on the first day. The early development scripts, which are not part of this export, learned this the hard way. One version trained the values and the readout but left the query and key matrices fixed. The attention weights never moved, and the head converged to mean-pooling.

The script checks the hand-derived gradient against finite differences. It nudges single weights up and down, measures the loss directly, and compares. The relative error is on the order of one part in a hundred thousand. The derivation is right, and you do not have to take that on faith.

**Training.** Before training, attention from position 0 is spread almost evenly across the five words. After a few epochs it puts more than 90% of its weight on the subject. Test accuracy reaches 100%, including on the sentences where subject and object disagree. The gradient found the subject.

The instrument works. The measurement is printed.

---

## III. The Value Vector Carries the Answer

Once a head looks in the right place, what comes out is the value vector of the word it looks at. The attention pattern decides *where*. The value decides *what*.

That separation matters, because a head can look in exactly the right place and still read back the wrong thing. To see it, the last part of the script makes one deliberate change. It adds a rank-1 matrix to `W_V`, chosen so that the value vector for *heron* is moved toward the average value vector of the machines:

```
W_V  ←  W_V + α · u (v_target − v_heron)ᵀ,      u = x_heron / |x_heron|²
```

Nothing about where the head looks is changed. The heron's logit margin, which starts strongly on the BIRD side, falls in a straight line as α grows. It crosses zero near α = 0.5. From then on, the head reads a heron as a machine.

Then look at the other birds. They move too. Each of them shares about a fifth of its embedding direction with *heron*, and each absorbs about a fifth of the shift. They keep their answers, with less margin. The edit leaks along the geometry the corpus built. In a space where the neighbors overlapped more, the same edit would flip them as well. Specificity is something you measure, not something you assume.

Hold on to those three ideas: where a head looks, what it reads back, and how a rank-1 edit to `W_V` redirects the second without touching the first. They are the whole vocabulary of the study that follows.

---

## IV. The Real Failure

SCAN is a benchmark of simple commands that translate into action sequences: *jump twice* becomes `JUMP JUMP`. In the `add_prim_jump` split, the model sees the word *jump* only on its own during training, never inside a longer command. At test time it must handle *jump around left*, *jump twice after walk*, and so on. A fixed rule system using the same grammar gets every one right.

T5-small, fine-tuned on this split, fails 56.8% of the compound jump test commands (4,376 of 7,706). The failure has a consistent shape. Where the target requires `I_JUMP`, the model writes `I_WALK`. (Established: `../findings/METHODS_REPORT_S043_S051.md`)

The study asked where that substitution comes from, using the same three ideas as Section III.

**Where the heads look.** Two decoder cross-attention heads, layer 4 head 6 and layer 5 head 5, attend strongly to the *jump* token in the encoder. Routing does not separate the failures from the successes: the test returned p = 0.9208. The heads look at the right word in both cases. (Established: S-049)

**What they read back.** At the action slot, those same two heads contribute in the `I_WALK` direction. They look at *jump* and hand back something that points toward *walk*. This is the failure from Section III, found in a real model: correct attention, wrong value. (Established: S-050b, S-051, S-059b)

**Why a small push wins.** The logit decomposition shows the cross-attention contributions totaling −171.25 logit units toward `I_WALK`, and the feed-forward and embedding contributions pushing +165.25 back. The net margin is −6.005. Two large forces nearly cancel, and the leftover is small. A second factor finishes the job. The output embedding for `I_WALK` has a larger norm than the one for `I_JUMP`, about 520 against 424, which amplifies a small pro-walk residual into P(`I_WALK`) = 0.914. (Established: S-059b)

**The repair.** A rank-1 modification to `W_V` at the two heads, the same algebraic move as the heron edit, redirects the value read from the *jump* encoder state away from the pro-walk direction. At α = 1.0 it improves the mean fail-group margin by +2.90 logit units and flips 16 of 30 sampled failures to `I_JUMP`, while changing about 8% of each head's `W_V` Frobenius norm. The dose-response is monotonic and close to linear. (Established: S-061)

The repair is partial. Fourteen of the thirty sampled failures do not flip, and the mean fail-group margin stays on the walk side. The specificity test, whether the edit disturbs unrelated commands like *walk around*, was never run. Only two correct jump examples were checked for damage, and neither broke. Two is not enough to call the edit specific. (Established as indeterminate: S-062)

---

## V. The Glass Track

Alongside the T5 work ran a second track, the G-track, built in plain NumPy like `glass_head.py` but larger: small multi-layer toy transformers whose heads were assigned their roles by design. A designed model has one advantage a trained model never has. You know the right answer before you measure, so you can tell whether the instrument is telling the truth.

The G-track reproduced the structural shapes seen in T5 at a scale where they could be read directly. A phase diagram showed sharp transitions in how a head's attention collapses as one embedding component is dialed up and down (G-054). A diffuse global head turned out to be diagnostic but not causal: ablating it changed downstream behavior by 4%, the same as ablating a near-identity control (G-055). A head that actively subtracts a direction was causal, and the mechanism was inversion rather than suppression (G-056). The near-cancellation structure was present in the toy, but the toy's base signal was too small for most of the pre-registered thresholds, so most of those hypotheses returned nulls. The nulls are recorded as nulls (G-057).

The two tracks do not prove each other. They point at the same shapes from two directions, one where every parameter is visible and one where the failure is real.

---

## VI. What We Got Wrong

Several things were retired along the way, and the archive keeps them on purpose. Most are recorded in `../RETIREMENTS_AND_METHOD_LESSONS.md`.

The first logit decomposition ignored the final normalization layer. Its numbers looked plausible and were wrong by orders of magnitude. The lesson was to validate an instrument on a known case before using it on an unknown one. S-059b is the corrected version.

Activation patching, copying activations from one input into another, produced large apparent causal effects that were partly an artifact of the patch itself. The study moved to weight-level edits, which let the model's own normalization run unchanged.

Ablating the largest heads in the decomposition made the failure *worse*, not better. Those heads were compensating, not causing. In a near-cancellation regime, a big contribution is not the same as blame, and the sign of an ablation has to be checked before it is read. (S-060, summarized in `../findings/METHODS_REPORT_S061.md`)

A closed-form calibration formula from the toy fit a handful of examples perfectly and was retired as overfit. A theoretical attribution that framed the early work was also removed from the public export. It shaped the questions I asked. It is not load-bearing for any result reported here.

Each of these broke in a specific place, and each break specified the next instrument. That is how the work moved.

---

## VII. Your Turn

The script is meant to be changed. Here are five changes worth making, roughly in order of difficulty.

**1. Remove the positions.** Delete the one-hot position code in `encode()` and adjust `d_in`. Predict the result before you run it. Then explain why the head cannot recover.

**2. Swap the embeddings for noise.** Replace the PPMI-SVD vectors with random unit vectors. Does the head still learn the task? What happens to the leak in Part 6, and why?

**3. Freeze the softmax path.** In `backward()`, set `d_S` to zero so that `W_Q` and `W_K` never update. Train again and look at the attention pattern. This is the mistake the early archive made.

**4. Make the edit leak.** Give two birds more shared private context in the corpus, so their embeddings overlap more. Rerun Part 6 and find the α at which the edit to *heron* also flips the neighbor. Then try to design an edit that flips *heron* without moving the neighbor, by projecting the neighbor's direction out of `u`.

**5. Rebuild the jump split.** Change the task so that one noun appears in training only in object position, never as subject. Test it as a subject. This is a small version of `add_prim_jump`. Find out whether the head fails, and if it does, whether it fails in where it looks or in what it reads back.

If you find something interesting, or something wrong in this document, I would like to hear about it.

---

## VIII. What This Does Not Establish

The toy script shows mechanics. It says nothing about T5-small, SCAN or any production model. The leak fractions and margins it prints belong to one designed head on one designed corpus.

The T5 results hold for one model, at one checkpoint, on one split. The repair flips 16 of 30 sampled failures and leaves 14. Its specificity is untested. None of it is claimed to generalize to other models, other tasks or larger scales, and none of it explains why compositional generalization fails in general. It locates one failure in one place and shows that a small, targeted change in that place moves it.

Nothing here is new mathematics. Attention, PPMI, SVD, logit decomposition and rank-1 weight edits all predate this work. What is offered is the assembly, the audit trail, and a path someone else can walk.

---

## IX. Where to Go Next

The full study, in reading order: `../README.md` for the summary, `../paper/` for the paper and its figures (every figure regenerates from a committed CSV), `../findings/` for the sealed methods reports behind each number quoted here, and `../workbench/experiments/g_track_export/` for the NumPy toy experiments, which run on a CPU with no checkpoint.

For background: Lake and Baroni (2018) introduced SCAN. Raffel et al. (2020) introduced T5. Vaswani et al. (2017) introduced the transformer attention used throughout. Levy and Goldberg (2014) connected PPMI-SVD embeddings to neural word embeddings. Meng et al. (2022) used rank-1 weight edits to locate and change factual associations in language models, the closest published relative of the repair in Section IV.

---

*The heads looked at the right word. They read back the wrong thing. The repair is partial, and the record says so.*

**Troy Teno | November 2026**
*Companion document — Not Peer Reviewed | Open Access*
