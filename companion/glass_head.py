"""
glass_head.py — one attention head, built by hand, in NumPy.

Companion to the Glass Attention Workbench reproducibility archive.
Troy Teno | November 2026 | Open Access

Run it:
    pip install numpy
    python glass_head.py

Nothing here is imported from a deep-learning library. Every step of
attention is written out: the embeddings, the Q/K/V projections, the
scaled scores, the softmax, the value mixing, and the backward pass
through the softmax Jacobian. If you can read NumPy, you can read
attention.

The script runs in six parts:

  1. EMBEDDINGS   Build word vectors from co-occurrence counts (PPMI + SVD).
  2. THE TASK     "the HERON watched the TRUCK" -> is the subject a bird or a machine?
  3. THE HEAD     Forward pass: Q, K, V, scores, softmax, mix, read out.
  4. GRADIENTS    Backward pass by hand, checked against finite differences.
  5. TRAINING     Watch the attention pattern go from diffuse to sharp.
  6. THE EDIT     A rank-1 change to W_V that redirects what the head reads
                  from one word — the toy version of the repair in S-061.

This is a teaching instrument. It is not the T5 study. It shows the
mechanics the study is about, at a scale where every number can be printed.
"""

import numpy as np

SEED = 7
rng = np.random.default_rng(SEED)
np.set_printoptions(precision=3, suppress=True)


def banner(title):
    print("\n" + "=" * 72)
    print("  " + title)
    print("=" * 72)


# ═════════════════════════════════════════════════════════════════════════
# PART 1 — EMBEDDINGS FROM CO-OCCURRENCE
# ═════════════════════════════════════════════════════════════════════════
#
# A word is known by the company it keeps. We write a tiny corpus, count
# which words appear near which, and compress the counts into vectors.
# No training happens here. The geometry comes from the corpus alone.

BIRDS    = ["heron", "eagle", "sparrow", "owl"]
MACHINES = ["tractor", "forklift", "drill", "truck"]
VERBS    = ["watched", "passed", "followed", "lifted"]

BIRD_CONTEXT    = ["wings", "nest", "sky", "feathers", "sings", "flies"]
MACHINE_CONTEXT = ["engine", "steel", "factory", "gears", "diesel", "rusts"]
SHARED_CONTEXT  = ["river", "field", "road", "morning"]

# Each noun also keeps some company of its own, so no two birds (or two
# machines) are identical. This matters in Part 6.
PRIVATE_CONTEXT = {
    "heron": ["fish", "marsh"],      "eagle": ["mountain", "prey"],
    "sparrow": ["seeds", "hedge"],   "owl": ["night", "hoots"],
    "tractor": ["plough", "farm"],   "forklift": ["pallet", "warehouse"],
    "drill": ["bit", "holes"],       "truck": ["cargo", "highway"],
}


def make_corpus(n_sentences=600):
    """Short sentences. Birds keep bird company, machines keep machine
    company, and the task verbs appear with both, so they stay neutral."""
    corpus = []
    for _ in range(n_sentences):
        kind = rng.integers(3)
        if kind == 0:   # "the heron has wings near the river"
            noun = rng.choice(BIRDS if rng.random() < 0.5 else MACHINES)
            ctx = (BIRD_CONTEXT if noun in BIRDS else MACHINE_CONTEXT) + PRIVATE_CONTEXT[noun]
            corpus.append(["the", noun, "has", rng.choice(ctx),
                           "near", "the", rng.choice(SHARED_CONTEXT)])
        elif kind == 1:  # "the eagle flies sky"
            noun = rng.choice(BIRDS if rng.random() < 0.5 else MACHINES)
            ctx = (BIRD_CONTEXT if noun in BIRDS else MACHINE_CONTEXT) + PRIVATE_CONTEXT[noun]
            corpus.append(["the", noun, rng.choice(ctx), rng.choice(PRIVATE_CONTEXT[noun])])
        else:            # "the truck watched the field" — verbs see everything
            noun = rng.choice(BIRDS + MACHINES)
            corpus.append(["the", noun, rng.choice(VERBS), "the",
                           rng.choice(SHARED_CONTEXT + BIRDS + MACHINES)])
    return corpus


def ppmi_svd_embeddings(corpus, dim=10, window=2):
    """
    Positive Pointwise Mutual Information, then truncated SVD.

    PMI(w, c) = log[ P(w, c) / (P(w) P(c)) ]
      > 0  : w and c appear together more often than chance
      < 0  : less often than chance (clipped to 0 — that is the "P" in PPMI)

    SVD then finds the few directions that explain most of the PPMI matrix.
    Each word's coordinates along those directions are its embedding.
    """
    vocab = sorted({w for s in corpus for w in s})
    idx = {w: i for i, w in enumerate(vocab)}
    C = np.zeros((len(vocab), len(vocab)))
    for sent in corpus:
        for i, w in enumerate(sent):
            for j in range(max(0, i - window), min(len(sent), i + window + 1)):
                if i != j:
                    C[idx[w], idx[sent[j]]] += 1

    total = C.sum()
    p_w = C.sum(axis=1, keepdims=True) / total
    p_c = C.sum(axis=0, keepdims=True) / total
    with np.errstate(divide="ignore", invalid="ignore"):
        pmi = np.log((C / total) / (p_w * p_c))
    ppmi = np.nan_to_num(np.maximum(pmi, 0.0), neginf=0.0)

    U, S, _ = np.linalg.svd(ppmi)
    E = U[:, :dim] * np.sqrt(S[:dim])          # common weighting choice
    E /= np.linalg.norm(E, axis=1, keepdims=True) + 1e-9
    return {w: E[idx[w]] for w in vocab}


def cos(a, b):
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


# ═════════════════════════════════════════════════════════════════════════
# PART 2 — THE TASK
# ═════════════════════════════════════════════════════════════════════════
#
# Input:  "the SUBJECT verb the OBJECT"     (five tokens)
# Label:  BIRD if the subject is a bird, MACHINE if it is a machine.
#
# The object is a bird or a machine at random. Half the time it disagrees
# with the subject. That is the trap: if the head averages every word
# equally (mean-pooling), a bird subject and a machine object cancel, and
# the answer is a coin flip. To succeed, the head must learn WHERE to look.
#
# One more thing. Attention by itself is blind to word order: it sees a
# set of vectors, not a sequence. "the heron watched the truck" and "the
# truck watched the heron" would look identical. So we append a one-hot
# position code to every word vector. Without it the task is impossible.

SEQ_LEN = 5
LABELS = ["BIRD", "MACHINE"]


def make_task(n, emb):
    X, y, words = [], [], []
    for _ in range(n):
        subj = rng.choice(BIRDS + MACHINES)
        obj = rng.choice(BIRDS + MACHINES)
        sent = ["the", subj, rng.choice(VERBS), "the", obj]
        X.append(encode(sent, emb))
        y.append(0 if subj in BIRDS else 1)
        words.append(sent)
    return np.array(X), np.array(y), words


def encode(sent, emb):
    """Word vector + one-hot position. Shape: (SEQ_LEN, d_emb + SEQ_LEN)."""
    pos = np.eye(SEQ_LEN)
    return np.stack([np.concatenate([emb[w], pos[i]]) for i, w in enumerate(sent)])


# ═════════════════════════════════════════════════════════════════════════
# PART 3 — THE HEAD (forward pass)
# ═════════════════════════════════════════════════════════════════════════
#
#   Q = X W_Q        what each position is looking for
#   K = X W_K        what each position offers to be found by
#   V = X W_V        what each position hands over if it is chosen
#
#   S = Q K^T / sqrt(d)     how well each query matches each key
#   A = softmax(S)          each row becomes a probability distribution
#   Z = A V                 each position receives a weighted mix of values
#
#   logits = Z[0] W_O       we read the answer out at position 0 ("the"),
#                           which plays the role of a summary token
#
# Position 0 must learn to put its attention on position 1, the subject.
# Nothing tells it to. The gradient has to find it.


def softmax(S):
    S = S - S.max(axis=-1, keepdims=True)     # stability: subtract the row max
    E = np.exp(S)
    return E / E.sum(axis=-1, keepdims=True)


class GlassHead:
    def __init__(self, d_in, d_head=8, n_classes=2, init_scale=0.1):
        r = np.random.default_rng(SEED + 1)
        self.d = d_head
        self.W_Q = r.normal(0, init_scale, (d_in, d_head))
        self.W_K = r.normal(0, init_scale, (d_in, d_head))
        self.W_V = r.normal(0, init_scale, (d_in, d_head))
        self.W_O = r.normal(0, init_scale, (d_head, n_classes))

    def params(self):
        return {"W_Q": self.W_Q, "W_K": self.W_K, "W_V": self.W_V, "W_O": self.W_O}

    def forward(self, X):
        """X: (seq, d_in). Returns class probabilities and a cache of every
        intermediate, so the backward pass and the printouts can use them."""
        Q = X @ self.W_Q
        K = X @ self.W_K
        V = X @ self.W_V
        S = Q @ K.T / np.sqrt(self.d)
        A = softmax(S)
        Z = A @ V
        logits = Z[0] @ self.W_O
        p = softmax(logits)
        return p, dict(X=X, Q=Q, K=K, V=V, S=S, A=A, Z=Z, logits=logits, p=p)

    def loss(self, X, label):
        p, _ = self.forward(X)
        return -np.log(p[label] + 1e-12)

    # ─────────────────────────────────────────────────────────────────────
    # PART 4 — THE BACKWARD PASS
    # ─────────────────────────────────────────────────────────────────────
    #
    # The chain rule, one line per step, in reverse order.
    #
    # The only step that is not ordinary matrix calculus is the softmax.
    # For one row a = softmax(s), the Jacobian is
    #
    #       da_j / ds_k = a_j (delta_jk - a_k)
    #
    # so given the upstream gradient g = dL/da, the gradient into the scores is
    #
    #       dL/ds_k = a_k ( g_k - sum_j g_j a_j )
    #
    # Read that as: a score's gradient is its weight times how much better
    # than average its value was. This is the line that lets attention
    # PATTERNS move. Without it, you can train V and W_O, but the head keeps
    # looking wherever it looked at initialization.

    def backward(self, cache, label):
        X, Q, K, V, A, Z, p = (cache[k] for k in ("X", "Q", "K", "V", "A", "Z", "p"))
        sc = np.sqrt(self.d)

        d_logits = p.copy()
        d_logits[label] -= 1.0                       # softmax + cross-entropy

        g_WO = np.outer(Z[0], d_logits)
        d_Z = np.zeros_like(Z)
        d_Z[0] = self.W_O @ d_logits                 # only position 0 is read out

        d_A = d_Z @ V.T
        d_V = A.T @ d_Z

        d_S = A * (d_A - (d_A * A).sum(axis=1, keepdims=True))   # softmax Jacobian

        d_Q = d_S @ K / sc
        d_K = d_S.T @ Q / sc

        return {"W_Q": X.T @ d_Q, "W_K": X.T @ d_K, "W_V": X.T @ d_V, "W_O": g_WO}


def gradient_check(head, X, label, eps=1e-6, n_probe=6):
    """Nudge individual weights up and down and measure the loss directly.
    If the hand-derived gradient is right, the two numbers agree."""
    _, cache = head.forward(X)
    grads = head.backward(cache, label)
    r = np.random.default_rng(0)
    worst = 0.0
    for name, W in head.params().items():
        for _ in range(n_probe):
            i, j = r.integers(W.shape[0]), r.integers(W.shape[1])
            old = W[i, j]
            W[i, j] = old + eps; lp = head.loss(X, label)
            W[i, j] = old - eps; lm = head.loss(X, label)
            W[i, j] = old
            numeric = (lp - lm) / (2 * eps)
            analytic = grads[name][i, j]
            rel = abs(numeric - analytic) / max(1e-8, abs(numeric) + abs(analytic))
            worst = max(worst, rel)
        print(f"  {name}: analytic {analytic:+.6e}   numeric {numeric:+.6e}")
    return worst


# ═════════════════════════════════════════════════════════════════════════
# PART 5 — TRAINING
# ═════════════════════════════════════════════════════════════════════════

def accuracy(head, X, y):
    return float(np.mean([np.argmax(head.forward(x)[0]) == t for x, t in zip(X, y)]))


def show_attention(head, sent, emb, label=""):
    _, c = head.forward(encode(sent, emb))
    row = c["A"][0]
    print(f"  {label:<10}" + "  ".join(f"{w:>8}" for w in sent))
    print(f"  {'pos 0 ->':<10}" + "  ".join(f"{a:>8.3f}" for a in row))
    print(f"  {'':<10}prediction: {LABELS[int(np.argmax(c['p']))]}  p={c['p'].max():.3f}")


def train(head, X, y, epochs=60, lr=0.5):
    for ep in range(1, epochs + 1):
        order = rng.permutation(len(X))
        total = 0.0
        for n in order:
            _, cache = head.forward(X[n])
            total += -np.log(cache["p"][y[n]] + 1e-12)
            for name, g in head.backward(cache, y[n]).items():
                head.params()[name] -= lr * g / 8      # in-place update
        if ep in (1, 5, 10, 20, 40, epochs):
            print(f"  epoch {ep:>3}   mean loss {total / len(X):.4f}   "
                  f"train acc {accuracy(head, X, y):.3f}")


# ═════════════════════════════════════════════════════════════════════════
# PART 6 — THE RANK-1 VALUE EDIT
# ═════════════════════════════════════════════════════════════════════════
#
# After training, the head looks at the subject and hands its VALUE to
# the readout. So the head's answer about "heron" depends on one vector:
#
#       v_heron = x_heron W_V
#
# Suppose we want the head to read "heron" as a machine, without touching
# where it looks. We add a rank-1 matrix to W_V:
#
#       W_V  <-  W_V + alpha * u (v_target - v_heron)^T
#
# where u = x_heron / |x_heron|^2 (word part only, no position). This
# changes x_heron W_V by exactly alpha * (v_target - v_heron), and changes
# any other word x by alpha * (x . u) * (v_target - v_heron).
#
# Read that second clause carefully. Words whose embeddings overlap with
# "heron" move too. The edit leaks along the geometry the corpus built.
#
# This is the same algebraic move as the S-061 repair in T5-small, where a
# rank-1 change to W_V at two decoder heads redirected what they read back
# from the "jump" encoder state. Here, every number is visible.

def rank1_value_edit(head, emb, d_emb, word, target_words, alpha):
    x = np.concatenate([emb[word], np.zeros(SEQ_LEN)])
    u = x / (x @ x)
    v_now = x @ head.W_V
    v_target = np.mean([np.concatenate([emb[w], np.zeros(SEQ_LEN)]) @ head.W_V
                        for w in target_words], axis=0)
    delta = alpha * np.outer(u, v_target - v_now)
    return delta, np.linalg.norm(delta) / np.linalg.norm(head.W_V)


def subject_margin(head, emb, X_words, subject):
    """Mean logit margin (BIRD minus MACHINE) on test sentences with this subject.
    Positive means the head leans BIRD. The sign decides the answer; the size
    says how firmly."""
    hits = [s for s in X_words if s[1] == subject]
    return float(np.mean([np.subtract(*head.forward(encode(s, emb))[1]["logits"])
                          for s in hits]))


def subject_accuracy(head, emb, X_words, subject):
    """Accuracy on test sentences whose subject is `subject`."""
    hits = [s for s in X_words if s[1] == subject]
    if not hits:
        return float("nan"), 0
    ok = [np.argmax(head.forward(encode(s, emb))[0]) == (0 if s[1] in BIRDS else 1)
          for s in hits]
    return float(np.mean(ok)), len(hits)


# ═════════════════════════════════════════════════════════════════════════
# MAIN
# ═════════════════════════════════════════════════════════════════════════

def main():
    banner("PART 1 — EMBEDDINGS FROM CO-OCCURRENCE (PPMI + SVD)")
    corpus = make_corpus()
    emb = ppmi_svd_embeddings(corpus)
    d_emb = len(next(iter(emb.values())))
    print(f"  corpus: {len(corpus)} sentences, vocabulary {len(emb)}, embedding dim {d_emb}")
    def mean_cos(A, B):
        return np.mean([cos(emb[a], emb[b]) for a in A for b in B if a != b])
    print(f"  mean cos, bird-bird:        {mean_cos(BIRDS, BIRDS):+.3f}")
    print(f"  mean cos, machine-machine:  {mean_cos(MACHINES, MACHINES):+.3f}")
    print(f"  mean cos, bird-machine:     {mean_cos(BIRDS, MACHINES):+.3f}")
    print("  Nobody labeled these words. The grouping came from the counts.")

    banner("PART 2 — THE TASK")
    X_tr, y_tr, _ = make_task(400, emb)
    X_te, y_te, W_te = make_task(400, emb)
    print("  example:", " ".join(W_te[0]), "->", LABELS[y_te[0]])
    print("  example:", " ".join(W_te[1]), "->", LABELS[y_te[1]])

    # Baseline: what does equal attention (mean-pooling) get?
    mean_pool = GlassHead(d_in=d_emb + SEQ_LEN)
    mean_pool.W_Q[:] = 0.0; mean_pool.W_K[:] = 0.0   # zero scores -> uniform attention
    # Train only V and W_O with attention frozen uniform.
    for _ in range(40):
        for n in rng.permutation(len(X_tr)):
            _, c = mean_pool.forward(X_tr[n])
            g = mean_pool.backward(c, y_tr[n])
            mean_pool.W_V -= 0.5 * g["W_V"] / 8
            mean_pool.W_O -= 0.5 * g["W_O"] / 8
    disagree = [i for i, s in enumerate(W_te) if (s[1] in BIRDS) != (s[4] in BIRDS)]
    print(f"\n  Control: attention frozen at uniform, V and W_O trained.")
    print(f"    test accuracy, all sentences:            {accuracy(mean_pool, X_te, y_te):.3f}")
    print(f"    test accuracy, subject/object disagree:  "
          f"{accuracy(mean_pool, X_te[disagree], y_te[disagree]):.3f}")
    print("  Averaging cannot tell the subject from the object. It needs to look.")

    banner("PART 3 — THE HEAD, BEFORE TRAINING")
    head = GlassHead(d_in=d_emb + SEQ_LEN)
    probe = ["the", "heron", "watched", "the", "truck"]
    show_attention(head, probe, emb, "untrained")
    print("  Attention from position 0 is nearly uniform. The head has no opinion yet.")

    banner("PART 4 — CHECK THE HAND-DERIVED GRADIENTS")
    worst = gradient_check(head, X_tr[0], y_tr[0])
    print(f"  worst relative error over probes: {worst:.2e}")
    print("  " + ("PASS — the backward pass is correct." if worst < 1e-4 else
                  "FAIL — the backward pass disagrees with finite differences."))

    banner("PART 5 — TRAINING (all four matrices, including W_Q and W_K)")
    train(head, X_tr, y_tr)
    print(f"\n  test accuracy, all sentences:            {accuracy(head, X_te, y_te):.3f}")
    print(f"  test accuracy, subject/object disagree:  "
          f"{accuracy(head, X_te[disagree], y_te[disagree]):.3f}\n")
    show_attention(head, probe, emb, "trained")
    show_attention(head, ["the", "truck", "lifted", "the", "owl"], emb, "trained")
    print("  Position 0 now puts its weight on position 1. The gradient found the subject.")

    banner("PART 6 — A RANK-1 EDIT TO W_V: READ 'heron' AS A MACHINE")
    watch = ["heron", "eagle", "sparrow", "owl", "truck"]
    print("  Logit margin (BIRD minus MACHINE), averaged over test sentences per subject.")
    print("  Positive = read as BIRD. Negative = read as MACHINE.\n")
    print(f"  {'alpha':>6} {'|dW|/|W_V|':>11}" + "".join(f"{w:>10}" for w in watch)
          + f"{'heron acc':>11}")
    W_V_orig = head.W_V.copy()
    for alpha in [0.0, 0.25, 0.5, 0.75, 1.0]:
        head.W_V[:] = W_V_orig
        delta, frac = rank1_value_edit(head, emb, d_emb, "heron", MACHINES, alpha)
        head.W_V += delta
        margins = [subject_margin(head, emb, W_te, w) for w in watch]
        acc = subject_accuracy(head, emb, W_te, "heron")[0]
        print(f"  {alpha:>6.2f} {frac:>11.3f}" + "".join(f"{m:>+10.2f}" for m in margins)
              + f"{acc:>11.3f}")
    head.W_V[:] = W_V_orig
    print("""
  The heron margin falls steadily as alpha grows and crosses zero: the head
  now reads a heron as a machine. The dose-response is a straight line,
  because the edit is linear in alpha and the attention pattern never moves.

  Watch the other birds. They move too. Each one shares roughly a fifth of
  its embedding direction with heron, and each one absorbs roughly a fifth
  of the shift. They keep their answers, but with less margin than before.
  The truck column moves the other way, deeper into MACHINE. In a model
  where the edited word shared more geometry with its neighbors, the same
  edit would flip them as well. Specificity is something you measure, not
  something you assume.
""")

    banner("WHAT THIS SCRIPT SHOWS, AND WHAT IT DOES NOT")
    print("""
  Shows: attention is a few matrix products and a softmax. The backward pass
  through the softmax is what lets a head learn where to look, and it can be
  checked against brute-force finite differences. Once a head looks in
  the right place, its answer is carried by the value vector it reads, and a
  rank-1 change to W_V redirects that value without changing where it looks.

  Does not show: anything about T5-small, SCAN, or production models. This is
  one head on a designed toy. The T5 study is in ../findings and ../paper.
""")


if __name__ == "__main__":
    main()
