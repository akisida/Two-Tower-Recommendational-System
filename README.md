# Two-Tower Recommender on MovieLens (from scratch)

A two-tower retrieval model built from scratch in PyTorch - no recommender
libraries, just embeddings and a dot product. Trained and evaluated on
MovieLens 100k, with a focus on **diagnosing why the first version failed** rather
than only reporting a final number.

**Result: Recall@50 ≈ 21%** with genre content features + regularization on the
item tower (id-only + logQ is ≈ 17%; ≈ 14% without logQ), on a time-based held-out
test set. Random baseline ≈ 0.9%, so ~23× better than chance. All headline numbers
are means over multiple random seeds.

---

## TL;DR

- **Model:** two embedding tables (users, movies). A user and a movie are
  scored by the dot product of their embeddings. Trained with in-batch
  softmax (each user's own movie is the positive; other movies in the batch
  are negatives).
- **Data:** MovieLens (`ml-latest-small`) - 610 users × 9724 movies, 100k
  ratings, **1.7% density**.
- **Six experiments** with real findings (below), including two rigorously-tested
  negative results and a rigorously-confirmed positive one - not just a trained model.

---

## Problem setup

- **Positives:** ratings ≥ 4.0 (treated as "user liked this"). Lower ratings
  are dropped for v1.
- **Negatives:** sampled implicitly via in-batch negatives - other users'
  movies in the same batch. This is cheap and works here *because* the data
  is sparse: a random unseen movie is very unlikely to be a false negative.
- **Train/test split:** leave-last-out per user, ordered by timestamp
  (train = a user's earlier likes, test = their later ones). Splitting by time
  rather than at random avoids leaking the future into training.

## Model

```
user_id  → [user embedding table]  → u  ┐
                                        ├─ score = (u · m) / sqrt(d)
movie_id → [movie embedding table] → m  ┘
```

- Embedding dim `d = 32`.
- Loss: cross-entropy over the batch score matrix `U @ Mᵀ`, where the diagonal
  is the positive pair - i.e. softmax with in-batch negatives.
- Optimizer: Adam, lr 1e-3, batch size 256.

---

## Experiment 1 - why the first version was *worse than random*

The first training run scored **below the random baseline** on Recall@50.
The evaluation code was correct (verified by reproducing it in isolation), so
the problem was the training recipe, not a bug. Three causes, none sufficient
alone:

| Change | Recall@50 |
|---|---|
| original (lr 1e-4, batch 32) | 0.15% |
| lr 1e-4 → 1e-3 | 0.83% |
| batch 32 → 256 | 0.09% |
| add `/ sqrt(d)` score scaling | 0.36% |
| **all three combined** | **10.2%** |

**What each fixes:**
1. **Learning rate** was far too low - the model was badly undertrained even
   after 200k steps.
2. **Initialization.** Dot products of `N(0, 1)` 32-dim vectors have std ≈ 6,
   so the softmax starts overconfident and the initial loss is inflated
   (~13 instead of `ln(32)` ≈ 3.5). Scaling scores by `1/sqrt(d)` fixes this -
   the same trick as scaled dot-product attention.
3. **In-batch popularity bias.** With a small batch, popular movies appear as
   negatives more often and get pushed down - but popular movies are exactly
   what generalizes on the test set. A larger batch reduces this bias.
   (The principled fix is logQ / sampled-softmax correction, Yi et al. 2019.)

**Takeaway:** a metric means nothing without a baseline. `0.15%` looked like a
"weak model" until compared to the `0.9%` random level, which made it obvious
something was broken.

## Experiment 2 - embedding size

Sweeping `n_dim` with everything else fixed (`ndim_sweep.py`):

| n_dim | train loss | Recall@50 |
|---|---|---|
| 16 | 3.54 | 11.99% |
| **32** | 3.08 | **13.32%** |
| 64 | 2.74 | 12.53% |
| 128 | 2.71 | 10.86% |

Train loss keeps decreasing with capacity, but test Recall **peaks at 32 and
then declines** - a textbook overfitting curve on a small dataset (~48k
positive interactions). `n_dim = 32` is the sweet spot here.

## Experiment 3 - logQ sampling-bias correction (Yi et al. 2019)

The popularity bias from Experiment 1 has a principled fix. In-batch negatives
are sampled in proportion to item popularity, so popular items are penalised too
often. The correction subtracts each candidate's log-frequency from its logit:

```
corrected_logit = raw_logit - log(Q(item))
```

Since `exp(s - log Q) = exp(s) / Q`, each candidate's contribution to the softmax
is divided by how often it appears - cancelling the over-sampling of popular
items. `Q` is just each movie's frequency in the training set (no need for the
paper's streaming estimator - the catalogue here is small and fixed). The
correction is applied in **training only**, not at evaluation.

| | Recall@50 (mean of 5 seeds) |
|---|---|
| without logQ | ~14% |
| **with logQ** | **~17%** |

+3 pp, well above the seed-to-seed noise (±0.3 pp) - a real effect.

## Experiment 4 - best-model selection (tested, no gain)

Does keeping the checkpoint with the lowest **validation** loss beat using the
final one? A single run suggested +2.4 pp - but that compared the best checkpoint
of one run against the last checkpoint of a *different* run (confounded).

Proper test: 5 seeds, validation loss measured every 5000 steps, `best` vs `last`
compared **within each run**:

| | mean | std |
|---|---|---|
| last | 17.26% | 0.28 |
| best | 17.11% | 0.35 |
| **delta (best - last)** | **-0.15 pp** | 0.23 |

The delta scatters around zero (per-seed: -0.14, -0.32, -0.47, +0.04, +0.14), so
best-model selection gives **no reliable improvement here**. At `n_dim = 32` the
model does not overfit within 200k steps, so there is no better checkpoint to
catch. The exciting single-run number was noise.

**Takeaway:** a single run can mislead; multiple seeds separate signal from noise,
and an honestly-reported negative result is worth more than a cherry-picked one.

## Experiment 5 - a 2025 refinement that didn't transfer

A recent paper (Khrylchenko et al., *Correcting the LogQ Correction*, 2025) points out
a flaw in the standard logQ correction: the positive item in the denominator is **not**
sampled - it is always present with probability 1 - so subtracting `log Q` from it is
wrong. (Experiment 3's code did exactly that - it subtracted the correction from the
diagonal too.) The fix undoes the correction on the positive only:
`scores[i, i] += log_q[i]`.

Paired test (4 seeds, identical init and minibatch stream for both variants, so the
delta reflects only the fix, not run-to-run noise):

| | mean | std |
|---|---|---|
| standard logQ | 17.33% | 0.27 |
| positive-fixed | 16.84% | 0.54 |
| **delta (fixed - standard)** | **-0.49 pp** | 0.36 |

3 of 4 seeds are negative - the fix gives **no benefit here, and slightly hurts**. The
correction is theoretically sound, but the paper's gains were on MovieLens-1M and
industrial data (larger, more skewed, measured at Recall@100/@1000). On this small
dataset at Recall@50, the effect doesn't survive the noise.

**Takeaway:** not every SOTA refinement transfers. The large bias correction (logQ,
+3 pp) is what matters; fine-grained refinements on top don't move the needle at this
scale - and testing that honestly is the point.

This was re-tested once content features were added - see the end of Experiment 6.

## Experiment 6 - content features (genres), the first confirmed win

Until now the model used only IDs - it could recommend a movie only if similar
*users* had liked it. Adding **genres** to the item tower lets it generalise by
*content*: an unseen movie that is genre-close to a user's likes now scores highly.

Each movie's genres are a **multi-hot** vector (19 genres). The item tower embeds
the id and the genres separately, concatenates them, and projects back to `d = 32`
so both towers still output the same dimension for the dot product:

```
movie_id    → [id embedding]     ─┐
                                   ├─ concat → dropout → Linear(2d → d) → m
genres(19)  → Linear(19 → d)     ─┘
```

Naively adding this **hurt** (14.2% - worse than id-only): the extra parameters
overfit a small dataset. The fix is regularization - `Dropout(0.3)` on the concat
and `weight_decay = 1e-4`. So the finding is not "add features" but "add features
**and** hold capacity in check".

Paired test (3 seeds, same init and minibatch stream per seed, best-by-val
checkpoint for both variants, so the delta is within-run):

| | mean | std |
|---|---|---|
| id-only + logQ | 17.25% | 0.30 |
| **genres + dropout + weight_decay + logQ** | **20.91%** | 0.56 |
| **delta (content - id)** | **+3.66 pp** | 0.71 |

All three seeds are positive (+3.17, +3.15, +4.67) and the mean is ~5× the std -
unlike Experiments 4 and 5, this signal survives the noise. **Content features are
a real +3.7 pp here, but only with regularization.**

**Takeaway:** the same multi-seed rigor that killed two exciting single-run numbers
also *confirms* a real one - the method cuts both ways, which is the point.

### Revisiting Experiment 5 with features

Experiment 5 left an open question: the logQ positive-fix didn't help on the id-only
model, but might it pay off *with* content features? Re-ran the same paired test, this
time with both variants on the content+regularization model (3 seeds):

| | mean | std |
|---|---|---|
| standard logQ | 20.91% | 0.56 |
| positive-fixed | 20.67% | 0.57 |
| **delta (fixed - standard)** | **-0.24 pp** | 0.04 |

The hypothesis is refuted: the fix still doesn't help. But note the **std of 0.04** -
all three seeds land in a tight band (-0.29, -0.22, -0.20). On the id-only model this
delta was -0.49 ± 0.36 (lost in noise); the paired design plus the more stable content
model resolves it precisely as a small, *consistent* negative. The positive-in-denominator
bias is simply too small to matter at this scale, and features don't change that.

**Takeaway:** the project's pattern holds on every model tried - the big correction (logQ)
and content features move the needle (+3-4 pp each); fine-grained refinements don't.

```bash
pip install torch pandas kagglehub scikit-learn matplotlib
```

- `MovieLens.ipynb` - end-to-end: data prep, model, training, logQ correction,
  Recall@K.
- `ndim_sweep.py` - Experiment 2: embedding-size sweep.
- `rigor_bestmodel.py` - Experiment 4: 5-seed best-vs-last comparison.
- `logq_positive_fix.py` - Experiment 5: 4-seed paired standard-vs-positive-fixed logQ.
- `genres_multiseed.py` - Experiment 6: 3-seed paired id-only-vs-genres+regularization.
- `posfix_features_multiseed.py` - Experiment 5 revisited: positive-fix on the content model.

The dataset is pulled via `kagglehub` (`abhikjha/movielens-100k`, the
`ml-latest-small` files). The trained model and id↔index mappings are saved
together in `model_and_dicts.pt`.

## What I learned / limitations

- Recommenders are trained on **positives + sampled negatives**, and evaluated
  by **ranking a held-out positive against the un-seen catalog** - not by
  accuracy.
- **Always evaluate against a baseline**, and **reproduce a suspicious result
  in isolation** before blaming the environment.
- **A single run can mislead** - average over several seeds before trusting a
  difference, and compare variants *within* the same run.
- **Content features help, but only with regularization** - genres added +3.7 pp,
  yet adding them naively (no dropout/weight decay) *hurt*, because extra capacity
  overfits a small dataset.
- **Limitations:** features are still only genres (no text/tags); the dataset is
  small, which caps capacity. Natural next steps: richer side features, metrics
  beyond Recall@K (e.g. NDCG), and sequence models (attention / SASRec).
