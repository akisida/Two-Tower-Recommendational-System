# Two-Tower Recommender on MovieLens (from scratch)

A two-tower retrieval model built from scratch in PyTorch - no recommender
libraries, just embeddings and a dot product. Trained and evaluated on
MovieLens 100k, with a focus on **diagnosing why the first version failed** rather
than only reporting a final number.

**Result: Recall@50 ≈ 13%** on a time-based held-out test set
(random baseline ≈ 0.9%, so ~15× better than chance).

---

## TL;DR

- **Model:** two embedding tables (users, movies). A user and a movie are
  scored by the dot product of their embeddings. Trained with in-batch
  softmax (each user's own movie is the positive; other movies in the batch
  are negatives).
- **Data:** MovieLens (`ml-latest-small`) - 610 users × 9724 movies, 100k
  ratings, **1.7% density**.
- **Two experiments** with real findings (below), not just a trained model.

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

---

## How to run

```bash
pip install torch pandas kagglehub scikit-learn matplotlib
```

- `MovieLens.ipynb` - end-to-end: data prep, model, training, Recall@K, and
  both experiments.
- `ndim_sweep.py` - standalone embedding-size sweep.

The dataset is pulled via `kagglehub` (`abhikjha/movielens-100k`, the
`ml-latest-small` files). The trained model and id↔index mappings are saved
together in `model_and_dicts.pt`.

## What I learned / limitations

- Recommenders are trained on **positives + sampled negatives**, and evaluated
  by **ranking a held-out positive against the un-seen catalog** - not by
  accuracy.
- **Always evaluate against a baseline**, and **reproduce a suspicious result
  in isolation** before blaming the environment.
- **Limitations:** the model uses only user/movie IDs - no side features
  (genres, text). The dataset is small, which caps how far embedding size and
  model capacity can go. Natural next steps: content features, a proper
  popularity correction, and metrics beyond Recall@K (e.g. NDCG).
