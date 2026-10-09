# SASRec on MovieLens (from scratch) - and an honest comparison with the two-tower

A **self-attentive sequential recommender** (SASRec, Kang & McAuley 2018) built from
scratch in PyTorch - the same causal self-attention as a GPT, but the "tokens" are
movies and the task is *predict the next movie a user will like*. Built as a direct
follow-up to the [two-tower retrieval model](../README.md) in this repo, and compared
against it **under one matched protocol** rather than quoting two incomparable numbers.

**Headline:** it depends on **scale**. On `ml-small` (~24k interactions) a plain two-tower
wins (HR@50 15.8 vs 11.8). On **MovieLens-1M** (~575k interactions) SASRec **doubles** it
(HR@50 41.2 vs 20.5). The honest result is a *crossover* - a static model wins on small
data, a sequence model wins decisively once there's enough history to learn from.

---

## The reframe: why a sequence model at all

The two-tower model gives each user **one static embedding** - an average of all their
tastes, order-free. SASRec instead represents a user by the **sequence** of their likes
and runs causal self-attention over it, so the user vector is *computed from recent
behaviour* and is recency-aware. Concretely:

```
SASRec  =  (a transformer as a DYNAMIC user tower)  +  (item embeddings = the item tower)
```

The transformer **replaces** `user_emb[userId]`. The prediction at each position scores
the hidden state against the item vocabulary - the same "score items for a user" idea as
the two-tower, but the user side now comes from attention over a sequence.

## Model

- **Tokens = movie ids.** Item embedding table, `padding_idx=0` (id 0 reserved for padding).
- **Causal self-attention** (multi-head) + **positional embeddings** + residual/LayerNorm
  blocks - adapted from a from-scratch nanoGPT.
- **Right-padding** of variable-length histories: with right-pad the causal mask already
  stops real tokens from attending to padding, so no separate attention mask is needed -
  padding is handled by `cross_entropy(..., ignore_index=0)` in the loss only.
- Prediction at a position = logits over all items; trained next-item at every position.
- Config here: `d=64`, 2 heads, 2 blocks, dropout 0.3 - deliberately small (tiny dataset,
  weak GPU). **Best checkpoint selected by validation HR@50.**

## Evaluation protocol (the part that makes the comparison honest)

The first instinct - compare SASRec's number to the two-tower's `Recall@50 ≈ 21%` - is
**wrong**: they used different protocols (the two-tower used an 80/10/10 split with
*several* held-out items per user; a leave-one-out single-target metric is a different,
harder thing). So both models are re-run under **one** protocol:

- **Leave-two-out:** a user's last like = test, second-to-last = validation, rest = train.
- **HR@50 / NDCG@50** on the single held-out item, masking already-seen items.
- **Best-by-val** early stopping on *both* models, 3 seeds, mean ± std.

## Experiment A - matched comparison (context window = 50)

| model | test HR@50 | NDCG@50 |
|---|---|---|
| **two-tower** | **15.84 ± 0.43** | **4.81 ± 0.01** |
| SASRec | 11.84 ± 1.05 | 3.88 ± 0.36 |
| random | ~0.79% | - |

At a context window of 50, the two-tower wins by ~4 pp HR@50, and is far more stable
(± 0.43 vs ± 1.05 - the transformer's extra capacity is noisy on ~24k interactions).
Early stopping barely moved SASRec, so this is *not* mid-training overfitting.
`compare_loo.py`.

## Experiment B - the max_len sweep (what Experiment A missed)

Was SASRec simply **starved of context**? Sweeping the history window:

| max_len | test HR@50 | NDCG@50 |
|---|---|---|
| 20 | 8.28 ± 0.28 | 2.66 ± 0.05 |
| 50 | 11.84 ± 1.05 | 3.88 ± 0.36 |
| **100** | **13.43 ± 0.51** | **4.50 ± 0.11** |
| *(two-tower)* | *15.84 ± 0.43* | *4.81* |

HR@50 rises **monotonically** 20 → 50 → 100, and the gap to the two-tower narrows from
~4 pp (at 50) to **~2.4 pp** (at 100) - still trending up at 100. (`max_len=200` was not
finished - it is ~16× the attention cost of 50 and the run was stopped.) `maxlen_sweep.py`.

**Revised, honest conclusion:** the right statement is *not* "a transformer loses on small
data." It is: **SASRec is context-length-sensitive - with a short window it is undercooked
and loses, with a longer window it nearly matches a two-tower that is itself a strong,
remarkably stable baseline.** The median user here has 38 likes, so `max_len=50` truncated
a large fraction of users; lengthening the window recovers that signal.

## Experiment C - scale (MovieLens-1M): the crossover

The real question behind all of this: *do sequence models need scale?* Re-running the
exact same matched protocol on **MovieLens-1M** (6,035 users, ~575k positive
interactions - ~25× `ml-small`), with scaled-up configs (SASRec `d=128`, `max_len=200`;
two-tower `d=64`):

| dataset | two-tower HR@50 | SASRec HR@50 | winner |
|---|---|---|---|
| ml-small (~24k interactions) | **15.84 ± 0.43** | 11.84 ± 1.05 | two-tower |
| **ML-1M (~575k interactions)** | 20.54 ± 0.25 | **41.19 ± 0.10** | **SASRec, ~2×** |

On ML-1M SASRec **doubles** the two-tower (HR@50 41.2 vs 20.5; NDCG@50 15.5 vs 6.1), and
it is now *more* stable than the two-tower (± 0.10 vs ± 0.25) - the exact opposite of the
small-data regime, where it was the noisy one (± 1.05). `compare_loo_ml1m.py`.

**The headline finding is the crossover:** a static two-tower wins on small data, a
sequence model wins - decisively - once there is enough history to learn from. Not "which
model is better" but "which model for how much data".

## What I learned

- **Compare like with like.** A single headline number is meaningless across different
  eval protocols; the only honest "who wins" is both models under one split and one metric.
- **Don't record a conclusion before the full curve.** The first comparison (window=50)
  suggested a clean "two-tower wins"; the sweep showed that was partly an artifact of too
  short a context. The sweep changed the story - twice a plateau was predicted and twice the
  data disagreed.
- A simple, well-tuned baseline is hard to beat cheaply on small data - a result worth
  stating plainly rather than tuning away.
- **Model choice is data-dependent.** The same two models swap places between ~24k and
  ~575k interactions. "SASRec vs two-tower" has no context-free answer; the scale does.

## Files

- `SasRec.ipynb` - end-to-end: data prep, leave-two-out split, padded sequences, the SASRec
  model, training with best-by-val, leave-one-out HR@50 / NDCG@50 eval, save/load.
- `compare_loo.py` - Experiment A: matched-protocol two-tower vs SASRec, best-by-val, 3 seeds.
- `maxlen_sweep.py` - Experiment B: SASRec HR@50 / NDCG@50 across context windows.

Data is pulled via `kagglehub` (`abhikjha/movielens-100k`, the `ml-latest-small` files).
