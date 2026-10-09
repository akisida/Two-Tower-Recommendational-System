"""Matched-protocol comparison: two-tower vs SASRec on MovieLens.
Leave-TWO-out: test = last like, val = 2nd-last, train = the rest.
BOTH models use best-checkpoint-by-val-HR@50 (early stopping), then are
scored on the held-out TEST item with ONE shared metric (HR@50 / NDCG@50,
masking already-seen items). 3 seeds, mean/std. The honest 'who wins'."""
import pandas as pd, torch, torch.nn as nn, torch.nn.functional as F
import math, collections, copy, statistics as st
from torch.optim import Adam, AdamW
import kagglehub

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('device:', device, flush=True)

# ---------------- data (leave-two-out) ----------------
p = kagglehub.dataset_download('abhikjha/movielens-100k') + '/ml-latest-small'
ratings = pd.read_csv(p + '/ratings.csv')
pos = ratings[ratings['rating'] >= 4.0].copy()
u2i = {u: i for i, u in enumerate(pos['userId'].unique())}
m2i = {m: i + 1 for i, m in enumerate(pos['movieId'].unique())}   # 1-based, 0 = pad
pos['u'] = pos['userId'].map(u2i); pos['m'] = pos['movieId'].map(m2i)
pos = pos.sort_values(['u', 'timestamp'])
num_users = len(u2i); num_movies = len(m2i)

seqs = pos.groupby('u')['m'].apply(list).to_dict()
train_seqs, val_item, test_item = {}, {}, {}
for u, s in seqs.items():
    if len(s) < 3:
        continue                                                 # need train + val + test
    train_seqs[u] = s[:-2]; val_item[u] = s[-2]; test_item[u] = s[-1]
users = list(train_seqs.keys())
seen_train    = {u: train_seqs[u] for u in users}                # mask for val eval
seen_trainval = {u: train_seqs[u] + [val_item[u]] for u in users}  # mask for test eval
print(f'users={len(users)}  movies={num_movies}', flush=True)

tt_u, tt_m = [], []
for u in users:
    for m in train_seqs[u]:
        tt_u.append(u); tt_m.append(m)
tt_u = torch.tensor(tt_u); tt_m = torch.tensor(tt_m)
cnt = collections.Counter(tt_m.tolist())
Q = torch.zeros(num_movies + 1)
for m, c in cnt.items():
    Q[m] = c / len(tt_m)

max_len = 50
seq_tensor = torch.tensor([(train_seqs[u][-max_len:] + [0] * max_len)[:max_len] for u in users])

# ---------------- shared metric ----------------
@torch.no_grad()
def hr_ndcg(score_of_user, target_of, seen_of, k=50):
    hr = ndcg = 0.0
    for u in users:
        scores = score_of_user(u).clone()
        scores[seen_of[u]] = -1e9
        scores[0] = -1e9
        topk = torch.topk(scores, k).indices.tolist()
        t = target_of[u]
        if t in topk:
            hr += 1.0
            ndcg += 1.0 / math.log2(topk.index(t) + 2)
    return hr / len(users) * 100, ndcg / len(users) * 100

# ---------------- two-tower (best-by-val) ----------------
def train_two_tower(seed, steps=100000, bs=256, d=32, eval_every=10000):
    torch.manual_seed(seed)
    U = nn.Embedding(num_users, d).to(device)
    M = nn.Embedding(num_movies + 1, d, padding_idx=0).to(device)
    opt = Adam(list(U.parameters()) + list(M.parameters()), lr=1e-3, weight_decay=1e-4)
    Qd = Q.to(device); ar = torch.arange(bs, device=device)
    g = torch.Generator().manual_seed(seed)
    def score(u):
        with torch.no_grad():
            return (U(torch.tensor(u, device=device)) @ M.weight.data.T).detach().cpu()
    best_val, best = -1.0, None
    for i in range(steps):
        if i % eval_every == 0:
            U.eval(); M.eval()
            vh, _ = hr_ndcg(score, val_item, seen_train)
            if vh > best_val:
                best_val = vh
                best = (copy.deepcopy(U.state_dict()), copy.deepcopy(M.state_dict()))
            U.train(); M.train()
        ix = torch.randint(0, len(tt_u), (bs,), generator=g)
        ub = tt_u[ix].to(device); mb = tt_m[ix].to(device)
        s = (U(ub) @ M(mb).T) / math.sqrt(d) - torch.log(Qd[mb])
        loss = F.cross_entropy(s, ar)
        opt.zero_grad(); loss.backward(); opt.step()
    U.load_state_dict(best[0]); M.load_state_dict(best[1]); U.eval(); M.eval()
    return score

# ---------------- SASRec (best-by-val) ----------------
class Head(nn.Module):
    def __init__(self, d, hs, maxlen, drop):
        super().__init__()
        self.k = nn.Linear(d, hs, bias=False); self.q = nn.Linear(d, hs, bias=False); self.v = nn.Linear(d, hs, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(maxlen, maxlen))); self.drop = nn.Dropout(drop)
    def forward(self, x):
        B, T, C = x.shape
        k = self.k(x); q = self.q(x)
        w = q @ k.transpose(-2, -1) * k.shape[-1] ** -0.5
        w = w.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        w = self.drop(F.softmax(w, dim=-1))
        return w @ self.v(x)
class MHA(nn.Module):
    def __init__(self, d, nhead, maxlen, drop):
        super().__init__()
        hs = d // nhead
        self.heads = nn.ModuleList([Head(d, hs, maxlen, drop) for _ in range(nhead)])
        self.proj = nn.Linear(hs * nhead, d); self.drop = nn.Dropout(drop)
    def forward(self, x):
        return self.drop(self.proj(torch.cat([h(x) for h in self.heads], dim=-1)))
class FF(nn.Module):
    def __init__(self, d, drop):
        super().__init__(); self.net = nn.Sequential(nn.Linear(d, 4 * d), nn.ReLU(), nn.Linear(4 * d, d), nn.Dropout(drop))
    def forward(self, x): return self.net(x)
class Block(nn.Module):
    def __init__(self, d, nhead, maxlen, drop):
        super().__init__()
        self.sa = MHA(d, nhead, maxlen, drop); self.ff = FF(d, drop)
        self.l1 = nn.LayerNorm(d); self.l2 = nn.LayerNorm(d)
    def forward(self, x):
        x = x + self.sa(self.l1(x)); x = x + self.ff(self.l2(x)); return x
class SAS(nn.Module):
    def __init__(self, nitems, d=64, nhead=2, nlayer=2, maxlen=50, drop=0.3):
        super().__init__()
        self.tok = nn.Embedding(nitems, d, padding_idx=0); self.pos = nn.Embedding(maxlen, d)
        self.blocks = nn.ModuleList([Block(d, nhead, maxlen, drop) for _ in range(nlayer)])
        self.ln = nn.LayerNorm(d); self.head = nn.Linear(d, nitems)
    def forward(self, idx):
        B, T = idx.shape
        x = self.tok(idx) + self.pos(torch.arange(T, device=idx.device))
        for b in self.blocks: x = b(x)
        return self.head(self.ln(x))

def train_sasrec(seed, steps=5000, bs=128, d=64, eval_every=500):
    torch.manual_seed(seed)
    model = SAS(num_movies + 1, d=d, maxlen=max_len).to(device)
    opt = AdamW(model.parameters(), lr=3e-4)
    sd = seq_tensor.to(device); inp, tgt = sd[:, :-1], sd[:, 1:]
    g = torch.Generator().manual_seed(seed)
    def score_ctx(ctx):
        s = ctx[-max_len:]; k = len(s)
        x = torch.tensor([s + [0] * (max_len - k)], device=device)
        with torch.no_grad():
            lg = model(x)
        return lg[0, k - 1].detach().cpu()
    val_score = lambda u: score_ctx(train_seqs[u])
    best_val, best = -1.0, None
    for i in range(steps):
        if i % eval_every == 0:
            model.eval()
            vh, _ = hr_ndcg(val_score, val_item, seen_train)
            if vh > best_val:
                best_val = vh; best = copy.deepcopy(model.state_dict())
            model.train()
        ix = torch.randint(0, inp.shape[0], (bs,), generator=g).to(device)
        logits = model(inp[ix])
        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), tgt[ix].reshape(-1), ignore_index=0)
        opt.zero_grad(); loss.backward(); opt.step()
    model.load_state_dict(best); model.eval()
    return lambda u: score_ctx(train_seqs[u] + [val_item[u]])   # test context = train + val

# ---------------- run ----------------
print('model       seed | test HR@50   NDCG@50', flush=True)
res = {'two-tower': ([], []), 'SASRec': ([], [])}
for seed in [1, 2, 3]:
    sc = train_two_tower(seed)
    hr, nd = hr_ndcg(sc, test_item, seen_trainval)
    res['two-tower'][0].append(hr); res['two-tower'][1].append(nd)
    print(f'two-tower   {seed}    | {hr:6.2f}     {nd:5.2f}', flush=True)
    sc = train_sasrec(seed)
    hr, nd = hr_ndcg(sc, test_item, seen_trainval)
    res['SASRec'][0].append(hr); res['SASRec'][1].append(nd)
    print(f'SASRec      {seed}    | {hr:6.2f}     {nd:5.2f}', flush=True)
print('---', flush=True)
print(f'random HR@50 baseline ~ {50 / num_movies * 100:.2f}%', flush=True)
for name, (H, N) in res.items():
    print(f'{name:10s}  test HR@50 {st.mean(H):5.2f} +/- {st.pstdev(H):.2f}   '
          f'NDCG@50 {st.mean(N):5.2f} +/- {st.pstdev(N):.2f}', flush=True)
