"""SASRec max_len sweep on MovieLens (leave-two-out, best-by-val, HR@50/NDCG@50).
Does a longer interaction context help on this small dataset?"""
import pandas as pd, torch, torch.nn as nn, torch.nn.functional as F
import math, copy, statistics as st
from torch.optim import AdamW
import kagglehub

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('device:', device, flush=True)

p = kagglehub.dataset_download('abhikjha/movielens-100k') + '/ml-latest-small'
ratings = pd.read_csv(p + '/ratings.csv')
pos = ratings[ratings['rating'] >= 4.0].copy()
m2i = {m: i + 1 for i, m in enumerate(pos['movieId'].unique())}   # 1-based, 0 = pad
u2i = {u: i for i, u in enumerate(pos['userId'].unique())}
pos['u'] = pos['userId'].map(u2i); pos['m'] = pos['movieId'].map(m2i)
pos = pos.sort_values(['u', 'timestamp'])
num_movies = len(m2i)

seqs = pos.groupby('u')['m'].apply(list).to_dict()
train_seqs, val_item, test_item = {}, {}, {}
for u, s in seqs.items():
    if len(s) < 3:
        continue
    train_seqs[u] = s[:-2]; val_item[u] = s[-2]; test_item[u] = s[-1]
users = list(train_seqs.keys())
seen_train    = {u: train_seqs[u] for u in users}
seen_trainval = {u: train_seqs[u] + [val_item[u]] for u in users}
print(f'users={len(users)}  movies={num_movies}', flush=True)

@torch.no_grad()
def hr_ndcg(score_of_user, target_of, seen_of, k=50):
    hr = ndcg = 0.0
    for u in users:
        scores = score_of_user(u).clone()
        scores[seen_of[u]] = -1e9; scores[0] = -1e9
        topk = torch.topk(scores, k).indices.tolist()
        t = target_of[u]
        if t in topk:
            hr += 1.0; ndcg += 1.0 / math.log2(topk.index(t) + 2)
    return hr / len(users) * 100, ndcg / len(users) * 100

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
    def __init__(self, nitems, d, nhead, nlayer, maxlen, drop):
        super().__init__()
        self.tok = nn.Embedding(nitems, d, padding_idx=0); self.pos = nn.Embedding(maxlen, d)
        self.blocks = nn.ModuleList([Block(d, nhead, maxlen, drop) for _ in range(nlayer)])
        self.ln = nn.LayerNorm(d); self.head = nn.Linear(d, nitems)
    def forward(self, idx):
        B, T = idx.shape
        x = self.tok(idx) + self.pos(torch.arange(T, device=idx.device))
        for b in self.blocks: x = b(x)
        return self.head(self.ln(x))

def run(seed, max_len, steps=5000, bs=128, d=64, nhead=2, nlayer=2, drop=0.3, eval_every=500):
    torch.manual_seed(seed)
    seq_t = torch.tensor([(train_seqs[u][-max_len:] + [0] * max_len)[:max_len] for u in users]).to(device)
    inp, tgt = seq_t[:, :-1], seq_t[:, 1:]
    model = SAS(num_movies + 1, d, nhead, nlayer, max_len, drop).to(device)
    opt = AdamW(model.parameters(), lr=3e-4)
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
    return hr_ndcg(lambda u: score_ctx(train_seqs[u] + [val_item[u]]), test_item, seen_trainval)

print('--- SASRec max_len sweep (test HR@50 / NDCG@50) ---', flush=True)
for ml in [20, 50, 100, 200]:
    H, N = [], []
    for seed in [1, 2, 3]:
        hr, nd = run(seed, ml)
        H.append(hr); N.append(nd)
        print(f'max_len={ml:3d} seed{seed}: HR {hr:5.2f}  NDCG {nd:5.2f}', flush=True)
    print(f'=> max_len={ml:3d}: HR@50 {st.mean(H):5.2f} +/- {st.pstdev(H):.2f}   '
          f'NDCG@50 {st.mean(N):5.2f} +/- {st.pstdev(N):.2f}', flush=True)
