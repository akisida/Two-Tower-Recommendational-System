import os, glob, math, collections, copy, statistics as st
import pandas as pd, torch, torch.nn as nn, torch.nn.functional as F
from torch.optim import Adam, AdamW

# ----------------- CONFIG (tweak here) -----------------
SEEDS        = [1, 2, 3]
MAX_LEN      = 200          # SASRec context window (ML-1M users are long)
# SASRec
SAS_D        = 128          # embedding dim (was 64 on ml-small)
SAS_HEADS    = 2            # try 4 (head_size 32) if you like
SAS_LAYERS   = 2
SAS_DROPOUT  = 0.2
SAS_STEPS    = 40000
SAS_BATCH    = 64           # keep <=64 at max_len=200 to stay inside 4GB VRAM
SAS_LR       = 1e-3
# two-tower
TT_D         = 64           # was 32 on ml-small; more data supports more capacity
TT_STEPS     = 150000
TT_BATCH     = 512
TT_LR        = 1e-3
TT_WD        = 1e-5
EVAL_EVERY_SAS = 4000       # how often to check val HR for best-model selection
EVAL_EVERY_TT  = 15000
VAL_SUBSET     = 1500       # users sampled for val monitoring (full test eval at the end)
# -------------------------------------------------------

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('device:', device, flush=True)

# ----------------- data: MovieLens-1M -----------------
try:
    import kagglehub
    base = kagglehub.dataset_download('odedgolden/movielens-1m-dataset')
except Exception as e:
    print('kagglehub failed, falling back to ML1M_DIR:', e, flush=True)
    base = os.environ.get('ML1M_DIR', '.')
cands = glob.glob(os.path.join(base, '**', 'ratings.dat'), recursive=True)
assert cands, "ratings.dat not found. Set ML1M_DIR to the folder with ratings.dat."
ratings = pd.read_csv(cands[0], sep='::', engine='python',
                      names=['userId', 'movieId', 'rating', 'timestamp'], encoding='latin-1')
print('ratings:', len(ratings), flush=True)

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
        continue
    train_seqs[u] = s[:-2]; val_item[u] = s[-2]; test_item[u] = s[-1]
users = list(train_seqs.keys())
seen_train    = {u: train_seqs[u] for u in users}
seen_trainval = {u: train_seqs[u] + [val_item[u]] for u in users}
print(f'users={len(users)}  movies={num_movies}  interactions={len(pos)}', flush=True)

# fixed subset for val monitoring (keeps best-model selection cheap)
_g = torch.Generator().manual_seed(0)
_sub_ix = torch.randperm(len(users), generator=_g)[:min(VAL_SUBSET, len(users))].tolist()
val_users = [users[i] for i in _sub_ix]

tt_u, tt_m = [], []
for u in users:
    for m in train_seqs[u]:
        tt_u.append(u); tt_m.append(m)
tt_u = torch.tensor(tt_u); tt_m = torch.tensor(tt_m)
cnt = collections.Counter(tt_m.tolist())
Q = torch.zeros(num_movies + 1)
for m, c in cnt.items():
    Q[m] = c / len(tt_m)

seq_tensor = torch.tensor([(train_seqs[u][-MAX_LEN:] + [0] * MAX_LEN)[:MAX_LEN] for u in users])

# ----------------- shared metric -----------------
@torch.no_grad()
def hr_ndcg(score_of_user, target_of, seen_of, user_list, k=50):
    hr = ndcg = 0.0
    for u in user_list:
        scores = score_of_user(u).clone()
        scores[seen_of[u]] = -1e9; scores[0] = -1e9
        topk = torch.topk(scores, k).indices.tolist()
        t = target_of[u]
        if t in topk:
            hr += 1.0; ndcg += 1.0 / math.log2(topk.index(t) + 2)
    return hr / len(user_list) * 100, ndcg / len(user_list) * 100

# ----------------- two-tower -----------------
def train_two_tower(seed):
    torch.manual_seed(seed)
    U = nn.Embedding(num_users, TT_D).to(device)
    M = nn.Embedding(num_movies + 1, TT_D, padding_idx=0).to(device)
    opt = Adam(list(U.parameters()) + list(M.parameters()), lr=TT_LR, weight_decay=TT_WD)
    Qd = Q.to(device); ar = torch.arange(TT_BATCH, device=device)
    g = torch.Generator().manual_seed(seed)
    def score(u):
        with torch.no_grad():
            return (U(torch.tensor(u, device=device)) @ M.weight.data.T).detach().cpu()
    best_val, best = -1.0, None
    for i in range(TT_STEPS):
        if i % EVAL_EVERY_TT == 0:
            U.eval(); M.eval()
            vh, _ = hr_ndcg(score, val_item, seen_train, val_users)
            if vh > best_val:
                best_val = vh; best = (copy.deepcopy(U.state_dict()), copy.deepcopy(M.state_dict()))
            U.train(); M.train()
        ix = torch.randint(0, len(tt_u), (TT_BATCH,), generator=g)
        ub = tt_u[ix].to(device); mb = tt_m[ix].to(device)
        s = (U(ub) @ M(mb).T) / math.sqrt(TT_D) - torch.log(Qd[mb])
        loss = F.cross_entropy(s, ar)
        opt.zero_grad(); loss.backward(); opt.step()
    U.load_state_dict(best[0]); M.load_state_dict(best[1]); U.eval(); M.eval()
    return score

# ----------------- SASRec -----------------
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

def train_sasrec(seed):
    torch.manual_seed(seed)
    model = SAS(num_movies + 1, SAS_D, SAS_HEADS, SAS_LAYERS, MAX_LEN, SAS_DROPOUT).to(device)
    opt = AdamW(model.parameters(), lr=SAS_LR)
    sd = seq_tensor.to(device); inp, tgt = sd[:, :-1], sd[:, 1:]
    g = torch.Generator().manual_seed(seed)
    def score_ctx(ctx):
        s = ctx[-MAX_LEN:]; k = len(s)
        x = torch.tensor([s + [0] * (MAX_LEN - k)], device=device)
        with torch.no_grad():
            lg = model(x)
        return lg[0, k - 1].detach().cpu()
    val_score = lambda u: score_ctx(train_seqs[u])
    best_val, best = -1.0, None
    for i in range(SAS_STEPS):
        if i % EVAL_EVERY_SAS == 0:
            model.eval()
            vh, _ = hr_ndcg(val_score, val_item, seen_train, val_users)
            if vh > best_val:
                best_val = vh; best = copy.deepcopy(model.state_dict())
            model.train()
        ix = torch.randint(0, inp.shape[0], (SAS_BATCH,), generator=g).to(device)
        logits = model(inp[ix])
        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), tgt[ix].reshape(-1), ignore_index=0)
        opt.zero_grad(); loss.backward(); opt.step()
    model.load_state_dict(best); model.eval()
    return lambda u: score_ctx(train_seqs[u] + [val_item[u]])

# ----------------- run (full test eval over ALL users) -----------------
print('model       seed | test HR@50   NDCG@50', flush=True)
res = {'two-tower': ([], []), 'SASRec': ([], [])}
for seed in SEEDS:
    sc = train_two_tower(seed)
    hr, nd = hr_ndcg(sc, test_item, seen_trainval, users)
    res['two-tower'][0].append(hr); res['two-tower'][1].append(nd)
    print(f'two-tower   {seed}    | {hr:6.2f}     {nd:5.2f}', flush=True)
    sc = train_sasrec(seed)
    hr, nd = hr_ndcg(sc, test_item, seen_trainval, users)
    res['SASRec'][0].append(hr); res['SASRec'][1].append(nd)
    print(f'SASRec      {seed}    | {hr:6.2f}     {nd:5.2f}', flush=True)
print('---', flush=True)
print(f'random HR@50 baseline ~ {50 / num_movies * 100:.3f}%', flush=True)
for name, (H, N) in res.items():
    print(f'{name:10s}  test HR@50 {st.mean(H):6.2f} +/- {st.pstdev(H):.2f}   '
          f'NDCG@50 {st.mean(N):5.2f} +/- {st.pstdev(N):.2f}', flush=True)
