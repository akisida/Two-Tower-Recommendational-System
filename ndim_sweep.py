# Эксперимент: как размер эмбеддинга влияет на Recall@50
import pandas as pd, torch, torch.nn as nn, torch.nn.functional as F, math
from torch.optim import Adam

import kagglehub
p = kagglehub.dataset_download('abhikjha/movielens-100k') + '/ml-latest-small'
ratings = pd.read_csv(p + '/ratings.csv'); movies = pd.read_csv(p + '/movies.csv')
pos_ratings = ratings[ratings['rating'] >= 4.0]
result = pd.merge(ratings, movies, on='movieId').drop(columns=['genres'])
pos_result = pd.merge(pos_ratings, movies, on='movieId').drop(columns=['genres'])
u2i = {id: i for i, id in enumerate(result['userId'].drop_duplicates())}
m2i = {id: i for i, id in enumerate(result['movieId'].drop_duplicates())}
for df in (result, pos_result):
    df['userIdx'] = df['userId'].map(u2i); df['movieIdx'] = df['movieId'].map(m2i)
pos_result = pos_result.sort_values(['userIdx', 'timestamp'])
pos_result['row'] = pos_result.groupby('userIdx').cumcount()
pos_result = pos_result.merge(pos_result.groupby('userIdx').size().reset_index(name='tot'), on='userIdx')
pos_result['pct'] = pos_result['row'] / pos_result['tot']
train = pos_result[pos_result.pct < 0.8]; test = pos_result[pos_result.pct >= 0.9]

tu = torch.tensor(train['userIdx'].values); tm = torch.tensor(train['movieIdx'].values)
tr_by_user = {u: torch.tensor(g['movieIdx'].values) for u, g in train.groupby('userIdx')}
te_by_user = {u: set(g['movieIdx'].values.tolist()) for u, g in test.groupby('userIdx')}
users = list(te_by_user.keys())

def recall_at_50(ue, me):
    s = 0.0
    with torch.no_grad():
        for u in users:
            sc = ue(torch.tensor(u)) @ me.weight.T
            sc[tr_by_user[u]] = float('-inf')
            _, idx = torch.topk(sc, 50)
            s += len(set(idx.tolist()) & te_by_user[u]) / len(te_by_user[u])
    return s / len(users) * 100

def train_and_eval(n_dim, steps=150000, lr=1e-3, bs=256):
    g = torch.Generator().manual_seed(2147483647)
    ue = nn.Embedding(len(u2i), n_dim); me = nn.Embedding(len(m2i), n_dim)
    opt = Adam(list(ue.parameters()) + list(me.parameters()), lr=lr)
    for i in range(steps):
        ix = torch.randint(0, len(tu), (bs,), generator=g)
        sc = (ue(tu[ix]) @ me(tm[ix]).T) / math.sqrt(n_dim)
        loss = F.cross_entropy(sc, torch.arange(bs))
        opt.zero_grad(); loss.backward(); opt.step()
    return loss.item(), recall_at_50(ue, me)

print('n_dim | train_loss | Recall@50', flush=True)
print('-' * 34, flush=True)
for n_dim in [16, 32, 64, 128]:
    loss, rec = train_and_eval(n_dim)
    print(f'{n_dim:5d} | {loss:10.3f} | {rec:.2f}%', flush=True)
