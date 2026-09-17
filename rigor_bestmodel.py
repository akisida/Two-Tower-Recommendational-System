# Строгая проверка best-model-by-val: 5 сидов, val каждые 5000, last vs best
import pandas as pd, torch, torch.nn as nn, torch.nn.functional as F, math, copy
from torch.optim import Adam
import kagglehub

p = kagglehub.dataset_download('abhikjha/movielens-100k') + '/ml-latest-small'
ratings = pd.read_csv(p+'/ratings.csv'); movies = pd.read_csv(p+'/movies.csv')
pos = ratings[ratings['rating']>=4.0]
result = pd.merge(ratings, movies, on='movieId').drop(columns=['genres'])
pos_result = pd.merge(pos, movies, on='movieId').drop(columns=['genres'])
u2i={id:i for i,id in enumerate(result['userId'].drop_duplicates())}
m2i={id:i for i,id in enumerate(result['movieId'].drop_duplicates())}
for df in (result, pos_result):
    df['userIdx']=df['userId'].map(u2i); df['movieIdx']=df['movieId'].map(m2i)
pos_result=pos_result.sort_values(['userIdx','timestamp'])
pos_result['row']=pos_result.groupby('userIdx').cumcount()
pos_result=pos_result.merge(pos_result.groupby('userIdx').size().reset_index(name='tot'),on='userIdx')
pos_result['pct']=pos_result['row']/pos_result['tot']
train=pos_result[pos_result.pct<0.8]; val=pos_result[(pos_result.pct>=0.8)&(pos_result.pct<0.9)]; test=pos_result[pos_result.pct>=0.9]

n_dim, num_u, num_m = 32, len(u2i), len(m2i)
tu=torch.tensor(train['userIdx'].values); tm=torch.tensor(train['movieIdx'].values)
Q=torch.tensor((train['movieIdx'].value_counts()/len(train)).reindex(range(num_m),fill_value=0).values, dtype=torch.float32)

# fixed val batch (same across seeds)
vu_all=torch.tensor(val['userIdx'].values); vm_all=torch.tensor(val['movieIdx'].values)
gv=torch.Generator().manual_seed(0); vix=torch.randint(0,len(vu_all),(1024,),generator=gv)
val_u=vu_all[vix]; val_m=vm_all[vix]

tr_by_user={u:torch.tensor(g['movieIdx'].values) for u,g in train.groupby('userIdx')}
te_by_user={u:set(g['movieIdx'].values.tolist()) for u,g in test.groupby('userIdx')}
users=list(te_by_user.keys())

def val_loss(ue,me):
    with torch.no_grad():
        s=(ue(val_u)@me(val_m).T)/math.sqrt(n_dim) - torch.log(Q[val_m].clamp(min=1e-9))
        return F.cross_entropy(s, torch.arange(len(val_u))).item()

def recall(ue,me):
    s=0.0
    with torch.no_grad():
        for u in users:
            sc=ue(torch.tensor(u))@me.weight.T
            sc[tr_by_user[u]]=float('-inf')
            _,idx=torch.topk(sc,50)
            s+=len(set(idx.tolist())&te_by_user[u])/len(te_by_user[u])
    return s/len(users)*100

def run_seed(seed):
    torch.manual_seed(seed)
    g=torch.Generator().manual_seed(seed)
    ue=nn.Embedding(num_u,n_dim); me=nn.Embedding(num_m,n_dim)
    opt=Adam(list(ue.parameters())+list(me.parameters()),lr=1e-3)
    best_val=float('inf'); best_state=None
    for i in range(200000):
        ix=torch.randint(0,len(tu),(256,),generator=g)
        s=(ue(tu[ix])@me(tm[ix]).T)/math.sqrt(n_dim) - torch.log(Q[tm[ix]])
        loss=F.cross_entropy(s, torch.arange(256))
        opt.zero_grad(); loss.backward(); opt.step()
        if i % 5000 == 0:
            vl=val_loss(ue,me)
            if vl<best_val: best_val=vl; best_state={'u':copy.deepcopy(ue.state_dict()),'m':copy.deepcopy(me.state_dict())}
    r_last=recall(ue,me)
    ue.load_state_dict(best_state['u']); me.load_state_dict(best_state['m'])
    r_best=recall(ue,me)
    return r_last, r_best

print('seed | last%  best%  delta', flush=True)
deltas=[]; lasts=[]; bests=[]
for seed in [1,2,3,4,5]:
    rl,rb=run_seed(seed)
    deltas.append(rb-rl); lasts.append(rl); bests.append(rb)
    print(f'{seed:4d} | {rl:5.2f}  {rb:5.2f}  {rb-rl:+.2f}', flush=True)

import statistics as st
print('---', flush=True)
print(f'last: mean {st.mean(lasts):.2f} std {st.pstdev(lasts):.2f}', flush=True)
print(f'best: mean {st.mean(bests):.2f} std {st.pstdev(bests):.2f}', flush=True)
print(f'delta(best-last): mean {st.mean(deltas):+.2f} std {st.pstdev(deltas):.2f}', flush=True)
