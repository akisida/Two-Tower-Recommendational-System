# Строго: standard logQ vs positive-uncorrected logQ (paired, 4 seeds)
import pandas as pd, torch, torch.nn as nn, torch.nn.functional as F, math
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
train=pos_result[pos_result.pct<0.8]; test=pos_result[pos_result.pct>=0.9]
n_dim,num_u,num_m=32,len(u2i),len(m2i)
tu=torch.tensor(train['userIdx'].values); tm=torch.tensor(train['movieIdx'].values)
Q=torch.tensor((train['movieIdx'].value_counts()/len(train)).reindex(range(num_m),fill_value=0).values,dtype=torch.float32)
tr_by_user={u:torch.tensor(g['movieIdx'].values) for u,g in train.groupby('userIdx')}
te_by_user={u:set(g['movieIdx'].values.tolist()) for u,g in test.groupby('userIdx')}
users=list(te_by_user.keys())

def recall(ue,me):
    s=0.0
    with torch.no_grad():
        for u in users:
            sc=ue(torch.tensor(u))@me.weight.T
            sc[tr_by_user[u]]=float('-inf')
            _,idx=torch.topk(sc,50)
            s+=len(set(idx.tolist())&te_by_user[u])/len(te_by_user[u])
    return s/len(users)*100

def run(seed, fix_positive):
    torch.manual_seed(seed); g=torch.Generator().manual_seed(seed)
    ue=nn.Embedding(num_u,n_dim); me=nn.Embedding(num_m,n_dim)
    opt=Adam(list(ue.parameters())+list(me.parameters()),lr=1e-3)
    ar=torch.arange(256)
    for i in range(200000):
        ix=torch.randint(0,len(tu),(256,),generator=g)
        lq=torch.log(Q[tm[ix]])
        s=(ue(tu[ix])@me(tm[ix]).T)/math.sqrt(n_dim) - lq
        if fix_positive:
            s[ar,ar]=s[ar,ar]+lq          # undo correction on positive
        loss=F.cross_entropy(s, ar)
        opt.zero_grad(); loss.backward(); opt.step()
    return recall(ue,me)

print('seed | standard  fixed   delta(fixed-standard)', flush=True)
d=[]; st=[]; fx=[]
for seed in [1,2,3,4]:
    rs=run(seed, False); rf=run(seed, True)
    d.append(rf-rs); st.append(rs); fx.append(rf)
    print(f'{seed:4d} | {rs:7.2f}  {rf:6.2f}  {rf-rs:+.2f}', flush=True)
import statistics as s
print('---', flush=True)
print(f'standard: mean {s.mean(st):.2f} std {s.pstdev(st):.2f}', flush=True)
print(f'fixed:    mean {s.mean(fx):.2f} std {s.pstdev(fx):.2f}', flush=True)
print(f'delta(fixed-standard): mean {s.mean(d):+.2f} std {s.pstdev(d):.2f}', flush=True)
