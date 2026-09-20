# Paired multi-seed: id-only vs content(genres)+regularization
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
for df in (result,pos_result):
    df['userIdx']=df['userId'].map(u2i); df['movieIdx']=df['movieId'].map(m2i)
pos_result=pos_result.sort_values(['userIdx','timestamp'])
pos_result['row']=pos_result.groupby('userIdx').cumcount()
pos_result=pos_result.merge(pos_result.groupby('userIdx').size().reset_index(name='tot'),on='userIdx')
pos_result['pct']=pos_result['row']/pos_result['tot']
train=pos_result[pos_result.pct<0.8]; val=pos_result[(pos_result.pct>=0.8)&(pos_result.pct<0.9)]; test=pos_result[pos_result.pct>=0.9]
n_dim,num_u,num_m=32,len(u2i),len(m2i)
tu=torch.tensor(train['userIdx'].values); tm=torch.tensor(train['movieIdx'].values)
Q=torch.tensor((train['movieIdx'].value_counts()/len(train)).reindex(range(num_m),fill_value=0).values,dtype=torch.float32)

# genres aligned to movieIdx
gd=movies.set_index('movieId')['genres'].str.get_dummies('|').drop(columns=['(no genres listed)'],errors='ignore').reset_index()
gd['movieIdx']=gd['movieId'].map(m2i)
gd=gd.dropna(subset=['movieIdx']).sort_values('movieIdx')
gcols=[c for c in gd.columns if c not in ('movieId','movieIdx')]
genres=torch.tensor(gd[gcols].values,dtype=torch.float32); n_genres=genres.shape[1]

vu_all=torch.tensor(val['userIdx'].values); vm_all=torch.tensor(val['movieIdx'].values)
gv=torch.Generator().manual_seed(0); vix=torch.randint(0,len(vu_all),(1024,),generator=gv)
val_u=vu_all[vix]; val_m=vm_all[vix]
tr_by_user={u:torch.tensor(g['movieIdx'].values) for u,g in train.groupby('userIdx')}
te_by_user={u:set(g['movieIdx'].values.tolist()) for u,g in test.groupby('userIdx')}
users=list(te_by_user.keys())

class Item(nn.Module):
    def __init__(self,p_drop):
        super().__init__()
        self.movie_emb=nn.Embedding(num_m,n_dim)
        if p_drop is not None:
            self.genre_W=nn.Linear(n_genres,n_dim,bias=False)
            self.drop=nn.Dropout(p_drop); self.proj=nn.Linear(2*n_dim,n_dim)
        self.content=p_drop is not None
    def forward(self,ix):
        e=self.movie_emb(ix)
        if not self.content: return e
        ge=self.genre_W(genres[ix])
        return self.proj(self.drop(torch.cat([e,ge],1)))

def run(seed, content):
    torch.manual_seed(seed); g=torch.Generator().manual_seed(seed)
    ue=nn.Embedding(num_u,n_dim)
    it=Item(0.3 if content else None)
    wd=1e-4 if content else 0.0
    opt=Adam(list(ue.parameters())+list(it.parameters()),lr=1e-3,weight_decay=wd)
    best_val=float('inf'); best=None; ar=torch.arange(256)
    for i in range(200000):
        ue.train(); it.train()
        ix=torch.randint(0,len(tu),(256,),generator=g)
        s=(ue(tu[ix])@it(tm[ix]).T)/math.sqrt(n_dim)-torch.log(Q[tm[ix]])
        loss=F.cross_entropy(s,ar); opt.zero_grad(); loss.backward(); opt.step()
        if i%10000==0:
            ue.eval(); it.eval()
            with torch.no_grad():
                sv=(ue(val_u)@it(val_m).T)/math.sqrt(n_dim)-torch.log(Q[val_m].clamp(min=1e-9))
                vl=F.cross_entropy(sv,torch.arange(len(val_u))).item()
            if vl<best_val: best_val=vl; best={'u':copy.deepcopy(ue.state_dict()),'i':copy.deepcopy(it.state_dict())}
    ue.load_state_dict(best['u']); it.load_state_dict(best['i'])
    ue.eval(); it.eval()
    with torch.no_grad():
        allit=it(torch.arange(num_m))
        s=0.0
        for u in users:
            sc=ue(torch.tensor(u))@allit.T
            sc[tr_by_user[u]]=float('-inf')
            _,idx=torch.topk(sc,50)
            s+=len(set(idx.tolist())&te_by_user[u])/len(te_by_user[u])
    return s/len(users)*100

print('seed | id-only  content  delta', flush=True)
d=[]; a=[]; b=[]
for seed in [1,2,3]:
    ri=run(seed,False); rc=run(seed,True)
    d.append(rc-ri); a.append(ri); b.append(rc)
    print(f'{seed:4d} | {ri:6.2f}  {rc:6.2f}  {rc-ri:+.2f}', flush=True)
import statistics as st
print('---', flush=True)
print(f'id-only: mean {st.mean(a):.2f} std {st.pstdev(a):.2f}', flush=True)
print(f'content: mean {st.mean(b):.2f} std {st.pstdev(b):.2f}', flush=True)
print(f'delta(content-id): mean {st.mean(d):+.2f} std {st.pstdev(d):.2f}', flush=True)
