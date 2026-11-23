"""Build reference-paired datasets. Usage: python build.py dev,val  (test is built only after freeze)."""
import sys, pandas as pd, numpy as np, duckdb
from config import *
splits=sys.argv[1].split(',')
c=duckdb.connect(); P=pd.read_parquet(f'{DISC}/pts.parquet'); P=P[P.dsplit.isin(splits)]
c.register('p',P[['id','hadm_id','intime','outtime','dsplit']])
allitems={**CHEM,**OTHER,REF['ica']:'ica',REF['lact']:'lact'}
case=' '.join(f"when {k} then '{v}'" for k,v in allitems.items())
L=c.execute(f"""select p.id, p.dsplit, date_diff('second',p.intime,l.charttime)/3600.0 h, case itemid {case} end a, valuenum v
  from '{O}/m_labv.parquet' l join p using(hadm_id) where itemid in ({','.join(map(str,allitems))}) and valuenum is not null""").df()
inp=pd.read_parquet(f'{DISC}/inp_long.parquet'); pr=pd.read_parquet(f'{DISC}/proc_long.parquet')
P['los_h']=(P.outtime-P.intime).dt.total_seconds()/3600
W=L.pivot_table(index=['id','dsplit','h'],columns='a',values='v',aggfunc='mean').reset_index().sort_values('h')
chemcols=list(CHEM.values())
chem=W.dropna(subset=['na','k','cl','hco3','bun','cr','glu'])[['id','h']+chemcols].rename(columns={'h':'h_chem'}).sort_values('h_chem')
def nearest(D,col,tol_h,name):
    x=W.dropna(subset=[col])[['id','h',col]].rename(columns={'h':'h_'+name}).sort_values('h_'+name)
    return pd.merge_asof(D.sort_values('h_ref'),x,left_on='h_ref',right_on='h_'+name,by='id',direction='nearest',tolerance=tol_h)
out={}
for an in ['ica','lact']:
    R=W.dropna(subset=[an])[['id','dsplit','h',an]+(['ph','pco2'] if an else [])].rename(columns={'h':'h_ref'})
    R=R.merge(P[['id','los_h']],on='id'); R=R[(R.h_ref>=0)&(R.h_ref<=R.los_h)].drop(columns='los_h')    # ICU stay only (exposures recorded only in ICU)
    n_ref=len(R)
    D=pd.merge_asof(R.sort_values('h_ref'),chem,left_on='h_ref',right_on='h_chem',by='id',direction='nearest',tolerance=PAIR_CHEM_MIN/60)
    D=D.dropna(subset=['na'])
    if an=='ica': D=D.dropna(subset=['ca'])
    D['gap_min']=(D.h_chem-D.h_ref)*60
    for col in ['phos','mg','hgb','wbc','plt']: D=nearest(D,col,PAIR_OTHER_H,col)
    D=nearest(D,'alb',PAIR_ALB_H,'alb')
    for k,its in ITEMS.items():
        x=inp[inp.itemid.isin(its)]; gp={i:g[['s','e']].values for i,g in x.groupby('id')}
        D[k+'6']=[int(gp.get(r.id) is not None and ((gp[r.id][:,0]<=r.h_ref)&(gp[r.id][:,1]>=r.h_ref-EXPO_H)).any()) for r in D.itertuples()]
    rr=pr[pr.itemid.isin(ITEMS['rrt'])]; gp={i:g[['s','e']].values for i,g in rr.groupby('id')}
    D['rrt_active']=[int(gp.get(r.id) is not None and ((gp[r.id][:,0]<=r.h_ref)&(gp[r.id][:,1]>=r.h_ref)).any()) for r in D.itertuples()]
    D['ag']=D.na-D.cl-D.hco3; D['ca_mmol']=D.ca*CA_MG_TO_MMOL if 'ca' in D else np.nan
    D['day']=(D.h_ref//24).astype(int)
    flow={'references in ICU':n_ref,'paired with chemistry +/-60 min':len(D)}
    D=D[D.rrt_active==0]; flow['excluding active RRT']=len(D)
    out[an]=(D,flow)
    D.to_parquet(f'pairs_{an}_{"_".join(splits)}.parquet'); print(an,flow,'patients',D.id.nunique(),D.groupby('dsplit').id.nunique().to_dict())
# all ICU chemistry panels (for selection/measurement-propensity analysis)
chem2=chem.merge(W[['id','dsplit']].drop_duplicates(),on='id').merge(P[['id','los_h']],on='id'); chem2=chem2[(chem2.h_chem>=0)&(chem2.h_chem<=chem2.los_h)]
for an in ['ica','lact']:
    ref=W.dropna(subset=[an])[['id','h']]; rg={i:g.h.values for i,g in ref.groupby('id')}
    chem2['has_'+an]=[int(rg.get(r.id) is not None and (np.abs(rg[r.id]-r.h_chem)<=1).any()) for r in chem2.itertuples()]
chem2.to_parquet(f'panels_{"_".join(splits)}.parquet'); print('ICU chemistry panels',len(chem2),'| with iCa',chem2.has_ica.mean().round(3),'| with lactate',chem2.has_lact.mean().round(3))
