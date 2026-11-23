import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score, mean_absolute_error
from config import *
def is_clf(name): return '(classifier)' in name
def score(name,an,p):   # higher = more likely abnormal
    if is_clf(name): return p
    return -p if an=='ica' else p
def pos(an,y): return (y<THR['ica']) if an=='ica' else (y>THR['lact'])
def point(name,an,y,p):
    m=~np.isnan(p); y,p=y[m],p[m]; t=pos(an,y).astype(int); s=score(name,an,p); r={'n':int(m.sum())}
    if not is_clf(name):
        r['MAE']=mean_absolute_error(y,p); r['bias']=float(np.mean(p-y)); r['cal_slope']=float(np.polyfit(p,y,1)[0]) if np.std(p)>0 else np.nan
        pr=pos(an,p)
    else:
        pr=p>=0.5
        lp=np.log(np.clip(p,1e-4,1-1e-4)/(1-np.clip(p,1e-4,1-1e-4)))
        import statsmodels.api as sm
        try: c=sm.GLM(t,sm.add_constant(lp),family=sm.families.Binomial()).fit().params; r['cal_int'],r['cal_slope']=float(c[0]),float(c[1])
        except Exception: pass
    r['AUROC']=roc_auc_score(t,s) if 0<t.sum()<len(t) else np.nan; r['AUPRC']=average_precision_score(t,s) if t.sum()>0 else np.nan
    r['sens']=float(pr[t==1].mean()) if t.sum() else np.nan; r['spec']=float((~pr)[t==0].mean()) if (t==0).sum() else np.nan; r['prev']=float(t.mean())
    return r
def boot_table(D,an,P,yname,refs,rng,B=NBOOT,mask=None):
    y=D[yname].values; g=D.id.values
    if mask is not None: y,g,P=y[mask],g[mask],{k:v[mask] for k,v in P.items()}
    ids=np.unique(g); idx={i:np.where(g==i)[0] for i in ids}
    base={k:point(k,an,y,v) for k,v in P.items()}
    bs={k:[] for k in P}; dl={(k,r):[] for k in P for r in refs if k!=r}
    for _ in range(B):
        s=np.concatenate([idx[i] for i in rng.choice(ids,len(ids))])
        pts={k:point(k,an,y[s],v[s]) for k,v in P.items()}
        for k in P: bs[k].append(pts[k])
        for (k,r) in dl:
            common=~np.isnan(P[k][s])&~np.isnan(P[r][s]); ys=y[s][common]
            a=point(k,an,ys,P[k][s][common]); b=point(r,an,ys,P[r][s][common])
            dl[(k,r)].append({'dAUROC':a['AUROC']-b['AUROC'],'dMAE':(b.get('MAE',np.nan)-a.get('MAE',np.nan))})
    rows=[]
    for k in P:
        b=pd.DataFrame(bs[k]); row={'model':k,**{m:base[k].get(m,np.nan) for m in ['n','prev','MAE','bias','cal_int','cal_slope','AUROC','AUPRC','sens','spec']}}
        for m in ['MAE','bias','cal_slope','AUROC','AUPRC','sens','spec']:
            if m in b: row[m+'_lo'],row[m+'_hi']=np.nanpercentile(b[m],2.5),np.nanpercentile(b[m],97.5)
        rows.append(row)
    drow=[]
    for (k,r),v in dl.items():
        v=pd.DataFrame(v); common=~np.isnan(P[k])&~np.isnan(P[r])
        a=point(k,an,y[common],P[k][common]); b=point(r,an,y[common],P[r][common])
        drow.append({'model':k,'vs':r,'n_identical_obs':int(common.sum()),'dAUROC':a['AUROC']-b['AUROC'],'dAUROC_lo':np.nanpercentile(v.dAUROC,2.5),'dAUROC_hi':np.nanpercentile(v.dAUROC,97.5),
                     'MAE_reduction':b.get('MAE',np.nan)-a.get('MAE',np.nan),'MAE_red_lo':np.nanpercentile(v.dMAE,2.5),'MAE_red_hi':np.nanpercentile(v.dMAE,97.5)})
    return pd.DataFrame(rows),pd.DataFrame(drow)
