"""Audit + subgroup + sensitivity analyses on dev (OOF) and val. Test set not touched."""
import pandas as pd, numpy as np, sys, warnings; warnings.filterwarnings('ignore')
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.inspection import permutation_importance
from config import *; from models import *; from metrics import *
rng=np.random.default_rng(SEED+1); B=300
PRIM={'ica':'Ridge regression','lact':'Gradient boosting'}; BASE={'ica':'B1 total Ca (calibrated)','lact':'B2 anion gap + HCO3 (calibrated)'}
def cb(y,g,pa,pb,an,w=None):
    """cluster bootstrap: AUROC A, AUROC B, delta; MAE A, MAE B"""
    m=~np.isnan(pa)&~np.isnan(pb); y,g,pa,pb=y[m],g[m],pa[m],pb[m]; w=None if w is None else w[m]
    t=pos(an,y).astype(int); sa=-pa if an=='ica' else pa; sb=-pb if an=='ica' else pb
    if t.sum()<5 or t.sum()>len(t)-5: return dict(n=len(y),npos=int(t.sum()))
    f=lambda s,i: roc_auc_score(t[i],s[i],sample_weight=None if w is None else w[i])
    ids=np.unique(g); ix={i:np.where(g==i)[0] for i in ids}; d=[]
    for _ in range(B):
        s=np.concatenate([ix[i] for i in rng.choice(ids,len(ids))])
        if 0<t[s].sum()<len(s): d.append(f(sa,s)-f(sb,s))
    a=np.arange(len(y))
    return dict(n=len(y),pts=len(ids),npos=int(t.sum()),AUC_model=f(sa,a),AUC_base=f(sb,a),dAUC=f(sa,a)-f(sb,a),lo=np.percentile(d,2.5),hi=np.percentile(d,97.5),
                MAE_model=np.mean(np.abs(pa-y)),MAE_base=np.mean(np.abs(pb-y)))
out=[]
def rec(an,split,analysis,stratum,r): out.append(dict(analyte=an,split=split,analysis=analysis,stratum=stratum,**r)); print(an,split,analysis,stratum,{k:(round(v,3) if isinstance(v,float) else v) for k,v in r.items()})
for an in ['ica','lact']:
    D=pd.read_parquet(f'pairs_{an}_dev_val.parquet'); dev=D[D.dsplit=='dev'].reset_index(drop=True); val=D[D.dsplit=='val'].reset_index(drop=True)
    O=pd.read_parquet(f'out/dev_{an}_oof.parquet'); Pv=fit_predict(dev,val,an)
    sets={'dev':(dev,{k:O[k].values for k in O if k not in('y','id')}),'val':(val,Pv)}
    # --- audit descriptives
    for s,(X,_) in sets.items():
        print(f'\n[{an} {s}] |gap| min median/IQR',np.percentile(np.abs(X.gap_min),[50,25,75,95]).round(0),' chem after ref %:',round((X.gap_min>0).mean()*100,1),
              ' per-pt median pairs',X.groupby('id').size().median(),'max',X.groupby('id').size().max())
        print('  missing %:',(X[['alb','phos','mg','hgb','wbc','plt']].isna().mean()*100).round(1).to_dict(),' exposures %:',(X[['transfusion6','iv_calcium6','vasopressor6']].mean()*100).round(1).to_dict())
    # --- primary vs baseline, overall + subgroups
    for s,(X,P) in sets.items():
        y=X[an].values; g=X.id.values; pa,pb=P[PRIM[an]],P[BASE[an]]
        rec(an,s,'primary vs baseline','all',cb(y,g,pa,pb,an))
        if an=='ica':
            subs={'albumin <3.0 g/dL':X.alb<3.0,'albumin >=3.0':X.alb>=3.0,'albumin not measured':X.alb.isna(),'transfusion <=6h':X.transfusion6==1,'IV calcium <=6h':X.iv_calcium6==1,
                  'creatinine >=1.5':X.cr>=1.5,'pH <7.30 (stratifier only)':X.ph<7.30,'pH >7.45 (stratifier only)':X.ph>7.45}
        else:
            subs={'vasopressor <=6h':X.vasopressor6==1,'creatinine >=1.5':X.cr>=1.5,'HCO3 <22':X.hco3<22,'anion gap <=12':X.ag<=12,'anion gap <16':X.ag<16,'pH <7.30 (stratifier only)':X.ph<7.30}
        for k,m in subs.items():
            m=m.values; rec(an,s,'subgroup',k,cb(y[m],g[m],pa[m],pb[m],an))
        if an=='lact':  # severe hyperlactatemia >4: AUROC of continuous estimates
            t4=(y>THR['lact_severe']).astype(int)
            rec(an,s,'severe >4 mmol/L detection','all',dict(npos=int(t4.sum()),AUC_model=roc_auc_score(t4,pa),AUC_base=roc_auc_score(t4,pb)))
            m=X.ag.values<16; t=(y>2)[m]
            rec(an,s,'hyperlactatemia with AG<16 flagged','AG<16',dict(n=int(m.sum()),npos=int(t.sum()),model_sens=float((pa[m]>2)[t].mean()),AGrule_sens=0.0,model_spec=float((pa[m]<=2)[~t].mean())))
        if an=='ica':   # corrected-Ca raw rules on albumin-measured observations
            m=X.alb.notna().values; Y=y[m]<THR['ica']; ca=X.ca.values[m]; al=X.alb.values[m]
            for nm,v in [('total Ca <8.5 mg/dL',ca),('corrected 0.8 <8.5',ca+0.8*(4-al)),('Payne 1.0 <8.5',ca+(4-al))]:
                pr=v<8.5; lo=al<3.0
                rec(an,s,'uncalibrated bedside rule (albumin-measured obs)',nm,dict(n=int(m.sum()),npos=int(Y.sum()),sens=float(pr[Y].mean()),spec=float((~pr)[~Y].mean()),
                    false_reassure_lowalb=float((~pr)[Y&lo].mean()) if (Y&lo).sum() else np.nan,n_hypo_lowalb=int((Y&lo).sum())))
            pr=pa[m]<THR['ica']; lo=al<3.0
            rec(an,s,'uncalibrated bedside rule (albumin-measured obs)','primary model <1.12',dict(n=int(m.sum()),npos=int(Y.sum()),sens=float(pr[Y].mean()),spec=float((~pr)[~Y].mean()),false_reassure_lowalb=float((~pr)[Y&lo].mean()),n_hypo_lowalb=int((Y&lo).sum())))
    # --- sensitivity analyses on val (trained on dev), each vs the same baseline on identical obs
    y=val[an].values; g=val.id.values; pb=Pv[BASE[an]]
    pa=fit_predict(dev,val,an,miss_ind=False)[PRIM[an]]; rec(an,'val','sensitivity','no missingness indicators',cb(y,g,pa,pb,an))
    feats=[f for f in (F_ICA if an=='ica' else F_LACT) if not f.endswith('6')]
    pa=fit_predict(dev,val,an,miss_ind=False,feats=feats)[PRIM[an]]; rec(an,'val','sensitivity','chemistry only (no exposures, no indicators)',cb(y,g,pa,pb,an))
    core=['na','k','cl','hco3','bun','cr','glu','ca']+(['ag'] if an=='lact' else [])
    pa=fit_predict(dev,val,an,miss_ind=False,feats=core)[PRIM[an]]; rec(an,'val','sensitivity','basic metabolic panel only',cb(y,g,pa,pb,an))
    m=(val.gap_min<=0).values; dm=dev[dev.gap_min<=0]
    P2=fit_predict(dm,val[m],an); rec(an,'val','sensitivity','chemistry drawn at/before reference only',cb(y[m],g[m],P2[PRIM[an]],P2[BASE[an]],an))
    one=lambda X: X.loc[X.assign(a=X.gap_min.abs()).groupby(['id','day']).a.idxmin()]
    d1,v1=one(dev),one(val).reset_index(drop=True); P3=fit_predict(d1,v1,an)
    rec(an,'val','sensitivity','one pair per patient-day',cb(v1[an].values,v1.id.values,P3[PRIM[an]],P3[BASE[an]],an))
    # IPW: propensity of measurement from panel chemistry (dev panels), weights applied to val pairs
    Pn=pd.read_parquet('panels_dev_val.parquet'); flag='has_ica' if an=='ica' else 'has_lact'
    pf=['na','k','cl','hco3','bun','cr','glu','ca']; Pn['dd']=Pn.h_chem//24
    pd_=Pn[Pn.dsplit=='dev'].dropna(subset=pf); lr=make_pipeline(StandardScaler(),LogisticRegression(max_iter=2000)).fit(pd_[pf+['dd']],pd_[flag])
    vv=val.assign(dd=val.day).dropna(subset=pf); ps=lr.predict_proba(vv[pf+['dd']])[:,1]; w=np.clip(1/ps,None,np.percentile(1/ps,99))
    ii=vv.index.values; rec(an,'val','sensitivity','IPW for measurement selection',cb(y[ii],g[ii],Pv[PRIM[an]][ii],pb[ii],an,w=w))
    pv=Pn[Pn.dsplit=='val'].dropna(subset=pf); print(f'  selection AUROC (who gets {an} tested, val panels):',round(roc_auc_score(pv[flag],lr.predict_proba(pv[pf+['dd']])[:,1]),3),
        ' measured vs unmeasured panel means:\n',Pn.groupby(flag)[pf].mean().round(2).to_string())
    # permutation importance of primary model on val
    Xtr,Xte=design(dev,val,F_ICA if an=='ica' else F_LACT)
    mdl=make_pipeline(StandardScaler(),RidgeCV(alphas=np.logspace(-3,3,25))) if an=='ica' else HistGradientBoostingRegressor(max_depth=3,learning_rate=0.05,max_iter=300,random_state=0)
    yt=dev[an].values if an=='ica' else np.log(dev[an].values); yv=val[an].values if an=='ica' else np.log(val[an].values)
    mdl.fit(Xtr,yt); pi=permutation_importance(mdl,Xte,yv,scoring='neg_mean_absolute_error',n_repeats=20,random_state=0)
    imp=pd.DataFrame({'feature':Xte.columns,'imp':pi.importances_mean,'sd':pi.importances_std}).sort_values('imp',ascending=False)
    imp.to_csv(f'out/val_{an}_perm_importance.csv',index=False); print(imp.head(10).round(4).to_string(index=False))
pd.DataFrame(out).to_csv('out/aux_results.csv',index=False)
