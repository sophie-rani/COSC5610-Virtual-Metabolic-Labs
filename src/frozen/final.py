"""FINAL single test-set evaluation. Train = dev+val, evaluate = test. Run once after FREEZE_MANIFEST is written."""
import pandas as pd, numpy as np, json, hashlib, warnings, os; warnings.filterwarnings('ignore')
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.linear_model import LogisticRegression
from sklearn.inspection import permutation_importance
from config import *; from models import *; from metrics import *
assert os.path.exists('FREEZE_MANIFEST.json'), 'freeze first'
man=json.load(open('FREEZE_MANIFEST.json'))
for f,h in man['sha256'].items(): assert hashlib.sha256(open(f,'rb').read()).hexdigest()==h, f'{f} changed after freeze'
os.makedirs('final',exist_ok=True); rng=np.random.default_rng(SEED+99)
PRIM={'ica':'Ridge regression','lact':'Gradient boosting'}; BASE={'ica':'B1 total Ca (calibrated)','lact':'B2 anion gap + HCO3 (calibrated)'}
REFS={'ica':['B1 total Ca (calibrated)','B2 albumin-corrected Ca, 0.8 (calibrated)'],'lact':['B2 anion gap + HCO3 (calibrated)']}
def cb(y,g,pa,pb,an,w=None,B=NBOOT):
    m=~np.isnan(pa)&~np.isnan(pb); y,g,pa,pb=y[m],g[m],pa[m],pb[m]; w=None if w is None else w[m]
    t=pos(an,y).astype(int); sa=-pa if an=='ica' else pa; sb=-pb if an=='ica' else pb
    if t.sum()<5 or t.sum()>len(t)-5: return dict(n=len(y),pts=len(np.unique(g)),npos=int(t.sum()),note='too few events')
    f=lambda s,i: roc_auc_score(t[i],s[i],sample_weight=None if w is None else w[i])
    ids=np.unique(g); ix={i:np.where(g==i)[0] for i in ids}; d=[]; dm=[]
    for _ in range(B):
        s=np.concatenate([ix[i] for i in rng.choice(ids,len(ids))])
        if 0<t[s].sum()<len(s): d.append(f(sa,s)-f(sb,s)); dm.append(np.mean(np.abs(pb[s]-y[s]))-np.mean(np.abs(pa[s]-y[s])))
    a=np.arange(len(y))
    return dict(n=len(y),pts=len(ids),npos=int(t.sum()),AUC_model=f(sa,a),AUC_base=f(sb,a),dAUC=f(sa,a)-f(sb,a),dAUC_lo=np.percentile(d,2.5),dAUC_hi=np.percentile(d,97.5),
                MAE_model=np.mean(np.abs(pa-y)),MAE_base=np.mean(np.abs(pb-y)),MAE_red_lo=np.percentile(dm,2.5),MAE_red_hi=np.percentile(dm,97.5))
out=[]; 
def rec(an,analysis,stratum,r): out.append(dict(analyte=an,analysis=analysis,stratum=stratum,**r))
PRED={}
for an in ['ica','lact']:
    TR=pd.read_parquet(f'pairs_{an}_dev_val.parquet').reset_index(drop=True); TE=pd.read_parquet(f'pairs_{an}_test.parquet').reset_index(drop=True)
    assert set(TR.id)&set(TE.id)==set(), 'patient overlap'
    P=fit_predict(TR,TE,an); PRED[an]=(TE,P)
    T,Dl=boot_table(TE,an,P,an,REFS[an],rng,NBOOT); T.to_csv(f'final/test_{an}_models.csv',index=False); Dl.to_csv(f'final/test_{an}_deltas.csv',index=False)
    pd.DataFrame({**P,'y':TE[an].values,'id':TE.id.values}).to_parquet(f'final/test_{an}_pred.parquet')
    y=TE[an].values; g=TE.id.values; pa,pb=P[PRIM[an]],P[BASE[an]]
    rec(an,'PRIMARY','all',cb(y,g,pa,pb,an))
    if an=='ica':
        m=TE.alb.notna().values; rec(an,'KEY SECONDARY vs corrected Ca 0.8 (albumin-measured)','albumin measured',cb(y[m],g[m],pa[m],P['B2 albumin-corrected Ca, 0.8 (calibrated)'][m],an))
        rec(an,'secondary: corrected Ca 0.8 vs total Ca (albumin-measured)','albumin measured',cb(y[m],g[m],P['B2 albumin-corrected Ca, 0.8 (calibrated)'][m],pb[m],an))
        rec(an,'secondary: Payne 1.0 vs total Ca (albumin-measured)','albumin measured',cb(y[m],g[m],P['B3 Payne-corrected Ca, 1.0 (calibrated)'][m],pb[m],an))
        subs={'albumin <3.0 g/dL':TE.alb<3.0,'albumin >=3.0':TE.alb>=3.0,'albumin not measured':TE.alb.isna(),'transfusion <=6h':TE.transfusion6==1,'IV calcium <=6h':TE.iv_calcium6==1,
              'creatinine >=1.5':TE.cr>=1.5,'pH <7.30 (stratifier only)':TE.ph<7.30,'pH >7.45 (stratifier only)':TE.ph>7.45}
        Y=y[m]<THR['ica']; ca=TE.ca.values[m]; al=TE.alb.values[m]; lo=al<3.0
        for nm,v in [('total Ca <8.5 mg/dL',ca<8.5),('corrected 0.8 <8.5 mg/dL',ca+0.8*(4-al)<8.5),('Payne 1.0 <8.5 mg/dL',ca+(4-al)<8.5),('primary model <1.12 mmol/L',pa[m]<THR['ica'])]:
            rec(an,'bedside rule (albumin-measured obs)',nm,dict(n=int(m.sum()),npos=int(Y.sum()),sens=float(v[Y].mean()),spec=float((~v)[~Y].mean()),
                n_hypo_lowalb=int((Y&lo).sum()),false_reassurance_lowalb=float((~v)[Y&lo].mean()) if (Y&lo).sum() else np.nan))
    else:
        subs={'vasopressor <=6h':TE.vasopressor6==1,'creatinine >=1.5':TE.cr>=1.5,'HCO3 <22':TE.hco3<22,'anion gap <=12':TE.ag<=12,'anion gap <16':TE.ag<16,'pH <7.30 (stratifier only)':TE.ph<7.30}
        t4=(y>THR['lact_severe']).astype(int)
        rec(an,'severe >4 mmol/L detection (AUROC)','all',dict(n=len(y),npos=int(t4.sum()),AUC_model=roc_auc_score(t4,pa),AUC_base=roc_auc_score(t4,pb)))
        m=TE.ag.values<16; t=(y>2)[m]
        rec(an,'hyperlactatemia when AG<16','AG<16',dict(n=int(m.sum()),npos=int(t.sum()),model_sens=float((pa[m]>2)[t].mean()),model_spec=float((pa[m]<=2)[~t].mean()),AG16_rule_sens=0.0))
        ag=TE.ag.values>=16; tt=y>2; rec(an,'AG>=16 rule (all obs)','all',dict(n=len(y),npos=int(tt.sum()),sens=float(ag[tt].mean()),spec=float((~ag)[~tt].mean())))
    for k,mm in subs.items(): mm=mm.values; rec(an,'subgroup',k,cb(y[mm],g[mm],pa[mm],pb[mm],an,B=1000))
    # prespecified sensitivity analyses
    rec(an,'sensitivity','no missingness indicators',cb(y,g,fit_predict(TR,TE,an,miss_ind=False)[PRIM[an]],pb,an,B=1000))
    feats=[f for f in (F_ICA if an=='ica' else F_LACT) if not f.endswith('6')]
    rec(an,'sensitivity','chemistry only (no exposures/indicators)',cb(y,g,fit_predict(TR,TE,an,miss_ind=False,feats=feats)[PRIM[an]],pb,an,B=1000))
    core=['na','k','cl','hco3','bun','cr','glu','ca']+(['ag'] if an=='lact' else [])
    rec(an,'sensitivity','basic metabolic panel only',cb(y,g,fit_predict(TR,TE,an,miss_ind=False,feats=core)[PRIM[an]],pb,an,B=1000))
    m=(TE.gap_min<=0).values; P2=fit_predict(TR[TR.gap_min<=0],TE[m],an); rec(an,'sensitivity','chemistry at/before reference',cb(y[m],g[m],P2[PRIM[an]],P2[BASE[an]],an,B=1000))
    one=lambda X: X.loc[X.assign(a=X.gap_min.abs()).groupby(['id','day']).a.idxmin()].reset_index(drop=True)
    t1,e1=one(TR),one(TE); P3=fit_predict(t1,e1,an); rec(an,'sensitivity','one pair per patient-day',cb(e1[an].values,e1.id.values,P3[PRIM[an]],P3[BASE[an]],an,B=1000))
    Pn=pd.read_parquet('panels_dev_val.parquet'); flag='has_ica' if an=='ica' else 'has_lact'; pf=['na','k','cl','hco3','bun','cr','glu','ca']
    Pn=Pn.dropna(subset=pf).assign(dd=lambda d:d.h_chem//24); lr=make_pipeline(StandardScaler(),LogisticRegression(max_iter=2000)).fit(Pn[pf+['dd']],Pn[flag])
    vv=TE.assign(dd=TE.day).dropna(subset=pf); ps=lr.predict_proba(vv[pf+['dd']])[:,1]; w=np.clip(1/ps,None,np.percentile(1/ps,99)); ii=vv.index.values
    rec(an,'sensitivity','IPW for who gets measured',cb(y[ii],g[ii],pa[ii],pb[ii],an,w=w,B=1000))
    Pt=pd.read_parquet('panels_test.parquet').dropna(subset=pf).assign(dd=lambda d:d.h_chem//24)
    rec(an,'selection model (who is tested) AUROC on test panels','-',dict(AUROC=roc_auc_score(Pt[flag],lr.predict_proba(Pt[pf+['dd']])[:,1]),n_panels=len(Pt),frac_tested=Pt[flag].mean()))
    Xtr,Xte=design(TR,TE,F_ICA if an=='ica' else F_LACT)
    mdl=make_pipeline(StandardScaler(),RidgeCV(alphas=np.logspace(-3,3,25))) if an=='ica' else HistGradientBoostingRegressor(max_depth=3,learning_rate=0.05,max_iter=300,random_state=0)
    tf=(lambda v:v) if an=='ica' else np.log; mdl.fit(Xtr,tf(TR[an].values))
    pi=permutation_importance(mdl,Xte,tf(y),scoring='neg_mean_absolute_error',n_repeats=30,random_state=0)
    pd.DataFrame({'feature':Xte.columns,'importance':pi.importances_mean,'sd':pi.importances_std}).sort_values('importance',ascending=False).to_csv(f'final/test_{an}_perm_importance.csv',index=False)
R=pd.DataFrame(out); R.to_csv('final/test_secondary.csv',index=False)
pd.set_option('display.width',260); pd.set_option('display.max_columns',30)
for an in ['ica','lact']:
    print(f'\n===== {an} TEST'); print(pd.read_csv(f'final/test_{an}_models.csv')[['model','n','prev','MAE','MAE_lo','MAE_hi','bias','cal_slope','AUROC','AUROC_lo','AUROC_hi','AUPRC','sens','spec']].round(3).to_string(index=False))
    print(pd.read_csv(f'final/test_{an}_deltas.csv').round(3).to_string(index=False))
print(R.round(3).to_string(index=False))
