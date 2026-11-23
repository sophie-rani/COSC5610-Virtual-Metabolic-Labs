"""Phase 3: PCA of shared metabolic variation + association/sequential rules (association only). Dev discovery, val replication."""
import pandas as pd, numpy as np, warnings; warnings.filterwarnings('ignore')
from sklearn.decomposition import PCA; from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from config import *
CORE=['na','k','cl','hco3','bun','cr','glu','ca','phos','mg']
IC=pd.read_parquet('pairs_ica_dev_val.parquet'); L=pd.read_parquet('pairs_lact_dev_val.parquet')
# ---- PCA on the iCa pairs and lactate pairs separately is redundant; fit PCA on dev union of panels used in either task
U=pd.concat([IC[['id','dsplit','h_chem']+CORE],L[['id','dsplit','h_chem']+CORE]]).drop_duplicates(['id','h_chem']).dropna()
dev=U[U.dsplit=='dev']; sc=StandardScaler().fit(dev[CORE]); pca=PCA().fit(sc.transform(dev[CORE]))
load=pd.DataFrame(pca.components_[:5].T,index=CORE,columns=[f'PC{i+1}' for i in range(5)]).round(2)
print('explained var',pca.explained_variance_ratio_[:6].round(3)); print(load.to_string()); load.to_csv('out/pca_loadings.csv')
rows=[]
for an,D,resid_base in [('ica',IC,['ca_mmol']),('lact',L,['ag','hco3'])]:
    D=D.dropna(subset=CORE).copy(); Z=pca.transform(sc.transform(D[CORE]))[:,:5]
    for i in range(5): D[f'PC{i+1}']=Z[:,i]
    y=D[an] if an=='ica' else np.log(D[an])
    for s in ['dev','val']:
        m=(D.dsplit==s).values
        r=y[m]-LinearRegression().fit(D.loc[D.dsplit=='dev',resid_base],y[D.dsplit=='dev']).predict(D.loc[m,resid_base])  # residual vs baseline (fit on dev)
        for i in range(5):
            c=np.corrcoef(D.loc[m,f'PC{i+1}'],y[m])[0,1]; cr=np.corrcoef(D.loc[m,f'PC{i+1}'],r)[0,1]
            rows.append(dict(analyte=an,split=s,pc=f'PC{i+1}',r_with_analyte=c,r_with_residual_beyond_baseline=cr))
P=pd.DataFrame(rows); P.to_csv('out/pca_assoc.csv',index=False); print(P.pivot_table(index=['analyte','pc'],columns='split',values=['r_with_analyte','r_with_residual_beyond_baseline']).round(2).to_string())
# complexing-anion check: partial association of iCa with AG and phosphate given total Ca
for s in ['dev','val']:
    d=IC[(IC.dsplit==s)].dropna(subset=['ca','ag','phos'])
    import statsmodels.formula.api as smf
    f=smf.ols('ica~ca+ag+phos+alb_f+nacl',data=d.assign(alb_f=d.alb.fillna(d.alb.median()),nacl=d.na-d.cl)).fit(cov_type='cluster',cov_kwds={'groups':d.id})
    print(s,'iCa ~ total Ca + AG + phos + albumin + (Na-Cl)  [cluster SE]\n',f.summary2().tables[1][['Coef.','[0.025','0.975]']].round(4).to_string())
# ---- association rules (pair level; items within the same 6 h window) and next-day sequential rules
from mlxtend.frequent_patterns import apriori, association_rules
def items(D,an):
    T=pd.DataFrame({'transfusion<=6h':D.transfusion6==1,'IV calcium<=6h':D.iv_calcium6==1,'vasopressor<=6h':D.vasopressor6==1,
                    'Cl>110':D.cl>110,'HCO3<22':D.hco3<22,'AG>=16':D.ag>=16,'Cr>=1.5':D.cr>=1.5,'phos>4.5':D.phos>4.5,'albumin<3.0':D.alb<3.0})
    T['iCa<1.12' if an=='ica' else 'lactate>2']=(D.ica<1.12) if an=='ica' else (D.lact>2)
    return T.astype(bool)
R=[]
for an,D in [('ica',IC),('lact',L)]:
    tgt='iCa<1.12' if an=='ica' else 'lactate>2'
    Td,Tv=items(D[D.dsplit=='dev'],an),items(D[D.dsplit=='val'],an)
    fi=apriori(Td,min_support=0.03,use_colnames=True,max_len=3); ar=association_rules(fi,metric='lift',min_threshold=1.2)
    ar=ar[ar.consequents.apply(lambda c:c=={tgt})]
    for _,r in ar.iterrows():
        a=list(r.antecedents); ma=Tv[a].all(1); conf_v=Tv.loc[ma,tgt].mean() if ma.sum()>=10 else np.nan
        R.append(dict(analyte=an,kind='concurrent (6 h)',antecedent=' & '.join(sorted(a)),consequent=tgt,support_dev=r.support,conf_dev=r.confidence,lift_dev=r.lift,
                      n_val=int(ma.sum()),conf_val=conf_v,lift_val=conf_v/Tv[tgt].mean() if ma.sum()>=10 else np.nan))
    # sequential: exposure in 6 h before a pair on day d -> abnormal reference on day d+1 (patients with references on both days)
    for s in ['dev','val']:
        X=D[D.dsplit==s]; g=X.groupby(['id','day']).agg(tr=('transfusion6','max'),iv=('iv_calcium6','max'),vp=('vasopressor6','max'),
              ab=(an,(lambda v:(v<1.12).any()) if an=='ica' else (lambda v:(v>2).any()))).reset_index()
        nx=g.assign(day=g.day-1)[['id','day','ab']].rename(columns={'ab':'ab_next'}); G=g.merge(nx,on=['id','day'])
        for e,nm in [('tr','transfusion'),('iv','IV calcium'),('vp','vasopressor')]:
            G2=G[G.ab==0]  # restrict to days that were NOT abnormal, so the rule is about new abnormality
            if G2[e].sum()>=5:
                R.append(dict(analyte=an,kind=f'sequential day d -> d+1 (new abnormality), {s}',antecedent=nm+' day d',consequent=tgt+' day d+1',
                              n_val=int(G2[e].sum()),conf_dev=G2.loc[G2[e]==1,'ab_next'].mean(),lift_dev=G2.loc[G2[e]==1,'ab_next'].mean()/max(G2.ab_next.mean(),1e-9),support_dev=len(G2)))
R=pd.DataFrame(R); R.to_csv('out/rules.csv',index=False)
pd.set_option('display.width',250); print(R.sort_values(['analyte','kind','lift_dev'],ascending=[True,True,False]).round(3).to_string(index=False))
