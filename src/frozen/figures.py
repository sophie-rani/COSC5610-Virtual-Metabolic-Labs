import pandas as pd, numpy as np, matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False,'axes.edgecolor':'#52514e',
  'axes.labelcolor':'#0b0b0b','xtick.color':'#52514e','ytick.color':'#52514e','axes.grid':True,'grid.color':'#e6e5e0','grid.linewidth':0.6,'figure.facecolor':'#fcfcfb','axes.facecolor':'#fcfcfb','savefig.dpi':200})
C={'model':'#2a78d6','base':'#eb6834','corr':'#1baf7a','payne':'#eda100','ink':'#0b0b0b','muted':'#52514e'}
S={'ica':dict(title='Ionized calcium (iCa < 1.12 mmol/L)',m='Ridge regression',b=[('B1 total Ca (calibrated)','Total Ca',C['base']),('B2 albumin-corrected Ca, 0.8 (calibrated)','Corrected Ca 0.8*',C['corr']),('B3 Payne-corrected Ca, 1.0 (calibrated)','Payne Ca 1.0*',C['payne'])],pos=lambda y:y<1.12,sgn=-1,unit='mmol/L'),
   'lact':dict(title='Lactate (> 2 mmol/L)',m='Gradient boosting',b=[('B2 anion gap + HCO3 (calibrated)','AG + HCO3',C['base']),('B1 anion gap (calibrated)','AG alone',C['payne'])],pos=lambda y:y>2,sgn=1,unit='mmol/L')}
# Fig 1 ROC
fig,ax=plt.subplots(1,2,figsize=(10,4.6))
for a,an in zip(ax,['ica','lact']):
    P=pd.read_parquet(f'final/test_{an}_pred.parquet'); s=S[an]; t=s['pos'](P.y.values)
    for k,lab,col in [(s['m'],'Soft sensor ('+s['m'].split(' ')[0]+')',C['model'])]+s['b']:
        m=~P[k].isna().values; fpr,tpr,_=roc_curve(t[m],s['sgn']*P[k].values[m]); auc=roc_auc_score(t[m],s['sgn']*P[k].values[m])
        a.plot(fpr,tpr,color=col,lw=2,label=f'{lab}  AUROC {auc:.2f}')
    a.plot([0,1],[0,1],color='#c3c2b7',lw=1,ls='--'); a.set_title(s['title'],loc='left',fontsize=11,color=C['ink']); a.set_xlabel('1 − specificity'); a.set_ylabel('Sensitivity')
    a.legend(frameon=False,fontsize=8.5,loc='lower right')
ax[0].text(0,-0.2,'*albumin-measured subset (n=172)',transform=ax[0].transAxes,fontsize=8,color=C['muted'])
fig.suptitle('Figure 1. Held-out test set (260 patients): detection of ionized hypocalcemia and hyperlactatemia',x=0.01,ha='left',fontsize=11.5)
fig.tight_layout(); fig.savefig('final/fig1_roc.png'); plt.close()
# Fig 2 binned calibration of estimates
fig,ax=plt.subplots(1,2,figsize=(10,4.6))
for a,an in zip(ax,['ica','lact']):
    P=pd.read_parquet(f'final/test_{an}_pred.parquet'); s=S[an]
    for k,lab,col in [(s['m'],'Soft sensor',C['model']),(s['b'][0][0],s['b'][0][1],C['base'])]:
        d=pd.DataFrame({'p':P[k],'y':P.y}).dropna(); d['bin']=pd.qcut(d.p,10,labels=False,duplicates='drop'); g=d.groupby('bin').mean()
        a.plot(g.p,g.y,'-o',color=col,lw=2,ms=6,label=lab,markeredgecolor='#fcfcfb',markeredgewidth=1.5)
    lo,hi=(0.95,1.3) if an=='ica' else (0.5,8); a.plot([lo,hi],[lo,hi],color='#c3c2b7',ls='--',lw=1); a.set_xlim(lo,hi); a.set_ylim(lo,hi)
    if an=='lact':
        from matplotlib.ticker import FixedLocator,NullLocator,FormatStrFormatter
        a.set_xscale('log'); a.set_yscale('log')
        for ax_ in (a.xaxis,a.yaxis): ax_.set_major_locator(FixedLocator([0.5,1,2,4,8])); ax_.set_minor_locator(NullLocator()); ax_.set_major_formatter(FormatStrFormatter('%g'))
    th=1.12 if an=='ica' else 2; a.axhline(th,color='#c3c2b7',lw=0.8); a.axvline(th,color='#c3c2b7',lw=0.8)
    a.set_title(s['title'].split(' (')[0]+' — decile calibration',loc='left',fontsize=11); a.set_xlabel(f'Estimated ({s["unit"]})'); a.set_ylabel(f'Measured ({s["unit"]})'); a.legend(frameon=False,fontsize=9,loc='upper left')
fig.suptitle('Figure 2. Estimated vs measured values by decile of estimate (test set); dashed = identity',x=0.01,ha='left',fontsize=11.5); fig.tight_layout(); fig.savefig('final/fig2_calibration.png'); plt.close()
# Fig 3 forest of delta AUROC
R=pd.read_csv('final/test_secondary.csv'); R=R[R.analysis.isin(['PRIMARY','KEY SECONDARY vs corrected Ca 0.8 (albumin-measured)','subgroup','sensitivity'])&R.dAUC.notna()]
fig,ax=plt.subplots(1,2,figsize=(13,6.4),sharex=True)
for a,an in zip(ax,['ica','lact']):
    d=R[R.analyte==an].reset_index(drop=True); lab=[('PRIMARY (all)' if r.analysis=='PRIMARY' else ('vs corrected Ca 0.8' if 'KEY' in r.analysis else r.stratum))+f'  (n={int(r.n)})' for r in d.itertuples()]
    yy=np.arange(len(d))[::-1]
    for i,r in enumerate(d.itertuples()):
        col=C['model'] if r.analysis in('PRIMARY',) or 'KEY' in r.analysis else (C['muted'] if r.analysis=='sensitivity' else C['corr'])
        a.plot([r.dAUC_lo,r.dAUC_hi],[yy[i]]*2,color=col,lw=2); a.plot(r.dAUC,yy[i],'o',color=col,ms=7,markeredgecolor='#fcfcfb',markeredgewidth=1.5)
    a.axvline(0,color=C['ink'],lw=0.8); a.set_yticks(yy); a.set_yticklabels(lab,fontsize=8.5); a.grid(axis='y',visible=False)
    a.set_title(('iCa: Ridge vs total Ca' if an=='ica' else 'Lactate: GBM vs AG + HCO3'),loc='left',fontsize=11); a.set_xlabel('ΔAUROC, model − baseline (95% CI)')
fig.suptitle('Figure 3. Added discrimination of the soft sensor: primary (blue), subgroups (green), sensitivity analyses (gray) — test set',x=0.01,ha='left',fontsize=11.5)
fig.tight_layout(); fig.savefig('final/fig3_forest.png'); plt.close()
# Fig 4 bedside rules
B=pd.read_csv('final/test_secondary.csv'); B=B[B.analysis.str.startswith('bedside')]
names=['total Ca <8.5 mg/dL','corrected 0.8 <8.5 mg/dL','Payne 1.0 <8.5 mg/dL','primary model <1.12 mmol/L']; lab=['Total Ca\n< 8.5 mg/dL','Corrected Ca 0.8\n< 8.5 mg/dL','Payne Ca 1.0\n< 8.5 mg/dL','Soft sensor\n< 1.12 mmol/L']
B=B.set_index('stratum').loc[names]; cols=[C['base'],C['corr'],C['payne'],C['model']]
fig,ax=plt.subplots(1,3,figsize=(11,3.9))
for a,(col,tt) in zip(ax,[('sens','Sensitivity for iCa < 1.12'),('spec','Specificity'),('false_reassurance_lowalb','False reassurance: albumin < 3.0\nand true hypocalcemia (n=42)')]):
    v=B[col].values; a.barh(range(4)[::-1],v,color=cols,height=0.6); a.set_xlim(0,1)
    for i,x in enumerate(v): a.text(x+0.02,3-i,f'{x:.0%}',va='center',fontsize=9,color=C['ink'])
    a.set_yticks(range(4)[::-1]); a.set_yticklabels(lab if a is ax[0] else ['']*4,fontsize=8.5); a.set_title(tt,loc='left',fontsize=10); a.grid(axis='y',visible=False)
fig.suptitle('Figure 4. Uncalibrated bedside rules on albumin-measured test observations (n=172, 75 with iCa < 1.12)',x=0.01,ha='left',fontsize=11.5)
fig.tight_layout(); fig.savefig('final/fig4_bedside_rules.png'); plt.close()
print('ok')
