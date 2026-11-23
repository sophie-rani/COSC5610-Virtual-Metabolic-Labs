"""Development: 5-fold patient-grouped CV on dev; then single evaluation on val (train dev -> val)."""
import pandas as pd, numpy as np, json, sys
from config import *; from models import *; from metrics import *
rng=np.random.default_rng(SEED); B=int(sys.argv[1]) if len(sys.argv)>1 else 500
res={}
for an,refs in [('ica',['B1 total Ca (calibrated)','B2 albumin-corrected Ca, 0.8 (calibrated)']),('lact',['B2 anion gap + HCO3 (calibrated)'])]:
    D=pd.read_parquet(f'pairs_{an}_dev_val.parquet'); dev=D[D.dsplit=='dev'].reset_index(drop=True); val=D[D.dsplit=='val'].reset_index(drop=True)
    P=oof(dev,an); T,Dl=boot_table(dev,an,P,an,refs,rng,B)
    T.to_csv(f'out/dev_{an}_models.csv',index=False); Dl.to_csv(f'out/dev_{an}_deltas.csv',index=False)
    pd.DataFrame({**P,'y':dev[an].values,'id':dev.id.values}).to_parquet(f'out/dev_{an}_oof.parquet')
    Pv=fit_predict(dev,val,an); Tv,Dv=boot_table(val,an,Pv,an,refs,rng,B)
    Tv.to_csv(f'out/val_{an}_models.csv',index=False); Dv.to_csv(f'out/val_{an}_deltas.csv',index=False)
    print(f'\n===== {an}: dev n={len(dev)} pts={dev.id.nunique()} | val n={len(val)} pts={val.id.nunique()}')
    cols=['model','n','prev','MAE','bias','cal_slope','AUROC','AUPRC','sens','spec']
    print('DEV (CV)'); print(T[cols].round(3).to_string(index=False)); print(Dl.round(3).to_string(index=False))
    print('VAL'); print(Tv[cols].round(3).to_string(index=False)); print(Dv.round(3).to_string(index=False))
