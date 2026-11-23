"""Virtual Metabolic Laboratories in sTBI - frozen configuration."""
O='/home/claude/na/out'; DISC='/home/claude/disc'
SPLIT_SALT='discovery-2026-11-18b'          # dev 60 / val 20 / test 20 by subject (pts.parquet dsplit)
REF={'ica':50808,'lact':50813}             # Blood Gas category, mmol/L
CHEM={50983:'na',50971:'k',50902:'cl',50882:'hco3',51006:'bun',50912:'cr',50931:'glu',50893:'ca'}
OTHER={50970:'phos',50960:'mg',51222:'hgb',51301:'wbc',51265:'plt',50862:'alb',50820:'ph',50818:'pco2'}
PAIR_CHEM_MIN=60          # chemistry panel nearest within +/-60 min of reference (primary); sensitivity: -120..0 min
PAIR_OTHER_H=6            # phos/mg/CBC within +/-6 h
PAIR_ALB_H=24             # albumin within +/-24 h (measured only; never imputed for baseline formulas)
EXPO_H=6                  # treatment exposures in the 6 h BEFORE the reference draw
ITEMS={'transfusion':[225168,220996,220970,225170,225171],'iv_calcium':[221456,227525,229618,229640],
       'vasopressor':[221906,221749,222315,221662,221289,229617,229630,229632],'rrt':[225802,225803,225805,225809,225955]}
THR={'ica':1.12,'ica_severe':1.00,'lact':2.0,'lact_severe':4.0}
CA_MG_TO_MMOL=0.2495
F_ICA=['ca','alb','na','k','cl','hco3','bun','cr','glu','phos','mg','hgb','plt','transfusion6','iv_calcium6']
F_LACT=['na','k','cl','hco3','ag','bun','cr','glu','ca','phos','mg','hgb','wbc','plt','alb','vasopressor6','transfusion6']
SEED=20261119; NBOOT=2000
