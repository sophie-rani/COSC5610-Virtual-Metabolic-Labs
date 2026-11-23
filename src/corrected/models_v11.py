"""Post hoc methodological revision v1.1 of frozen models.py.
Only change: Ridge alpha is chosen by patient-grouped CV (GroupKFold on patient id) instead of
RidgeCV's leave-one-ROW-out, so repeated observations of one patient never sit on both sides of an
internal split (Saeb 2017, GigaScience, doi:10.1093/gigascience/gix019). Same alpha grid and the same
criterion (mean squared error). Every other model keeps its fixed a-priori hyperparameters; none uses
internal validation (HistGB early stopping is off for n < 10,000 — checked in the notebook)."""
import numpy as np, warnings; warnings.filterwarnings('ignore')
from sklearn.linear_model import LinearRegression, Ridge, LogisticRegression
from sklearn.model_selection import GridSearchCV, GroupKFold
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.svm import SVR
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import HistGradientBoostingRegressor, HistGradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from config import *
from models import design          # unchanged frozen design(): training medians + missingness indicators
ALPHAS = np.logspace(-3, 3, 25)    # frozen grid
def grouped_ridge():
    return GridSearchCV(make_pipeline(StandardScaler(), Ridge()), {'ridge__alpha': ALPHAS},
                        cv=GroupKFold(5), scoring='neg_mean_squared_error')
def fit_predict(tr, te, an, miss_ind=True, feats=None, return_alpha=False):
    out = {}
    if an == 'ica':
        y = tr.ica.values; feats = feats or F_ICA
        out['B1 total Ca (calibrated)'] = LinearRegression().fit(tr[['ca_mmol']], y).predict(te[['ca_mmol']])
        for k, f in [('B2 albumin-corrected Ca, 0.8 (calibrated)', 0.8), ('B3 Payne-corrected Ca, 1.0 (calibrated)', 1.0)]:
            ctr = (tr.ca + f*(4-tr.alb))*CA_MG_TO_MMOL; cte = (te.ca + f*(4-te.alb))*CA_MG_TO_MMOL; m = ctr.notna()
            p = np.full(len(te), np.nan); mm = cte.notna().values
            if m.sum() > 20: p[mm] = LinearRegression().fit(ctr[m].values.reshape(-1,1), y[m.values]).predict(cte[mm].values.reshape(-1,1))
            out[k] = p
    else:
        y = np.log(tr.lact.values); feats = feats or F_LACT
        out['B1 anion gap (calibrated)'] = np.exp(LinearRegression().fit(tr[['ag']], y).predict(te[['ag']]))
        out['B2 anion gap + HCO3 (calibrated)'] = np.exp(LinearRegression().fit(tr[['ag','hco3']], y).predict(te[['ag','hco3']]))
    pos = (lambda v: v < THR['ica']) if an == 'ica' else (lambda v: v > THR['lact'])
    Xtr, Xte = design(tr, te, feats, miss_ind)
    rid = grouped_ridge().fit(Xtr, y, groups=tr.id.values)                     # <- the v1.1 change
    regs = {'Ridge regression': rid,
            'Regression tree (depth 4)': DecisionTreeRegressor(max_depth=4, min_samples_leaf=30, random_state=0),
            'SVR (RBF)': make_pipeline(StandardScaler(), SVR(C=1.0, epsilon=0.02 if an=='ica' else 0.1)),
            'Gradient boosting': HistGradientBoostingRegressor(max_depth=3, learning_rate=0.05, max_iter=300, random_state=0)}
    for k, m in regs.items():
        if k != 'Ridge regression': m.fit(Xtr, y)
        p = m.predict(Xte); out[k] = p if an == 'ica' else np.exp(p)
    yb = pos(tr[an].values).astype(int)
    clfs = {'Logistic regression (classifier)': make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=5000)),
            'Naive Bayes (classifier)': make_pipeline(StandardScaler(), GaussianNB()),
            'Decision tree (classifier)': DecisionTreeClassifier(max_depth=4, min_samples_leaf=30, random_state=0),
            'Gradient boosting (classifier)': HistGradientBoostingClassifier(max_depth=3, learning_rate=0.05, max_iter=300, random_state=0)}
    for k, m in clfs.items(): out[k] = m.fit(Xtr, yb).predict_proba(Xte)[:, 1]
    if return_alpha: return out, rid.best_params_['ridge__alpha']
    return out
def oof(D, an, folds=5, **kw):
    res = None
    for tr, te in GroupKFold(folds).split(D, groups=D.id):
        p = fit_predict(D.iloc[tr], D.iloc[te], an, **kw)
        if res is None: res = {k: np.full(len(D), np.nan) for k in p}
        for k, v in p.items(): res[k][te] = v
    return res
