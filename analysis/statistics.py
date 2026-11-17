# Statistical comparisons computed from the notebook results.
# Usage: python analysis/statistics.py <results folder>
# The results folder must contain results_original, results_smotenc, results_adasyn, results_ctgan
# and results_shap_lightgbm (as written by the notebooks).
import os, sys
import numpy as np
import pandas as pd
from scipy.stats import friedmanchisquare, wilcoxon, spearmanr, binomtest
from sklearn.metrics import roc_auc_score, balanced_accuracy_score, recall_score

pd.set_option('display.width', 250)
R = sys.argv[1] if len(sys.argv) > 1 else 'results'
OUT = os.path.join(R, 'statistics')
os.makedirs(OUT, exist_ok=True)
M = ['original', 'smotenc', 'adasyn', 'ctgan']


def holm(p):
    p = np.asarray(p); o = np.argsort(p); adj = np.empty_like(p); run = 0
    for r, i in enumerate(o):
        run = max(run, (len(p) - r) * p[i]); adj[i] = min(1, run)
    return adj


# cross-validation: Friedman test across the four training configurations, then Wilcoxon signed-rank
# tests of each oversampler against no oversampling (Holm-adjusted), on the 15 paired folds
cv = pd.concat([pd.read_csv(f'{R}/results_{m}/cv_scores.csv') for m in M])
rows = []
for model in cv.model.unique():
    for met in ['balanced_accuracy', 'roc_auc', 'pr_auc', 'brier']:
        piv = cv[(cv.feature_set == 'all') & (cv.model == model)].pivot_table(index='fold', columns='method',
                                                                              values=met)[M]
        fr = friedmanchisquare(*[piv[m] for m in M]).pvalue
        adj = holm([wilcoxon(piv[m], piv['original']).pvalue for m in M[1:]])
        rows.append({'model': model, 'metric': met, 'n_folds': len(piv), 'friedman_p': fr,
                     **{f'{m}-orig': piv[m].mean() - piv['original'].mean() for m in M[1:]},
                     **{f'p_{m}': a for m, a in zip(M[1:], adj)}})
st = pd.DataFrame(rows)
print('Cross-validation: mean difference from no oversampling and Holm-adjusted Wilcoxon p-values')
print(st.round(4).to_string(index=False))
st.to_csv(f'{OUT}/cv_tests.csv', index=False)

# test set: McNemar test and paired bootstrap intervals of the differences
y = pd.read_csv(f'{R}/results_original/test_predictions.csv')['y_true'].values
P = {m: pd.read_csv(f'{R}/results_{m}/test_predictions.csv') for m in M}
rng = np.random.default_rng(42)
idx = [rng.integers(0, len(y), len(y)) for _ in range(1000)]
out = []
for model in ['CatBoost', 'LightGBM', 'LogisticRegression']:
    po = P['original'][f'all|{model}'].values
    for m in M[1:]:
        pm = P[m][f'all|{model}'].values
        co, cm = (po >= .5) == y, (pm >= .5) == y
        b, c = int((co & ~cm).sum()), int((~co & cm).sum())
        dauc = [roc_auc_score(y[i], pm[i]) - roc_auc_score(y[i], po[i]) for i in idx]
        dba = [balanced_accuracy_score(y[i], pm[i] >= .5) - balanced_accuracy_score(y[i], po[i] >= .5) for i in idx]
        out.append({'model': model, 'method': m, 'only_original_correct': b, 'only_method_correct': c,
                    'mcnemar_p': binomtest(min(b, c), b + c, 0.5).pvalue,
                    'd_bal_acc': balanced_accuracy_score(y, pm >= .5) - balanced_accuracy_score(y, po >= .5),
                    'd_bal_acc_lo': np.percentile(dba, 2.5), 'd_bal_acc_hi': np.percentile(dba, 97.5),
                    'd_roc_auc': roc_auc_score(y, pm) - roc_auc_score(y, po),
                    'd_auc_lo': np.percentile(dauc, 2.5), 'd_auc_hi': np.percentile(dauc, 97.5)})
t = pd.DataFrame(out)
print('\nTest set: oversampled minus no oversampling (all features)')
print(t.round(4).to_string(index=False))
t.to_csv(f'{OUT}/test_tests.csv', index=False)

# threshold adjustment for models trained on the original data: threshold = training prevalence
prev = 9065 / 53808
tm = []
for model in ['LogisticRegression', 'RandomForest', 'LightGBM', 'XGBoost', 'CatBoost']:
    p = P['original'][f'all|{model}'].values
    for thr in [0.5, prev]:
        yh = p >= thr
        tm.append({'model': model, 'threshold': round(thr, 4), 'bal_acc': balanced_accuracy_score(y, yh),
                   'sensitivity': recall_score(y, yh), 'specificity': recall_score(1 - y, 1 - yh)})
    for m in M[1:]:
        p2 = P[m][f'all|{model}'].values >= .5
        tm.append({'model': model, 'threshold': f'{m} @0.5', 'bal_acc': balanced_accuracy_score(y, p2),
                   'sensitivity': recall_score(y, p2), 'specificity': recall_score(1 - y, 1 - p2)})
tm = pd.DataFrame(tm)
print(f'\nThreshold adjustment (threshold = training prevalence {prev:.4f})')
print(tm.round(4).to_string(index=False))
tm.to_csv(f'{OUT}/threshold_adjustment.csv', index=False)


# expected calibration error, 10 equal-width bins
def ece(y, p, k=10):
    b = np.minimum((p * k).astype(int), k - 1)
    return sum(abs(p[b == i].mean() - y[b == i].mean()) * (b == i).mean() for i in range(k) if (b == i).any())


e = pd.DataFrame({m: {mo: ece(y, P[m][f'all|{mo}'].values) for mo in ['CatBoost', 'LightGBM', 'LogisticRegression']}
                  for m in M})
print('\nExpected calibration error (all features)')
print(e.round(4))
e.to_csv(f'{OUT}/ece.csv')

# stability of LightGBM SHAP importances across the 15 folds, and agreement between training configurations
S = f'{R}/results_shap_lightgbm'
feats = [c for c in pd.read_csv(f'{S}/cv_shap_importance_lightgbm_original.csv').columns
         if c not in ('method', 'fold', 'cv_balanced_accuracy')]
rows, imp = [], {}
for m in M:
    s = pd.read_csv(f'{S}/cv_shap_importance_lightgbm_{m}.csv')[feats].values
    rh = [spearmanr(s[i], s[j])[0] for i in range(len(s)) for j in range(i + 1, len(s))]
    top = [set(np.argsort(-r)[:5]) for r in s]
    ov = np.mean([len(top[i] & top[j]) / 5 for i in range(len(s)) for j in range(i + 1, len(s))])
    imp[m] = pd.read_csv(f'{S}/shap_importance_test_lightgbm_{m}.csv').set_index('feature')['mean_abs_shap'].reindex(feats)
    rows.append({'method': m, 'folds': len(s), 'rho_mean': np.mean(rh), 'rho_min': np.min(rh), 'top5_overlap': ov})
stab = pd.DataFrame(rows)
agree = pd.DataFrame({a: {b: spearmanr(imp[a], imp[b])[0] for b in M} for a in M})
print('\nSHAP stability across folds (LightGBM)')
print(stab.round(3).to_string(index=False))
print('\nSpearman correlation of test-set SHAP importances between training configurations')
print(agree.round(3))
stab.to_csv(f'{OUT}/shap_stability.csv', index=False)
agree.to_csv(f'{OUT}/shap_agreement.csv')
