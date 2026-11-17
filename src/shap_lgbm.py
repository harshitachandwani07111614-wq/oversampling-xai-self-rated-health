# SHAP explanations of LightGBM for each type of training data.
# - test set: the model trained on the full (oversampled) training set is explained on all test records
# - stability: importances are recomputed in each of the 15 cross-validation folds on 2,000 test records
# TreeSHAP values come from LightGBM (pred_contrib=True) and are on the log-odds scale; the one-hot
# columns of Diabetes are summed so that each of the 18 features has one value.
import os, json, time
import numpy as np
import pandas as pd
from sklearn.model_selection import RepeatedStratifiedKFold
import pipeline as P

FEATS = P.ALL_FEATURES


def lgbm_shap(model, X):
    pre, clf = model[0], model[-1]
    Xt = pre.transform(X[FEATS])
    contrib = clf.booster_.predict(Xt, pred_contrib=True)
    names = pre.get_feature_names_out()
    sv = np.zeros((len(X), len(FEATS)))
    for j, name in enumerate(names):
        kind, col = name.split('__', 1)
        feat = col if kind == 'remainder' else next(f for f in P.NOMINAL if col.startswith(f + '_'))
        sv[:, FEATS.index(feat)] += contrib[:, j]
    return sv, contrib[:, -1]


def run_cv_shap(method, Xtr, ytr, Xte, out, seed=P.SEED):
    # resumable: finished folds are appended to the csv file
    path = f'{out}/cv_shap_importance_lightgbm_{method}.csv'
    done = set(pd.read_csv(path)['fold']) if os.path.exists(path) else set()
    rskf = RepeatedStratifiedKFold(n_splits=P.CFG['n_splits'], n_repeats=P.CFG['n_repeats'], random_state=seed)
    n_folds = P.CFG['n_splits'] * P.CFG['n_repeats']
    shap_X = Xte.iloc[:2000]
    for k, (tr, va) in enumerate(rskf.split(Xtr, ytr)):
        if k in done:
            continue
        t0 = time.time()
        Xo, yo = P.OVERSAMPLERS[method](Xtr.iloc[tr].reset_index(drop=True), ytr.iloc[tr].reset_index(drop=True),
                                        seed + k)
        m = P.make_model('LightGBM', FEATS, seed).fit(Xo[FEATS], yo)
        ba = P.metrics(ytr.iloc[va].values, m.predict_proba(Xtr.iloc[va][FEATS])[:, 1])['balanced_accuracy']
        sv, _ = lgbm_shap(m, shap_X)
        row = {'method': method, 'fold': k, 'cv_balanced_accuracy': ba, **dict(zip(FEATS, np.abs(sv).mean(0)))}
        pd.DataFrame([row]).to_csv(path, mode='a', header=not os.path.exists(path), index=False)
        P.log(f'  {method}: CV fold {k + 1}/{n_folds} done in {time.time() - t0:.0f}s (balanced acc {ba:.4f})')
    return pd.read_csv(path)


def run_test_shap(method, Xtr, ytr, Xte, yte, out, seed=P.SEED):
    Xo, yo = P.OVERSAMPLERS[method](Xtr, ytr, seed)
    m = P.make_model('LightGBM', FEATS, seed).fit(Xo[FEATS], yo)
    p = m.predict_proba(Xte[FEATS])[:, 1]
    check = {'method': method, 'test_balanced_accuracy': P.metrics(yte.values, p)['balanced_accuracy']}
    P.log(f'  {method}: {check}')
    sv, base = lgbm_shap(m, Xte)
    assert np.allclose(sv.sum(1) + base, np.log(p / (1 - p)), atol=1e-6), 'SHAP values do not add up to the log-odds'
    np.savez_compressed(f'{out}/shap_values_test_lightgbm_{method}.npz', values=sv, base=base, prob=p,
                        data=Xte[FEATS].values, y=yte.values)
    pd.DataFrame({'feature': FEATS, 'mean_abs_shap': np.abs(sv).mean(0), 'mean_shap': sv.mean(0)}) \
        .sort_values('mean_abs_shap', ascending=False) \
        .to_csv(f'{out}/shap_importance_test_lightgbm_{method}.csv', index=False)
    return check


def run(methods, Xtr, ytr, Xte, yte, out):
    checks = []
    for method in methods:
        P.log(f'=== LightGBM trained with {method} ===')
        checks.append(run_test_shap(method, Xtr, ytr, Xte, yte, out))
        run_cv_shap(method, Xtr, ytr, Xte, out)
        with open(f'{out}/checks_{method}.json', 'w') as f:
            json.dump(checks[-1], f, indent=2)
    imp = pd.DataFrame({m: pd.read_csv(f'{out}/shap_importance_test_lightgbm_{m}.csv').set_index('feature')
                        .mean_abs_shap for m in methods})
    print(imp.sort_values(methods[0], ascending=False).round(3).to_string())
    print(pd.DataFrame(checks).to_string(index=False))
    return imp
