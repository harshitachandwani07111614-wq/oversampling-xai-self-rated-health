# Shared code for all notebooks: data preparation, oversampling, models, metrics,
# cross-validation, test-set evaluation, synthetic-data quality, EDA and sensitivity analyses.
import os, json, time, random, hashlib, platform, warnings
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.model_selection import train_test_split, RepeatedStratifiedKFold, StratifiedKFold, cross_val_predict
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import (confusion_matrix, f1_score, matthews_corrcoef, roc_auc_score,
                             average_precision_score, brier_score_loss)
from imblearn.over_sampling import SMOTENC, ADASYN

warnings.filterwarnings('ignore')

SEED = 42
TARGET = 'General_Health'
YES_NO = ['Exercise', 'Heart_Disease', 'Skin_Cancer', 'Other_Cancer', 'Depression', 'Arthritis', 'Smoking_History']
BINARY = YES_NO + ['Sex']
ORDINAL = {
    'Checkup': ['Never', '5 or more years ago', 'Within the past 5 years', 'Within the past 2 years',
                'Within the past year'],
    'Age_Category': ['18-24', '25-29', '30-34', '35-39', '40-44', '45-49', '50-54', '55-59', '60-64',
                     '65-69', '70-74', '75-79', '80+'],
}
NOMINAL = {'Diabetes': ['No', 'No, pre-diabetes or borderline diabetes',
                        'Yes, but female told only during pregnancy', 'Yes']}
CAT_COLS = BINARY + list(ORDINAL) + list(NOMINAL)
NUM_COLS = ['Height_(cm)', 'Weight_(kg)', 'BMI', 'Alcohol_Consumption', 'Fruit_Consumption',
            'Green_Vegetables_Consumption', 'FriedPotato_Consumption']
# decimals in the raw data, used to round synthetic values
NUM_DECIMALS = {'Height_(cm)': 0, 'Weight_(kg)': 2, 'BMI': 2, 'Alcohol_Consumption': 0, 'Fruit_Consumption': 0,
                'Green_Vegetables_Consumption': 0, 'FriedPotato_Consumption': 0}
N_LEVELS = {**{c: 2 for c in BINARY}, **{c: len(v) for c, v in {**ORDINAL, **NOMINAL}.items()}}
DIAG_COLS = ['Heart_Disease', 'Diabetes', 'Arthritis', 'Depression', 'Skin_Cancer', 'Other_Cancer']
ALL_FEATURES = CAT_COLS + NUM_COLS
FEATURE_SETS = {'all': ALL_FEATURES, 'drop_weight': [c for c in ALL_FEATURES if c != 'Weight_(kg)']}
MODEL_NAMES = ['LogisticRegression', 'DecisionTree', 'RandomForest', 'LightGBM', 'XGBoost', 'CatBoost']
SCHEMES = {'extremes': {'Excellent': 0, 'Poor': 1},
           'cdc': {'Excellent': 0, 'Very Good': 0, 'Good': 0, 'Fair': 1, 'Poor': 1}}

CFG = dict(quick=False, n_splits=5, n_repeats=3, bootstrap=1000, ctgan_epochs=86, ga_pop=10, ga_gens=20)

# Google Drive folder used on Colab; set RESULTS_DIR to override
DRIVE_DIR = '/content/drive/MyDrive/brfss_imbalance'

try:
    import torch
    HAS_GPU = torch.cuda.is_available()
except ImportError:
    HAS_GPU = False


def log(msg):
    print(time.strftime('%H:%M:%S'), msg, flush=True)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
    except ImportError:
        pass


def configure(quick=False):
    # quick=True: small test run to check that a notebook works
    CFG['quick'] = quick
    if quick:
        CFG.update(n_splits=2, n_repeats=1, bootstrap=50, ctgan_epochs=2, ga_pop=4, ga_gens=2)


def setup(name):
    if os.environ.get('RESULTS_DIR'):
        base = os.environ['RESULTS_DIR']
    elif os.path.isdir('/content/drive/MyDrive'):
        base = DRIVE_DIR
    elif os.path.isdir('/content'):
        base = '/content'
    else:
        base = 'results'
    out = os.path.join(base, f'results_{name}' + ('_QUICK' if CFG['quick'] else ''))
    os.makedirs(out, exist_ok=True)
    save_environment(out)
    log(f'Results folder: {out} | GPU: {HAS_GPU}')
    return out


def save_environment(out):
    import sklearn, imblearn
    env = {'python': platform.python_version(), 'platform': platform.platform(), 'cpu_count': os.cpu_count(),
           'gpu': None, 'numpy': np.__version__, 'pandas': pd.__version__, 'scikit-learn': sklearn.__version__,
           'imbalanced-learn': imblearn.__version__, 'config': CFG}
    for pkg in ['catboost', 'xgboost', 'lightgbm', 'shap', 'optuna', 'ctgan', 'torch']:
        try:
            env[pkg] = __import__(pkg).__version__
        except ImportError:
            env[pkg] = None
    if HAS_GPU:
        import torch
        env['gpu'] = torch.cuda.get_device_name(0)
    with open(f'{out}/environment.json', 'w') as f:
        json.dump(env, f, indent=2)


def find_file(name):
    dirs = [os.environ.get('RESULTS_DIR', ''), DRIVE_DIR, '/content', 'data', '../data', 'results', '.']
    for d in dirs:
        p = os.path.join(d, name)
        if d and os.path.exists(p):
            return p
    raise FileNotFoundError(f'{name} not found (searched {dirs}).')


# data
def encode(df):
    out = df.copy()
    for c in YES_NO:
        out[c] = (out[c] == 'Yes').astype(int)
    out['Sex'] = (out['Sex'] == 'Male').astype(int)
    for c, levels in {**ORDINAL, **NOMINAL}.items():
        out[c] = out[c].map({v: i for i, v in enumerate(levels)})
    assert out[CAT_COLS].notna().all().all(), 'unexpected category value'
    out[CAT_COLS] = out[CAT_COLS].astype(int)
    return out[ALL_FEATURES + [TARGET]]


def prepare(scheme='extremes', test_size=0.2, seed=SEED):
    # raw file -> binary target -> remove duplicates -> encode -> one stratified train/test split
    raw = pd.read_csv(find_file('dataset.csv'))
    m = SCHEMES[scheme]
    df = raw[raw[TARGET].isin(m)].copy()
    df[TARGET] = df[TARGET].map(m)
    n_before = len(df)
    df = encode(df.drop_duplicates().reset_index(drop=True))
    X, y = df.drop(columns=TARGET), df[TARGET]
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=test_size, stratify=y, random_state=seed)
    if CFG['quick']:
        Xtr, _, ytr, _ = train_test_split(Xtr, ytr, train_size=4000, stratify=ytr, random_state=seed)
        Xte, _, yte, _ = train_test_split(Xte, yte, train_size=2000, stratify=yte, random_state=seed)
    info = {'scheme': scheme, 'rows_before_dedup': n_before, 'duplicates_removed': n_before - len(df),
            'rows': len(df), 'train_rows': len(Xtr), 'test_rows': len(Xte),
            'train_class_counts': ytr.value_counts().sort_index().to_dict(),
            'test_class_counts': yte.value_counts().sort_index().to_dict(),
            'test_fingerprint': hashlib.md5(np.sort(Xte.index.values).astype('int64').tobytes()).hexdigest()[:12]}
    log(f"Data: {info}")
    return (Xtr.reset_index(drop=True), Xte.reset_index(drop=True),
            ytr.reset_index(drop=True), yte.reset_index(drop=True), info)


# oversampling (applied to training data only)
# Each oversampler returns the input rows first, followed by the synthetic minority rows.
def _round_numeric(X):
    for c, d in NUM_DECIMALS.items():
        X[c] = X[c].round(d)
    return X


def os_original(X, y, seed):
    return X.copy(), y.copy()


def os_smotenc(X, y, seed):
    sc = StandardScaler().fit(X[NUM_COLS])
    Z = X.copy()
    Z[NUM_COLS] = sc.transform(X[NUM_COLS])
    sm = SMOTENC(categorical_features=[Z.columns.get_loc(c) for c in CAT_COLS], random_state=seed)
    Zr, yr = sm.fit_resample(Z, y)
    Zr = pd.DataFrame(np.asarray(Zr, dtype=float), columns=X.columns)
    Zr[NUM_COLS] = sc.inverse_transform(Zr[NUM_COLS])
    Zr[CAT_COLS] = Zr[CAT_COLS].round().astype(int)
    return _round_numeric(Zr), pd.Series(np.asarray(yr).astype(int), name=TARGET)


def _onehot(Xcat):
    return pd.concat([pd.DataFrame(np.eye(N_LEVELS[c])[Xcat[c].values],
                                   columns=[f'{c}__{i}' for i in range(N_LEVELS[c])]) for c in CAT_COLS], axis=1)


def os_adasyn(X, y, seed):
    # categorical features are one-hot encoded for the neighbour search; each synthetic
    # one-hot block is mapped back to the category with the largest value
    sc = StandardScaler().fit(X[NUM_COLS])
    Z = pd.concat([_onehot(X[CAT_COLS]), pd.DataFrame(sc.transform(X[NUM_COLS]), columns=NUM_COLS)], axis=1)
    Zr, yr = ADASYN(random_state=seed).fit_resample(Z, y)
    Zr = pd.DataFrame(np.asarray(Zr, dtype=float), columns=Z.columns)
    out = pd.DataFrame({c: Zr[[f'{c}__{i}' for i in range(N_LEVELS[c])]].values.argmax(1) for c in CAT_COLS})
    out[NUM_COLS] = sc.inverse_transform(Zr[NUM_COLS])
    return _round_numeric(out[X.columns]), pd.Series(np.asarray(yr).astype(int), name=TARGET)


def os_ctgan(X, y, seed):
    # CTGAN is trained on the minority (Poor health) rows of X only
    import torch
    from ctgan import CTGAN
    set_seed(seed)
    minority = X[y.values == 1].reset_index(drop=True)
    n_new = int((y == 0).sum() - (y == 1).sum())
    model = CTGAN(epochs=CFG['ctgan_epochs'], batch_size=500, generator_lr=2e-4, discriminator_lr=2e-4,
                  cuda=torch.cuda.is_available(), verbose=False)
    model.set_random_state(seed)
    model.fit(minority, discrete_columns=CAT_COLS)
    syn = model.sample(n_new)[X.columns]
    syn[CAT_COLS] = syn[CAT_COLS].astype(int)
    syn[NUM_COLS] = syn[NUM_COLS].astype(float).clip(lower=X[NUM_COLS].min(), upper=X[NUM_COLS].max(), axis=1)
    Xr = pd.concat([X.reset_index(drop=True), _round_numeric(syn)], ignore_index=True)
    return Xr, pd.Series(np.r_[y.values, np.ones(n_new, int)], name=TARGET)


OVERSAMPLERS = {'original': os_original, 'smotenc': os_smotenc, 'adasyn': os_adasyn, 'ctgan': os_ctgan}


def categories_valid(X):
    return all(X[c].between(0, N_LEVELS[c] - 1).all() and (X[c] == X[c].round()).all() for c in CAT_COLS)


def oversample_training(method, Xtr, ytr, out, seed=SEED):
    t = time.time()
    Xo, yo = OVERSAMPLERS[method](Xtr, ytr, seed)
    report = {'method': method, 'train_before': ytr.value_counts().sort_index().to_dict(),
              'train_after': yo.value_counts().sort_index().to_dict(), 'synthetic_rows': len(Xo) - len(Xtr),
              'original_rows_unchanged': bool(np.allclose(Xo.iloc[:len(Xtr)].values, Xtr.values)),
              'all_categorical_values_valid': bool(categories_valid(Xo)), 'seconds': round(time.time() - t, 1)}
    with open(f'{out}/oversampling_report.json', 'w') as f:
        json.dump(report, f, indent=2, default=int)
    log(f'Oversampling: {report}')
    return Xo, yo


# models and metrics
def make_model(name, feats, seed=SEED):
    nominal = [c for c in NOMINAL if c in feats]
    rest = [c for c in feats if c not in nominal]
    if name == 'CatBoost':
        from catboost import CatBoostClassifier
        p = dict(random_seed=seed, verbose=0, thread_count=-1, allow_writing_files=False,
                 cat_features=[c for c in CAT_COLS if c in feats])
        if HAS_GPU:
            p['task_type'] = 'GPU'
        return CatBoostClassifier(**p)
    if name == 'LogisticRegression':
        pre = ColumnTransformer([('onehot', OneHotEncoder(handle_unknown='ignore'), nominal),
                                 ('scale', StandardScaler(), rest)])
        return make_pipeline(pre, LogisticRegression(max_iter=2000, random_state=seed))
    pre = ColumnTransformer([('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False), nominal)],
                            remainder='passthrough')
    if name == 'DecisionTree':
        clf = DecisionTreeClassifier(random_state=seed)
    elif name == 'RandomForest':
        clf = RandomForestClassifier(random_state=seed, n_jobs=-1)
    elif name == 'LightGBM':
        from lightgbm import LGBMClassifier
        clf = LGBMClassifier(random_state=seed, n_jobs=-1, verbose=-1)
    elif name == 'XGBoost':
        from xgboost import XGBClassifier
        clf = XGBClassifier(random_state=seed, n_jobs=-1, tree_method='hist', eval_metric='logloss',
                            device='cuda' if HAS_GPU else 'cpu')
    return make_pipeline(pre, clf)


def metrics(y, p, thr=0.5):
    # positive class = Poor health
    y = np.asarray(y)
    yp = (np.asarray(p) >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, yp, labels=[0, 1]).ravel()
    sens = tp / (tp + fn) if tp + fn else 0.0
    spec = tn / (tn + fp) if tn + fp else 0.0
    return {'accuracy': (tp + tn) / len(y), 'balanced_accuracy': (sens + spec) / 2, 'sensitivity': sens,
            'specificity': spec, 'precision_poor': tp / (tp + fp) if tp + fp else 0.0,
            'f1_poor': f1_score(y, yp, zero_division=0), 'macro_f1': f1_score(y, yp, average='macro'),
            'mcc': matthews_corrcoef(y, yp), 'g_mean': float(np.sqrt(sens * spec)),
            'roc_auc': roc_auc_score(y, p), 'pr_auc': average_precision_score(y, p),
            'brier': brier_score_loss(y, p), 'tn': int(tn), 'fp': int(fp), 'fn': int(fn), 'tp': int(tp)}


CI_METRICS = ['accuracy', 'balanced_accuracy', 'sensitivity', 'specificity', 'precision_poor', 'f1_poor',
              'macro_f1', 'mcc', 'g_mean', 'roc_auc', 'pr_auc', 'brier']


def bootstrap_ci(y, p, seed=SEED):
    # 95% percentile bootstrap intervals over resamples of the test set
    y, p = np.asarray(y), np.asarray(p)
    rng = np.random.default_rng(seed)
    draws = []
    for _ in range(CFG['bootstrap']):
        i = rng.integers(0, len(y), len(y))
        m = metrics(y[i], p[i])
        draws.append([m[k] for k in CI_METRICS])
    lo, hi = np.percentile(np.array(draws), [2.5, 97.5], axis=0)
    return {**{f'{k}_lo': a for k, a in zip(CI_METRICS, lo)}, **{f'{k}_hi': b for k, b in zip(CI_METRICS, hi)}}


# repeated cross-validation on the training data
def run_cv(method, Xtr, ytr, out, seed=SEED):
    # the oversampler is refitted on the training part of every fold; the validation part is real data
    path = f'{out}/cv_scores.csv'
    done = set(pd.read_csv(path)['fold']) if os.path.exists(path) else set()
    rskf = RepeatedStratifiedKFold(n_splits=CFG['n_splits'], n_repeats=CFG['n_repeats'], random_state=seed)
    n_folds = CFG['n_splits'] * CFG['n_repeats']
    for k, (tr, va) in enumerate(rskf.split(Xtr, ytr)):
        if k in done:
            log(f'CV fold {k + 1}/{n_folds} already done, skipped')
            continue
        t0 = time.time()
        Xo, yo = OVERSAMPLERS[method](Xtr.iloc[tr].reset_index(drop=True), ytr.iloc[tr].reset_index(drop=True),
                                      seed + k)
        Xv, yv = Xtr.iloc[va], ytr.iloc[va].values
        rows = []
        for fs, feats in FEATURE_SETS.items():
            for name in MODEL_NAMES:
                m = make_model(name, feats, seed)
                t = time.time()
                m.fit(Xo[feats], yo)
                p = m.predict_proba(Xv[feats])[:, 1]
                rows.append({'method': method, 'fold': k, 'repeat': k // CFG['n_splits'], 'feature_set': fs,
                             'model': name, 'fit_seconds': round(time.time() - t, 1), **metrics(yv, p)})
        pd.DataFrame(rows).to_csv(path, mode='a', header=not os.path.exists(path), index=False)
        log(f'CV fold {k + 1}/{n_folds} done in {time.time() - t0:.0f}s')
    cv = pd.read_csv(path)
    summary = cv.groupby(['feature_set', 'model'])[
        ['balanced_accuracy', 'sensitivity', 'specificity', 'mcc', 'roc_auc', 'pr_auc', 'accuracy']].agg(
        ['mean', 'std']).round(4)
    summary.to_csv(f'{out}/cv_summary.csv')
    print(summary.to_string())
    return cv


# final evaluation on the test set
def run_test(method, Xo, yo, Xte, yte, out, seed=SEED):
    # fit on the (oversampled) training set, evaluate once on the real test set
    preds, rows = pd.DataFrame({'y_true': yte.values}), []
    for fs, feats in FEATURE_SETS.items():
        for name in MODEL_NAMES:
            m = make_model(name, feats, seed)
            m.fit(Xo[feats], yo)
            p = m.predict_proba(Xte[feats])[:, 1]
            preds[f'{fs}|{name}'] = p
            rows.append({'method': method, 'feature_set': fs, 'model': name, **metrics(yte, p),
                         **bootstrap_ci(yte, p, seed)})
            log(f'Test: {fs:11s} {name:18s} bal.acc {rows[-1]["balanced_accuracy"]:.4f} '
                f'[{rows[-1]["balanced_accuracy_lo"]:.4f}, {rows[-1]["balanced_accuracy_hi"]:.4f}]  '
                f'sens {rows[-1]["sensitivity"]:.4f}  acc {rows[-1]["accuracy"]:.4f}')
    res = pd.DataFrame(rows)
    res.to_csv(f'{out}/test_metrics.csv', index=False)
    preds.to_csv(f'{out}/test_predictions.csv', index=False)
    return res, preds


# quality of the synthetic records
def synthetic_quality(Xtr, ytr, Xo, Xte, yte, out, seed=SEED):
    real = Xtr[ytr.values == 1].reset_index(drop=True)
    syn = Xo.iloc[len(Xtr):].reset_index(drop=True)
    holdout = Xte[yte.values == 1].reset_index(drop=True)
    num_rows, cat_rows = [], []
    for c in NUM_COLS:
        sd = real[c].std() or 1.0
        num_rows.append({'feature': c, 'ks_statistic': stats.ks_2samp(real[c], syn[c]).statistic,
                         'wasserstein_in_sd': stats.wasserstein_distance(real[c], syn[c]) / sd,
                         'mean_real': real[c].mean(), 'mean_synthetic': syn[c].mean()})
    for c in CAT_COLS:
        pr = real[c].value_counts(normalize=True).reindex(range(N_LEVELS[c]), fill_value=0)
        ps = syn[c].value_counts(normalize=True).reindex(range(N_LEVELS[c]), fill_value=0)
        cat_rows.append({'feature': c, 'total_variation_distance': 0.5 * (pr - ps).abs().sum()})
    num_df, cat_df = pd.DataFrame(num_rows), pd.DataFrame(cat_rows)
    num_df.to_csv(f'{out}/quality_numeric.csv', index=False)
    cat_df.to_csv(f'{out}/quality_categorical.csv', index=False)

    cr, cs = real.corr().values, syn.corr().values
    iu = np.triu_indices_from(cr, 1)
    corr_diff = float(np.nanmean(np.abs(cr[iu] - cs[iu])))

    # real-vs-synthetic classifier (ROC-AUC 0.5 = indistinguishable)
    n = min(len(real), len(syn), 5000)
    D = pd.concat([real.sample(n, random_state=seed), syn.sample(n, random_state=seed)], ignore_index=True)
    lab = np.r_[np.zeros(n), np.ones(n)]
    pd_ = cross_val_predict(HistGradientBoostingClassifier(random_state=seed), D, lab,
                            cv=StratifiedKFold(5, shuffle=True, random_state=seed), method='predict_proba')[:, 1]
    disc_auc = roc_auc_score(lab, pd_)

    # distance to the closest real training record, compared with real test records
    def space(A):
        return np.c_[_onehot(A[CAT_COLS]).values, (A[NUM_COLS].values - real[NUM_COLS].mean().values) /
                     real[NUM_COLS].std().values]
    nn = NearestNeighbors(n_neighbors=1).fit(space(real))
    dcr_syn = nn.kneighbors(space(syn))[0].ravel()
    dcr_hold = nn.kneighbors(space(holdout))[0].ravel()
    exact_copies = int(syn.merge(real.drop_duplicates(), how='inner').shape[0])

    # train on synthetic minority + real majority, test on real data (TSTR)
    maj = Xtr[ytr.values == 0]
    Xs = pd.concat([maj, syn], ignore_index=True)
    ys = np.r_[np.zeros(len(maj)), np.ones(len(syn))]
    m = make_model('CatBoost', ALL_FEATURES, seed).fit(Xs, ys)
    tstr = metrics(yte, m.predict_proba(Xte)[:, 1])

    summary = {'synthetic_rows': len(syn), 'real_minority_train_rows': len(real),
               'mean_ks_numeric': num_df.ks_statistic.mean(), 'mean_wasserstein_sd_numeric': num_df.wasserstein_in_sd.mean(),
               'mean_tvd_categorical': cat_df.total_variation_distance.mean(), 'mean_abs_corr_difference': corr_diff,
               'discriminator_auc': disc_auc, 'median_dcr_synthetic': float(np.median(dcr_syn)),
               'median_dcr_real_holdout': float(np.median(dcr_hold)),
               'dcr_ratio_synthetic_to_holdout': float(np.median(dcr_syn) / np.median(dcr_hold)),
               'exact_copies_of_real_rows': exact_copies,
               'tstr_balanced_accuracy': tstr['balanced_accuracy'], 'tstr_sensitivity': tstr['sensitivity'],
               'tstr_roc_auc': tstr['roc_auc'], 'tstr_pr_auc': tstr['pr_auc']}
    with open(f'{out}/quality_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
    print(num_df.round(4).to_string(index=False))
    print(cat_df.round(4).to_string(index=False))
    log(f'Synthetic data quality: {json.dumps({k: round(v, 4) if isinstance(v, float) else v for k, v in summary.items()})}')
    return summary


# dataset description and EDA statistics
def run_eda(out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    raw = pd.read_csv(find_file('dataset.csv'))
    order = ['Excellent', 'Very Good', 'Good', 'Fair', 'Poor']
    gh = raw[TARGET].value_counts().reindex(order)
    mapping = pd.DataFrame({'category': order, 'count': gh.values, 'percent': (gh.values / len(raw) * 100).round(2),
                            'use': ['Excellent health (0)', 'excluded', 'excluded', 'excluded', 'Poor health (1)']})
    mapping.to_csv(f'{out}/eda_target_mapping.csv', index=False)
    print(mapping.to_string(index=False))
    df = raw[raw[TARGET].isin(['Excellent', 'Poor'])].drop_duplicates().reset_index(drop=True)
    y = (df[TARGET] == 'Poor').astype(int)
    log(f'Raw rows {len(raw)}, raw duplicates {raw.duplicated().sum()}, study rows after de-duplication {len(df)}')

    cat_rows = []
    for c in CAT_COLS:
        t = pd.crosstab(df[c], y)
        chi, p, dof, _ = stats.chi2_contingency(t)
        cat_rows.append({'feature': c, 'chi2': chi, 'dof': dof, 'p_value': p,
                         'cramers_v': np.sqrt(chi / (t.values.sum() * (min(t.shape) - 1))),
                         'percent_poor_by_level': json.dumps((df.groupby(c)[TARGET].apply(
                             lambda s: (s == 'Poor').mean() * 100)).round(1).to_dict())})
    num_rows = []
    X = df[NUM_COLS]
    for c in NUM_COLS:
        a, b = X[c][y == 1], X[c][y == 0]
        u, p = stats.mannwhitneyu(a, b)
        others = X.drop(columns=c)
        r2 = LinearRegression().fit(others, X[c]).score(others, X[c])
        num_rows.append({'feature': c, 'skewness': stats.skew(X[c]), 'median_poor': a.median(),
                         'median_excellent': b.median(), 'mannwhitney_p': p,
                         'rank_biserial_r': 2 * u / (len(a) * len(b)) - 1, 'vif': 1 / (1 - r2)})
    cat_df, num_df = pd.DataFrame(cat_rows), pd.DataFrame(num_rows)
    cat_df.to_csv(f'{out}/eda_categorical_tests.csv', index=False)
    num_df.to_csv(f'{out}/eda_numeric_tests.csv', index=False)
    print(cat_df.drop(columns='percent_poor_by_level').round(4).to_string(index=False))
    print(num_df.round(4).to_string(index=False))

    fig, axes = plt.subplots(4, 3, figsize=(15, 16))
    for ax, c in zip(axes.ravel(), CAT_COLS):
        levels = {**ORDINAL, **NOMINAL}.get(c, sorted(df[c].unique()))
        pd.crosstab(df[c], df[TARGET]).reindex(levels)[['Excellent', 'Poor']].plot.bar(ax=ax, rot=45)
        ax.set_title(c); ax.set_xlabel('')
    axes.ravel()[-1].axis('off')
    plt.tight_layout(); plt.savefig(f'{out}/eda_categorical_by_class.png', dpi=300); plt.close('all')
    fig, axes = plt.subplots(2, 4, figsize=(18, 8))
    for ax, c in zip(axes.ravel(), NUM_COLS):
        for lab in ['Excellent', 'Poor']:
            ax.hist(df.loc[df[TARGET] == lab, c], bins=40, alpha=0.5, density=True, label=lab)
        ax.set_title(f'{c} (skew {stats.skew(df[c]):.2f})'); ax.legend()
    axes.ravel()[-1].axis('off')
    plt.tight_layout(); plt.savefig(f'{out}/eda_numeric_by_class.png', dpi=300); plt.close('all')
    log('EDA tables and figures saved')


# sensitivity analyses (target definition, diagnoses)
def run_sensitivity(out, seed=SEED):
    rows = []
    analyses = [('cdc_dichotomy', 'cdc', ALL_FEATURES),
                ('extremes_all_features', 'extremes', ALL_FEATURES),
                ('extremes_without_diagnoses', 'extremes', [c for c in ALL_FEATURES if c not in DIAG_COLS])]
    data = {}
    for label, scheme, feats in analyses:
        if scheme not in data:
            data[scheme] = prepare(scheme)
        Xtr, Xte, ytr, yte, info = data[scheme]
        for method in ['original', 'smotenc']:
            Xo, yo = OVERSAMPLERS[method](Xtr, ytr, seed)
            for name in ['LogisticRegression', 'CatBoost']:
                m = make_model(name, feats, seed).fit(Xo[feats], yo)
                p = m.predict_proba(Xte[feats])[:, 1]
                rows.append({'analysis': label, 'scheme': scheme, 'n_features': len(feats), 'method': method,
                             'model': name, 'train_rows': len(Xtr), 'test_rows': len(Xte),
                             'test_minority_share': float(yte.mean()), **metrics(yte, p),
                             **bootstrap_ci(yte, p, seed)})
                log(f'{label:28s} {method:8s} {name:18s} bal.acc {rows[-1]["balanced_accuracy"]:.4f} '
                    f'sens {rows[-1]["sensitivity"]:.4f} roc_auc {rows[-1]["roc_auc"]:.4f}')
            pd.DataFrame(rows).to_csv(f'{out}/sensitivity_results.csv', index=False)
    return pd.DataFrame(rows)
