# Recalibration of models trained on oversampled data.
# Calibration maps are fitted on out-of-fold predictions for the training data (oversampling inside
# each fold); the test set is only used to evaluate the recalibrated probabilities.
import json
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.isotonic import IsotonicRegression
from sklearn.calibration import calibration_curve
from sklearn.metrics import (brier_score_loss, log_loss, roc_auc_score, average_precision_score,
                             balanced_accuracy_score, recall_score)
import pipeline as P

EPS = 1e-6
VARIANTS = ['uncalibrated', 'platt', 'isotonic', 'prior_correction']


def logit(p):
    p = np.clip(np.asarray(p, float), EPS, 1 - EPS)
    return np.log(p / (1 - p))


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def ece(y, p, k=10):
    # expected calibration error, 10 equal-width bins
    y, p = np.asarray(y), np.asarray(p)
    b = np.minimum((p * k).astype(int), k - 1)
    return float(sum(abs(p[b == i].mean() - y[b == i].mean()) * (b == i).mean() for i in range(k) if (b == i).any()))


def slope_intercept(y, p):
    # logistic regression of the outcome on logit(p); ideal slope 1, intercept 0
    lr = LogisticRegression(C=1e6, max_iter=1000).fit(logit(p).reshape(-1, 1), y)
    return float(lr.coef_[0, 0]), float(lr.intercept_[0])


def prob_scores(y, p):
    slope, intercept = slope_intercept(y, p)
    return {'ece': ece(y, p), 'brier': brier_score_loss(y, p), 'log_loss': log_loss(y, np.clip(p, EPS, 1 - EPS)),
            'calibration_slope': slope, 'calibration_intercept': intercept, 'mean_predicted': float(np.mean(p)),
            'observed_rate': float(np.mean(y)), 'roc_auc': roc_auc_score(y, p), 'pr_auc': average_precision_score(y, p)}


def oof_predictions(method, model, Xtr, ytr, seed=P.SEED):
    n_splits = 2 if P.CFG['quick'] else 5
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    oof = np.zeros(len(ytr))
    for k, (a, b) in enumerate(skf.split(Xtr, ytr)):
        Xo, yo = P.OVERSAMPLERS[method](Xtr.iloc[a].reset_index(drop=True), ytr.iloc[a].reset_index(drop=True), seed + k)
        m = P.make_model(model, P.ALL_FEATURES, seed).fit(Xo[P.ALL_FEATURES], yo)
        oof[b] = m.predict_proba(Xtr.iloc[b][P.ALL_FEATURES])[:, 1]
        P.log(f'  {method}: out-of-fold predictions, fold {k + 1}/{n_splits} done')
    return oof


def fit_calibrators(oof, ytr, train_prevalence, oversampled_prevalence):
    platt = LogisticRegression(C=1e6, max_iter=1000).fit(logit(oof).reshape(-1, 1), ytr)
    iso = IsotonicRegression(out_of_bounds='clip', y_min=0, y_max=1).fit(oof, ytr)
    # prior correction: shift the log-odds back from the balanced to the training class prior
    shift = np.log(train_prevalence / (1 - train_prevalence)) - np.log(oversampled_prevalence / (1 - oversampled_prevalence))
    return {'uncalibrated': lambda p: np.asarray(p, float),
            'platt': lambda p: platt.predict_proba(logit(p).reshape(-1, 1))[:, 1],
            'isotonic': lambda p: iso.predict(np.asarray(p, float)),
            'prior_correction': lambda p: sigmoid(logit(p) + shift)}


def bootstrap(y, probs, n=500, seed=P.SEED):
    # 95% intervals of ECE and Brier score, and of their change from the uncalibrated model
    rng = np.random.default_rng(seed)
    y = np.asarray(y)
    draws = {v: {'ece': [], 'brier': []} for v in probs}
    for _ in range(n):
        i = rng.integers(0, len(y), len(y))
        for v, p in probs.items():
            draws[v]['ece'].append(ece(y[i], p[i]))
            draws[v]['brier'].append(brier_score_loss(y[i], p[i]))
    res = {}
    for v in probs:
        for k in ('ece', 'brier'):
            a = np.array(draws[v][k]); d = a - np.array(draws['uncalibrated'][k])
            res[v] = {**res.get(v, {}), f'{k}_lo': np.percentile(a, 2.5), f'{k}_hi': np.percentile(a, 97.5),
                      f'{k}_change_lo': np.percentile(d, 2.5), f'{k}_change_hi': np.percentile(d, 97.5)}
    return res


def run(methods, model, Xtr, ytr, Xte, yte, out, seed=P.SEED):
    rows, preds = [], pd.DataFrame({'y_true': yte.values})
    yte_ = yte.values
    for method in methods:
        P.log(f'=== {model} trained with {method} ===')
        oof = oof_predictions(method, model, Xtr, ytr, seed)
        Xo, yo = P.OVERSAMPLERS[method](Xtr, ytr, seed)
        final = P.make_model(model, P.ALL_FEATURES, seed).fit(Xo[P.ALL_FEATURES], yo)
        p_test = final.predict_proba(Xte[P.ALL_FEATURES])[:, 1]
        cal = fit_calibrators(oof, ytr.values, float(ytr.mean()), float(yo.mean()))
        probs = {v: cal[v](p_test) for v in VARIANTS}
        decision = (p_test >= 0.5).astype(int)   # classifications of the model at 0.5 are kept
        cis = bootstrap(yte_, probs)
        for v in VARIANTS:
            s = prob_scores(yte_, probs[v])
            rows.append({'method': method, 'model': model, 'variant': v, **s, **cis[v],
                         'balanced_accuracy_original_model_at_0.5': balanced_accuracy_score(yte_, decision),
                         'sensitivity_original_model_at_0.5': recall_score(yte_, decision),
                         'balanced_accuracy_if_calibrated_probs_cut_at_0.5':
                             balanced_accuracy_score(yte_, (probs[v] >= 0.5).astype(int))})
            preds[f'{method}|{v}'] = probs[v]
        pd.DataFrame(rows).to_csv(f'{out}/recalibration_metrics.csv', index=False)
        pd.DataFrame({'y_true': ytr.values, 'oof': oof}).to_csv(f'{out}/oof_{method}.csv', index=False)
        preds.to_csv(f'{out}/recalibrated_test_predictions.csv', index=False)
        _plot(yte_, probs, method, model, out)
    res = pd.DataFrame(rows)
    show = ['method', 'variant', 'ece', 'ece_lo', 'ece_hi', 'brier', 'calibration_slope', 'calibration_intercept',
            'mean_predicted', 'observed_rate', 'roc_auc', 'balanced_accuracy_original_model_at_0.5']
    print(res[show].round(4).to_string(index=False))
    with open(f'{out}/recalibration_info.json', 'w') as f:
        json.dump({'model': model, 'methods': methods, 'calibration_data': 'out-of-fold predictions on the training '
                   'data (oversampling inside each fold)'}, f, indent=2)
    return res


def _plot(y, probs, method, model, out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(4.6, 4.4))
    ax.plot([0, 1], [0, 1], ls='--', color='#52514e', lw=0.8)
    colors = {'uncalibrated': '#eb6834', 'platt': '#2a78d6', 'isotonic': '#1baf7a', 'prior_correction': '#eda100'}
    for v, p in probs.items():
        fy, fx = calibration_curve(y, p, n_bins=10, strategy='quantile')
        ax.plot(fx, fy, marker='o', ms=4, lw=2, color=colors[v], label=f'{v} (ECE {ece(y, p):.3f})')
    ax.set_xlabel('Mean predicted probability of Poor health'); ax.set_ylabel('Observed share with Poor health')
    ax.set_title(f'{model} trained with {method}', fontsize=9); ax.legend(fontsize=7, frameon=False)
    fig.savefig(f'{out}/calibration_recalibrated_{method}.png', dpi=300, bbox_inches='tight')
    plt.close(fig)
