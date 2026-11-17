# Hyperparameter search for LightGBM trained with ADASYN: genetic algorithm, random search and
# Bayesian optimisation (Optuna TPE) with the same number of evaluated configurations.
# Fitness = balanced accuracy on a stratified validation split of the training data (real records);
# the test set is used once per optimiser, after the search.
import json, random, time, itertools
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import make_pipeline
import pipeline as P

# 5 x 4 x 4 x 4 x 4 = 1,280 combinations; the LightGBM defaults are inside the space
SPACE = {'n_estimators': [100, 200, 300, 400, 500], 'num_leaves': [15, 31, 63, 127],
         'learning_rate': [0.01, 0.05, 0.1, 0.2], 'min_child_samples': [10, 20, 50, 100],
         'reg_lambda': [0.0, 1.0, 5.0, 10.0]}


def make_lgbm(params, seed):
    # same model as pipeline.make_model('LightGBM'), with the given hyperparameters
    from lightgbm import LGBMClassifier
    feats = P.ALL_FEATURES
    nominal = [c for c in P.NOMINAL if c in feats]
    pre = ColumnTransformer([('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False), nominal)],
                            remainder='passthrough')
    return make_pipeline(pre, LGBMClassifier(random_state=seed, n_jobs=-1, verbose=-1, **params))


def tune(method, Xtr, ytr, Xte, yte, out, seed):
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    feats = P.ALL_FEATURES
    Xa, Xv, ya, yv = train_test_split(Xtr, ytr, test_size=0.2, stratify=ytr, random_state=seed)
    Xo, yo = P.OVERSAMPLERS[method](Xa.reset_index(drop=True), ya.reset_index(drop=True), seed)
    yv = yv.values
    hist_path = f'{out}/tuning_history_seed{seed}.csv'
    history = []

    def evaluator(opt):
        cache = {}

        def f(params):
            key = tuple(params[k] for k in SPACE)
            if key not in cache:
                t = time.time()
                m = make_lgbm(dict(params), seed).fit(Xo[feats], yo)
                cache[key] = P.metrics(yv, m.predict_proba(Xv[feats])[:, 1])['balanced_accuracy']
                history.append({'seed': seed, 'optimizer': opt, 'evaluation': len(cache), 'score': cache[key],
                                'best_so_far': max(cache.values()), 'fit_seconds': round(time.time() - t, 1),
                                **params})
                if len(cache) % 10 == 0:
                    P.log(f'  {opt}: {len(cache)} evaluations, best validation balanced acc {max(cache.values()):.4f}')
                    pd.DataFrame(history).to_csv(hist_path, index=False)
            return cache[key]
        return f, cache

    # genetic algorithm: elitism (2), one-point crossover, mutation rate 0.1 per hyperparameter
    rnd, nrng = random.Random(seed), np.random.RandomState(seed)
    f_ga, c_ga = evaluator('GA')
    population = [{k: rnd.choice(v) for k, v in SPACE.items()} for _ in range(P.CFG['ga_pop'])]
    for g in range(P.CFG['ga_gens']):
        fit = [f_ga(ind) for ind in population]
        new = [population[i] for i in np.argsort(fit)[::-1][:2]]
        while len(new) < P.CFG['ga_pop']:
            p1, p2 = rnd.choice(new), rnd.choice(new)
            cp = nrng.randint(1, len(SPACE))
            c1 = dict(list(p1.items())[:cp] + list(p2.items())[cp:])
            c2 = dict(list(p2.items())[:cp] + list(p1.items())[cp:])
            for ch in (c1, c2):
                for k in SPACE:
                    if nrng.rand() < 0.1:
                        ch[k] = rnd.choice(SPACE[k])
            new += [c1, c2]
        population = new[:P.CFG['ga_pop']]
    for ind in population:
        f_ga(ind)
    budget = len(c_ga)
    P.log(f'GA finished: {budget} unique configurations evaluated (budget for random search and TPE)')

    f_rs, c_rs = evaluator('RandomSearch')
    combos = list(itertools.product(*SPACE.values()))
    for i in random.Random(seed + 1).sample(range(len(combos)), budget):
        f_rs(dict(zip(SPACE, combos[i])))

    f_tpe, c_tpe = evaluator('BayesianTPE')
    study = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(seed=seed))
    trials = 0
    while len(c_tpe) < budget and trials < budget * 5:
        tr = study.ask()
        study.tell(tr, f_tpe({k: tr.suggest_categorical(k, v) for k, v in SPACE.items()}))
        trials += 1
    pd.DataFrame(history).to_csv(hist_path, index=False)

    # refit the best configuration of each optimiser on the full training data and evaluate on the test set
    Xf, yf = P.OVERSAMPLERS[method](Xtr, ytr, seed)
    default_val = P.metrics(yv, make_lgbm({}, seed).fit(Xo[feats], yo).predict_proba(Xv[feats])[:, 1])['balanced_accuracy']
    candidates = {'Default LightGBM': {}}
    for opt, cache in [('GA', c_ga), ('RandomSearch', c_rs), ('BayesianTPE', c_tpe)]:
        candidates[opt] = dict(zip(SPACE, max(cache, key=cache.get)))
    rows, preds = [], pd.DataFrame({'y_true': yte.values})
    for opt, params in candidates.items():
        p = make_lgbm(params, seed).fit(Xf[feats], yf).predict_proba(Xte[feats])[:, 1]
        cache = {'GA': c_ga, 'RandomSearch': c_rs, 'BayesianTPE': c_tpe}.get(opt)
        rows.append({'method': method, 'model': 'LightGBM', 'seed': seed, 'optimizer': opt,
                     'evaluations': len(cache) if cache else 0,
                     'best_validation_balanced_accuracy': max(cache.values()) if cache else default_val,
                     'params': json.dumps(params), **P.metrics(yte, p), **P.bootstrap_ci(yte, p, seed)})
        preds[opt] = p
        P.log(f'Test ({opt}): bal.acc {rows[-1]["balanced_accuracy"]:.4f} sens {rows[-1]["sensitivity"]:.4f} '
              f'params {params}')
    res = pd.DataFrame(rows)
    res.to_csv(f'{out}/tuning_test_seed{seed}.csv', index=False)
    preds.to_csv(f'{out}/tuning_test_predictions_seed{seed}.csv', index=False)
    return res
