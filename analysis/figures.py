# Figures of the paper, built from the raw data and the notebook results.
# Usage: python analysis/figures.py <results folder> <dataset.csv> [output folder]
import os, json, sys
import numpy as np, pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

EX = sys.argv[1] if len(sys.argv) > 1 else 'results'
RAW = sys.argv[2] if len(sys.argv) > 2 else os.path.join('data', 'dataset.csv')
OUT = sys.argv[3] if len(sys.argv) > 3 else os.path.join(EX, 'figures')
os.makedirs(OUT, exist_ok=True)

INK, INK2, GRID, SURF = '#0b0b0b', '#52514e', '#e4e3df', '#ffffff'
BLUE, ORANGE, AQUA, YELLOW = '#2a78d6', '#eb6834', '#1baf7a', '#eda100'
METHODS = ['original', 'smotenc', 'adasyn', 'ctgan']
LABEL = {'original': 'No oversampling', 'smotenc': 'SMOTENC', 'adasyn': 'ADASYN', 'ctgan': 'CTGAN'}
COLOR = dict(zip(METHODS, [BLUE, ORANGE, AQUA, YELLOW]))
CLS = {0: ('Excellent health', BLUE), 1: ('Poor health', ORANGE)}
MODELS = ['LogisticRegression', 'DecisionTree', 'RandomForest', 'LightGBM', 'XGBoost', 'CatBoost']
MSHORT = {'LogisticRegression': 'LR', 'DecisionTree': 'DT', 'RandomForest': 'RF', 'LightGBM': 'LightGBM',
          'XGBoost': 'XGBoost', 'CatBoost': 'CatBoost'}

plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 9, 'axes.edgecolor': INK2, 'axes.labelcolor': INK,
                     'xtick.color': INK2, 'ytick.color': INK2, 'axes.spines.top': False, 'axes.spines.right': False,
                     'axes.grid': True, 'grid.color': GRID, 'grid.linewidth': 0.6, 'axes.axisbelow': True,
                     'figure.facecolor': SURF, 'axes.facecolor': SURF, 'legend.frameon': False, 'savefig.dpi': 300})


def save(fig, name):
    fig.savefig(os.path.join(OUT, name), bbox_inches='tight')
    plt.close(fig)
    print('saved', name)


def panel(ax, letter):
    ax.text(-0.12, 1.04, f'({letter})', transform=ax.transAxes, fontsize=10, fontweight='bold', color=INK)


# data (same preparation as pipeline.py)
raw = pd.read_csv(RAW)
df = raw[raw.General_Health.isin(['Excellent', 'Poor'])].drop_duplicates().reset_index(drop=True)
df['y'] = (df.General_Health == 'Poor').astype(int)
NUM = ['Height_(cm)', 'Weight_(kg)', 'BMI', 'Alcohol_Consumption', 'Fruit_Consumption',
       'Green_Vegetables_Consumption', 'FriedPotato_Consumption']
print('rows', len(df), 'mean height', round(df['Height_(cm)'].mean(), 1))

# Fig 1: workflow
fig, ax = plt.subplots(figsize=(9.6, 5.6))
ax.set_xlim(0, 100); ax.set_ylim(0, 64); ax.axis('off')


def box(x, y, w, h, text, fc='#f3f2ef', ec=INK2, bold=False, size=7.0):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0.4,rounding_size=1.2', fc=fc, ec=ec, lw=0.9))
    ax.text(x + w / 2, y + h / 2, text, ha='center', va='center', fontsize=size, color=INK,
            fontweight='bold' if bold else 'normal', wrap=True)


def arrow(x1, y1, x2, y2, color=INK2):
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1), arrowprops=dict(arrowstyle='-|>', color=color, lw=1.0))


box(1, 52, 22, 10, 'BRFSS (Kaggle)\n308,854 respondents\n19 variables')
box(27, 52, 24, 10, 'Target: self-rated health\nExcellent (0) vs Poor (1)\n67,260 after de-duplication')
box(55, 52, 22, 10, 'Encoding\nbinary 0/1, ordinal order,\nnative categorical\n(CatBoost)')
box(81, 52, 18, 10, 'Stratified split\n80 / 20\n(seed 42)', fc='#e6effb', ec=BLUE, bold=True)
arrow(23.6, 57, 26.4, 57); arrow(51.6, 57, 54.4, 57); arrow(77.6, 57, 80.4, 57)

ax.add_patch(FancyBboxPatch((1, 6), 70, 40, boxstyle='round,pad=0.4,rounding_size=1.5', fc='#fbfbfa',
                            ec=INK2, lw=0.8, ls='--'))
ax.text(3, 43.5, 'Training data: 53,808 real records (9,065 Poor)', fontsize=8, fontweight='bold', color=INK)
box(3, 31, 20, 9, 'Oversampling\n(training data only)\nnone · SMOTENC ·\nADASYN · CTGAN', fc='#fdebe3', ec=ORANGE)
box(27, 31, 20, 9, '6 classifiers\nLR · DT · RF\nLightGBM · XGBoost\nCatBoost')
box(51, 31, 18, 9, 'Repeated stratified\n5-fold CV × 3\n(oversampling\ninside each fold)')
arrow(23.6, 35.5, 26.4, 35.5); arrow(47.6, 35.5, 50.4, 35.5)
box(3, 9, 30, 17, 'Hyperparameter search\n(final model: LightGBM + ADASYN)\nGA vs random search\nvs Bayesian (TPE)\nequal budget, 3 seeds\nfitness: balanced accuracy\non a real validation split')
box(37, 9, 32, 17, 'Synthetic-data quality\nKS · Wasserstein · TVD\ncorrelation · discriminator\ndistance to closest record · TSTR')
arrow(13, 30.4, 13, 26.6)

box(75, 26, 24, 16, 'Test data\n13,452 real records\n(2,266 Poor)\nsame for all methods\nused once\nnever oversampled', fc='#e6effb', ec=BLUE, bold=False)
arrow(90, 51.4, 90, 42.6, BLUE)
box(75, 4, 24, 17, 'Evaluation\nbalanced accuracy,\nsensitivity, MCC,\nROC/PR-AUC, calibration\n95% CI, paired tests\nSHAP + stability')
arrow(87, 25.4, 87, 21.6, BLUE)
arrow(71.6, 26, 74.4, 16, INK2)
ax.text(71.0, 13.0, 'final\nmodels', fontsize=6.3, color=INK2, ha='center')
save(fig, 'fig01_workflow.png')

# Fig 2: Height distribution
fig, axs = plt.subplots(1, 2, figsize=(7.2, 2.8))
axs[0].hist(df['Height_(cm)'], bins=40, color=BLUE, edgecolor=SURF, linewidth=0.4)
axs[0].axvline(df['Height_(cm)'].mean(), color=INK, lw=1, ls='--')
axs[0].text(df['Height_(cm)'].mean() + 2, axs[0].get_ylim()[1] * 0.9, f"mean {df['Height_(cm)'].mean():.1f} cm",
            fontsize=8, color=INK)
axs[0].set_xlabel('Height (cm)'); axs[0].set_ylabel('Respondents'); panel(axs[0], 'a')
axs[1].boxplot(df['Height_(cm)'], vert=False, widths=0.5, patch_artist=True,
               boxprops=dict(facecolor='#cde2fb', edgecolor=BLUE), medianprops=dict(color=INK),
               flierprops=dict(marker='o', markersize=2, markerfacecolor=INK2, markeredgecolor='none', alpha=0.4))
axs[1].set_yticks([]); axs[1].set_xlabel('Height (cm)'); panel(axs[1], 'b')
save(fig, 'fig02_height.png')

# Fig 3: target distribution
fig, ax = plt.subplots(figsize=(4.2, 2.8))
vc = df.y.value_counts().sort_index()
bars = ax.bar([CLS[0][0], CLS[1][0]], vc.values, color=[BLUE, ORANGE], width=0.55)
for b, v in zip(bars, vc.values):
    ax.text(b.get_x() + b.get_width() / 2, v + 900, f'{v:,} ({v / len(df) * 100:.1f}%)', ha='center', fontsize=8,
            color=INK)
ax.set_ylabel('Respondents'); ax.set_ylim(0, vc.max() * 1.15); ax.grid(axis='x', visible=False)
save(fig, 'fig03_target.png')

# Fig 4: Age and Checkup by class
ages = ['18-24', '25-29', '30-34', '35-39', '40-44', '45-49', '50-54', '55-59', '60-64', '65-69', '70-74', '75-79', '80+']
chk = ['Never', '5 or more years ago', 'Within the past 5 years', 'Within the past 2 years', 'Within the past year']
fig, axs = plt.subplots(1, 2, figsize=(7.2, 3.0), gridspec_kw={'width_ratios': [1.6, 1]})
for ax, col, order, letter in [(axs[0], 'Age_Category', ages, 'a'), (axs[1], 'Checkup', chk, 'b')]:
    ct = pd.crosstab(df[col], df.y).reindex(order)
    x = np.arange(len(order))
    ax.bar(x, ct[0], color=BLUE, label=CLS[0][0], width=0.8)
    ax.bar(x, ct[1], bottom=ct[0], color=ORANGE, label=CLS[1][0], width=0.8, edgecolor=SURF, linewidth=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(order if col == 'Age_Category' else ['Never', '≥5 years', '<5 years', '<2 years', '<1 year'],
                       rotation=45, ha='right', fontsize=7.5)
    ax.set_ylabel('Respondents'); ax.grid(axis='x', visible=False); panel(ax, letter)
axs[0].legend(fontsize=7.5, loc='upper left')
save(fig, 'fig04_age_checkup.png')

# Fig 5: Weight vs BMI
fig, ax = plt.subplots(figsize=(4.4, 3.4))
s = df.sample(8000, random_state=42)
for k in [0, 1]:
    t = s[s.y == k]
    ax.scatter(t['Weight_(kg)'], t['BMI'], s=4, alpha=0.35, color=CLS[k][1], label=CLS[k][0], edgecolors='none')
r = df[['Weight_(kg)', 'BMI']].corr().iloc[0, 1]
ax.text(0.03, 0.95, f'Pearson r = {r:.2f}', transform=ax.transAxes, fontsize=8, color=INK, va='top')
ax.set_xlabel('Weight (kg)'); ax.set_ylabel('BMI (kg/m²)')
ax.legend(fontsize=7.5, markerscale=3, loc='lower right')
save(fig, 'fig05_weight_bmi.png')
print('weight-bmi r', round(r, 3))

# Fig 6: correlations
fig, axs = plt.subplots(1, 2, figsize=(7.8, 3.4), gridspec_kw={'width_ratios': [1.25, 1], 'wspace': 0.75})
short = ['Height', 'Weight', 'BMI', 'Alcohol', 'Fruit', 'Green veg.', 'Fried potato']
C = df[NUM].corr()
print(C.round(2))
im = axs[0].imshow(C.values, cmap='RdBu_r', vmin=-1, vmax=1)
axs[0].set_xticks(range(7)); axs[0].set_xticklabels(short, rotation=45, ha='right', fontsize=7.5)
axs[0].set_yticks(range(7)); axs[0].set_yticklabels(short, fontsize=7.5); axs[0].grid(False)
for i in range(7):
    for j in range(7):
        axs[0].text(j, i, f'{C.values[i, j]:.2f}', ha='center', va='center', fontsize=6.3,
                    color='white' if abs(C.values[i, j]) > 0.6 else INK)
fig.colorbar(im, ax=axs[0], fraction=0.046, pad=0.03)
panel(axs[0], 'a')
ct = df[NUM].corrwith(df.y).values
order = np.argsort(ct)
axs[1].barh(np.array(short)[order], ct[order], color=[ORANGE if v > 0 else BLUE for v in ct[order]], height=0.6)
axs[1].axvline(0, color=INK2, lw=0.8)
axs[1].set_xlabel('Point-biserial correlation with Poor health'); axs[1].tick_params(axis='y', labelsize=7.5)
axs[1].grid(axis='y', visible=False); panel(axs[1], 'b')
save(fig, 'fig06_correlation.png')
print('corr with target', dict(zip(short, ct.round(3))))

# Fig 7: features by class
cat = {'Exercise': ['No', 'Yes'], 'Heart_Disease': ['No', 'Yes'], 'Skin_Cancer': ['No', 'Yes'],
       'Other_Cancer': ['No', 'Yes'], 'Depression': ['No', 'Yes'], 'Arthritis': ['No', 'Yes'],
       'Smoking_History': ['No', 'Yes'], 'Sex': ['Female', 'Male']}
fig = plt.figure(figsize=(7.4, 8.4))
gs = fig.add_gridspec(3, 4, height_ratios=[1.0, 1.0, 1.35], hspace=0.75, wspace=0.45)
ax = fig.add_subplot(gs[0, :2])
names = list(cat)
pct = np.array([[df[df[c] == v].y.mean() * 100 for v in cat[c]] for c in names])
x = np.arange(len(names))
ax.bar(x - 0.2, pct[:, 0], 0.38, color='#9ec5f4', label='No / Female')
ax.bar(x + 0.2, pct[:, 1], 0.38, color=ORANGE, label='Yes / Male')
ax.set_xticks(x); ax.set_xticklabels([n.replace('_', ' ') for n in names], rotation=45, ha='right', fontsize=7)
ax.set_ylabel('% with Poor health'); ax.axhline(df.y.mean() * 100, color=INK2, lw=0.8, ls='--')
ax.legend(fontsize=6.5, loc='upper left'); ax.grid(axis='x', visible=False); panel(ax, 'a')
for k, (col, order, labels) in enumerate([
        ('Age_Category', ages, ages),
        ('Diabetes', ['No', 'No, pre-diabetes or borderline diabetes', 'Yes, but female told only during pregnancy', 'Yes'],
         ['No', 'Pre-diabetes', 'Gestational', 'Yes'])]):
    ax = fig.add_subplot(gs[0, 2] if k else gs[1, :2])
    p = df.groupby(col).y.mean().reindex(order) * 100
    ax.bar(range(len(order)), p.values, color=ORANGE, width=0.7)
    ax.set_xticks(range(len(order))); ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=7)
    ax.set_ylabel('% with Poor health'); ax.axhline(df.y.mean() * 100, color=INK2, lw=0.8, ls='--')
    ax.grid(axis='x', visible=False); panel(ax, 'c' if k else 'b')
ax = fig.add_subplot(gs[0, 3])
p = df.groupby('Checkup').y.mean().reindex(chk) * 100
ax.bar(range(5), p.values, color=ORANGE, width=0.7)
ax.set_xticks(range(5)); ax.set_xticklabels(['Never', '≥5 y', '<5 y', '<2 y', '<1 y'], rotation=45, ha='right',
                                            fontsize=7)
ax.axhline(df.y.mean() * 100, color=INK2, lw=0.8, ls='--'); ax.grid(axis='x', visible=False); panel(ax, 'd')
ax = fig.add_subplot(gs[1, 2:]); ax.axis('off')
ax.text(0, 0.5, 'Dashed line: overall share of\nPoor health (16.8%).\n\nPanels (a)-(d): categorical features;\n'
        'panel (e): numerical features\n(density by class; skewness in titles).', fontsize=7.5, color=INK2,
        va='center')
sub = gs[2, :].subgridspec(2, 4, hspace=0.9, wspace=0.35)
for k, c in enumerate(NUM):
    ax = fig.add_subplot(sub[k // 4, k % 4])
    hi = df[c].quantile(0.995)
    bins = np.linspace(df[c].min(), hi, 35)
    for cl in [0, 1]:
        v = df.loc[df.y == cl, c]
        ax.hist(v[v <= hi], bins=bins, density=True, histtype='step', lw=1.4,
                color=CLS[cl][1], label=CLS[cl][0])
    ax.set_title(f"{short[k]} (skew {df[c].skew():.2f})", fontsize=7.5, color=INK)
    ax.tick_params(labelsize=6.5); ax.set_yticks([])
    if k == 0:
        panel(ax, 'e')
    if k == 6:
        ax.legend(fontsize=6.5, loc='upper left', bbox_to_anchor=(1.15, 1.0))
save(fig, 'fig07_features_by_class.png')

# Fig 8: test-set performance
tm = pd.concat([pd.read_csv(os.path.join(EX, f'results_{m}', 'test_metrics.csv')) for m in METHODS])
tm = tm[tm.feature_set == 'all']
fig, axs = plt.subplots(1, 3, figsize=(7.4, 3.2), sharey=True)
for ax, (met, title), letter in zip(axs, [('balanced_accuracy', 'Balanced accuracy'), ('sensitivity', 'Sensitivity'),
                                         ('roc_auc', 'ROC-AUC')], 'abc'):
    for i, m in enumerate(METHODS):
        t = tm[tm.method == m].set_index('model').reindex(MODELS)
        y = np.arange(len(MODELS)) + (i - 1.5) * 0.18
        ax.errorbar(t[met], y, xerr=[t[met] - t[f'{met}_lo'], t[f'{met}_hi'] - t[met]], fmt='o', ms=4.5,
                    color=COLOR[m], ecolor=COLOR[m], elinewidth=1, capsize=0, label=LABEL[m],
                    markeredgecolor=SURF, markeredgewidth=0.6)
    ax.set_title(title, fontsize=9, color=INK); ax.grid(axis='y', visible=False); panel(ax, letter)
    if met == 'roc_auc':
        ax.set_xlim(0.78, 0.97)
axs[0].set_yticks(range(len(MODELS))); axs[0].set_yticklabels([MSHORT[m] for m in MODELS]); axs[0].invert_yaxis()
axs[0].legend(fontsize=7, loc='lower center', bbox_to_anchor=(1.7, -0.36), ncol=4)
save(fig, 'fig08_test_performance.png')

# Fig 9: calibration
from sklearn.calibration import calibration_curve
preds = {m: pd.read_csv(os.path.join(EX, f'results_{m}', 'test_predictions.csv')) for m in METHODS}
fig, axs = plt.subplots(1, 3, figsize=(7.6, 3.0), sharey=True)
for ax, model, letter in [(axs[0], 'CatBoost', 'a'), (axs[1], 'LogisticRegression', 'b')]:
    ax.plot([0, 1], [0, 1], ls='--', color=INK2, lw=0.8)
    for m in METHODS:
        fy, fx = calibration_curve(preds[m].y_true, preds[m][f'all|{model}'], n_bins=10, strategy='quantile')
        ax.plot(fx, fy, marker='o', ms=4, lw=2, color=COLOR[m], label=LABEL[m], markeredgecolor=SURF)
    ax.set_title('CatBoost' if model == 'CatBoost' else 'Logistic Regression', fontsize=9, color=INK)
    ax.set_xlabel('Mean predicted probability\nof Poor health'); panel(ax, letter)
axs[0].set_ylabel('Observed share with Poor health'); axs[0].legend(fontsize=7, loc='upper left')
# (c) recalibration of the final model (LightGBM + ADASYN); the ADASYN colour marks the uncalibrated model
rec = pd.read_csv(os.path.join(EX, 'results_recalibration', 'recalibrated_test_predictions.csv'))
ax = axs[2]
ax.plot([0, 1], [0, 1], ls='--', color=INK2, lw=0.8)
for v, lab, style in [('uncalibrated', 'Uncalibrated', dict(color=COLOR['adasyn'], marker='o', ls='-')),
                      ('platt', 'Platt scaling', dict(color=INK, marker='s', ls='-')),
                      ('isotonic', 'Isotonic regression', dict(color=INK, marker='^', ls='--')),
                      ('prior_correction', 'Prior correction', dict(color=INK2, marker='D', ls=':'))]:
    fy, fx = calibration_curve(rec.y_true, rec[f'adasyn|{v}'], n_bins=10, strategy='quantile')
    ax.plot(fx, fy, ms=4, lw=1.8, label=lab, markeredgecolor=SURF, **style)
ax.set_title('LightGBM with ADASYN', fontsize=9, color=INK)
ax.set_xlabel('Mean predicted probability\nof Poor health'); panel(ax, 'c'); ax.legend(fontsize=6.5, loc='lower right')
save(fig, 'fig09_calibration.png')

# Fig 10: optimiser convergence
# final model (LightGBM trained with ADASYN), one panel per seed
fig, axs = plt.subplots(1, 3, figsize=(7.2, 2.6), sharey=True)
OPT = {'GA': ('Genetic algorithm', BLUE), 'RandomSearch': ('Random search', ORANGE), 'BayesianTPE': ('Bayesian (TPE)', AQUA)}
for ax, seed, letter in zip(axs, (42, 43, 44), 'abc'):
    h = pd.read_csv(os.path.join(EX, 'results_tuning_lightgbm_adasyn', f'tuning_history_seed{seed}.csv'))
    for o, (lab, col) in OPT.items():
        t = h[h.optimizer == o]
        ax.step(t.evaluation, t.best_so_far, where='post', color=col, lw=1.8, label=lab)
    ax.set_title(f'({letter}) Seed {seed}', fontsize=8.5, color=INK, loc='left'); ax.set_xlabel('Evaluations', fontsize=7.5)
    ax.tick_params(labelsize=7)
axs[0].set_ylabel('Best validation\nbalanced accuracy', fontsize=7.5)
axs[0].legend(fontsize=7, loc='lower center', bbox_to_anchor=(1.65, -0.55), ncol=3)
fig.subplots_adjust(wspace=0.12)
save(fig, 'fig10_optimisers.png')

# Figs 11-13: SHAP of LightGBM (final model's classifier)
SH = os.path.join(EX, 'results_shap_lightgbm')
SHF = ['Exercise', 'Heart_Disease', 'Skin_Cancer', 'Other_Cancer', 'Depression', 'Arthritis', 'Smoking_History', 'Sex',
       'Checkup', 'Age_Category', 'Diabetes', 'Height_(cm)', 'Weight_(kg)', 'BMI', 'Alcohol_Consumption',
       'Fruit_Consumption', 'Green_Vegetables_Consumption', 'FriedPotato_Consumption']
NICE = {f: f.replace('_Consumption', '').replace('_', ' ') for f in SHF}
NICE['Green_Vegetables_Consumption'] = 'Green vegetables'
NICE['FriedPotato_Consumption'] = 'Fried potato'
shz = {m: np.load(os.path.join(SH, f'shap_values_test_lightgbm_{m}.npz')) for m in METHODS}
from matplotlib.colors import LinearSegmentedColormap
FEATCMAP = LinearSegmentedColormap.from_list('feat', [BLUE, '#c9c8c4', ORANGE])


def beeswarm(ax, z, title, n=3000, seed=42):
    sv, X = z['values'], z['data']
    order = np.argsort(np.abs(sv).mean(0))          # least important at the bottom
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(sv), size=min(n, len(sv)), replace=False)
    for row, j in enumerate(order):
        x = sv[idx, j]
        v = pd.Series(X[idx, j]).rank(pct=True).values  # colour = feature value (low to high) within the feature
        hist, edges = np.histogram(x, bins=60)
        dens = hist[np.clip(np.digitize(x, edges) - 1, 0, len(hist) - 1)] / max(hist.max(), 1)
        y = row + (rng.random(len(x)) - 0.5) * 0.8 * dens
        ax.scatter(x, y, c=v, cmap=FEATCMAP, s=2.2, linewidths=0, rasterized=True)
    ax.set_yticks(range(len(order))); ax.set_yticklabels([NICE[SHF[j]] for j in order], fontsize=7)
    ax.axvline(0, color=INK2, lw=0.7); ax.grid(axis='y', visible=False)
    ax.set_xlabel('SHAP value (log-odds of Poor health)', fontsize=7.5); ax.tick_params(axis='x', labelsize=7)
    ax.set_title(title, fontsize=8.5, color=INK, loc='left')


# Fig 11: summary plots, (a) no oversampling, (b) ADASYN (final model)
fig, axs = plt.subplots(1, 2, figsize=(7.4, 4.6))
beeswarm(axs[0], shz['original'], '(a) No oversampling')
beeswarm(axs[1], shz['adasyn'], '(b) ADASYN (final model)')
fig.subplots_adjust(wspace=0.75, right=0.88)
sm = plt.cm.ScalarMappable(cmap=FEATCMAP); sm.set_array([])
cb = fig.colorbar(sm, cax=fig.add_axes([0.91, 0.25, 0.015, 0.5]), ticks=[0, 1])
cb.ax.set_yticklabels(['Low', 'High'], fontsize=7); cb.set_label('Feature value', fontsize=7.5)
save(fig, 'fig11_shap_summary.png')

# Fig 12: importance rank across methods
imp = pd.DataFrame({m: pd.read_csv(os.path.join(SH, f'shap_importance_test_lightgbm_{m}.csv')).set_index('feature')
                    ['mean_abs_shap'] for m in METHODS})
imp = imp.sort_values('original', ascending=False)
rank = imp.rank(ascending=False).astype(int)
fig, ax = plt.subplots(figsize=(5.6, 5.4))
im = ax.imshow(imp.values, cmap='Blues', aspect='auto')
for i in range(imp.shape[0]):
    for j in range(imp.shape[1]):
        v = imp.values[i, j]
        ax.text(j, i, f'{v:.2f}  (#{rank.values[i, j]})', ha='center', va='center', fontsize=6.8,
                color='white' if v > imp.values.max() * 0.6 else INK)
ax.set_xticks(range(4)); ax.set_xticklabels([LABEL[m] for m in METHODS], fontsize=8)
ax.set_yticks(range(len(imp))); ax.set_yticklabels([NICE[f] for f in imp.index], fontsize=7.5)
ax.grid(False); ax.xaxis.tick_top()
cb = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.03); cb.set_label('Mean |SHAP| (log-odds)', fontsize=7.5)
save(fig, 'fig12_shap_across_methods.png')


# Fig 13: local explanations of the final model and the height dependence
def fmt_value(f, v):
    if f in ('Exercise', 'Heart_Disease', 'Skin_Cancer', 'Other_Cancer', 'Depression', 'Arthritis', 'Smoking_History'):
        return 'yes' if v == 1 else 'no'
    if f == 'Sex':
        return 'male' if v == 1 else 'female'
    return f'{v:g}'


def waterfall(ax, z, i, title, k=8):
    sv, X, base = z['values'][i], z['data'][i], float(z['base'][i])
    order = np.argsort(-np.abs(sv))[:k]
    rest = sv.sum() - sv[order].sum()
    labels = [f'{NICE[SHF[j]]} = {fmt_value(SHF[j], X[j])}' for j in order] + [f'{len(SHF) - k} other features']
    vals = list(sv[order]) + [rest]
    y = np.arange(len(vals))[::-1]
    ax.barh(y, vals, color=[ORANGE if v > 0 else BLUE for v in vals], height=0.62)
    for yy, v in zip(y, vals):
        ax.text(v + (0.03 if v > 0 else -0.03), yy, f'{v:+.2f}', va='center', ha='left' if v > 0 else 'right',
                fontsize=6.5, color=INK)
    ax.set_yticks(y); ax.set_yticklabels(labels, fontsize=6.8); ax.axvline(0, color=INK2, lw=0.7)
    lim = max(np.abs(vals)) * 1.45; ax.set_xlim(-lim, lim); ax.grid(axis='y', visible=False)
    ax.tick_params(axis='x', labelsize=6.8)
    lo = base + sv.sum()
    ax.set_xlabel(f'SHAP value (log-odds)\nbase {base:.2f}, output {lo:.2f} (p = {1 / (1 + np.exp(-lo)):.2f})',
                  fontsize=6.8)
    ax.set_title(title, fontsize=8.5, color=INK, loc='left')


za = shz['adasyn']
yv, pv = za['y'], za['prob']
TP = int(np.where((yv == 1) & (pv >= .5))[0][0]); FN = int(np.where((yv == 1) & (pv < .5))[0][0])
fig = plt.figure(figsize=(7.4, 6.8))
gs = fig.add_gridspec(2, 2, height_ratios=[1, 0.9], hspace=0.65, wspace=1.0)
waterfall(fig.add_subplot(gs[0, 0]), za, TP, '(a) Poor health, identified')
waterfall(fig.add_subplot(gs[0, 1]), za, FN, '(b) Poor health, missed')
ax = fig.add_subplot(gs[1, :])
h = SHF.index('Height_(cm)')
for m, col, lab in [('original', COLOR['original'], 'No oversampling'), ('adasyn', COLOR['adasyn'], 'ADASYN (final model)')]:
    z = shz[m]
    ax.scatter(z['data'][:, h] + np.random.default_rng(1).uniform(-0.4, 0.4, len(z['data'])), z['values'][:, h],
               s=2, color=col, alpha=0.35, linewidths=0, rasterized=True, label=lab)
ax.axhline(0, color=INK2, lw=0.7)
ax.set_xlabel('Height (cm)', fontsize=7.5); ax.set_ylabel('SHAP value of height\n(log-odds)', fontsize=7.5)
ax.tick_params(labelsize=7); ax.set_title('(c) Dependence of the prediction on height', fontsize=8.5, color=INK, loc='left')
leg = ax.legend(fontsize=7, markerscale=4, loc='upper right')
for lh in leg.legend_handles:
    lh.set_alpha(1)
save(fig, 'fig13_local.png')
print('local cases', TP, FN)
