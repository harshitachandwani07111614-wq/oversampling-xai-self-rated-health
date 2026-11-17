# Class imbalance, oversampling and explainability for self-rated health prediction

Code for the paper *Explainable AI in Healthcare Informatics for Prediction under Class Imbalance Using Generative
Oversampling* (H. Chandwani, A. Sinha, S. Sabherwal).

The study predicts poor versus excellent self-rated general health from 18 lifestyle, demographic and
chronic-condition variables of the 2021 Behavioral Risk Factor Surveillance System (BRFSS). It compares training
without oversampling and with SMOTENC, ADASYN and CTGAN for six classifiers, and examines decision-threshold
adjustment, recalibration, hyperparameter optimisation and SHAP explanations.

## Data

The data are the *Cardiovascular Diseases Risk Prediction Dataset* on Kaggle:
https://www.kaggle.com/datasets/alphiree/cardiovascular-diseases-risk-prediction-dataset

Download the CSV file (308,854 rows) and save it as `dataset.csv`. The data are not included in this repository.

## Repository structure

```
src/
  pipeline.py        data preparation, oversampling, models, metrics, cross-validation, test evaluation,
                     synthetic-data quality, EDA and sensitivity analyses
  recalibration.py   Platt scaling, isotonic regression and prior correction
  tuning_lgbm.py     genetic algorithm, random search and Bayesian optimisation for LightGBM
  shap_lgbm.py       TreeSHAP explanations of LightGBM and their stability
notebooks/           Google Colab notebooks (run in the order below)
analysis/
  statistics.py      statistical tests, threshold adjustment, calibration error, SHAP stability
  figures.py         figures of the paper
```

Each notebook writes the modules it needs from `src/` with `%%writefile`, so a notebook can be opened in Google
Colab and run on its own.

## Running the experiments

1. Put `dataset.csv` in Google Drive under `MyDrive/brfss_imbalance/` (or upload it to the Colab session).
2. Open a notebook in Google Colab, run the installation cell, restart the session once
   (Runtime > Restart session) and run all cells. Results are written to `MyDrive/brfss_imbalance/results_<name>/`;
   interrupted runs continue where they stopped. Set `QUICK = True` for a short test run.
3. Copy the result folders into one local folder (for example `results/`) and run

   ```
   pip install -r requirements.txt
   python analysis/statistics.py results
   python analysis/figures.py results dataset.csv
   ```

| Notebook | Output | Paper |
|---|---|---|
| `01_eda` | target mapping, association tests | Tables 3 and 5 |
| `02_models_original` | models trained without oversampling | Tables 6, 9, 10, A1, A2; Figs. 8 and 9 |
| `03_models_smotenc` | models trained with SMOTENC, synthetic-data quality | Tables 6, 7, 9, 10, A1, A2 |
| `04_models_adasyn` | models trained with ADASYN, synthetic-data quality | Tables 6, 7, 9, 10, A1, A2 |
| `05_models_ctgan` | models trained with CTGAN, synthetic-data quality (GPU recommended) | Tables 6, 7, 9, 10, A1, A2 |
| `06_sensitivity` | other target definition, model without diagnoses | Table 13 |
| `07_recalibration` | recalibration of LightGBM | Table 11, Fig. 9c |
| `08_tuning_lightgbm_adasyn` | hyperparameter optimisation of the final model | Table 12, Fig. 10 |
| `09_shap_lightgbm` | SHAP values and their stability | Table 14, Figs. 11-13 |
| `analysis/statistics.py` | Friedman, Wilcoxon and McNemar tests, threshold adjustment, ECE | Tables 10 and 11 |
| `analysis/figures.py` | all figures | Figs. 1-13 |

Approximate run times on a free Colab CPU runtime: 5 minutes (01), 1-1.5 hours (02-04, 06), 10 minutes (07),
20-40 minutes per seed (08), 20 minutes (09 without CTGAN). CTGAN (05 and the CTGAN part of 09) takes 1-2 hours on a
GPU runtime.

## Reproducibility

- The data are split once (stratified, 80/20, seed 42). All notebooks print a fingerprint of the test set, which
  should be `f20ed8bc6322`.
- Oversampling is applied to the training data only, and inside every cross-validation fold to the training part of
  the fold.
- The library versions used for the paper are pinned in `requirements.txt` and in the installation cell of each
  notebook. The SMOTENC sample depends on the NumPy version: with NumPy 2.4.6 instead of 2.1.3, the test balanced
  accuracy of LightGBM with SMOTENC is 0.878 instead of 0.880. All other results are unaffected.

## License

MIT, see `LICENSE`.
