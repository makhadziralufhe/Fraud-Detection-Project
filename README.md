# Fraud Detection under Heavy Class Imbalance

Digital and app fraud is rising fast in South Africa. SABRIC recorded 110,074 digital banking crime incidents in 2025, with client claims of R2.41bn, up 29.2% on 2024. Banking apps made up about 89% of cases. In most of them the app was not hacked: criminals tricked the victim first, then used the app to add beneficiaries and move money.

This project builds and compares fraud detection models on two public datasets where fraud is extremely rare (about 0.1% to 0.2% of transactions). Models are judged on precision, recall and PR-AUC, not accuracy.

## Datasets

| Dataset | Rows | Fraud rate | Source |
|---|---|---|---|
| Credit Card Fraud (ULB) | 284,807 | 0.173% (492 frauds) | Loaded in the notebook with `tf.keras.utils.get_file` |
| PaySim mobile-money transactions | 500,000 sampled (of about 6.3M) | 0.129% (647 frauds in the sample) | Loaded in the notebook with `kagglehub` (`ealaxi/paysim1`) |

Nothing needs to be downloaded by hand. If PaySim cannot be loaded, the notebook says so and continues with the credit card data.

## Repository contents

```
fraud_detection_imbalance.ipynb   the whole project
README.md
```

Running the notebook creates a `results/` folder next to it with the charts and CSVs.

## Getting started

```bash
git clone <your-repo-url>
cd <repo-folder>
pip install tensorflow xgboost imbalanced-learn scikit-learn pandas numpy matplotlib kagglehub jupyter
jupyter notebook fraud_detection_imbalance.ipynb
```

Run all cells. The first code cell installs `imbalanced-learn`, `xgboost` and `kagglehub` if they are missing. If you see an `imblearn` or `jinja2` error, restart the kernel once after the install cell.

Settings at the top of the notebook:

- `RUN_PAYSIM` turns the PaySim part on or off.
- `PAYSIM_ROWS` sets how many PaySim rows to sample (default 500,000, to keep memory low).
- `SEED` is fixed at 42.

## Method

1. **Explore** the imbalance and the transaction types where fraud occurs.
2. **Split chronologically**: 70% train, 15% validation, 15% test. Fraud is predicted forward in time, so a random split would leak the future into training.
3. **Compare five approaches**:
   - Logistic Regression with class weights
   - Logistic Regression with SMOTE
   - Random Forest (balanced)
   - XGBoost (`scale_pos_weight`, early stopping on PR-AUC)
   - Keras MLP (class weights, early stopping on validation PR-AUC)
4. **Evaluate** with PR-AUC (with a bootstrap 95% confidence interval), ROC-AUC, precision, recall, F1, and recall at a 0.5% alert budget (the share of fraud caught if analysts can review only 0.5% of transactions). Accuracy is reported only to show why it misleads.
5. **Explain** model decisions with XGBoost TreeSHAP values.
6. **Ablate** on PaySim: retrain without the balance-reconciliation features to check how much they drive the result.

## Results

Best model per dataset on the held-out test set, from a full run of the notebook:

| Dataset | Best model (by PR-AUC) | PR-AUC | Precision | Recall | Recall at 0.5% alerts |
|---|---|---|---|---|---|
| Credit card | Keras MLP (class weights) | 0.774 | 1.000 | 0.615 | 0.808 |
| PaySim | Random Forest (balanced) | 1.000 | 1.000 | 0.994 | 1.000 |

Main findings:

- **Accuracy is misleading.** Predicting "always legitimate" scores 99.88% accuracy on the credit card data and catches no fraud (recall 0.0).
- **On credit card data the top three models are statistically close.** Keras MLP (0.774), Random Forest (0.772) and XGBoost (0.757) have heavily overlapping confidence intervals (for example 0.670 to 0.867 for the MLP), so no single winner should be claimed. A no-skill model scores about 0.001, so these are strong results.
- **The PaySim score is too good to trust on its own.** PaySim is simulated, and its balance-reconciliation features almost give the fraud away. Without them, XGBoost PR-AUC drops from 1.000 to 0.860. The 0.860 figure is the more realistic one.

## Limitations

- The credit card features are anonymised (PCA components), so the SHAP explanations cannot be mapped to business concepts.
- PaySim is synthetic, so its scores will overstate real-world performance.
- Neither dataset contains the social-engineering stage that drives most real app fraud (victim manipulation, then new beneficiaries). Models here detect anomalous transactions, not the scam itself.
- Results come from a single chronological split. They show how the models rank against each other, not guaranteed production performance.
- Real deployments need ongoing monitoring for drift, since fraud tactics change.

## Requirements

Python 3.10 or later with: tensorflow (2.15+), xgboost (2.0+), imbalanced-learn (0.12+), scikit-learn (1.3+), pandas (2.0+), numpy, matplotlib, kagglehub and jupyter.

## References

- SABRIC annual crime statistics, 2025
- Machine Learning Group, ULB. Credit Card Fraud Detection dataset (Kaggle: `mlg-ulb/creditcardfraud`)
- Lopez-Rojas, E. et al. PaySim: a financial mobile money simulator (Kaggle: `ealaxi/paysim1`)
