# 💳 Credit Card Fraud Detection using Machine Learning

A comparative study of four classical machine learning algorithms — **Logistic Regression, Decision Tree, K-Nearest Neighbors, and Support Vector Machine** — applied to detecting fraudulent credit card transactions on a highly imbalanced, real-world dataset.

## 📌 Problem Statement

Credit card fraud accounts for billions of dollars in losses every year. The goal of this project is to build and evaluate multiple ML classifiers that can flag fraudulent transactions accurately, while dealing with the core challenge of fraud-detection datasets: **severe class imbalance** (fraudulent transactions make up a tiny fraction of all transactions).

## 📊 Dataset

- **Source:** [Kaggle — Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
- **Size:** 284,807 transactions made by European cardholders over two days in September 2013
- **Class distribution:** 492 fraudulent transactions (**0.172%**) vs. 284,315 legitimate transactions — a highly imbalanced dataset
- **Features:** 31 columns total
  - `V1`–`V28`: anonymized principal components from a PCA transformation (original features withheld for confidentiality)
  - `Time`: seconds elapsed since the first transaction in the dataset
  - `Amount`: transaction amount
  - `Class`: target label (`0` = legitimate, `1` = fraud)

## 🗂️ Repository Structure

```
Credit-Card-Fraud-Detection/
├── Credit Card Fraud Detection Using Machine Learning/
│   └── Code/
│       ├── Credit Card Fraud Detection - Logistic Regression.ipynb
│       ├── Credit Card Fraud Detection - Decision Tree.ipynb
│       ├── Credit Card Fraud Detection - K-Nearest Neighbor.ipynb
│       └── Credit Card Fraud Detection - Support Vector Machines.ipynb
├── Presentation/
│   └── Report-Presentation.pptx
├── LICENSE
└── README.md
```

## 🧪 Methodology

Each notebook explores a different algorithm and a different strategy for handling class imbalance:

| Model | Imbalance handling | Train/Test split | Key steps |
|---|---|---|---|
| **Logistic Regression** | Random undersampling of the majority class to match the 492 fraud cases (984 balanced rows), fixed `random_state=42` | 80/20, stratified, `random_state=2` | Baseline linear classifier on the balanced subset; now reports precision/recall/F1/confusion matrix, not just accuracy |
| **Decision Tree** | None (trained on full imbalanced data) | 70/30, stratified, `random_state=0` | `criterion='entropy'`, `max_depth=8` (pruned to reduce overfitting on the full imbalanced dataset); evaluated with precision/recall/F1 |
| **K-Nearest Neighbors** | None (trained on full imbalanced data) | 70/30, stratified, `random_state=0` | Features scaled with `StandardScaler` **fit on the training fold only** (previously fit on the full dataset before splitting); optimal *k* selected via an error-rate curve across k = 1–39 |
| **SVM** | `imblearn.RandomUnderSampler` (0.5 sampling ratio, `random_state=42`), applied **only to the training fold** after splitting | 80/20, stratified | `Time` dropped, `Amount` standardized; evaluated with ROC-AUC and precision-recall curves in addition to accuracy; test set kept at the real-world class balance |

## 📈 Results

> ⚠️ **Note on accuracy:** Because fraud is only 0.17% of the dataset, overall accuracy is a misleading metric on the full (imbalanced) data — a model that predicts "not fraud" for every transaction would still score ~99.8%. Precision, recall, and F1-score on the **fraud class specifically** are more meaningful indicators of real-world performance.

> 📌 **Note on the numbers below:** the fixes described in [Known Limitations](#-known-limitations--fixes-applied) (stratified splits, fixed random seeds, leakage-safe scaling/undersampling, tree pruning) changed how each model is trained, so the metrics from the original version of this project are no longer accurate. Re-run each notebook against `creditcard.csv` to regenerate this table with current numbers.

| Model | Accuracy | Fraud-class Precision | Fraud-class Recall | Fraud-class F1 |
|---|---|---|---|---|
| Logistic Regression (balanced subset) | *re-run notebook* | *re-run notebook* | *re-run notebook* | *re-run notebook* |
| Decision Tree (full imbalanced data, pruned) | *re-run notebook* | *re-run notebook* | *re-run notebook* | *re-run notebook* |
| K-Nearest Neighbors (full imbalanced data) | *re-run notebook* | *re-run notebook* | *re-run notebook* | *re-run notebook* |
| SVM (undersampled train fold only) | *re-run notebook* | *re-run notebook* | *re-run notebook* | *re-run notebook* |

Across all four models, SVM and Decision Tree previously offered the best balance between catching fraud (recall) and minimizing false alarms (precision), while Logistic Regression on the undersampled data was a strong, interpretable baseline — expect broadly similar rankings after re-running, though exact numbers will shift slightly now that splits are stratified/seeded and the Decision Tree is pruned.

## 🩹 Known limitations & fixes applied

This project started as a comparative study, not a production system, and the original notebooks had a few real methodological gaps. The following have since been fixed directly in the notebooks:

- **Pre-split leakage (KNN, SVM):** the `StandardScaler` (KNN) and `RandomUnderSampler` (SVM) were previously fit on the *entire* dataset before the train/test split, meaning test-set distribution information leaked into training. Both now fit only on the training fold.
- **No pruning (Decision Tree):** the tree was grown to full depth with no `max_depth`, `min_samples_leaf`, or cost-complexity pruning, and was likely overfitting. `max_depth=8` has been added.
- **No reproducibility (all notebooks):** several `train_test_split` calls and `legit.sample()`/`RandomUnderSampler()` calls had no `random_state`, so results changed on every run. All are now seeded.
- **No stratification on the full imbalanced data (Decision Tree, KNN):** splits on the full 284,807-row dataset weren't stratified, so the already-tiny fraud count could shift between train/test by chance. Both now use `stratify=y`.
- **Incomplete evaluation (Logistic Regression):** only `accuracy_score` was reported, despite this project's own thesis that accuracy is misleading for fraud detection. `classification_report` and `confusion_matrix` have been added.

**Still open (not yet addressed):**
- No cross-validation — each notebook still uses a single train/test split, so metrics have no variance estimate.
- No hyperparameter tuning (`GridSearchCV`/`RandomizedSearchCV`) — models are close to library defaults.
- No `sklearn.Pipeline`/`ColumnTransformer` — preprocessing steps are manual rather than chained into a single leakage-safe object.
- No saved model artifact or inference script — this remains a set of training/evaluation notebooks, not a servable system.
- No duplicate-row check (`df.duplicated().sum()`) — this public dataset is known to contain ~1,081 duplicate rows.

## 🛠️ Tech Stack

- **Language:** Python
- **Libraries:** pandas, NumPy, scikit-learn, imbalanced-learn (`imblearn`), seaborn, matplotlib

## 🚀 How to Run

1. Clone the repository:
   ```bash
   git clone https://github.com/aayus401/Credit-Card-Fraud-Detection.git
   cd Credit-Card-Fraud-Detection
   ```
2. Download `creditcard.csv` from the [Kaggle dataset page](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) and place it in the same folder as the notebooks.
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Open any notebook in the `Code/` folder and run all cells.

## 🔮 Future Work

- Apply cost-sensitive learning or SMOTE-based oversampling as an alternative to undersampling, which discards a large amount of legitimate-transaction data
- Incorporate ensemble methods (Random Forest, XGBoost) for comparison
- Add cross-validation instead of a single train/test split for more robust metric estimates
- Explore feature engineering using transaction location/velocity (e.g., flagging geographically implausible transaction sequences)

## 📜 License

MIT — feel free to fork, star, and use in your own portfolio.

## 👤 Author

**Aayush** ([@aayus401](https://github.com/aayus401))

