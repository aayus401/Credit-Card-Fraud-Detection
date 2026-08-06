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
├── code/
│   ├── Logistic Regression.ipynb
│   ├── Decision tree.ipynb
│   ├── K-Nearest Neighbor(KNN).ipynb
│   └── Support Vector Machines.ipynb
├── LICENSE
└── README.md
```

## 🧪 Methodology

Each notebook explores a different algorithm and a different strategy for handling class imbalance:

| Model | Imbalance handling | Train/Test split | Key steps |
|---|---|---|---|
| **Logistic Regression** | Random undersampling of the majority class to match the 492 fraud cases (984 balanced rows), fixed `random_state=42` | 80/20, stratified, `random_state=2` | Features scaled with `StandardScaler` (fit on train fold only); `LogisticRegression(max_iter=1000)`; evaluated with accuracy plus `classification_report`/`confusion_matrix` |
| **Decision Tree** | None (trained on the full imbalanced data) | 70/30, stratified, `random_state=0` | `criterion='entropy'`, `max_depth=8` (pruned to reduce overfitting); evaluated with `classification_report`/`confusion_matrix` |
| **K-Nearest Neighbors** | Random undersampling to a balanced 492/492 subset, fixed `random_state=42` | 70/30, stratified, `random_state=0` | Features scaled with `StandardScaler` **fit on the training fold only**; optimal *k* selected via an error-rate curve across k = 1–39 |
| **SVM** | `imblearn.RandomUnderSampler` (0.5 sampling ratio), applied **only to the training fold** after splitting | 80/20, stratified, `random_state=1` | `Time` dropped, `Amount` standardized; `SVC(probability=True, random_state=2)`; evaluated with accuracy/precision/recall/F1, a confusion-matrix heatmap, and ROC-AUC and precision-recall curves; test set kept at the real-world class balance |

## 📈 Results

> ⚠️ **Note on accuracy:** Because fraud is only 0.17% of the full dataset, overall accuracy is a misleading metric on unbalanced data — a model that predicts "not fraud" for every transaction would still score ~99.8%. The numbers below for Decision Tree are on the full imbalanced test set; Logistic Regression and KNN are evaluated on a balanced held-out split, so their accuracy figures are directly comparable to precision/recall.

| Model | Test set | Accuracy | Fraud-class Precision | Fraud-class Recall | Fraud-class F1 |
|---|---|---|---|---|---|
| Logistic Regression (balanced subset) | 197 rows (98 fraud) | 0.929 | 0.97 | 0.89 | 0.93 |
| Decision Tree (full imbalanced data, pruned) | 85,443 rows (148 fraud) | 0.999 | 0.87 | 0.72 | 0.79 |
| K-Nearest Neighbors (balanced subset, k=1 selected) | 296 rows (148 fraud) | 0.91 | 0.94 | 0.87 | 0.91 |
| SVM (undersampled train fold only) | *re-run notebook* | *re-run notebook* | *re-run notebook* | *re-run notebook* | *re-run notebook* |

Across the three re-run models, Logistic Regression and KNN — both trained on a balanced subset — catch fraud at a noticeably higher rate (recall 0.87–0.89) than the Decision Tree trained on the full imbalanced data (recall 0.72), which is the expected trade-off: training on realistic class proportions keeps overall accuracy near 100%, but the model sees far fewer fraud examples and misses more of them. SVM numbers aren't filled in yet — its evaluation cells were rebuilt after a fix and haven't been executed against `creditcard.csv`.

## 🩹 Known limitations & fixes applied

This project started as a comparative study, not a production system, and the original notebooks had a few real methodological gaps. The following have since been fixed directly in the notebooks:

- **Pre-split leakage (KNN, SVM):** the `StandardScaler` (KNN) and `RandomUnderSampler` (SVM) were previously fit on the *entire* dataset before the train/test split, meaning test-set distribution information leaked into training. Both now fit only on the training fold.
- **No pruning (Decision Tree):** the tree was grown to full depth with no `max_depth`, `min_samples_leaf`, or cost-complexity pruning, and was likely overfitting. `max_depth=8` has been added.
- **No reproducibility (all notebooks):** several `train_test_split` calls and `legit.sample()`/`RandomUnderSampler()` calls had no `random_state`, so results changed on every run. All are now seeded.
- **No stratification:** splits weren't stratified, so the fraud count could shift between train/test by chance. All four notebooks now use `stratify=y`.
- **Incomplete evaluation (Logistic Regression):** only `accuracy_score` was reported, despite this project's own thesis that accuracy is misleading for fraud detection. `classification_report` and `confusion_matrix` have been added and are now printed.

**Still open (not yet addressed):**
- No cross-validation — each notebook still uses a single train/test split, so metrics have no variance estimate.
- No hyperparameter tuning (`GridSearchCV`/`RandomizedSearchCV`) — models are close to library defaults.
- No `sklearn.Pipeline`/`ColumnTransformer` — preprocessing steps are manual rather than chained into a single leakage-safe object.
- No saved model artifact or inference script — this remains a set of training/evaluation notebooks, not a servable system.
- No duplicate-row check (`df.duplicated().sum()`) — this public dataset is known to contain duplicate rows.
- SVM notebook's evaluation cells haven't been re-executed since being rebuilt — its results row above is still blank.

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
   pip install pandas numpy scikit-learn imbalanced-learn seaborn matplotlib
   ```
4. Open any notebook in the `code/` folder and run all cells.

## 🔮 Future Work

- Apply cost-sensitive learning or SMOTE-based oversampling as an alternative to undersampling, which discards a large amount of legitimate-transaction data
- Incorporate ensemble methods (Random Forest, XGBoost) for comparison
- Add cross-validation instead of a single train/test split for more robust metric estimates
- Explore feature engineering using transaction location/velocity (e.g., flagging geographically implausible transaction sequences)

## 📜 License

MIT — feel free to fork, star, and use in your own portfolio.

## 👤 Author

**Aayush** ([@aayus401](https://github.com/aayus401))

