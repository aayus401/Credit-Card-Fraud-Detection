# 💳 Credit Card Fraud Detection using Machine Learning

A comparative study of four classical machine learning algorithms — **Logistic Regression, Decision Tree, K-Nearest Neighbors, and Support Vector Machine** — applied to detecting fraudulent credit card transactions on a highly imbalanced, real-world dataset.

## 📌 Problem Statement

Credit card fraud accounts for billions of dollars in losses every year. This project builds and evaluates multiple ML classifiers that flag fraudulent transactions, while dealing with the core challenge of fraud-detection datasets: **severe class imbalance** (fraudulent transactions make up a tiny fraction of all transactions).

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
| **Logistic Regression** | Random undersampling of the majority class to match the 492 fraud cases (984 balanced rows) | 80/20, stratified | Features scaled with `StandardScaler`; `LogisticRegression(max_iter=1000)`; evaluated with accuracy, `classification_report`, and `confusion_matrix` |
| **Decision Tree** | None — trained on the full imbalanced data | 70/30, stratified | `criterion='entropy'`, `max_depth=8`; evaluated with `classification_report` and `confusion_matrix` |
| **K-Nearest Neighbors** | Random undersampling to a balanced 492/492 subset | 70/30, stratified | Features scaled with `StandardScaler`; optimal *k* selected via an error-rate curve across k = 1–39 |
| **SVM** | `imblearn.RandomUnderSampler` (0.5 sampling ratio) applied to the training fold only — the test fold is left at the real-world class balance | 80/20, stratified | `Time` dropped, `Amount` standardized; `SVC(probability=True)`; evaluated with accuracy, precision, recall, F1, a confusion-matrix heatmap, and ROC-AUC / precision-recall curves |

## 📈 Results

> ⚠️ **Note on accuracy:** Fraud is only 0.17% of the full dataset, so overall accuracy is a misleading metric on unbalanced data — a model that predicts "not fraud" for every transaction would still score ~99.8%. Precision, recall, and F1 on the fraud class are the metrics that actually matter here.

| Model | Test set | Accuracy | Fraud-class Precision | Fraud-class Recall | Fraud-class F1 | ROC-AUC |
|---|---|---|---|---|---|---|
| Logistic Regression (balanced subset) | 197 rows (98 fraud) | 0.929 | 0.97 | 0.89 | 0.93 | — |
| Decision Tree (full imbalanced data) | 85,443 rows (148 fraud) | 0.999 | 0.87 | 0.72 | 0.79 | — |
| K-Nearest Neighbors (balanced subset, k=1) | 296 rows (148 fraud) | 0.91 | 0.94 | 0.87 | 0.91 | — |
| SVM (train undersampled, test left imbalanced) | ~56,962 rows (real-world imbalance) | 0.989 | 0.13 | 0.91 | 0.23 | 0.982 |

The four models are deliberately evaluated under different conditions, which is part of the comparison:

- **Logistic Regression and KNN** are trained and tested on balanced subsets, so precision and recall are both directly meaningful — both models land in the 0.87–0.97 range across the board.
- **Decision Tree** is trained and tested on the full imbalanced data. Accuracy stays near 100% (expected, since 99.8% of rows are legitimate), but recall drops to 0.72 — with far fewer fraud examples to learn from, it misses more of them.
- **SVM** is evaluated the most realistically: trained on an undersampled fold but tested against the actual imbalanced data. Recall is high (0.91 — it catches most fraud) but precision is low (0.13 — undersampling the training data biases it toward flagging fraud too readily, producing many false alarms). The 0.98 ROC-AUC shows the model still ranks fraud vs. non-fraud transactions well; the low precision is a decision-threshold issue rather than a sign the model learned poorly.

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
- Tune the SVM's undersampling ratio and decision threshold to raise precision without giving up much recall
- Explore feature engineering using transaction location/velocity (e.g., flagging geographically implausible transaction sequences)

## 📜 License

MIT — feel free to fork, star, and use in your own portfolio.

## 👤 Author

**Aayush** ([@aayus401](https://github.com/aayus401))

